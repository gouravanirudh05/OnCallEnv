"""Baseline inference script for OnCallEnv."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI

from env.env import OnCallEnv
from env.models import Action, ActionType


DEFAULT_SEEDS = {1: 42, 2: 123, 3: 7}
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class EpisodeResult:
    task_id: int
    rewards: List[float]
    done: bool
    error: Optional[str]

    @property
    def score(self) -> float:
        if not self.rewards:
            return 0.0
        return max(0.0, min(1.0, sum(self.rewards) / max(1, len(self.rewards))))


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_reward(value: float) -> str:
    return f"{value:.2f}"


def _render_action(action: Action) -> str:
    payload = action.model_dump()
    return json.dumps(payload, sort_keys=True)


def _log_start(task_name: str, model: str) -> None:
    print(f"[START] task={task_name} env=oncall-env model={model}")


def _log_step(
    step: int, action: Action, reward: float, done: bool, error: Optional[str]
) -> None:
    error_field = error if error else "null"
    print(
        "[STEP]  "
        f"step={step} "
        f"action={_render_action(action)} "
        f"reward={_format_reward(reward)} "
        f"done={_format_bool(done)} "
        f"error={error_field}"
    )


def _log_end(result: EpisodeResult) -> None:
    rewards_str = ",".join(_format_reward(val) for val in result.rewards)
    success = result.error is None
    print(
        "[END]   "
        f"success={_format_bool(success)} "
        f"steps={len(result.rewards)} "
        f"score={result.score:.3f} "
        f"rewards={rewards_str}"
    )


def _build_prompt(observation) -> str:
    payload = observation.model_dump()
    action_schema = Action.model_json_schema()
    return (
        "You are an on-call SRE assistant. "
        "Return exactly one JSON object that matches the Action schema. "
        "No extra text.\n\n"
        f"Observation:\n{json.dumps(payload, sort_keys=True)}\n\n"
        f"Action schema:\n{json.dumps(action_schema, sort_keys=True)}"
    )


def _parse_action(raw: str) -> Action:
    return Action.model_validate_json(raw)


def _fallback_action() -> Action:
    return Action(action_type=ActionType.HOLD, target_id="noop")


def run_episode(task_id: int, seed: int, client: OpenAI, model: str) -> EpisodeResult:
    env = OnCallEnv()
    observation = env.reset(task_id=task_id, seed=seed)
    task_name = observation.context.get("task", f"task-{task_id}")

    _log_start(task_name, model)
    rewards: List[float] = []
    error: Optional[str] = None
    done = False
    max_steps = observation.budget_remaining

    try:
        while not done and len(rewards) < max_steps:
            prompt = _build_prompt(observation)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            parse_error: Optional[str] = None
            try:
                action = _parse_action(content)
            except Exception as exc:
                parse_error = str(exc)
                action = _fallback_action()
            observation, reward, done, info = env.step(action)
            rewards.append(reward)
            error = info.get("error") if not info.get("valid", True) else None
            if parse_error:
                error = parse_error
            _log_step(
                env.state().get("step", len(rewards)), action, reward, done, error
            )
    except Exception as exc:  # pragma: no cover
        error = str(exc)

    result = EpisodeResult(task_id=task_id, rewards=rewards, done=done, error=error)
    _log_end(result)
    return result


def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("API_BASE_URL", DEFAULT_BASE_URL)
    model = os.getenv("MODEL_NAME", DEFAULT_MODEL)
    _ = os.getenv("HF_TOKEN")

    if not api_key:
        print("OPENAI_API_KEY is required", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key, base_url=base_url)

    for task_id, seed in DEFAULT_SEEDS.items():
        run_episode(task_id=task_id, seed=seed, client=client, model=model)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
