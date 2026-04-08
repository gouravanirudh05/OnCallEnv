"""Baseline inference script for OnCallEnv (OpenEnv hackathon spec)."""

from __future__ import annotations

import json
import os
from typing import List, Optional
from openai import OpenAI

from env.env import OnCallEnv
from env.grader import grade_task
from env.models import Action, ActionType, EventLabel, Severity

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

DEFAULT_SEEDS = {1: 42, 2: 123, 3: 7}
TASK_IDS = (1, 2, 3)
MAX_STEPS = {1: 20, 2: 60, 3: 40}
TEMPERATURE = 0.2
MAX_TOKENS = 220


def _log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}")


def _log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = "true" if done else "false"
    print(
        "[STEP]  "
        f"step={step} "
        f"action={action} "
        f"reward={reward:.2f} "
        f"done={done_val} "
        f"error={error_val}"
    )


def _log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    success_val = "true" if success else "false"
    print(
        "[END]   "
        f"success={success_val} "
        f"steps={steps} "
        f"score={score:.2f} "
        f"rewards={rewards_str}"
    )


def _safe_json_loads(raw: str) -> dict:
    raw = raw.strip()
    if not raw:
        raise ValueError("empty response")
    if raw.startswith("```"):
        raw = raw.strip("`\n ")
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(raw[start : end + 1])


def _render_action(action: Action) -> str:
    return json.dumps(action.model_dump(), sort_keys=True)


def _build_prompt(observation) -> str:
    task = observation.context.get("task", "unknown")
    instructions = ""
    if task == "severity_classification":
        instructions = (
            'Pick one alert and return a JSON action with action_type="classify_alert" '
            "and severity in [P0,P1,P2,P3]."
        )
    elif task == "alert_storm":
        instructions = (
            "Choose one action: label_event, silence_alert, or escalate. "
            "Label root_cause/symptom/false_positive/unrelated first; then silence false positives; "
            "escalate to the correct team with severity P0."
        )
    elif task == "timeline_labelling":
        instructions = (
            "Choose one action: investigate or label_event. Use investigate once per id. "
            "Label events as root_cause/symptom/contributing_factor/noise/unrelated/false_positive."
        )
    return (
        "You are an on-call SRE agent. Return exactly ONE JSON object describing the action.\n"
        f"Task: {task}\n"
        f"Instructions: {instructions}\n"
        "Observation JSON:\n"
        f"{json.dumps(observation.model_dump(), indent=2)}\n"
        "Return ONLY JSON."
    )


# ── JSON / action parsing ─────────────────────────────────────────────────────


def _fallback_action(observation) -> Action:
    task = observation.context.get("task", "unknown")
    target_id = observation.alerts[0].id if observation.alerts else "noop"

    if task == "severity_classification":
        return Action(
            action_type=ActionType.CLASSIFY_ALERT,
            target_id=target_id,
            severity=Severity.P2,
        )
    if task == "alert_storm":
        return Action(
            action_type=ActionType.LABEL_EVENT,
            target_id=target_id,
            event_label=EventLabel.UNRELATED,
        )
    if task == "timeline_labelling":
        return Action(
            action_type=ActionType.LABEL_EVENT,
            target_id=target_id,
            event_label=EventLabel.SYMPTOM,
        )
    return Action(action_type=ActionType.HOLD, target_id=target_id)


def _task_metadata(env: OnCallEnv, observation) -> dict:
    state = env.state()
    ground_truth = state.get("ground_truth", {})
    classified = state.get("classified_alerts", {})
    labelled = state.get("labelled_events", {})

    alert_ids = [alert.id for alert in observation.alerts]
    labelable_event_ids = list(ground_truth.get("event_labels", {}).keys())
    team_names = sorted({node.team for node in observation.service_graph})

    return {
        "task": observation.context.get("task", "unknown"),
        "alert_ids": alert_ids,
        "unclassified_alert_ids": [
            alert_id for alert_id in alert_ids if alert_id not in classified
        ],
        "labelable_event_ids": labelable_event_ids,
        "unlabelled_event_ids": [
            event_id for event_id in labelable_event_ids if event_id not in labelled
        ],
        "silenced_alert_ids": list(state.get("silenced_alerts", [])),
        "labelled_event_ids": list(labelled.keys()),
        "team_names": team_names,
    }


def _fallback_task2(observation, metadata: dict) -> Action:
    labelled = set(metadata["labelled_event_ids"])
    root_service = observation.context.get("root_cause_service")
    if root_service:
        root_alert = next(
            (alert for alert in observation.alerts if alert.service == root_service),
            None,
        )
        if root_alert and root_alert.id not in labelled:
            return Action(
                action_type=ActionType.LABEL_EVENT,
                target_id=root_alert.id,
                event_label=EventLabel.ROOT_CAUSE,
            )

    for alert in observation.alerts:
        if (
            alert.id not in labelled
            and alert.labels.get("signature") != "known_false_positive"
            and alert.id.startswith("alert-symptom-")
        ):
            return Action(
                action_type=ActionType.LABEL_EVENT,
                target_id=alert.id,
                event_label=EventLabel.SYMPTOM,
            )

    for alert in observation.alerts:
        if (
            alert.id not in labelled
            and alert.labels.get("signature") == "known_false_positive"
        ):
            return Action(
                action_type=ActionType.LABEL_EVENT,
                target_id=alert.id,
                event_label=EventLabel.FALSE_POSITIVE,
            )

    for alert in observation.alerts:
        if alert.id not in labelled:
            return Action(
                action_type=ActionType.LABEL_EVENT,
                target_id=alert.id,
                event_label=EventLabel.UNRELATED,
            )

    silenced = set(metadata["silenced_alert_ids"])
    for alert in observation.alerts:
        if (
            alert.id not in silenced
            and alert.labels.get("signature") == "known_false_positive"
        ):
            return Action(action_type=ActionType.SILENCE_ALERT, target_id=alert.id)

    if root_service:
        owner = next(
            (
                node.team
                for node in observation.service_graph
                if node.name == root_service
            ),
            None,
        )
        if owner:
            return Action(
                action_type=ActionType.ESCALATE,
                target_id=_fallback_action(observation).target_id,
                team=owner,
                severity=Severity.P0,
            )

    return Action(
        action_type=ActionType.HOLD, target_id=_fallback_action(observation).target_id
    )


def _get_action_from_model(client: OpenAI, observation) -> Action:
    prompt = _build_prompt(observation)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a careful SRE agent."},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    content = (response.choices[0].message.content or "").strip()
    data = _safe_json_loads(content)
    return Action.model_validate(data)


def _run_task(task_id: int, seed: int, client: OpenAI) -> List[float]:
    env = OnCallEnv()
    observation = env.reset(task_id=task_id, seed=seed)
    task_name = observation.context.get("task", f"task-{task_id}")
    _log_start(task=task_name, env_name="oncall-env", model=MODEL_NAME)

    rewards: List[float] = []
    done = False
    last_error: Optional[str] = None

    try:
        for step in range(1, MAX_STEPS[task_id] + 1):
            if done:
                break
            try:
                action = _get_action_from_model(client, observation)
            except Exception:
                action = _fallback_action(observation)

            observation, reward, done, info = env.step(action)
            last_error = info.get("error") if not info.get("valid", True) else None
            rewards.append(reward)
            _log_step(
                step=step,
                action=_render_action(action),
                reward=reward,
                done=done,
                error=last_error,
            )
    finally:
        score = 0.0
        if env._state is not None:
            score = grade_task(env._state)
        success = last_error is None
        _log_end(success=success, steps=len(rewards), score=score, rewards=rewards)

    return rewards


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> int:
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN is required for inference.")
    if LOCAL_IMAGE_NAME is None:
        raise SystemExit("LOCAL_IMAGE_NAME must be set (can be empty string).")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    for task_id in TASK_IDS:
        seed = DEFAULT_SEEDS[task_id]
        _run_task(task_id=task_id, seed=seed, client=client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
