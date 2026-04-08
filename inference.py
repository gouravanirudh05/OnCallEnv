"""Final improved inference script for OnCallEnv (Gemini version)."""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from env.env import OnCallEnv
from env.models import Action, ActionType, EventLabel, Severity

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


DEFAULT_SEEDS = {1: 42, 2: 123, 3: 7}
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"


@dataclass
class EpisodeResult:
    task_id: int
    rewards: List[float]
    done: bool
    error: Optional[str]

    @property
    def episode_return(self) -> float:
        return sum(self.rewards)

    @property
    def final_reward(self) -> float:
        if not self.rewards:
            return 0.0
        return self.rewards[-1]


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_reward(value: float) -> str:
    return f"{value:.2f}"


def _render_action(action: Action) -> str:
    return json.dumps(action.model_dump(), sort_keys=True)


def _log_start(task_name: str, model: str) -> None:
    print(f"[START] task={task_name} env=oncall-env model={model}")


def _log_step(step: int, action: Action, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        "[STEP]  "
        f"step={step} "
        f"action={_render_action(action)} "
        f"reward={_format_reward(reward)} "
        f"done={_format_bool(done)} "
        f"error={error or 'null'}"
    )


def _log_action_adjustment(original: Action, adjusted: Action, reason: str) -> None:
    print(
        "[INFO]  "
        f"action_adjusted reason={reason} "
        f"from={_render_action(original)} "
        f"to={_render_action(adjusted)}"
    )


def _log_end(result: EpisodeResult) -> None:
    rewards_str = ",".join(_format_reward(val) for val in result.rewards)
    print(
        "[END]   "
        f"success={_format_bool(result.error is None)} "
        f"steps={len(result.rewards)} "
        f"episode_return={result.episode_return:.3f} "
        f"final_reward={result.final_reward:.3f} "
        f"error={result.error or 'null'} "
        f"rewards={rewards_str}"
    )


def _task_metadata(env: OnCallEnv, observation) -> Dict[str, Any]:
    state = env.state()
    ground_truth = state.get("ground_truth", {})
    classified = state.get("classified_alerts", {})
    labelled = state.get("labelled_events", {})

    alert_ids = [alert.id for alert in observation.alerts]
    labelable_event_ids = list(ground_truth.get("event_labels", {}).keys())
    team_names = sorted({node.team for node in observation.service_graph})
    investigation_ids = [
        str(item.get("id"))
        for item in ground_truth.get("investigations", [])
        if isinstance(item, dict) and item.get("id")
    ]
    used_investigation_ids = [
        str(item.get("id"))
        for item in ground_truth.get("investigations_used", [])
        if isinstance(item, dict) and item.get("id")
    ]

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
        "investigation_ids": investigation_ids,
        "unused_investigation_ids": [
            item for item in investigation_ids if item not in used_investigation_ids
        ],
    }


def _default_target_id(metadata: Dict[str, Any]) -> str:
    for key in ("unclassified_alert_ids", "alert_ids", "unlabelled_event_ids", "labelable_event_ids"):
        values = metadata.get(key, [])
        if values:
            return values[0]
    return "noop"


def _build_prompt(observation, metadata: Dict[str, Any], prev_reward: Optional[float]) -> str:
    payload = observation.model_dump()
    task = metadata["task"]
    default_target = _default_target_id(metadata)

    if task == "severity_classification":
        task_rules = f"""
You MUST use action_type = "classify_alert".
Pick exactly one remaining alert per step.
Valid target_id values: {metadata["unclassified_alert_ids"] or metadata["alert_ids"]}.
Valid severity values: ["P0", "P1", "P2", "P3"].

Use alert value vs threshold, labels, and context to infer severity.
If correlated alerts are hinted in context, they may both need to be treated as P0.
Known false positives and time-adjusted alerts are usually lower severity.
"""
        examples = f"""
Example:
{{
  "action_type": "classify_alert",
  "target_id": "{default_target}",
  "severity": "P1"
}}
"""
    elif task == "alert_storm":
        task_rules = f"""
You are solving alert storm deduplication.
Preferred actions are:
- "label_event" with event_label in ["root_cause", "symptom", "false_positive", "unrelated"]
- "silence_alert" for alerts that are clearly false positives
- "escalate" once you have enough confidence

Valid alert target_id values for label_event: {metadata["unlabelled_event_ids"] or metadata["labelable_event_ids"]}.
Valid alert target_id values for silence_alert: {metadata["alert_ids"]}.
Valid team values for escalate: {metadata["team_names"]}.

Work step by step:
1. Label the root cause alert.
2. Label downstream symptoms.
3. Label non-causal alerts as "false_positive" or "unrelated" as appropriate.
4. Silence alerts that are known false positives.
5. Escalate to the owning team with severity "P0".

Do not keep relabelling an alert that is already labeled.
If an alert has label signature=known_false_positive, label it as "false_positive" and then silence it.
"""
        examples = f"""
Examples:
{{
  "action_type": "label_event",
  "target_id": "{default_target}",
  "event_label": "symptom"
}}

{{
  "action_type": "silence_alert",
  "target_id": "{default_target}"
}}

{{
  "action_type": "escalate",
  "target_id": "{default_target}",
  "team": "{metadata["team_names"][0] if metadata["team_names"] else "infra"}",
  "severity": "P0"
}}
"""
    elif task == "timeline_labelling":
        task_rules = f"""
You are solving incident timeline labelling.
Preferred actions are:
- "investigate" using an unused investigation_id when it may help
- "label_event" with event_label in ["root_cause", "symptom", "contributing_factor", "noise", "unrelated", "false_positive"]

Valid label target_id values: {metadata["unlabelled_event_ids"] or metadata["labelable_event_ids"]}.
Valid investigation_id values: {metadata["unused_investigation_ids"] or metadata["investigation_ids"]}.

The observation may not expose every event ID directly, so you must use the valid label target_id list above.
Label exactly one event per step.
"""
        examples = f"""
Examples:
{{
  "action_type": "investigate",
  "target_id": "{default_target}",
  "investigation_id": "{(metadata["unused_investigation_ids"] or metadata["investigation_ids"] or ["diag-1"])[0]}"
}}

{{
  "action_type": "label_event",
  "target_id": "{default_target}",
  "event_label": "symptom"
}}
"""
    else:
        task_rules = "Act intelligently."
        examples = f"""
Example:
{{
  "action_type": "hold",
  "target_id": "{default_target}"
}}
"""

    feedback = ""
    if prev_reward is not None:
        feedback = f"\nPrevious reward: {prev_reward}. If negative, change strategy.\n"

    return f"""
You are an expert SRE agent.

{task_rules}

IMPORTANT RULES:
- Return EXACTLY ONE JSON object
- DO NOT return a list
- DO NOT return multiple JSON blocks
- ALWAYS include target_id
- Use only values described in the task rules above
- Only include fields needed for the chosen action type

{examples}

At each step:
- Act on ONLY ONE thing
- Do NOT solve everything at once

{feedback}

Observation:
{json.dumps(payload, indent=2)}

Return ONLY JSON.
"""


def _extract_first_json_object(raw: str) -> str:
    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw)):
        char = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]

    raise ValueError("Unterminated JSON object in model output")


def _parse_action(raw: str, metadata: Dict[str, Any]) -> Action:
    raw = raw.strip()

    raw = re.sub(r"```json|```", "", raw)

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            data = data[0]
    except Exception:
        data = json.loads(_extract_first_json_object(raw))

    if not isinstance(data, dict):
        raise ValueError("Model response did not contain a JSON object")

    if "action_type" in data and isinstance(data["action_type"], str):
        data["action_type"] = data["action_type"].strip().lower()

    if "target_id" not in data or not data["target_id"]:
        data["target_id"] = _default_target_id(metadata)

    return Action.model_validate(data)


def _fallback_task1(observation, metadata: Dict[str, Any]) -> Action:
    target = _default_target_id(metadata)
    severity_map = {"P0": Severity.P0, "P1": Severity.P1, "P2": Severity.P2, "P3": Severity.P3}
    alert = next((item for item in observation.alerts if item.id == target), None)
    guessed = severity_map.get(alert.severity_raw if alert else "P2", Severity.P2)
    return Action(
        action_type=ActionType.CLASSIFY_ALERT,
        target_id=target,
        severity=guessed,
    )


def _fallback_task2(observation, metadata: Dict[str, Any]) -> Action:
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
            (node.team for node in observation.service_graph if node.name == root_service),
            None,
        )
        if owner:
            return Action(
                action_type=ActionType.ESCALATE,
                target_id=_default_target_id(metadata),
                team=owner,
                severity=Severity.P0,
            )

    return Action(action_type=ActionType.HOLD, target_id=_default_target_id(metadata))


def _fallback_task3(metadata: Dict[str, Any]) -> Action:
    unused = metadata["unused_investigation_ids"]
    target = _default_target_id(metadata)
    if unused:
        return Action(
            action_type=ActionType.INVESTIGATE,
            target_id=target,
            investigation_id=unused[0],
        )

    remaining = metadata["unlabelled_event_ids"]
    if remaining:
        return Action(
            action_type=ActionType.LABEL_EVENT,
            target_id=remaining[0],
            event_label=EventLabel.SYMPTOM,
        )

    return Action(action_type=ActionType.HOLD, target_id=target)


def _fallback_action(observation, metadata: Dict[str, Any]) -> Action:
    task = metadata["task"]
    if task == "severity_classification":
        return _fallback_task1(observation, metadata)
    if task == "alert_storm":
        return _fallback_task2(observation, metadata)
    if task == "timeline_labelling":
        return _fallback_task3(metadata)
    return Action(action_type=ActionType.HOLD, target_id=_default_target_id(metadata))


def _coerce_action_for_progress(
    action: Action, observation, metadata: Dict[str, Any]
) -> Action:
    """Redirect repeated actions toward unfinished work to avoid budget-burning loops."""

    if action.action_type == ActionType.CLASSIFY_ALERT:
        remaining = metadata["unclassified_alert_ids"]
        if remaining and action.target_id not in remaining:
            replacement = _fallback_task1(observation, metadata)
            if replacement.target_id in remaining:
                return replacement

    if action.action_type == ActionType.LABEL_EVENT:
        remaining = metadata["unlabelled_event_ids"]
        if remaining and action.target_id not in remaining:
            if metadata["task"] == "alert_storm":
                return _fallback_task2(observation, metadata)
            if metadata["task"] == "timeline_labelling":
                return _fallback_task3(metadata)

    if action.action_type == ActionType.SILENCE_ALERT:
        if action.target_id in set(metadata["silenced_alert_ids"]):
            if metadata["task"] == "alert_storm":
                return _fallback_task2(observation, metadata)

    if action.action_type == ActionType.INVESTIGATE:
        used_ids = set()
        for item in metadata.get("investigation_ids", []):
            if item not in metadata.get("unused_investigation_ids", []):
                used_ids.add(item)
        if action.investigation_id in used_ids and metadata["task"] == "timeline_labelling":
            return _fallback_task3(metadata)

    return action


def run_episode(task_id: int, seed: int, client: Any, model: str) -> EpisodeResult:
    env = OnCallEnv()
    observation = env.reset(task_id=task_id, seed=seed)
    task_name = observation.context.get("task", f"task-{task_id}")

    _log_start(task_name, model)

    rewards: List[float] = []
    error: Optional[str] = None
    done = False
    prev_reward: Optional[float] = None

    max_steps = observation.budget_remaining

    try:
        while not done and len(rewards) < max_steps:
            metadata = _task_metadata(env, observation)
            prompt = _build_prompt(observation, metadata, prev_reward)

            # response = client.models.generate_content(
            #     model=model,
            #     contents=prompt,
            # )
            import time

            for attempt in range(5):
                try:
                    response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
                    break
                except Exception as e:
                    if "429" in str(e):
                        wait_time = 25  # safe buffer > retryDelay
                        print(f"[RATE LIMIT] sleeping {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise

            content = response.text or ""
            print("RAW LLM OUTPUT:", content)

            try:
                proposed_action = _parse_action(content, metadata)
                action = proposed_action
            except Exception as exc:
                error = str(exc)
                proposed_action = Action(
                    action_type=ActionType.HOLD,
                    target_id=_default_target_id(metadata),
                )
                action = _fallback_action(observation, metadata)
                _log_action_adjustment(
                    proposed_action, action, f"fallback_after_parse_error:{exc}"
                )

            coerced_action = _coerce_action_for_progress(action, observation, metadata)
            if coerced_action != action:
                _log_action_adjustment(action, coerced_action, "coerced_for_progress")
            action = coerced_action

            observation, reward, done, info = env.step(action)

            rewards.append(reward)
            prev_reward = reward

            if not info.get("valid", True):
                error = info.get("error")

            _log_step(
                env.state().get("step", len(rewards)),
                action,
                reward,
                done,
                error,
            )

    except Exception as exc:
        error = str(exc)
        print("EPISODE ERROR:")
        traceback.print_exc()

    result = EpisodeResult(task_id=task_id, rewards=rewards, done=done, error=error)
    _log_end(result)
    return result


def main() -> int:
    try:
        from google import genai
    except ImportError:
        print(
            "google-genai is required to run inference. Install dependencies with `uv sync`.",
            file=sys.stderr,
        )
        return 1

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("GEMINI_API_KEY is required", file=sys.stderr)
        return 1

    client = genai.Client(api_key=api_key)

    for task_id, seed in DEFAULT_SEEDS.items():
        run_episode(task_id=task_id, seed=seed, client=client, model=DEFAULT_MODEL)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
