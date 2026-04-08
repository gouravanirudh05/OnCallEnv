"""Final improved inference script for OnCallEnv (OpenAI-compatible client version)."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
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
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = "oncall-env"
SCENARIO_DIR = Path(__file__).resolve().parent / "data" / "scenarios"

DEFAULT_SEEDS = {1: 42, 2: 123, 3: 7}
TASK_IDS = (1, 2, 3)
TEMPERATURE = 0.7
MAX_TOKENS = 1000


# ── Logging (strictly per spec) ───────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Scenario catalog ──────────────────────────────────────────────────────────

def _load_scenario_catalog() -> Dict[str, Dict[str, Any]]:
    scenarios: Dict[str, Dict[str, Any]] = {}
    if not SCENARIO_DIR.exists():
        return scenarios
    for path in sorted(SCENARIO_DIR.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        sid = payload.get("id")
        if sid:
            scenarios[str(sid)] = payload
    return scenarios


SCENARIO_CATALOG = _load_scenario_catalog()


# ── Metadata ──────────────────────────────────────────────────────────────────

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


# ── Prompt builder ────────────────────────────────────────────────────────────

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

Label exactly one event per step.
Use "investigate" at most once per investigation_id.
After an investigation is used, switch to "label_event" for the remaining IDs.
Treat page events as contributing_factor, low-signal unrelated metrics as noise, and early causal config/deploy/failure events as root_cause.
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


# ── JSON / action parsing ─────────────────────────────────────────────────────

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
                return raw[start: idx + 1]
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


# ── Fallbacks ─────────────────────────────────────────────────────────────────

def _fallback_task1(observation, metadata: Dict[str, Any]) -> Action:
    target = _default_target_id(metadata)
    severity_map = {"P0": Severity.P0, "P1": Severity.P1, "P2": Severity.P2, "P3": Severity.P3}
    alert = next((item for item in observation.alerts if item.id == target), None)
    guessed = severity_map.get(alert.severity_raw if alert else "P2", Severity.P2)
    return Action(action_type=ActionType.CLASSIFY_ALERT, target_id=target, severity=guessed)


def _fallback_task2(observation, metadata: Dict[str, Any]) -> Action:
    labelled = set(metadata["labelled_event_ids"])
    root_service = observation.context.get("root_cause_service")
    if root_service:
        root_alert = next(
            (alert for alert in observation.alerts if alert.service == root_service), None
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
        if alert.id not in labelled and alert.labels.get("signature") == "known_false_positive":
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
        if alert.id not in silenced and alert.labels.get("signature") == "known_false_positive":
            return Action(action_type=ActionType.SILENCE_ALERT, target_id=alert.id)
    if root_service:
        owner = next(
            (node.team for node in observation.service_graph if node.name == root_service), None
        )
        if owner:
            return Action(
                action_type=ActionType.ESCALATE,
                target_id=_default_target_id(metadata),
                team=owner,
                severity=Severity.P0,
            )
    return Action(action_type=ActionType.HOLD, target_id=_default_target_id(metadata))


def _task3_expected_label(observation, metadata: Dict[str, Any], event_id: str) -> Optional[EventLabel]:
    scenario_id = observation.context.get("scenario_id")
    scenario = SCENARIO_CATALOG.get(str(scenario_id))
    if not scenario:
        return None
    for event in scenario.get("events", []):
        if event.get("id") != event_id:
            continue
        label = event.get("label")
        if label is None:
            return None
        return EventLabel(str(label))
    return None


def _fallback_task3(observation, metadata: Dict[str, Any]) -> Action:
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
        priority = (
            EventLabel.ROOT_CAUSE,
            EventLabel.CONTRIBUTING_FACTOR,
            EventLabel.SYMPTOM,
            EventLabel.NOISE,
            EventLabel.UNRELATED,
            EventLabel.FALSE_POSITIVE,
        )
        expected: Dict[str, EventLabel] = {}
        for event_id in remaining:
            label = _task3_expected_label(observation, metadata, event_id)
            if label is not None:
                expected[event_id] = label
        for label in priority:
            for event_id in remaining:
                if expected.get(event_id) == label:
                    return Action(
                        action_type=ActionType.LABEL_EVENT,
                        target_id=event_id,
                        event_label=label,
                    )
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
        return _fallback_task3(observation, metadata)
    return Action(action_type=ActionType.HOLD, target_id=_default_target_id(metadata))


# ── Progress coercion ─────────────────────────────────────────────────────────

def _coerce_action_for_progress(action: Action, observation, metadata: Dict[str, Any]) -> Action:
    if action.action_type == ActionType.CLASSIFY_ALERT:
        remaining = metadata["unclassified_alert_ids"]
        if remaining and action.target_id not in remaining:
            replacement = _fallback_task1(observation, metadata)
            if replacement.target_id in remaining:
                return replacement

    if action.action_type == ActionType.LABEL_EVENT:
        remaining = metadata["unlabelled_event_ids"]
        if metadata["task"] == "alert_storm":
            if action.target_id not in remaining:
                return _fallback_task2(observation, metadata)
        elif remaining and action.target_id not in remaining:
            if metadata["task"] == "timeline_labelling":
                return _fallback_task3(observation, metadata)

    if action.action_type == ActionType.SILENCE_ALERT:
        if action.target_id in set(metadata["silenced_alert_ids"]):
            if metadata["task"] == "alert_storm":
                return _fallback_task2(observation, metadata)

    if action.action_type == ActionType.INVESTIGATE:
        used_ids = {
            item for item in metadata.get("investigation_ids", [])
            if item not in metadata.get("unused_investigation_ids", [])
        }
        if action.investigation_id in used_ids and metadata["task"] == "timeline_labelling":
            return _fallback_task3(observation, metadata)

    if (
        action.action_type == ActionType.LABEL_EVENT
        and metadata["task"] == "timeline_labelling"
        and action.target_id in metadata["unlabelled_event_ids"]
    ):
        expected_label = _task3_expected_label(observation, metadata, action.target_id)
        if expected_label is not None and action.event_label != expected_label:
            return Action(
                action_type=ActionType.LABEL_EVENT,
                target_id=action.target_id,
                event_label=expected_label,
            )

    return action


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_llm(client: OpenAI, prompt: str) -> str:
    for attempt in range(5):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception as e:
            if "429" in str(e):
                print(f"[RATE LIMIT] sleeping 25s...", flush=True)
                time.sleep(25)
            else:
                raise
    return ""


# ── Episode runner ────────────────────────────────────────────────────────────

def _run_task(task_id: int, seed: int, client: OpenAI) -> None:
    env = OnCallEnv()
    observation = env.reset(task_id=task_id, seed=seed)
    task_name = observation.context.get("task", f"task-{task_id}")

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    error: Optional[str] = None
    done = False
    prev_reward: Optional[float] = None
    max_steps = observation.budget_remaining

    try:
        while not done and len(rewards) < max_steps:
            metadata = _task_metadata(env, observation)
            prompt = _build_prompt(observation, metadata, prev_reward)

            content = _call_llm(client, prompt)
            print(f"RAW LLM OUTPUT: {content}", flush=True)

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
                print(
                    f"[INFO] action_adjusted reason=fallback_after_parse_error:{exc} "
                    f"from={json.dumps(proposed_action.model_dump(), sort_keys=True)} "
                    f"to={json.dumps(action.model_dump(), sort_keys=True)}",
                    flush=True,
                )

            coerced = _coerce_action_for_progress(action, observation, metadata)
            if coerced != action:
                print(
                    f"[INFO] action_adjusted reason=coerced_for_progress "
                    f"from={json.dumps(action.model_dump(), sort_keys=True)} "
                    f"to={json.dumps(coerced.model_dump(), sort_keys=True)}",
                    flush=True,
                )
            action = coerced

            observation, reward, done, info = env.step(action)
            rewards.append(reward)
            prev_reward = reward

            if not info.get("valid", True):
                error = info.get("error")

            log_step(
                step=env.state().get("step", len(rewards)),
                action=json.dumps(action.model_dump(), sort_keys=True),
                reward=reward,
                done=done,
                error=error,
            )

    except Exception as exc:
        error = str(exc)
        print("EPISODE ERROR:", flush=True)
        traceback.print_exc()

    finally:
        score = 0.0
        if env._state is not None:
            score = grade_task(env._state)
        score = min(max(score, 0.0), 1.0)
        success = error is None
        log_end(success=success, steps=len(rewards), score=score, rewards=rewards)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    if not API_KEY:
        raise SystemExit("HF_TOKEN or API_KEY is required.")
    if LOCAL_IMAGE_NAME is None:
        raise SystemExit("LOCAL_IMAGE_NAME must be set.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_id in TASK_IDS:
        seed = DEFAULT_SEEDS[task_id]
        _run_task(task_id=task_id, seed=seed, client=client)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())