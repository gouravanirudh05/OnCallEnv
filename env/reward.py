"""Dense reward shaping for OnCallEnv."""

from typing import Dict

from .grader import _task1_correct_score, _task1_misclassification_penalty
from .models import Action, ActionType
from .types import EpisodeState


def compute_reward(state: EpisodeState, action: Action, valid: bool) -> float:
    """Compute a per-step reward signal."""

    if not valid:
        return -0.02

    if action.action_type == ActionType.HOLD:
        return -0.05

    if action.action_type == ActionType.CLASSIFY_ALERT and action.severity is not None:
        true_sev = state.ground_truth.get("alerts", {}).get(action.target_id)
        if true_sev is None:
            return -0.02
        if action.severity.value == true_sev:
            return _task1_correct_score(true_sev)
        return _task1_misclassification_penalty(true_sev, action.severity.value)

    if action.action_type == ActionType.SILENCE_ALERT:
        true_label = state.ground_truth.get("event_labels", {}).get(action.target_id)
        if true_label == "false_positive":
            return 0.05
        if true_label == "root_cause":
            return -0.20
        return -0.02

    if action.action_type == ActionType.LABEL_EVENT and action.event_label is not None:
        true_label = state.ground_truth.get("event_labels", {}).get(action.target_id)
        if true_label is None:
            return -0.02
        return 0.08 if action.event_label.value == true_label else -0.05

    if action.action_type == ActionType.ESCALATE:
        escalation = state.ground_truth.get("escalation", {})
        if escalation and action.team == escalation.get("team"):
            if action.severity and action.severity.value == escalation.get("severity"):
                return 0.20
        return -0.05

    if action.action_type == ActionType.INVESTIGATE:
        investigations = state.ground_truth.get("investigations", [])
        used = state.ground_truth.setdefault("investigations_used", [])
        if len(used) >= len(investigations):
            used.append({"helpful": False})
            return -0.05
        used.append({"helpful": True})
        return 0.10

    return 0.01


def episode_bonus(state: EpisodeState) -> Dict[str, float]:
    """Compute episode-level bonus components."""

    return {
        "efficiency": 0.0,
        "zero_false_positive_errors": 0.0,
        "perfect_root_cause": 0.0,
    }
