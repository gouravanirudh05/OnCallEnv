"""Dense reward shaping for OnCallEnv."""

from typing import Dict

from .grader import (
    _task1_correct_score,
    _task1_misclassification_penalty,
    _task3_correct_score,
    _task3_mislabel_penalty,
    _task3_normalized,
)
from .models import Action, ActionType
from .types import EpisodeState


def compute_reward(state: EpisodeState, action: Action, valid: bool) -> float:
    """Compute a per-step reward signal."""

    if not valid:
        return -0.02 - (0.01 * state.invalid_actions)

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
        if action.event_label.value == true_label:
            return _task3_correct_score(true_label)
        return _task3_mislabel_penalty(true_label, action.event_label.value)

    if action.action_type == ActionType.ESCALATE:
        escalation = state.ground_truth.get("escalation", {})
        if escalation and action.team == escalation.get("team"):
            if action.severity and action.severity.value == escalation.get("severity"):
                return 0.20
        return -0.05

    if action.action_type == ActionType.INVESTIGATE:
        investigations = state.ground_truth.get("investigations", [])
        used = state.ground_truth.get("investigations_used", [])
        used_ids = {
            str(item.get("id"))
            for item in used
            if isinstance(item, dict) and item.get("id")
        }
        if action.investigation_id:
            valid_ids = {
                str(item.get("id"))
                for item in investigations
                if isinstance(item, dict) and item.get("id")
            }
            if action.investigation_id in used_ids:
                return -0.05
            return 0.10 if action.investigation_id in valid_ids else -0.05
        if len(used) >= len(investigations):
            return -0.05
        return 0.10

    return 0.01


def episode_bonus(state: EpisodeState) -> Dict[str, float]:
    """Compute episode-level bonus components."""
    bonuses = {
        "efficiency": 0.0,
        "zero_false_positive_errors": 0.0,
        "perfect_root_cause": 0.0,
    }

    initial_budget = state.budget + state.step
    if initial_budget > 0 and state.step <= int(initial_budget * 0.6):
        bonuses["efficiency"] = 0.15

    if state.task_id == 2:
        false_positive_ids = [
            alert_id
            for alert_id, label in state.ground_truth.get("event_labels", {}).items()
            if label == "false_positive"
        ]
        if false_positive_ids and all(
            alert_id in state.silenced_alerts for alert_id in false_positive_ids
        ):
            bonuses["zero_false_positive_errors"] = 0.10

        root_service = state.ground_truth.get("root_cause")
        if root_service:
            root_alert = next(
                (alert for alert in state.alerts if alert.service == root_service),
                None,
            )
            if root_alert and state.labelled_events.get(root_alert.id) == "root_cause":
                bonuses["perfect_root_cause"] = 0.20

    return bonuses
