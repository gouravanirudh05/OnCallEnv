from env.generator import generate_episode
from env.grader import _grade_task2
from env.models import Action, ActionType, EventLabel, Severity
from env.types import EpisodeState


def _apply_label(state: EpisodeState, alert_id: str, label: str) -> None:
    state.labelled_events[alert_id] = label


def test_task2_grader_root_and_escalation():
    state = generate_episode(task_id=2, seed=55, budget=60)
    labels = state.ground_truth.get("event_labels", {})
    root_service = state.ground_truth.get("root_cause")
    root_alert = next(
        (alert for alert in state.alerts if alert.service == root_service), None
    )
    assert root_alert is not None

    _apply_label(state, root_alert.id, "root_cause")

    for alert_id, label in labels.items():
        if label == "symptom":
            _apply_label(state, alert_id, "symptom")

    state.escalation = {
        "team": state.ground_truth["escalation"]["team"],
        "severity": state.ground_truth["escalation"]["severity"],
    }

    score = _grade_task2(state)
    assert score >= 0.6


def test_task2_silence_penalty():
    state = generate_episode(task_id=2, seed=66, budget=60)
    labels = state.ground_truth.get("event_labels", {})
    false_positive_ids = [
        alert_id for alert_id, label in labels.items() if label == "false_positive"
    ]
    assert false_positive_ids

    score_without_silence = _grade_task2(state)
    for alert_id in false_positive_ids:
        state.silenced_alerts.append(alert_id)
    score_with_silence = _grade_task2(state)
    assert score_with_silence >= score_without_silence
