"""Task graders for OnCallEnv."""

from typing import Dict

from .models import Severity
from .types import EpisodeState


def grade_task(state: EpisodeState) -> float:
    """Return final score in [0,1] for the current episode."""

    if state.task_id == 1:
        return _grade_task1(state)
    if state.task_id == 2:
        return _grade_task2(state)
    if state.task_id == 3:
        return _grade_task3(state)
    return 0.0


def _grade_task1(state: EpisodeState) -> float:
    scores = []
    for alert_id, true_sev in state.ground_truth.get("alerts", {}).items():
        predicted = state.classified_alerts.get(alert_id)
        if predicted is None:
            scores.append(0.0)
            continue

        if predicted == true_sev:
            scores.append(_task1_correct_score(true_sev))
            continue

        scores.append(_task1_misclassification_penalty(true_sev, predicted))

    if not scores:
        return 0.0

    raw = sum(scores)
    min_possible = -0.20 * len(scores)
    max_possible = 0.25 * len(scores)
    if max_possible == min_possible:
        return 0.0
    normalized = (raw - min_possible) / (max_possible - min_possible)
    return max(0.0, min(1.0, normalized))


def _grade_task2(state: EpisodeState) -> float:
    labels = state.ground_truth.get("event_labels", {})
    root_service = state.ground_truth.get("root_cause")
    escalation = state.ground_truth.get("escalation", {})

    root_score = 0.0
    if root_service is not None:
        root_alert = next(
            (alert for alert in state.alerts if alert.service == root_service), None
        )
        if root_alert is not None:
            predicted = state.labelled_events.get(root_alert.id)
            if predicted == "root_cause":
                root_score = 0.40

    symptom_alerts = [
        alert_id for alert_id, label in labels.items() if label == "symptom"
    ]
    symptom_correct = sum(
        1
        for alert_id in symptom_alerts
        if state.labelled_events.get(alert_id) == "symptom"
    )
    symptom_score = _safe_ratio(symptom_correct, len(symptom_alerts)) * 0.20

    false_positive_alerts = [
        alert_id for alert_id, label in labels.items() if label == "false_positive"
    ]
    silenced_correct = sum(
        1 for alert_id in false_positive_alerts if alert_id in state.silenced_alerts
    )
    silence_score = _safe_ratio(silenced_correct, len(false_positive_alerts)) * 0.20

    escalation_score = 0.0
    if escalation:
        predicted = state.escalation or {}
        if (
            predicted.get("team") == escalation.get("team")
            and predicted.get("severity") == escalation.get("severity")
        ):
            escalation_score = 0.20

    penalty = 0.0
    for alert_id in false_positive_alerts:
        if alert_id not in state.silenced_alerts:
            penalty -= 0.10

    root_alert_id = None
    if root_service is not None:
        root_alert_id = next(
            (alert.id for alert in state.alerts if alert.service == root_service),
            None,
        )
    if root_alert_id and root_alert_id in state.silenced_alerts:
        penalty -= 0.30

    score = root_score + symptom_score + silence_score + escalation_score + penalty
    return max(0.0, min(1.0, score))


def _grade_task3(state: EpisodeState) -> float:
    labels = state.ground_truth.get("event_labels", {})
    if not labels:
        return 0.0

    score = 0.0
    for event_id, true_label in labels.items():
        predicted = state.labelled_events.get(event_id)
        if predicted is None:
            continue

        if predicted == true_label:
            score += _task3_correct_score(true_label)
            continue

        score += _task3_mislabel_penalty(true_label, predicted)

    investigate_actions = state.ground_truth.get("investigations_used", [])
    for investigation in investigate_actions:
        if investigation.get("helpful"):
            score += 0.10
        else:
            score -= 0.05

    return max(0.0, min(1.0, score))


def _safe_ratio(correct: int, total: int) -> float:
    if total == 0:
        return 0.0
    return max(0.0, min(1.0, correct / total))


def _task1_correct_score(true_severity: str) -> float:
    if true_severity == Severity.P0.value:
        return 0.25
    if true_severity in (Severity.P1.value, Severity.P2.value):
        return 0.15
    return 0.10


def _task1_misclassification_penalty(true_severity: str, predicted: str) -> float:
    if true_severity == Severity.P0.value and predicted in {
        Severity.P2.value,
        Severity.P3.value,
    }:
        return -0.20
    if true_severity == Severity.P3.value and predicted in {
        Severity.P0.value,
        Severity.P1.value,
    }:
        return -0.10
    return -0.05


def _task3_correct_score(label: str) -> float:
    if label == "root_cause":
        return 0.15
    if label == "contributing_factor":
        return 0.10
    if label == "noise":
        return 0.08
    return 0.08


def _task3_mislabel_penalty(true_label: str, predicted: str) -> float:
    if true_label == "symptom" and predicted == "root_cause":
        return -0.15
    if true_label == "noise" and predicted == "symptom":
        return -0.05
    return -0.03
