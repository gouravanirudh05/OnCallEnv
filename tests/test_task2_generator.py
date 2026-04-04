from env.generator import generate_episode


def test_task2_event_labels_present():
    episode = generate_episode(task_id=2, seed=33, budget=60)
    labels = episode.ground_truth.get("event_labels", {})
    assert labels

    root_service = episode.ground_truth.get("root_cause")
    assert root_service is not None

    root_alert = next(
        (alert for alert in episode.alerts if alert.service == root_service), None
    )
    assert root_alert is not None
    assert labels[root_alert.id] == "root_cause"

    assert any(label == "symptom" for label in labels.values())


def test_task2_false_positives_flagged():
    episode = generate_episode(task_id=2, seed=44, budget=60)
    labels = episode.ground_truth.get("event_labels", {})
    false_positive_ids = [
        alert_id for alert_id, label in labels.items() if label == "false_positive"
    ]
    assert false_positive_ids
