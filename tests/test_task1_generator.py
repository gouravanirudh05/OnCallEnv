from env.generator import generate_episode


def test_task1_deterministic_alerts():
    first = generate_episode(task_id=1, seed=42, budget=20)
    second = generate_episode(task_id=1, seed=42, budget=20)

    assert [alert.model_dump() for alert in first.alerts] == [
        alert.model_dump() for alert in second.alerts
    ]
    assert first.ground_truth == second.ground_truth


def test_task1_alert_count_range():
    episode = generate_episode(task_id=1, seed=7, budget=20)
    assert 5 <= len(episode.alerts) <= 8


def test_task1_contains_flags():
    episode = generate_episode(task_id=1, seed=13, budget=20)
    gt = episode.ground_truth

    assert "correlated_incident" in gt
    assert "adjusted_alert_id" in gt
    assert "false_positive_alert_id" in gt

    alert_ids = {alert.id for alert in episode.alerts}
    assert set(gt["correlated_incident"]).issubset(alert_ids)
    assert gt["adjusted_alert_id"] in alert_ids
    assert gt["false_positive_alert_id"] in alert_ids


def test_task1_correlated_alerts_are_p0():
    episode = generate_episode(task_id=1, seed=21, budget=20)
    gt = episode.ground_truth
    correlated = gt["correlated_incident"]

    for alert_id in correlated:
        assert gt["alerts"][alert_id] == "P0"


def test_task1_different_seed_changes_alerts():
    first = generate_episode(task_id=1, seed=42, budget=20)
    second = generate_episode(task_id=1, seed=43, budget=20)
    assert [alert.model_dump() for alert in first.alerts] != [
        alert.model_dump() for alert in second.alerts
    ]
