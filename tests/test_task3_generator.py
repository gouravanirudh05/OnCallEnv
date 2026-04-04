from env.generator import generate_episode


def test_task3_uses_scenario_labels():
    episode = generate_episode(task_id=3, seed=77, budget=40)
    labels = episode.ground_truth.get("event_labels", {})
    assert labels
    assert "scenario_id" in episode.context


def test_task3_logs_and_alerts_present():
    episode = generate_episode(task_id=3, seed=88, budget=40)
    assert episode.logs or episode.alerts
