from env.generator import generate_episode
from env.grader import grade_task
from env.models import Action, ActionType, EventLabel, Severity
from env.reward import compute_reward


def test_reward_invalid_actions_increase_penalty():
    state = generate_episode(task_id=1, seed=11, budget=5)
    action = Action(action_type=ActionType.HOLD, target_id="noop")
    first = compute_reward(state, action, valid=False)
    state.invalid_actions += 1
    second = compute_reward(state, action, valid=False)
    assert second < first


def test_reward_investigation_id_helps_only_when_valid():
    state = generate_episode(task_id=3, seed=12, budget=5)
    investigations = state.ground_truth.get("investigations", [])
    if not investigations:
        return
    valid_id = str(investigations[0]["id"])
    action = Action(
        action_type=ActionType.INVESTIGATE,
        target_id="noop",
        investigation_id=valid_id,
    )
    reward = compute_reward(state, action, valid=True)
    assert reward > 0
    action = Action(
        action_type=ActionType.INVESTIGATE,
        target_id="noop",
        investigation_id="missing",
    )
    reward = compute_reward(state, action, valid=True)
    assert reward < 0


def test_grade_task_bounds():
    for task_id in (1, 2, 3):
        state = generate_episode(task_id=task_id, seed=13, budget=10)
        score = grade_task(state)
        assert 0.0 <= score <= 1.0


def test_task1_scoring_penalizes_misclassification():
    state = generate_episode(task_id=1, seed=14, budget=10)
    for alert_id in state.ground_truth.get("alerts", {}):
        state.classified_alerts[alert_id] = Severity.P3.value
    score = grade_task(state)
    assert score <= 0.6


def test_task3_label_score_changes_with_labels():
    state = generate_episode(task_id=3, seed=15, budget=10)
    labels = state.ground_truth.get("event_labels", {})
    for event_id in labels:
        state.labelled_events[event_id] = EventLabel.NOISE.value
    low_score = grade_task(state)
    for event_id, true_label in labels.items():
        state.labelled_events[event_id] = true_label
    high_score = grade_task(state)
    assert high_score >= low_score
