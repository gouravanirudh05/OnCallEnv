from env.generator import generate_episode
from env.grader import _grade_task3


def test_task3_grader_scores_labels():
    state = generate_episode(task_id=3, seed=99, budget=40)
    labels = state.ground_truth.get("event_labels", {})
    for event_id, label in labels.items():
        state.labelled_events[event_id] = label

    score = _grade_task3(state)
    assert score > 0.5


def test_task3_investigation_usage():
    state = generate_episode(task_id=3, seed=101, budget=40)
    state.ground_truth["investigations_used"] = [{"helpful": False}]
    score = _grade_task3(state)
    assert 0.0 <= score <= 1.0
