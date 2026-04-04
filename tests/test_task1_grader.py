from env.grader import _task1_correct_score, _task1_misclassification_penalty


def test_task1_correct_scores():
    assert _task1_correct_score("P0") == 0.25
    assert _task1_correct_score("P1") == 0.15
    assert _task1_correct_score("P2") == 0.15
    assert _task1_correct_score("P3") == 0.10


def test_task1_misclassification_penalties():
    assert _task1_misclassification_penalty("P0", "P2") == -0.20
    assert _task1_misclassification_penalty("P0", "P3") == -0.20
    assert _task1_misclassification_penalty("P3", "P0") == -0.10
    assert _task1_misclassification_penalty("P3", "P1") == -0.10
    assert _task1_misclassification_penalty("P2", "P1") == -0.05
