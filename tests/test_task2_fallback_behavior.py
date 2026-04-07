from inference import _fallback_task2, _task_metadata
from env.env import OnCallEnv
from env.models import ActionType, EventLabel


def test_fallback_task2_prioritizes_false_positive_label_before_silence():
    env = OnCallEnv()
    observation = env.reset(task_id=2, seed=123)

    root_action = _fallback_task2(observation, _task_metadata(env, observation))
    assert root_action.action_type == ActionType.LABEL_EVENT
    assert root_action.event_label == EventLabel.ROOT_CAUSE

    env.step(root_action)
    observation = env._build_observation(done=False, reward=0.0)

    symptom_action = _fallback_task2(observation, _task_metadata(env, observation))
    assert symptom_action.action_type == ActionType.LABEL_EVENT
    assert symptom_action.event_label == EventLabel.SYMPTOM

    env.step(symptom_action)
    observation = env._build_observation(done=False, reward=0.0)

    false_positive_action = _fallback_task2(observation, _task_metadata(env, observation))
    assert false_positive_action.action_type == ActionType.LABEL_EVENT
    assert false_positive_action.event_label == EventLabel.FALSE_POSITIVE
