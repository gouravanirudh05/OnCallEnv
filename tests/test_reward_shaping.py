from env.generator import generate_episode
from env.models import Action, ActionType, Severity
from env.reward import compute_reward, episode_bonus


def test_reward_invalid_action_penalty_increases():
    state = generate_episode(task_id=1, seed=42, budget=20)
    action = Action(action_type=ActionType.CLASSIFY_ALERT, target_id="missing")
    first = compute_reward(state, action, valid=False)
    state.invalid_actions += 1
    second = compute_reward(state, action, valid=False)
    assert second < first


def test_reward_efficiency_bonus():
    state = generate_episode(task_id=1, seed=42, budget=20)
    state.step = 10
    bonus = episode_bonus(state)
    assert 0.0 <= bonus["efficiency"] <= 0.15
