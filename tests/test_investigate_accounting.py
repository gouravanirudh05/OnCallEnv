from env.env import OnCallEnv
from env.generator import generate_episode
from env.models import Action, ActionType
from env.reward import compute_reward


def test_compute_reward_investigate_does_not_mutate_usage():
    state = generate_episode(task_id=3, seed=7, budget=40)
    investigation_id = state.ground_truth["investigations"][0]["id"]
    action = Action(
        action_type=ActionType.INVESTIGATE,
        target_id="ev-0",
        investigation_id=investigation_id,
    )

    reward = compute_reward(state, action, valid=True)

    assert reward == 0.10
    assert state.ground_truth.get("investigations_used", []) == []


def test_env_step_investigate_records_usage_once():
    env = OnCallEnv()
    obs = env.reset(task_id=3, seed=7)
    state = env.state()
    investigation_id = state["ground_truth"]["investigations"][0]["id"]

    action = Action(
        action_type=ActionType.INVESTIGATE,
        target_id=obs.alerts[0].id if obs.alerts else "ev-0",
        investigation_id=investigation_id,
    )
    env.step(action)

    investigations_used = env.state()["ground_truth"].get("investigations_used", [])
    assert len(investigations_used) == 1
    assert investigations_used[0]["id"] == investigation_id
    assert investigations_used[0]["helpful"] is True


def test_repeated_investigate_is_penalized_and_not_recorded_twice():
    env = OnCallEnv()
    obs = env.reset(task_id=3, seed=7)
    investigation_id = env.state()["ground_truth"]["investigations"][0]["id"]
    action = Action(
        action_type=ActionType.INVESTIGATE,
        target_id=obs.alerts[0].id if obs.alerts else "ev-0",
        investigation_id=investigation_id,
    )

    _, first_reward, _, _ = env.step(action)
    _, second_reward, _, _ = env.step(action)

    investigations_used = env.state()["ground_truth"].get("investigations_used", [])
    assert first_reward == 0.10
    assert second_reward == -0.05
    assert len(investigations_used) == 1
