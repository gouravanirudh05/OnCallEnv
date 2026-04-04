from env.models import Action, ActionType, Observation
from server.oncall_environment import OnCallEnvironment


def test_server_reset_defaults():
    server_env = OnCallEnvironment()
    observation = server_env.reset()
    assert isinstance(observation, Observation)
    assert observation.budget_remaining == 20
    assert observation.done is False
    assert observation.reward == 0.0


def test_server_step_updates_state():
    server_env = OnCallEnvironment()
    server_env.reset()
    action = Action(action_type=ActionType.HOLD, target_id="noop")
    observation = server_env.step(action)
    state = server_env.state
    assert isinstance(observation, Observation)
    assert state.step_count == 1
    payload = state.model_dump()
    assert "last_step" in payload
    assert payload["last_step"]["done"] == observation.done


def test_server_reset_sets_episode_id():
    server_env = OnCallEnvironment()
    server_env.reset(episode_id="episode-123")
    state = server_env.state
    assert state.episode_id == "episode-123"
