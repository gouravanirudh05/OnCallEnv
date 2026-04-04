import pytest

from env.models import Action, ActionType, Alert, Observation


def test_models_are_frozen():
    alert = Alert(
        id="alert-1",
        service="payments-api",
        metric="latency_ms",
        value=1.2,
        threshold=0.8,
        severity_raw="P2",
        timestamp=10,
        labels={"region": "us-east-1"},
    )
    with pytest.raises(Exception):
        alert.value = 2.0


def test_action_serialization_roundtrip():
    action = Action(action_type=ActionType.HOLD, target_id="noop")
    payload = action.model_dump()
    restored = Action(**payload)
    assert restored == action


def test_observation_serialization_roundtrip():
    obs = Observation(
        step=1,
        alerts=[],
        logs=[],
        service_graph=[],
        active_incidents=[],
        budget_remaining=10,
        context={"task": "severity_classification"},
        done=False,
        reward=0.0,
    )
    payload = obs.model_dump()
    restored = Observation(**payload)
    assert restored == obs
