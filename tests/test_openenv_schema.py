import json

from env.models import Action, Observation


def test_action_schema_has_required_fields():
    schema = Action.model_json_schema()
    properties = schema.get("properties", {})
    assert "action_type" in properties
    assert "target_id" in properties


def test_observation_schema_has_done_reward():
    schema = Observation.model_json_schema()
    properties = schema.get("properties", {})
    assert "done" in properties
    assert "reward" in properties


def test_action_json_round_trip():
    payload = {"action_type": "hold", "target_id": "noop"}
    action = Action.model_validate(payload)
    data = json.loads(action.model_dump_json())
    assert data["action_type"] == "hold"
    assert data["target_id"] == "noop"
