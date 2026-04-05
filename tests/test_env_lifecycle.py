import json

import pytest

from env.env import DEFAULT_BUDGET, OnCallEnv
from env.models import Action, ActionType, EventLabel, Severity


def test_reset_sets_budget_and_step():
    env = OnCallEnv()
    observation = env.reset(task_id=1, seed=1)
    assert observation.step == 0
    assert observation.budget_remaining == DEFAULT_BUDGET[1]
    assert observation.done is False
    assert observation.reward == 0.0


def test_step_decrements_budget_and_increments_step():
    env = OnCallEnv()
    observation = env.reset(task_id=1, seed=2)
    start_budget = observation.budget_remaining
    action = Action(action_type=ActionType.HOLD, target_id="noop")
    observation, _, _, _ = env.step(action)
    assert observation.step == 1
    assert observation.budget_remaining == start_budget - 1


def test_done_when_budget_exhausted():
    env = OnCallEnv()
    observation = env.reset(task_id=1, seed=3)
    action = Action(action_type=ActionType.HOLD, target_id="noop")
    for _ in range(observation.budget_remaining):
        observation, _, done, _ = env.step(action)
    assert done is True
    assert observation.done is True


def test_invalid_action_flagged():
    env = OnCallEnv()
    env.reset(task_id=1, seed=4)
    action = Action(action_type=ActionType.CLASSIFY_ALERT, target_id="x")
    _, _, _, info = env.step(action)
    assert info["valid"] is False
    assert info["error"]


def test_label_event_requires_event_label():
    env = OnCallEnv()
    env.reset(task_id=3, seed=5)
    action = Action(action_type=ActionType.LABEL_EVENT, target_id="event")
    _, _, _, info = env.step(action)
    assert info["valid"] is False


def test_escalate_requires_team_and_severity():
    env = OnCallEnv()
    env.reset(task_id=2, seed=6)
    action = Action(action_type=ActionType.ESCALATE, target_id="alert")
    _, _, _, info = env.step(action)
    assert info["valid"] is False


def test_investigate_tracks_id_when_present():
    env = OnCallEnv()
    env.reset(task_id=3, seed=7)
    action = Action(
        action_type=ActionType.INVESTIGATE,
        target_id="noop",
        investigation_id="investigation-1",
    )
    env.step(action)
    state = env.state()
    used = state["ground_truth"].get("investigations_used", [])
    assert used


def test_task_complete_after_all_classified():
    env = OnCallEnv()
    observation = env.reset(task_id=1, seed=8)
    for alert in observation.alerts:
        action = Action(
            action_type=ActionType.CLASSIFY_ALERT,
            target_id=alert.id,
            severity=Severity.P2,
        )
        observation, _, done, _ = env.step(action)
    assert done is True


def test_task3_labeling_completion():
    env = OnCallEnv()
    observation = env.reset(task_id=3, seed=9)
    event_ids = observation.context.get("event_ids")
    if event_ids:
        event_ids = json.loads(event_ids)
    else:
        event_ids = [alert.id for alert in observation.alerts]
    for event_id in event_ids:
        action = Action(
            action_type=ActionType.LABEL_EVENT,
            target_id=event_id,
            event_label=EventLabel.NOISE,
        )
        observation, _, done, _ = env.step(action)
    assert done is True
