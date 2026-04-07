"""OnCallEnv core environment implementation."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from .generator import TASK_NAMES, generate_episode
from .grader import grade_task
from .models import Action, ActionType, Observation
from .reward import compute_reward, episode_bonus
from .types import EpisodeState


DEFAULT_BUDGET = {1: 20, 2: 60, 3: 40}


class OnCallEnv:
    """OpenEnv-compatible environment for on-call incident management."""

    def __init__(self) -> None:
        self._state: Optional[EpisodeState] = None

    def reset(self, task_id: int, seed: int) -> Observation:
        budget = DEFAULT_BUDGET.get(task_id, 40)
        self._state = generate_episode(task_id=task_id, seed=seed, budget=budget)
        return self._build_observation(done=False, reward=0.0)

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        if self._state is None:
            raise RuntimeError("Environment must be reset before stepping")

        info = {"valid": True, "error": None, "step": self._state.step}
        valid, error = self._validate_action(action)
        if not valid:
            info["valid"] = False
            info["error"] = error
            self._state.invalid_actions += 1

        reward = compute_reward(self._state, action, valid)
        if valid:
            self._apply_action(action)

        self._state.step += 1
        self._state.budget = max(0, self._state.budget - 1)

        done = self._state.budget == 0 or self._task_complete()
        if done:
            reward += grade_task(self._state)
            reward += sum(episode_bonus(self._state).values())

        observation = self._build_observation(done=done, reward=reward)
        return observation, reward, done, info

    def state(self) -> Dict:
        if self._state is None:
            return {}
        return {
            "episode_id": self._state.episode_id,
            "task_id": self._state.task_id,
            "seed": self._state.seed,
            "step": self._state.step,
            "budget": self._state.budget,
            "ground_truth": self._state.ground_truth,
            "classified_alerts": self._state.classified_alerts,
            "labelled_events": self._state.labelled_events,
            "silenced_alerts": self._state.silenced_alerts,
            "escalation": self._state.escalation,
        }

    def _build_observation(
        self, *, done: bool = False, reward: Optional[float] = None
    ) -> Observation:
        if self._state is None:
            raise RuntimeError("Environment must be reset before observation")

        return Observation(
            step=self._state.step,
            alerts=self._state.alerts,
            logs=self._state.logs,
            service_graph=self._state.service_graph,
            active_incidents=self._state.active_incidents,
            budget_remaining=self._state.budget,
            context=self._state.context,
            done=done,
            reward=reward,
        )

    def _validate_action(self, action: Action) -> Tuple[bool, Optional[str]]:
        if self._state is None:
            return False, "environment not reset"

        if action.action_type == ActionType.CLASSIFY_ALERT and action.severity is None:
            return False, "severity required for classify_alert"

        if action.action_type == ActionType.LABEL_EVENT and action.event_label is None:
            return False, "event_label required for label_event"

        if action.action_type == ActionType.ESCALATE:
            if action.team is None:
                return False, "team required for escalate"
            if action.severity is None:
                return False, "severity required for escalate"

        if action.action_type == ActionType.REMEDIATE and action.remediation is None:
            return False, "remediation required for remediate"

        return True, None

    def _apply_action(self, action: Action) -> None:
        if self._state is None:
            return

        if (
            action.action_type == ActionType.CLASSIFY_ALERT
            and action.severity is not None
        ):
            self._state.classified_alerts[action.target_id] = action.severity.value
            return

        if (
            action.action_type == ActionType.LABEL_EVENT
            and action.event_label is not None
        ):
            self._state.labelled_events[action.target_id] = action.event_label.value
            return

        if action.action_type == ActionType.SILENCE_ALERT:
            if action.target_id not in self._state.silenced_alerts:
                self._state.silenced_alerts.append(action.target_id)
            return

        if action.action_type == ActionType.ESCALATE and action.team is not None:
            self._state.escalation = {
                "team": action.team,
                "target_id": action.target_id,
                "severity": action.severity.value if action.severity else None,
            }
            return

        if action.action_type == ActionType.INVESTIGATE:
            if action.investigation_id:
                used = self._state.ground_truth.setdefault("investigations_used", [])
                used_ids = {
                    str(item.get("id"))
                    for item in used
                    if isinstance(item, dict) and item.get("id")
                }
                if action.investigation_id in used_ids:
                    return
                investigations = self._state.ground_truth.get("investigations", [])
                valid_ids = {
                    str(item.get("id"))
                    for item in investigations
                    if isinstance(item, dict) and item.get("id")
                }
                used.append(
                    {
                        "id": action.investigation_id,
                        "helpful": action.investigation_id in valid_ids,
                    }
                )
            return

    def _task_complete(self) -> bool:
        if self._state is None:
            return False
        if self._state.task_id == 1:
            return len(self._state.classified_alerts) >= len(
                self._state.ground_truth.get("alerts", {})
            )
        if self._state.task_id in (2, 3):
            return len(self._state.labelled_events) >= len(
                self._state.ground_truth.get("event_labels", {})
            )
        return False
