"""Server-side OpenEnv wrapper for OnCallEnv."""

from typing import Dict, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from env.env import OnCallEnv
from env.models import Action, Observation


class OnCallEnvironment(Environment):
    """OpenEnv-compatible server wrapper around the core OnCallEnv."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._env = OnCallEnv()
        self._last_state: Optional[Dict] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[int] = None,
        **kwargs,
    ) -> Observation:
        resolved_task = task_id if task_id is not None else 1
        resolved_seed = seed if seed is not None else 42
        observation = self._env.reset(task_id=resolved_task, seed=resolved_seed)
        self._last_state = self._env.state()
        if episode_id:
            self._last_state["episode_id"] = episode_id
        return observation

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> Observation:  # type: ignore[override]
        observation, reward, done, info = self._env.step(action)
        self._last_state = self._env.state()
        self._last_state["last_step"] = {
            "reward": reward,
            "done": done,
            "info": info,
        }
        return observation

    @property
    def state(self) -> State:
        if self._last_state is None:
            return State()
        payload = dict(self._last_state)
        step_count = payload.pop("step", 0)
        episode_id = payload.pop("episode_id", None)
        return State(episode_id=episode_id, step_count=step_count, **payload)
