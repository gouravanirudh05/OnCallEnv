"""Internal episode state types for OnCallEnv."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import Alert, LogLine, ServiceNode


@dataclass
class EpisodeState:
    episode_id: str
    task_id: int
    seed: int
    step: int
    budget: int
    alerts: List[Alert] = field(default_factory=list)
    logs: List[LogLine] = field(default_factory=list)
    service_graph: List[ServiceNode] = field(default_factory=list)
    active_incidents: List[str] = field(default_factory=list)
    context: Dict[str, str] = field(default_factory=dict)
    ground_truth: Dict[str, object] = field(default_factory=dict)
    labelled_events: Dict[str, str] = field(default_factory=dict)
    classified_alerts: Dict[str, str] = field(default_factory=dict)
    silenced_alerts: List[str] = field(default_factory=list)
    escalation: Optional[Dict[str, str]] = None
