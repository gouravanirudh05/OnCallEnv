"""
Pydantic models for OnCallEnv.

Defines the action and observation schema used by the environment.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from openenv.core.env_server.types import Action as OpenEnvAction
    from openenv.core.env_server.types import Observation as OpenEnvObservation
except ImportError:  # pragma: no cover
    class OpenEnvAction(BaseModel):
        """Fallback Action base when openenv is unavailable."""

    class OpenEnvObservation(BaseModel):
        """Fallback Observation base when openenv is unavailable."""


class FrozenBaseModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class Alert(FrozenBaseModel):
    """Monitoring alert emitted by a service."""

    id: str = Field(..., description="Unique alert ID")
    service: str = Field(..., description="Service emitting the alert")
    metric: str = Field(..., description="Metric name")
    value: float = Field(..., description="Observed metric value")
    threshold: float = Field(..., description="Configured alert threshold")
    severity_raw: str = Field(
        ..., description="Raw monitoring severity (may be incorrect)"
    )
    timestamp: int = Field(..., description="Unix seconds relative to episode start")
    labels: Dict[str, str] = Field(
        default_factory=dict, description="Alert metadata labels"
    )


class LogLine(FrozenBaseModel):
    """Single log line from a service."""

    timestamp: int = Field(..., description="Unix seconds relative to episode start")
    service: str = Field(..., description="Service emitting the log line")
    level: str = Field(..., description="Log level (INFO/WARN/ERROR/FATAL)")
    message: str = Field(..., description="Log message")
    trace_id: Optional[str] = Field(
        default=None, description="Optional trace ID for correlation"
    )


class ServiceNode(FrozenBaseModel):
    """Service dependency graph node."""

    name: str = Field(..., description="Service name")
    depends_on: List[str] = Field(
        default_factory=list, description="Upstream dependencies"
    )
    team: str = Field(..., description="Owning team")
    tier: int = Field(..., description="Service tier (1=user-facing, 2=internal, 3=infra)")


class ActionType(str, Enum):
    CLASSIFY_ALERT = "classify_alert"
    LABEL_EVENT = "label_event"
    SILENCE_ALERT = "silence_alert"
    ESCALATE = "escalate"
    REMEDIATE = "remediate"
    INVESTIGATE = "investigate"
    HOLD = "hold"


class Severity(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class EventLabel(str, Enum):
    ROOT_CAUSE = "root_cause"
    SYMPTOM = "symptom"
    CONTRIBUTING_FACTOR = "contributing_factor"
    NOISE = "noise"
    UNRELATED = "unrelated"
    FALSE_POSITIVE = "false_positive"


class RemediationAction(str, Enum):
    ROLLBACK_DEPLOY = "rollback_deploy"
    RESTART_SERVICE = "restart_service"
    SCALE_REPLICAS = "scale_replicas"
    FLUSH_CACHE = "flush_cache"
    FAILOVER_DATABASE = "failover_database"
    NOTIFY_DOWNSTREAM = "notify_downstream"
    REROUTE_TRAFFIC = "reroute_traffic"


class Observation(OpenEnvObservation):
    """Observation returned to the agent each step."""

    model_config = ConfigDict(frozen=True)

    step: int = Field(..., description="Current step in the episode")
    alerts: List[Alert] = Field(default_factory=list, description="Active alerts")
    logs: List[LogLine] = Field(default_factory=list, description="Recent log lines")
    service_graph: List[ServiceNode] = Field(
        default_factory=list, description="Service dependency graph"
    )
    active_incidents: List[str] = Field(
        default_factory=list, description="Incident IDs currently open"
    )
    budget_remaining: int = Field(..., description="Steps remaining in the episode")
    context: Dict[str, str] = Field(
        default_factory=dict, description="Scenario metadata"
    )


class Action(OpenEnvAction):
    """Agent action, fully discrete."""

    model_config = ConfigDict(frozen=True)

    action_type: ActionType = Field(..., description="Action type")
    target_id: str = Field(..., description="Alert ID, event ID, or service name")
    severity: Optional[Severity] = Field(
        default=None, description="Severity for CLASSIFY_ALERT"
    )
    event_label: Optional[EventLabel] = Field(
        default=None, description="Label for LABEL_EVENT"
    )
    team: Optional[str] = Field(
        default=None, description="Team name for ESCALATE"
    )
    remediation: Optional[RemediationAction] = Field(
        default=None, description="Remediation action for REMEDIATE"
    )
