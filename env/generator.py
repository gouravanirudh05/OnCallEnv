"""
Deterministic episode generator for OnCallEnv.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from .models import Alert, LogLine, ServiceNode
from .types import EpisodeState


TASK_NAMES = {
    1: "severity_classification",
    2: "alert_storm",
    3: "timeline_labelling",
}


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SERVICE_GRAPH_PATH = DATA_DIR / "service_graph.yaml"
ALERT_TEMPLATES_PATH = DATA_DIR / "alert_templates.yaml"


def generate_episode(task_id: int, seed: int, budget: int) -> EpisodeState:
    """Generate a deterministic episode state for a given task and seed."""

    rng = random.Random(seed)
    episode_id = f"episode-{task_id}-{seed}"

    service_graph = _load_service_graph()
    templates = _load_alert_templates()
    alerts, logs, ground_truth, context = _generate_task_payload(
        rng, task_id, service_graph, templates
    )

    return EpisodeState(
        episode_id=episode_id,
        task_id=task_id,
        seed=seed,
        step=0,
        budget=budget,
        alerts=alerts,
        logs=logs,
        service_graph=service_graph,
        active_incidents=[],
        context=context,
        ground_truth=ground_truth,
    )


def _load_service_graph() -> List[ServiceNode]:
    payload = _load_yaml(SERVICE_GRAPH_PATH)
    services = payload.get("services", [])
    return [ServiceNode(**service) for service in services]


def _load_alert_templates() -> List[Dict[str, object]]:
    payload = _load_yaml(ALERT_TEMPLATES_PATH)
    return payload.get("templates", [])


def _generate_task_payload(
    rng: random.Random,
    task_id: int,
    service_graph: List[ServiceNode],
    templates: List[Dict[str, object]],
) -> Tuple[List[Alert], List[LogLine], Dict[str, object], Dict[str, str]]:
    """Task-specific payload generator."""

    if task_id not in TASK_NAMES:
        raise ValueError(f"Unknown task_id: {task_id}")

    context = {"task": TASK_NAMES[task_id]}

    if task_id == 1:
        return _generate_task1(rng, service_graph, templates, context)

    if task_id == 2:
        return _generate_task2(rng, service_graph, templates, context)

    if task_id == 3:
        return _generate_task3(rng, service_graph, context)

    alerts = _generate_alerts(rng, service_graph, templates, count=5)
    logs = _generate_logs(rng, service_graph, count=5)
    ground_truth = {
        "alerts": {alert.id: "P3" for alert in alerts},
        "event_labels": {},
        "root_cause": None,
    }
    return alerts, logs, ground_truth, context


def _generate_task1(
    rng: random.Random,
    service_graph: List[ServiceNode],
    templates: List[Dict[str, object]],
    context: Dict[str, str],
) -> Tuple[List[Alert], List[LogLine], Dict[str, object], Dict[str, str]]:
    """Task 1: severity classification."""

    alert_count = rng.randint(5, 8)
    alerts = _generate_alerts(rng, service_graph, templates, count=alert_count)

    # Choose two alerts to correlate into a P0 escalation.
    correlated = rng.sample(alerts, k=2)
    correlated_ids = {alert.id for alert in correlated}

    # Choose one alert for time-of-day adjustment.
    remaining = [alert for alert in alerts if alert.id not in correlated_ids]
    adjusted_alert = rng.choice(remaining)

    # Choose one alert to be a known false positive.
    remaining = [alert for alert in remaining if alert.id != adjusted_alert.id]
    false_positive_alert = rng.choice(remaining)

    ground_truth: Dict[str, str] = {}
    for alert in alerts:
        ground_truth[alert.id] = _infer_severity(alert, templates)

    # Override severities for task-specific twists.
    for alert_id in correlated_ids:
        ground_truth[alert_id] = "P0"

    # Combine correlated alerts into a P0 incident.
    context["correlated_alerts"] = json.dumps(sorted(list(correlated_ids)))
    context["correlated_incident_severity"] = "P0"

    ground_truth[adjusted_alert.id] = "P3"
    adjusted_alert.labels["pattern"] = "weekday_morning_high_traffic"
    context["adjusted_alert_id"] = adjusted_alert.id

    ground_truth[false_positive_alert.id] = "P3"
    false_positive_alert.labels["signature"] = "known_false_positive"
    context["false_positive_alert_id"] = false_positive_alert.id

    logs = _generate_logs(rng, service_graph, count=5)
    for alert in correlated:
        alert.severity_raw = "P2"

    ground_truth_bundle = {
        "alerts": ground_truth,
        "event_labels": {},
        "root_cause": None,
        "correlated_incident": sorted(list(correlated_ids)),
        "correlated_incident_severity": "P0",
        "adjusted_alert_id": adjusted_alert.id,
        "false_positive_alert_id": false_positive_alert.id,
    }
    return alerts, logs, ground_truth_bundle, context


def _generate_task2(
    rng: random.Random,
    service_graph: List[ServiceNode],
    templates: List[Dict[str, object]],
    context: Dict[str, str],
) -> Tuple[List[Alert], List[LogLine], Dict[str, object], Dict[str, str]]:
    """Task 2: alert storm deduplication."""

    root_candidates = _services_with_downstream(service_graph)
    root = rng.choice(root_candidates or service_graph)
    downstream = _collect_downstream(root.name, service_graph)
    unrelated = _pick_unrelated_services(rng, root.name, service_graph, count=2)

    alerts: List[Alert] = []
    event_labels: Dict[str, str] = {}

    def append_alert(alert: Alert, label: str) -> None:
        alerts.append(alert)
        event_labels[alert.id] = label

    # Root cause alert.
    root_alert = _build_alert_from_template(rng, root, templates, alert_id="alert-root")
    root_alert.severity_raw = rng.choice(["P1", "P2"])
    append_alert(root_alert, "root_cause")

    # Symptom alerts for downstream services.
    for idx, service_name in enumerate(sorted(downstream)):
        service = _find_service(service_graph, service_name)
        alert = _build_alert_from_template(
            rng, service, templates, alert_id=f"alert-symptom-{idx}"
        )
        append_alert(alert, "symptom")

    # Unrelated alerts.
    for idx, service in enumerate(unrelated):
        alert = _build_alert_from_template(
            rng, service, templates, alert_id=f"alert-unrelated-{idx}"
        )
        append_alert(alert, "unrelated")

    # False positives (labelled as false_positive, should be silenced).
    unrelated_ids = [
        alert_id for alert_id, label in event_labels.items() if label == "unrelated"
    ]
    symptom_ids = [
        alert_id for alert_id, label in event_labels.items() if label == "symptom"
    ]

    false_positive_pool = list(unrelated_ids)
    if not false_positive_pool and len(symptom_ids) > 1:
        false_positive_pool = symptom_ids[1:]

    false_positive_count = min(2, len(false_positive_pool))
    false_positive_candidates = rng.sample(false_positive_pool, k=false_positive_count)
    for alert_id in false_positive_candidates:
        event_labels[alert_id] = "false_positive"
        alert = next((item for item in alerts if item.id == alert_id), None)
        if alert is not None:
            alert.labels["signature"] = "known_false_positive"

    logs = _generate_logs(rng, service_graph, count=8)
    ground_truth = {
        "alerts": {alert.id: "P1" for alert in alerts},
        "event_labels": event_labels,
        "root_cause": root.name,
        "escalation": {
            "team": root.team,
            "severity": "P0",
        },
    }

    context["root_cause_service"] = root.name
    return alerts, logs, ground_truth, context


def _generate_task3(
    rng: random.Random,
    service_graph: List[ServiceNode],
    context: Dict[str, str],
) -> Tuple[List[Alert], List[LogLine], Dict[str, object], Dict[str, str]]:
    """Task 3: incident timeline labeling."""

    scenarios = _load_scenarios()
    if not scenarios:
        raise ValueError("No scenario templates found")

    scenario = rng.choice(scenarios)
    events = list(scenario["events"])
    rng.shuffle(events)

    event_labels: Dict[str, str] = {
        event["id"]: event["label"] for event in scenario["events"]
    }

    context["scenario_id"] = scenario["id"]

    logs = []
    alerts = []
    for event in events:
        if event["type"] == "log":
            logs.append(
                LogLine(
                    timestamp=int(event["timestamp"]),
                    service=event["service"],
                    level="ERROR",
                    message=event["detail"],
                    trace_id=None,
                )
            )
        if event["type"] == "alert":
            alerts.append(
                Alert(
                    id=event["id"],
                    service=event["service"],
                    metric="alert",
                    value=1.0,
                    threshold=1.0,
                    severity_raw="P1",
                    timestamp=int(event["timestamp"]),
                    labels={"scenario": scenario["id"]},
                )
            )

    investigations = scenario.get("investigations", [])
    ground_truth = {
        "alerts": {alert.id: "P2" for alert in alerts},
        "event_labels": event_labels,
        "root_cause": scenario.get("root_cause_service"),
        "investigations": investigations,
    }

    return alerts, logs, ground_truth, context


def _generate_alerts(
    rng: random.Random,
    service_graph: List[ServiceNode],
    templates: List[Dict[str, object]],
    count: int,
) -> List[Alert]:
    alerts: List[Alert] = []
    for idx in range(count):
        service = rng.choice(service_graph)
        template = rng.choice(templates)
        metric = str(template["metric"])
        base_threshold = float(template["base_threshold"])
        multiplier = rng.uniform(0.8, 1.4)
        threshold = round(base_threshold * multiplier, 3)
        value = round(threshold * rng.uniform(0.8, 3.0), 3)
        severity_raw = _sample_raw_severity(rng)
        labels = dict(template.get("labels", {}))
        labels["service_tier"] = str(service.tier)
        alerts.append(
            Alert(
                id=f"alert-{idx}",
                service=service.name,
                metric=metric,
                value=value,
                threshold=threshold,
                severity_raw=severity_raw,
                timestamp=idx * 10,
                labels={
                    **labels,
                    "region": rng.choice(["us-east-1", "eu-west-1"]),
                },
            )
        )
    return alerts


def _generate_logs(
    rng: random.Random, service_graph: List[ServiceNode], count: int
) -> List[LogLine]:
    logs: List[LogLine] = []
    for idx in range(count):
        service = rng.choice(service_graph)
        logs.append(
            LogLine(
                timestamp=idx * 8,
                service=service.name,
                level=rng.choice(["INFO", "WARN", "ERROR"]),
                message="synthetic log line",
                trace_id=None,
            )
        )
    return logs


def _build_alert_from_template(
    rng: random.Random,
    service: ServiceNode,
    templates: List[Dict[str, object]],
    alert_id: str,
) -> Alert:
    template = rng.choice(templates)
    metric = str(template["metric"])
    base_threshold = float(template["base_threshold"])
    multiplier = rng.uniform(0.8, 1.4)
    threshold = round(base_threshold * multiplier, 3)
    value = round(threshold * rng.uniform(1.0, 3.2), 3)
    labels = dict(template.get("labels", {}))
    labels["service_tier"] = str(service.tier)
    labels["incident_group"] = "storm"
    return Alert(
        id=alert_id,
        service=service.name,
        metric=metric,
        value=value,
        threshold=threshold,
        severity_raw=_sample_raw_severity(rng),
        timestamp=rng.randint(0, 180),
        labels={
            **labels,
            "region": rng.choice(["us-east-1", "eu-west-1"]),
        },
    )


def _collect_downstream(root: str, service_graph: List[ServiceNode]) -> List[str]:
    downstream: List[str] = []
    queue = [root]
    while queue:
        current = queue.pop(0)
        for service in service_graph:
            if current in service.depends_on and service.name not in downstream:
                downstream.append(service.name)
                queue.append(service.name)
    return downstream


def _pick_unrelated_services(
    rng: random.Random,
    root_name: str,
    service_graph: List[ServiceNode],
    count: int,
) -> List[ServiceNode]:
    related = {root_name}
    related.update(_collect_downstream(root_name, service_graph))
    candidates = [service for service in service_graph if service.name not in related]
    rng.shuffle(candidates)
    return candidates[:count]


def _find_service(service_graph: List[ServiceNode], name: str) -> ServiceNode:
    for service in service_graph:
        if service.name == name:
            return service
    raise ValueError(f"Unknown service: {name}")


def _services_with_downstream(service_graph: List[ServiceNode]) -> List[ServiceNode]:
    dependents = {service.name: 0 for service in service_graph}
    for service in service_graph:
        for dependency in service.depends_on:
            if dependency in dependents:
                dependents[dependency] += 1
    return [
        service for service in service_graph if dependents.get(service.name, 0) > 0
    ]


def _load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_scenarios() -> List[Dict[str, object]]:
    scenario_dir = DATA_DIR / "scenarios"
    scenarios: List[Dict[str, object]] = []
    for path in sorted(scenario_dir.glob("*.yaml")):
        payload = _load_yaml(path)
        if payload:
            scenarios.append(payload)
    return scenarios


def _sample_raw_severity(rng: random.Random) -> str:
    return rng.choice(["P0", "P1", "P2", "P3"])


def _infer_severity(alert: Alert, templates: List[Dict[str, object]]) -> str:
    template = next(
        (item for item in templates if item.get("metric") == alert.metric), None
    )
    if not template:
        return "P3"

    severity_map = template.get("severity_map", {})
    thresholds = {key: float(val) for key, val in severity_map.items()}
    if alert.value >= thresholds.get("P0", float("inf")):
        return "P0"
    if alert.value >= thresholds.get("P1", float("inf")):
        return "P1"
    if alert.value >= thresholds.get("P2", float("inf")):
        return "P2"
    if alert.value >= thresholds.get("P3", float("inf")):
        return "P3"
    return "P3"
