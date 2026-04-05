from pathlib import Path

import yaml

from env.generator import _load_scenarios, _load_service_graph


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_service_graph_dependencies_exist():
    services = _load_service_graph()
    names = {service.name for service in services}
    external = {"smtp-relay"}
    for service in services:
        for dep in service.depends_on:
            assert dep in names or dep in external


def test_service_graph_has_downstream():
    services = _load_service_graph()
    dependents = {service.name: 0 for service in services}
    for service in services:
        for dep in service.depends_on:
            dependents[dep] = dependents.get(dep, 0) + 1
    assert any(count > 0 for count in dependents.values())


def test_alert_templates_have_severity_map():
    payload = yaml.safe_load((DATA_DIR / "alert_templates.yaml").read_text())
    templates = payload.get("templates", [])
    assert templates
    for template in templates:
        severity_map = template.get("severity_map", {})
        assert "P0" in severity_map
        assert "P1" in severity_map
        assert "P2" in severity_map
        assert "P3" in severity_map


def test_scenarios_have_required_fields():
    scenarios = _load_scenarios()
    assert scenarios
    for scenario in scenarios:
        assert "id" in scenario
        assert "events" in scenario
        assert "root_cause_service" in scenario
        assert isinstance(scenario["events"], list)
        assert scenario["events"]
        ids = {event["id"] for event in scenario["events"]}
        assert len(ids) == len(scenario["events"])


def test_scenarios_use_multiple_labels():
    scenarios = _load_scenarios()
    labels = set()
    for scenario in scenarios:
        for event in scenario["events"]:
            labels.add(event["label"])
    assert len(labels) >= 3
