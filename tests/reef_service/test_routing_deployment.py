"""``reef serve`` hosts the routing stack; without a manifest nothing changes."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from reef.cli import main
from reef.service.deploy import orchestrator
from reef.service.deploy.config_utils import DeployConfigError, load_config
from reef.service.deploy.execution import validate_services
from reef.service.deploy.routing import attach_routing_services
from reef.service.deploy.service_config import service_config_from_mapping


def write_stack(tmp_path: Path, **reef_extra) -> Path:
    reef = {
        "recipe": "recipe",
        "host": "127.0.0.1",
        "port": 8977,
        "upstream_url": "http://127.0.0.1:18434",
        "upstream_model": "fake-model",
        "tokens": [],
    }
    reef.update(reef_extra)
    # An unversioned file is a custom process stack: it declares its own
    # services, and routing attaches beside them.
    config = {
        "reef": reef,
        "ready_timeout": 30,
        "services": [{"name": "reef", "command": ["python", "-m", "reef.service"]}],
    }
    path = tmp_path / "stack.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def write_manifest(tmp_path: Path) -> Path:
    manifest = {
        "targets": {"primary": {"endpoint": "http://127.0.0.1:18434", "model": "fake-model"}},
        "scenarios": {"coding": {"targets": ["primary"], "static": "primary"}},
    }
    path = tmp_path / "routing.yaml"
    path.write_text(yaml.safe_dump(manifest))
    return path


ROUTING_DEPLOYMENT = {
    "semantic-router": {"command": ["/bin/cat"], "ready_timeout": 5},
    "envoy": {"command": "/bin/true", "ready_timeout": 5},
}


@pytest.fixture
def captured_stack(monkeypatch):
    captured = {}

    class Stack:
        exit_code = 0

        def __init__(self, config, services, run_dir, ready_timeout_default, config_path, source_root=None):
            captured.update(config=load_config(config_path), services=services)

        def start(self):
            pass

        def block(self):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(orchestrator, "_Stack", Stack)
    return captured


def test_attach_is_a_noop_without_a_manifest() -> None:
    services = [{"name": "reef", "command": ["python", "-m", "reef.service"]}]
    config = {"reef": {"recipe": "recipe"}, "services": [dict(services[0])]}
    attach_routing_services(config)
    assert config["services"] == services


def test_attach_wires_the_routing_stack_before_the_http_service(tmp_path) -> None:
    manifest = write_manifest(tmp_path)
    config = {
        "reef": {
            "recipe": "recipe",
            "routing_manifest": str(manifest),
            "routing_deployment": ROUTING_DEPLOYMENT,
        },
        "services": [{"name": "reef", "command": ["python", "-m", "reef.service"]}],
    }
    attach_routing_services(config)
    names = [service["name"] for service in config["services"]]
    assert names == ["semantic-router", "envoy", "reef"]
    router, envoy, http = config["services"]
    assert router["command"] == ["/bin/cat"] and "depends_on" not in router
    assert envoy["command"] == "/bin/true" and envoy["depends_on"] == ["semantic-router"]
    assert http["depends_on"] == ["envoy"]
    validate_services(config, tmp_path / "stack.yaml")


def test_attach_rejects_broken_manifests_and_incomplete_deployments(tmp_path) -> None:
    manifest = write_manifest(tmp_path)
    reef = {"recipe": "recipe", "routing_manifest": str(manifest)}
    services = [{"name": "reef", "command": ["python", "-m", "reef.service"]}]

    with pytest.raises(DeployConfigError, match="routing_deployment must be an object"):
        attach_routing_services({"reef": dict(reef), "services": services})
    with pytest.raises(DeployConfigError, match="routing_deployment accepts only"):
        attach_routing_services(
            {"reef": {**reef, "routing_deployment": {"sidecar": {"command": "true"}}}, "services": services}
        )
    with pytest.raises(DeployConfigError, match="command is required"):
        attach_routing_services(
            {
                "reef": {**reef, "routing_deployment": {"semantic-router": {}, "envoy": {"command": "true"}}},
                "services": services,
            }
        )
    with pytest.raises(DeployConfigError, match="needs the Reef HTTP service"):
        attach_routing_services(
            {
                "reef": {**reef, "routing_deployment": ROUTING_DEPLOYMENT},
                "services": [{"name": "other", "command": "true"}],
            }
        )

    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump({"targets": {}, "scenarios": {}}))
    with pytest.raises(DeployConfigError, match="non-empty 'targets'"):
        attach_routing_services(
            {
                "reef": {**reef, "routing_manifest": str(broken), "routing_deployment": ROUTING_DEPLOYMENT},
                "services": services,
            }
        )
    with pytest.raises(DeployConfigError, match="not found"):
        attach_routing_services(
            {
                "reef": {
                    **reef,
                    "routing_manifest": str(tmp_path / "absent.yaml"),
                    "routing_deployment": ROUTING_DEPLOYMENT,
                },
                "services": services,
            }
        )


def test_service_config_carries_the_routing_fields(tmp_path) -> None:
    manifest = write_manifest(tmp_path)
    config = {
        "reef": {
            "recipe": "recipe",
            "routing_manifest": str(manifest),
            "routing_deployment": ROUTING_DEPLOYMENT,
        }
    }
    settings = service_config_from_mapping(config)
    assert settings.routing_manifest == str(manifest)
    assert settings.routing_deployment["semantic-router"]["command"] == ["/bin/cat"]
    defaults = service_config_from_mapping({"reef": {"recipe": "recipe"}})
    assert defaults.routing_manifest is None and defaults.routing_deployment == {}


def test_serve_hosts_routing_only_when_configured(tmp_path, captured_stack, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for key in list(__import__("os").environ):
        if key.startswith("REEF_"):
            monkeypatch.delenv(key)
    plain = write_stack(tmp_path)
    with pytest.raises(SystemExit):
        main(["serve", "-c", str(plain)])
    without_routing = captured_stack["services"]

    routed = write_stack(
        tmp_path,
        routing_manifest=str(write_manifest(tmp_path)),
        routing_deployment=ROUTING_DEPLOYMENT,
    )
    with pytest.raises(SystemExit):
        main(["serve", "-c", str(routed)])
    routed_services = captured_stack["services"]

    assert [service["name"] for service in without_routing] == ["reef"]
    names = [service["name"] for service in routed_services]
    assert names[:2] == ["semantic-router", "envoy"] and names[-1] == "reef"
    by_name = {service["name"]: service for service in routed_services}
    assert by_name["reef"]["depends_on"] == ["envoy"]
    assert by_name["envoy"]["depends_on"] == ["semantic-router"]
    validate_services({"services": routed_services}, tmp_path / "routing.yaml")
