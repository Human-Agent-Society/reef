"""A bad deployment stack fails as a typed config error before anything starts (#141, #143)."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest

import reef.service.deploy.orchestrator as orchestrator
from reef.cli import main as cli_main
from reef.service.deploy.config import DeployConfigError, load_config, validate_services

VALID = "services:\n  - name: worker\n    command: python -c 'print(1)'\n"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "stack.yaml"
    path.write_text(text)
    return path


@pytest.mark.unit
def test_malformed_yaml_is_a_config_error_naming_the_path(tmp_path: Path) -> None:
    path = _write(tmp_path, "services: [unclosed\n")
    with pytest.raises(DeployConfigError, match="not valid YAML") as caught:
        load_config(path)
    assert str(path) in str(caught.value)


@pytest.mark.unit
def test_non_object_root_is_a_config_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "- not\n- an\n- object\n")
    with pytest.raises(DeployConfigError, match="must be a YAML object at the root, not list"):
        load_config(path)


@pytest.mark.unit
def test_empty_file_loads_as_an_empty_config(tmp_path: Path) -> None:
    assert load_config(_write(tmp_path, "")) == {}


@pytest.mark.unit
def test_reef_python_defaults_to_the_launching_interpreter(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("REEF_PYTHON", raising=False)
    config = load_config(
        _write(tmp_path, "services:\n  - name: worker\n    command: '\"${REEF_PYTHON}\" -m worker'\n")
    )

    assert shlex.split(config["services"][0]["command"])[0] == sys.executable


@pytest.mark.unit
def test_reef_python_can_be_overridden_without_changing_bare_python(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REEF_PYTHON", "/opt/worker venv/bin/python")
    config = load_config(
        _write(
            tmp_path,
            "services:\n"
            "  - name: managed-worker\n"
            "    command: '\"${REEF_PYTHON}\" -m managed_worker'\n"
            "  - name: path-worker\n"
            "    command: python -m path_worker\n",
        )
    )

    assert shlex.split(config["services"][0]["command"])[0] == "/opt/worker venv/bin/python"
    assert config["services"][1]["command"] == "python -m path_worker"


@pytest.mark.unit
def test_auxiliary_services_are_optional_but_must_be_a_list_of_named_objects() -> None:
    assert validate_services({}, "stack.yaml") == []
    assert validate_services({"services": []}, "stack.yaml") == []
    with pytest.raises(DeployConfigError, match="'services' must be a list"):
        validate_services({"services": "worker"}, "stack.yaml")
    with pytest.raises(DeployConfigError, match=r"services\[0\] must be an object, not str"):
        validate_services({"services": ["not-an-object"]}, "stack.yaml")
    with pytest.raises(DeployConfigError, match=r"services\[1\] must have a non-empty 'name'"):
        validate_services({"services": [{"name": "a"}, {"command": "x"}]}, "stack.yaml")


@pytest.mark.unit
def test_reef_is_a_reserved_service_name() -> None:
    with pytest.raises(DeployConfigError, match="service name 'reef' is reserved"):
        validate_services({"services": [{"name": "reef", "command": "anything"}]}, "stack.yaml")


@pytest.mark.unit
def test_builtin_reef_service_uses_the_launcher_and_depends_on_auxiliary_services() -> None:
    config = {"reef": {"recipe": "recipe", "port": 9123, "process": {"ready_timeout": 45}}}
    auxiliaries = [{"name": "model"}, {"name": "trainer"}]

    service = orchestrator._builtin_reef_service(config, auxiliaries)

    assert shlex.split(service["command"]) == [sys.executable, "-m", "reef.service"]
    assert service["ready"] == "curl -sf http://127.0.0.1:9123/healthz"
    assert service["depends_on"] == ["model", "trainer"]
    assert service["ready_timeout"] == 45


@pytest.mark.unit
def test_builtin_reef_process_options_must_be_an_object() -> None:
    with pytest.raises(DeployConfigError, match=r"reef\.process must be an object"):
        orchestrator._builtin_reef_service({"reef": {"recipe": "recipe", "process": "invalid"}}, [])


@pytest.mark.unit
def test_duplicate_service_names_are_named() -> None:
    config = {"services": [{"name": "worker"}, {"name": "api"}, {"name": "worker"}, {"name": "api"}]}
    with pytest.raises(DeployConfigError, match="service names must be unique; duplicated: api, worker"):
        validate_services(config, "stack.yaml")


@pytest.mark.unit
def test_valid_services_pass_through_in_order() -> None:
    services = [{"name": "b"}, {"name": "a"}]
    assert validate_services({"services": services}, "stack.yaml") == services


@pytest.mark.unit
def test_invalid_stack_never_downloads_or_launches(tmp_path: Path, monkeypatch) -> None:
    def must_not_run(*args, **kwargs):
        raise RuntimeError("reached a stage that must not run for an invalid stack")

    monkeypatch.setattr(orchestrator, "resolve_model_paths", must_not_run)
    monkeypatch.setattr(orchestrator, "_Stack", must_not_run)
    path = _write(
        tmp_path, "services:\n  - name: worker\n    command: sleep 60\n  - name: worker\n    command: echo 1\n"
    )
    with pytest.raises(DeployConfigError, match="duplicated: worker"):
        orchestrator._run_orchestrator(str(path))
    assert not (tmp_path / "reef-stack").exists()


@pytest.mark.unit
def test_orchestrator_appends_the_builtin_reef_service(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class StackStub:
        exit_code = 0

        def __init__(self, config, services, run_dir, ready_timeout_default, config_path):
            captured["services"] = services

        def start(self):
            pass

        def block(self):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(orchestrator, "_Stack", StackStub)
    monkeypatch.setattr(orchestrator, "resolve_model_paths", lambda config: False)
    path = _write(
        tmp_path,
        f"run_dir: {tmp_path / 'run'}\n"
        "reef:\n"
        "  recipe: recipe\n"
        "services:\n"
        "  - name: model\n"
        "    command: model-server\n",
    )

    assert orchestrator._run_orchestrator(str(path)) == 0
    services = captured["services"]
    assert isinstance(services, list)
    assert [service["name"] for service in services] == ["model", "reef"]
    assert services[-1]["depends_on"] == ["model"]


@pytest.mark.unit
def test_cli_exits_2_without_a_traceback(tmp_path: Path, capsys) -> None:
    path = _write(tmp_path, "- not\n- an\n- object\n")
    with pytest.raises(SystemExit) as exit_info:
        cli_main(["serve", "-c", str(path)])
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "[reef] ERROR" in err and "must be a YAML object" in err
    assert "Traceback" not in err
