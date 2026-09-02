"""A bad deployment stack fails as a typed config error before anything starts (#141, #143)."""

from __future__ import annotations

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
        _write(tmp_path, 'services:\n  - name: worker\n    command: ["${REEF_PYTHON}", "-m", "worker"]\n')
    )

    assert config["services"][0]["command"] == [sys.executable, "-m", "worker"]


@pytest.mark.unit
def test_reef_python_can_be_overridden_without_changing_bare_python(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REEF_PYTHON", "/opt/worker venv/bin/python")
    config = load_config(
        _write(
            tmp_path,
            "services:\n"
            "  - name: managed-worker\n"
            '    command: ["${REEF_PYTHON}", "-m", "managed_worker"]\n'
            "  - name: path-worker\n"
            "    command: python -m path_worker\n",
        )
    )

    assert config["services"][0]["command"][0] == "/opt/worker venv/bin/python"
    assert config["services"][1]["command"] == "python -m path_worker"


@pytest.mark.unit
def test_services_must_be_a_non_empty_list_of_named_objects() -> None:
    with pytest.raises(DeployConfigError, match="non-empty 'services' list"):
        validate_services({}, "stack.yaml")
    with pytest.raises(DeployConfigError, match="non-empty 'services' list"):
        validate_services({"services": []}, "stack.yaml")
    with pytest.raises(DeployConfigError, match=r"services\[0\] must be an object, not str"):
        validate_services({"services": ["not-an-object"]}, "stack.yaml")
    with pytest.raises(DeployConfigError, match=r"services\[1\] must have a non-empty 'name'"):
        validate_services({"services": [{"name": "a", "command": "a"}, {"command": "x"}]}, "stack.yaml")


@pytest.mark.unit
@pytest.mark.parametrize("command", [None, "", [], [""], ["python", 3]])
def test_service_command_must_be_a_string_or_string_list(command) -> None:
    with pytest.raises(DeployConfigError, match="non-empty 'command' string or list of strings"):
        validate_services({"services": [{"name": "worker", "command": command}]}, "stack.yaml")


@pytest.mark.unit
def test_command_argv_supports_exact_lists_and_legacy_strings() -> None:
    config = {"reef": {"port": 9123}}

    assert orchestrator._command_argv(
        config,
        ["/opt/reef env/bin/python", "-m", "reef.service", "--port=${reef.port}", ""],
    ) == ["/opt/reef env/bin/python", "-m", "reef.service", "--port=9123", ""]
    assert orchestrator._command_argv(config, "python -m worker --port=${reef.port}") == [
        "python",
        "-m",
        "worker",
        "--port=9123",
    ]


@pytest.mark.unit
def test_duplicate_service_names_are_named() -> None:
    config = {
        "services": [
            {"name": "worker", "command": "worker"},
            {"name": "api", "command": "api"},
            {"name": "worker", "command": "worker"},
            {"name": "api", "command": "api"},
        ]
    }
    with pytest.raises(DeployConfigError, match="service names must be unique; duplicated: api, worker"):
        validate_services(config, "stack.yaml")


@pytest.mark.unit
def test_valid_services_pass_through_in_order() -> None:
    services = [{"name": "b", "command": "b"}, {"name": "a", "command": ["a", ""]}]
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
def test_orchestrator_uses_exactly_the_declared_services(tmp_path: Path, monkeypatch) -> None:
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
        "    command: model-server\n"
        "  - name: reef\n"
        '    command: ["${REEF_PYTHON}", "-m", "reef.service"]\n'
        "    depends_on: [model]\n",
    )

    assert orchestrator._run_orchestrator(str(path)) == 0
    services = captured["services"]
    assert isinstance(services, list)
    assert services == [
        {"name": "model", "command": "model-server"},
        {"name": "reef", "command": [sys.executable, "-m", "reef.service"], "depends_on": ["model"]},
    ]


@pytest.mark.unit
def test_cli_exits_2_without_a_traceback(tmp_path: Path, capsys) -> None:
    path = _write(tmp_path, "- not\n- an\n- object\n")
    with pytest.raises(SystemExit) as exit_info:
        cli_main(["serve", "-c", str(path)])
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "[reef] ERROR" in err and "must be a YAML object" in err
    assert "Traceback" not in err
