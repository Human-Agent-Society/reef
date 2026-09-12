"""Public deployment settings share one parser for CLI and YAML values."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from reef.cli import main as cli_main
from reef.service.deploy import orchestrator
from reef.service.deploy.config import interpolate_environment, load_config
from reef.service.deploy.orchestrator import _apply_overrides, _parse_overrides, build_serve_parser
from reef.service.deploy.settings import (
    ServiceSettings,
    normalize_service_config,
    parse_service_arguments,
    service_config_arguments,
    service_settings_from_config,
)


@pytest.mark.parametrize(
    ("name", "yaml_value", "cli_value"),
    [
        ("port", 9123, "9123"),
        ("inference_timeout_s", 12.5, "12.5"),
        ("upstream_model", "00123", "00123"),
        ("upstream_model", "true", "true"),
        ("upstream_model", "null", "null"),
        ("allow_implicit_scenario_creation", False, "false"),
        ("tokens", ["first", "second"], '["first", "second"]'),
        ("tokens", [], "[]"),
        (
            "inference_backend_config",
            {"nested": {"enabled": False, "ids": [1, 2]}},
            '{"nested": {"enabled": false, "ids": [1, 2]}}',
        ),
        ("inference_backend_config", {}, "{}"),
    ],
)
def test_yaml_and_cli_use_the_same_field_types(name, yaml_value, cli_value):
    yaml_values = parse_service_arguments({"reef": {name: yaml_value}})
    cli_values = vars(build_serve_parser().parse_args([f"--{name.replace('_', '-')}={cli_value}"]))
    assert yaml_values[name] == cli_values[name]
    assert type(yaml_values[name]) is type(cli_values[name])


def test_parser_defaults_come_from_settings():
    defaults = parse_service_arguments({})
    for argument in service_config_arguments():
        if argument.name in {"token", "recipe"}:
            continue
        field = next(field for field in fields(ServiceSettings) if field.name == argument.name)
        assert defaults[argument.name] == getattr(ServiceSettings(recipe="recipe"), field.name)


def test_recipe_setting_and_launcher_profile_are_independent_arguments():
    args = build_serve_parser().parse_args(["--recipe", "harness-evolve", "--reef.recipe", "custom-preset"])
    assert args.recipe == "harness-evolve"
    assert args.config_recipe == "custom-preset"
    assert parse_service_arguments({"reef": {"recipe": "custom-preset"}})["recipe"] == "custom-preset"


@pytest.mark.parametrize("flag", ["--port", "--reef.port"])
def test_cli_beats_yaml_and_omitted_cli_leaves_yaml_value(flag):
    config = {"reef": {"recipe": "recipe", "port": 9123}}
    assert service_settings_from_config(config).port == 9123
    updated = _apply_overrides(config, _parse_overrides([flag, "8123"]))
    assert service_settings_from_config(normalize_service_config(updated)).port == 8123
    assert config["reef"]["port"] == 9123


def test_last_cli_alias_wins_even_when_a_previous_spelling_is_repeated():
    overrides = _parse_overrides(["--port", "7000", "--reef.port", "8000", "--port", "9000"])
    config = _apply_overrides({"reef": {"recipe": "recipe", "port": 6000}}, overrides)
    assert service_settings_from_config(config).port == 9000


@pytest.mark.parametrize(
    "arguments",
    [
        ["--host"],
        ["--port", "--tokens", "[]"],
        ["--no-allow-implicit-scenario-creation=false"],
        ["--no-allow-implicit-scenario-creation", "true"],
    ],
)
def test_declared_cli_options_reject_missing_or_unexpected_values(arguments):
    with pytest.raises(ValueError, match="value"):
        _parse_overrides(arguments)


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["--allow-implicit-scenario-creation"], True),
        (["--allow_implicit_scenario_creation", "false"], False),
        (["--no-allow-implicit-scenario-creation"], False),
        (["--no-allow-implicit-scenario-creation", "--allow-implicit-scenario-creation"], True),
    ],
)
def test_boolean_flags_can_enable_and_disable_yaml_settings(arguments, expected):
    config = _apply_overrides(
        {"reef": {"recipe": "recipe", "allow_implicit_scenario_creation": not expected}}, _parse_overrides(arguments)
    )
    assert service_settings_from_config(config).allow_implicit_scenario_creation is expected


def test_empty_container_overrides_replace_yaml_values_and_survive_child_serialization():
    config = _apply_overrides(
        {"reef": {"recipe": "recipe", "tokens": ["old"], "inference_backend_config": {"old": 1}}},
        {"tokens": "[]", "inference-backend-config": "{}"},
    )
    normalized = normalize_service_config(config)
    child = yaml.safe_load(yaml.safe_dump(normalized))
    settings = service_settings_from_config(child)
    assert settings.tokens == ()
    assert settings.inference_backend_config == {}


def test_override_replaces_required_environment_reference_before_validation(monkeypatch):
    monkeypatch.delenv("REEF_CONFIG_TEST_PORT", raising=False)
    config = _apply_overrides(
        {
            "reef": {
                "recipe": "recipe",
                "port": "${REEF_CONFIG_TEST_PORT:?}",
                "inference_url": "http://localhost:${reef.port}",
            }
        },
        {"port": "9000"},
    )
    config = interpolate_environment(config, "stack.yaml")
    settings = service_settings_from_config(normalize_service_config(config))
    assert settings.port == 9000
    assert settings.inference_url == "http://localhost:9000"


def test_string_lists_resolve_config_references():
    config = {"reef": {"recipe": "recipe", "tokens": ["${auth.current}"]}, "auth": {"current": "credential"}}
    assert service_settings_from_config(config).tokens == ("credential",)


def test_non_reef_objects_and_generic_recipe_overrides_keep_their_ownership():
    config = _apply_overrides(
        {"reef": {"recipe": "recipe"}},
        {
            "training": '{"global_batch_size": 2}',
            "evaluation": '{"module": "example:Evaluator"}',
            "observability.wandb": '{"enabled": false}',
            "batch_size": "4",
            "execution.services.backend": "uni",
        },
    )
    normalized = normalize_service_config(config)
    settings = service_settings_from_config(normalized)
    assert settings.training_settings == {"global_batch_size": 2}
    assert settings.evaluation_settings == {"module": "example:Evaluator"}
    assert settings.wandb_config == {"enabled": False}
    assert settings.recipe_settings["batch_size"] == 4
    assert "batch_size" not in parse_service_arguments(normalized)
    assert normalized["execution"] == {"services": {"backend": "uni"}}


def test_retry_deadline_still_follows_inference_timeout():
    config = {"reef": {"recipe": "recipe", "inference_timeout_s": 42}}
    assert service_settings_from_config(normalize_service_config(config)).inference_retry_timeout_s == 42
    config["reef"]["inference_retry_timeout_s"] = 12
    assert service_settings_from_config(config).inference_retry_timeout_s == 12


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("port", "secret-invalid-integer"),
        ("port", "1.5"),
        ("port", True),
        ("inference_timeout_s", "nan"),
        ("allow_implicit_scenario_creation", "maybe"),
        ("tokens", [123]),
        ("inference_backend_config", [1]),
    ],
)
def test_invalid_public_values_report_the_field_without_echoing_values(name, value):
    with pytest.raises(ValueError, match=name) as caught:
        service_settings_from_config({"reef": {"recipe": "recipe", name: value}})
    assert "secret-invalid-integer" not in str(caught.value)


def test_cli_help_exposes_public_fields_without_optional_runtimes():
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from reef.service.deploy.orchestrator import build_serve_parser\n"
        "print(build_serve_parser().format_help())\n"
        "assert not any(name in sys.modules for name in ('torch', 'slime', 'sglang'))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--upstream-model" in result.stdout
    assert "--no-allow-implicit-scenario-creation" in result.stdout


def test_launcher_passes_typed_cli_values_to_commands_and_child(tmp_path: Path, monkeypatch):
    captured = {}

    class StackStub:
        exit_code = 0

        def __init__(self, config, services, run_dir, ready_timeout_default, config_path, source_root=None):
            captured["parent"] = config
            captured["child"] = load_config(config_path)

        def start(self):
            pass

        def block(self):
            pass

        def shutdown(self):
            pass

    path = tmp_path / "stack.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "reef": {"recipe": "recipe", "port": 9000, "upstream_model": "old", "tokens": ["old"]},
                "services": [{"name": "reef", "command": ["python", "-m", "reef.service"]}],
                "run_dir": str(tmp_path / "run"),
            }
        )
    )
    monkeypatch.setattr(orchestrator, "_Stack", StackStub)
    monkeypatch.setattr(orchestrator, "resolve_model_paths", lambda config: False)
    with pytest.raises(SystemExit) as caught:
        cli_main(
            [
                "serve",
                "-c",
                str(path),
                "--port",
                "8123",
                "--upstream-model",
                "001",
                "--tokens",
                "[]",
                "--inference-backend-config",
                json.dumps({"nested": [1, False]}),
            ]
        )
    assert caught.value.code == 0
    for config in captured.values():
        settings = service_settings_from_config(config)
        assert settings.port == 8123
        assert settings.upstream_model == "001"
        assert settings.tokens == ()
        assert settings.inference_backend_config == {"nested": [1, False]}


def test_invalid_cli_setting_fails_before_download_or_spawn(tmp_path, monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid settings reached a side effect")

    monkeypatch.setattr(orchestrator, "resolve_model_paths", unexpected)
    monkeypatch.setattr(orchestrator, "_Stack", unexpected)
    path = tmp_path / "stack.yaml"
    path.write_text("reef: {recipe: recipe}\nservices: [{name: reef, command: worker}]\n")
    with pytest.raises(SystemExit) as caught:
        cli_main(["serve", "-c", str(path), "--port", "credential-like-invalid-port"])
    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert "reef.port" in error
    assert "credential-like-invalid-port" not in error
    assert "Traceback" not in error
