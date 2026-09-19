"""``reef serve`` starts the generator from the ``generator`` section, before the HTTP service, under its own executor role."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from reef.record2dataset import DesignerTurn, FixedPrompt, HarnessPrompt, ReefDesigner
from reef.record2dataset import __main__ as generator_main
from reef.runtime.executor.config import ExecutorSettings, role_executor_settings, select_executor
from reef.service.deploy.config_utils import DeployConfigError
from reef.service.deploy.execution import service_executor_selection, validate_services
from reef.service.deploy.generator import attach_generator_service, generator_service, generator_settings
from reef.service.deploy.orchestrator import resolve_deployment_config
from reef.service.deploy.service_config import service_config_from_mapping

pytestmark = pytest.mark.unit


def config(**generator: object) -> dict[str, object]:
    return {
        "reef": {"host": "0.0.0.0", "port": 8900, "tokens": ["t"]},
        "generator": {"tasks-root": "/tmp/tasks", **generator},
        "services": [
            {"name": "reef", "command": ["python", "-m", "reef.service"], "endpoint": "http://127.0.0.1:8900"}
        ],
    }


def test_the_generator_is_started_before_the_http_service_which_depends_on_it() -> None:
    stack = config(port=8911, **{"ready-timeout": 90})
    attach_generator_service(stack)
    generator, http = stack["services"]
    assert generator == {
        "name": "generator",
        "role": "generator",
        "command": [sys.executable, "-m", "reef.record2dataset"],
        "endpoint": "http://{host}:8911",
        "ready": generator["ready"],
        "ready_timeout": 90,
    }
    assert generator["ready"][0] == sys.executable and generator["ready"][-1] == "http://127.0.0.1:8911/healthz"
    assert http["name"] == "reef" and http["depends_on"] == ["generator"]
    assert validate_services(stack, "stack.yaml")[0]["name"] == "generator"


def test_a_deployment_without_the_section_is_unchanged_and_a_bad_section_is_a_config_error() -> None:
    stack = config()
    del stack["generator"]
    attach_generator_service(stack)
    assert [service["name"] for service in stack["services"]] == ["reef"]
    assert generator_service(stack) is None
    with pytest.raises(DeployConfigError, match="generator: unknown config fields: bogus"):
        attach_generator_service(config(bogus=1))
    headless = config()
    headless["services"] = []
    with pytest.raises(DeployConfigError, match="needs the Reef HTTP service"):
        attach_generator_service(headless)


def test_the_generator_runs_under_reefs_interpreter_and_probes_its_bind_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REEF_PYTHON", raising=False)
    stack = config(host="10.0.0.5")
    attach_generator_service(stack)
    generator = stack["services"][0]
    assert generator["command"][0] == sys.executable and generator["ready"][-1] == "http://10.0.0.5:8910/healthz"
    monkeypatch.setenv("REEF_PYTHON", "/opt/serve/bin/python")
    assert generator_service(config())["command"][0] == "/opt/serve/bin/python", "the same interpreter as reef.service"
    with pytest.raises(DeployConfigError, match="unknown config fields: python"):
        attach_generator_service(config(python="/opt/py312/bin/python"))


def test_the_generator_role_selects_its_own_executor() -> None:
    stack = config()
    attach_generator_service(stack)
    generator = stack["services"][0]
    assert service_executor_selection(stack, generator).settings.backend == "uni"
    stack["execution"] = {"generator": "ray"}
    assert service_executor_selection(stack, generator).settings.backend == "ray"
    assert service_executor_selection(stack, stack["services"][1]).settings.backend == "uni", "reef keeps its role"
    assert role_executor_settings(stack, "generator").backend == "ray"
    with pytest.raises(ValueError, match=r"execution\.generator\.workers must be 1"):
        select_executor(ExecutorSettings(backend="auto", workers=2), role="generator")
    with pytest.raises(DeployConfigError, match="role must be services or generator"):
        validate_services({**stack, "services": [{**generator, "role": "training"}]}, "stack.yaml")


def test_a_versioned_deployment_with_a_generator_section_assembles_the_service(tmp_path: Path) -> None:
    deployment = {
        "schema-version": 2,
        "reef": {"host": "127.0.0.1", "port": 8900},
        "inference": {"upstream-url": "http://localhost:8000", "upstream-model": "m"},
        "generator": {"tasks-root": str(tmp_path / "tasks"), "port": 8912},
    }
    resolved, _ = resolve_deployment_config(deployment, None, tmp_path / "serve.yaml")
    assert [service["name"] for service in resolved["services"]] == ["generator", "reef"]
    assert resolved["services"][1]["depends_on"] == ["generator"]
    settings = service_config_from_mapping(resolved)
    assert settings.generator_settings == {"tasks-root": str(tmp_path / "tasks"), "port": 8912}
    assert settings.upstream_model == "m"


def test_the_service_carries_the_designer_model_when_the_section_names_one_not_otherwise(tmp_path: Path) -> None:
    deployment = {
        "schema-version": 2,
        "reef": {"host": "127.0.0.1", "port": 8900},
        "inference": {"upstream-url": "http://localhost:8000", "upstream-model": "m"},
        "generator": {"tasks-root": str(tmp_path / "tasks"), "designer-model": "openai/gpt-5"},
    }
    resolved, _ = resolve_deployment_config(deployment, None, tmp_path / "serve.yaml")
    settings = service_config_from_mapping(resolved)
    built = generator_main.generator_service(settings, generator_settings(settings.generator_settings))
    assert built.designer_model == "openai/gpt-5" and built.default_model == "m"
    del deployment["generator"]["designer-model"]
    resolved, _ = resolve_deployment_config(deployment, None, tmp_path / "serve.yaml")
    settings = service_config_from_mapping(resolved)
    built = generator_main.generator_service(settings, generator_settings(settings.generator_settings))
    assert built.designer_model is None and built.default_model == "m"


def test_the_service_pulls_the_designers_prompt_from_the_harness_release_the_section_names(tmp_path: Path) -> None:
    deployment = {
        "schema-version": 2,
        "reef": {"host": "127.0.0.1", "port": 8900},
        "inference": {"upstream-url": "http://localhost:8000", "upstream-model": "m"},
        "generator": {
            "tasks-root": str(tmp_path / "tasks"),
            "designer-url": "http://127.0.0.1:8901",
            "designer-token": "d",
            "designer-scenario": "designer",
            "designer-prompt": "harness",
        },
    }
    resolved, _ = resolve_deployment_config(deployment, None, tmp_path / "serve.yaml")
    settings = service_config_from_mapping(resolved)
    built = generator_main.generator_service(settings, generator_settings(settings.generator_settings))
    assert isinstance(built.prompts, HarnessPrompt) and built.prompts.scenario == "designer"
    assert isinstance(built.designer, ReefDesigner) and built.prompts.client is built.designer.client
    assert built.prompts.client.service_url == "http://127.0.0.1:8901" and built.prompts.client.token == "d"
    assert isinstance(built.turn, DesignerTurn) and built.turn.client is built.designer.client
    assert built.turn.is_harness_prompt and (built.turn.poll_s, built.turn.wait_s) == (5.0, 1800.0)
    del deployment["generator"]["designer-prompt"]
    resolved, _ = resolve_deployment_config(deployment, None, tmp_path / "serve.yaml")
    settings = service_config_from_mapping(resolved)
    built = generator_main.generator_service(settings, generator_settings(settings.generator_settings))
    assert isinstance(built.prompts, FixedPrompt), "fixed by default"
    assert (
        isinstance(built.turn, DesignerTurn) and not built.turn.is_harness_prompt
    ), "the runtime load id is the version"
    with pytest.raises(ValueError, match=r"harness needs generator\.designer-scenario"):
        generator_settings({"tasks-root": "/tmp/t", "designer-prompt": "harness"})
    with pytest.raises(ValueError, match=r"designer-prompt must be one of \('fixed', 'harness'\)"):
        generator_settings({"tasks-root": "/tmp/t", "designer-prompt": "evolved"})


def test_the_designer_wait_keys_set_the_turns_poll_and_limit_and_a_zero_is_refused(tmp_path: Path) -> None:
    deployment = {
        "schema-version": 2,
        "reef": {"host": "127.0.0.1", "port": 8900},
        "inference": {"upstream-url": "http://localhost:8000", "upstream-model": "m"},
        "generator": {"tasks-root": str(tmp_path / "tasks"), "designer-poll-s": 1, "designer-wait-s": 30},
    }
    resolved, _ = resolve_deployment_config(deployment, None, tmp_path / "serve.yaml")
    settings = service_config_from_mapping(resolved)
    generator = generator_settings(settings.generator_settings)
    assert (generator.designer_poll_s, generator.designer_wait_s) == (1.0, 30.0)
    built = generator_main.generator_service(settings, generator)
    assert built.turn is not None and (built.turn.poll_s, built.turn.wait_s) == (1.0, 30.0)
    assert (
        built.turn.client is built.designer.client and built.turn.client.service_url == "http://127.0.0.1:8900"
    ), "without designer-url the stack's own service answers for the Designer's version"
    for key in ("designer-poll-s", "designer-wait-s"):
        with pytest.raises(ValueError, match=f"generator.{key} must be positive"):
            generator_settings({"tasks-root": "/tmp/t", key: 0})


def test_the_service_carries_the_designer_scenario_when_the_section_names_one_not_otherwise(tmp_path: Path) -> None:
    deployment = {
        "schema-version": 2,
        "reef": {"host": "127.0.0.1", "port": 8900},
        "inference": {"upstream-url": "http://localhost:8000", "upstream-model": "m"},
        "generator": {
            "tasks-root": str(tmp_path / "tasks"),
            "designer-url": "http://127.0.0.1:8901",
            "designer-scenario": "designer",
        },
    }
    resolved, _ = resolve_deployment_config(deployment, None, tmp_path / "serve.yaml")
    settings = service_config_from_mapping(resolved)
    built = generator_main.generator_service(settings, generator_settings(settings.generator_settings))
    assert built.designer_scenario == "designer"
    del deployment["generator"]["designer-scenario"]
    resolved, _ = resolve_deployment_config(deployment, None, tmp_path / "serve.yaml")
    settings = service_config_from_mapping(resolved)
    built = generator_main.generator_service(settings, generator_settings(settings.generator_settings))
    assert built.designer_scenario is None, "without the key the proposal's own scenario is the Designer's"
