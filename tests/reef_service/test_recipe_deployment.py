"""Public config omits process definitions; method dependencies use the shared lifecycle."""

import json
import os
import sys
from types import SimpleNamespace

import pytest

from recipes.openclawrl.deployment import prepare_dependencies
from reef.service.deploy import orchestrator
from reef.service.deploy.config import DeployConfigError, interpolate_config
from reef.service.deploy.execution import validate_services
from reef.service.deploy.inference import command_line_config
from reef.service.deploy.orchestrator import _Stack, resolve_deployment_config
from reef.train.slime_backend.launch import driver_environment

OPENCLAW = "recipes.openclawrl.recipe:OpenClawRLRecipe"


def method_config():
    return {
        "schema-version": 2,
        "inference": {"model-path": "/models/policy"},
        "recipe": {
            "implementation": OPENCLAW,
            "config": {
                "prm": {"model-path": "/models/judge", "port": 23001},
                "user-simulator": {"model-path": "/models/user", "port": 30001},
            },
        },
    }


@pytest.mark.parametrize("section", ["service", "services"])
@pytest.mark.parametrize("payload", [{}, [], None])
def test_versioned_config_rejects_process_sections_even_when_empty(tmp_path, section, payload):
    raw = {**method_config(), section: payload}
    with pytest.raises(DeployConfigError, match="does not accept service/services"):
        resolve_deployment_config(raw, None, tmp_path / "serve.yaml")


def test_standalone_recipe_cannot_reintroduce_process_execution():
    from reef.recipe.config import recipe_config_from_mapping
    from reef.recipe.errors import RecipeConfigError

    raw = {**method_config(), "execution": {"services": "ray"}}
    with pytest.raises(RecipeConfigError, match="legacy process stacks"):
        recipe_config_from_mapping(raw)


@pytest.mark.parametrize("flag", ["service.port", "services", "execution.services.backend"])
@pytest.mark.parametrize("from_file", [False, True])
def test_cli_cannot_reintroduce_process_configuration(tmp_path, flag, from_file):
    raw = method_config() if from_file else command_line_config({})
    with pytest.raises(DeployConfigError, match=r"unknown configuration flag|legacy process stacks"):
        resolve_deployment_config(raw, {flag: "uni"}, tmp_path / "serve.yaml", standard=not from_file)


def test_legacy_processes_remain_explicit_and_do_not_run_recipe_hook(tmp_path, monkeypatch):
    from recipes.openclawrl.recipe import OpenClawRLRecipe

    def unexpected(*args):
        pytest.fail("legacy explicit process stacks must not prepare automatic dependencies")

    monkeypatch.setattr(OpenClawRLRecipe, "prepare_deployment", unexpected)
    raw = {"reef": {"recipe": OPENCLAW}, "services": [{"name": "custom", "command": ["custom"]}]}
    config, _ = resolve_deployment_config(raw, None, tmp_path / "legacy.yaml")
    assert config["services"] == raw["services"]


def test_method_dependency_fields_follow_cli_over_yaml(tmp_path):
    raw = method_config()
    config, _ = resolve_deployment_config(
        raw,
        {"recipe.config.prm.tensor-parallel-size": "2", "recipe.config.prm.options.mem-fraction-static": "0.7"},
        tmp_path / "serve.yaml",
    )
    prm, simulator, driver, http = validate_services(config, "test")
    assert prm["resources"] == {"num_gpus": 2}
    assert "--mem-fraction-static=0.7" in prm["command"]
    assert prm["command"][prm["command"].index("--model-path") + 1] == "/models/judge"
    assert simulator["resources"] == {"num_gpus": 1}
    assert driver["depends_on"] == [prm["name"], simulator["name"]]
    assert driver["executor"] == http["executor"] == "uni"
    assert http["depends_on"] == [driver["name"]]
    assert config["reef"]["prm_url"] == "${endpoints.prm-sglang}"
    assert config["reef"]["prm_tokenizer_path"] == "/models/judge"
    assert "services" not in raw and "prm-url" not in raw["recipe"]["config"]


@pytest.mark.parametrize(
    "fields,match",
    [
        ({"prm": {"model-path": ""}}, "requires model-path"),
        ({"prm": {"model-path": "demo", "port": 0}}, "invalid OpenClawRL"),
        ({"prm": {"model-path": "demo", "tensor-parallel-size": 0}}, "invalid OpenClawRL"),
        ({"prm": {"model-path": "demo", "ready-timeout": 0}}, "invalid OpenClawRL"),
        ({"prm": {"model-path": "demo", "command": "arbitrary"}}, "unknown config fields"),
        ({"prm": {"model-path": "demo", "options": {"model": "other"}}}, "managed by Reef"),
        ({"prm": {"model-path": "demo"}, "prm_url": "http://existing"}, "either a managed prm"),
        (
            {"prm": {"model-path": "a", "port": 9000}, "user_simulator": {"model-path": "b", "port": 9000}},
            "distinct ports",
        ),
    ],
)
def test_method_rejects_invalid_server_config(fields, match):
    with pytest.raises((ValueError, DeployConfigError), match=match):
        prepare_dependencies(fields)


def test_external_prm_has_no_managed_process_and_keeps_its_connection():
    config = {"prm_url": "http://external:23001", "prm_tokenizer_path": "/models/judge"}
    assert prepare_dependencies(config) == ()
    assert config["prm_url"] == "http://external:23001"


def test_declared_runtime_starts_http_without_upstream_fields(tmp_path):
    from reef.service.assembly import _serving_recipe
    from reef.service.deploy.service_config import service_settings_from_config

    raw = {
        "schema-version": 2,
        "recipe": {
            "implementation": "reef.recipe.base:Recipe",
            "runtime": {"type": "inference_proxy", "base-url": "http://localhost:8000"},
        },
    }
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    assert [process["name"] for process in validate_services(config, "test")] == ["reef"]
    recipe = _serving_recipe(raw["recipe"]["implementation"], service_settings_from_config(config), {}, None)
    assert recipe.runtime.base_url == "http://localhost:8000"


def test_recipe_hook_cannot_bind_core_settings(tmp_path, monkeypatch):
    from recipes.openclawrl.recipe import OpenClawRLRecipe

    def prepare(config):
        config["port"] = 9999
        return ()

    monkeypatch.setattr(OpenClawRLRecipe, "prepare_deployment", prepare)
    with pytest.raises(DeployConfigError, match="only declared recipe settings"):
        resolve_deployment_config(method_config(), None, tmp_path / "serve.yaml")


def test_training_environment_defaults_belong_to_backend_and_honor_overrides():
    assert driver_environment({}) == {"CUDA_DEVICE_MAX_CONNECTIONS": "1", "NCCL_NVLS_ENABLE": "0"}
    assert driver_environment({"CUDA_DEVICE_MAX_CONNECTIONS": "8", "NCCL_NVLS_ENABLE": "1"}) == {
        "CUDA_DEVICE_MAX_CONNECTIONS": "8",
        "NCCL_NVLS_ENABLE": "1",
    }


@pytest.mark.parametrize("fail_dependency", [False, True])
def test_method_dependencies_gate_driver_and_share_failure_cleanup(tmp_path, monkeypatch, fail_dependency):
    # Run real child processes and readiness checks; replace only GPU workloads/placement.
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    closed = []
    monkeypatch.setattr(
        orchestrator,
        "acquire_ray_runtime",
        lambda address: SimpleNamespace(address="127.0.0.1:6379", close=lambda: closed.append(True)),
    )
    raw = method_config()
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    for process in config["services"]:
        name = process["name"]
        marker = tmp_path / name
        dependencies = [str(tmp_path / dependency) for dependency in process.get("depends_on", [])]
        script = (
            "import json,time; from pathlib import Path; "
            f"paths={dependencies!r}; "
            "assert all(Path(p).exists() for p in paths); "
            f"Path({str(marker)!r}).write_text('ready'); time.sleep(120)"
        )
        if fail_dependency and name == "prm-sglang":
            script = "raise SystemExit(7)"
        process.update(
            executor="uni",
            command=[sys.executable, "-c", script],
            ready_timeout=10,
            ready=[
                sys.executable,
                "-c",
                f"from pathlib import Path; raise SystemExit(not Path({str(marker)!r}).exists())",
            ],
        )
        process.pop("resources", None)
    run_dir = tmp_path / "stack"
    run_dir.mkdir()
    stack = _Stack(config, validate_services(config, "test"), run_dir, 10, tmp_path / "config.yaml")
    try:
        if fail_dependency:
            with pytest.raises(RuntimeError, match=r"prm-sglang.*exited"):
                stack.start()
            assert not (tmp_path / "slime-driver").exists()
            assert not (tmp_path / "reef").exists()
        else:
            stack.start()
            assert all((tmp_path / process["name"]).exists() for process in config["services"])
            assert interpolate_config(stack.config, stack.config["reef"]["prm_url"]) == "http://127.0.0.1:23001"
    finally:
        stack.shutdown(grace=1)
    assert closed == [True]
    for path in run_dir.glob("*.worker.json"):
        for pid in json.loads(path.read_text())["pids"].values():
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
