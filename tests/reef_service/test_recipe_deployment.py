"""Reef owns inference/training; recipes consume independently managed endpoints."""

import http.client
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from reef.service.deploy import orchestrator
from reef.service.deploy.config_utils import DeployConfigError
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
                "prm-url": "http://external:23001",
                "prm-tokenizer-path": "/models/judge",
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


def test_legacy_processes_remain_explicit(tmp_path):
    raw = {"reef": {"recipe": OPENCLAW}, "services": [{"name": "custom", "command": ["custom"]}]}
    config, _ = resolve_deployment_config(raw, None, tmp_path / "legacy.yaml")
    assert config["services"] == raw["services"]


def test_recipe_endpoint_follows_cli_over_yaml_without_launching_a_process(tmp_path):
    from reef_service.runtime_stubs import StubTrainingRuntime

    from recipes.openclawrl.recipe import OpenClawRLRecipe
    from reef.service.assembly import _recipe_owned_settings
    from reef.service.deploy.service_config import service_config_from_mapping

    raw = method_config()
    config, _ = resolve_deployment_config(
        raw,
        {"recipe.config.prm-url": "http://other:9000", "recipe.config.prm-tokenizer-path": "/models/other"},
        tmp_path / "serve.yaml",
    )
    driver, http = validate_services(config, "test")
    assert driver["name"] == "slime-driver"
    assert http["name"] == "reef"
    assert not driver.get("depends_on")
    assert driver["executor"] == http["executor"] == "uni"
    assert http["depends_on"] == [driver["name"]]
    settings = service_config_from_mapping(config)
    recipe = OpenClawRLRecipe.from_environment(
        {},
        config=OpenClawRLRecipe.service_config(_recipe_owned_settings(settings), model_path=settings.model_path),
        runtime=StubTrainingRuntime(),
    )
    assert recipe.prm_url == "http://other:9000"
    assert recipe.prm_tokenizer_path == "/models/other"
    assert raw["recipe"]["config"]["prm-url"] == "http://external:23001"
    assert "services" not in raw


@pytest.mark.parametrize("field", ["prm", "user-simulator"])
@pytest.mark.parametrize("cli", [False, True])
def test_recipe_rejects_model_service_launch_options(tmp_path, field, cli):
    raw = method_config()
    overrides = None
    if cli:
        overrides = {f"recipe.config.{field}.model-path": "/models/auxiliary"}
    else:
        raw["recipe"]["config"][field] = {"model-path": "/models/auxiliary"}
    with pytest.raises(DeployConfigError, match="unknown"):
        resolve_deployment_config(raw, overrides, tmp_path / "serve.yaml")


def test_declared_runtime_starts_http_without_upstream_fields(tmp_path):
    from reef.service.assembly import _serving_recipe
    from reef.service.deploy.service_config import service_config_from_mapping

    raw = {
        "schema-version": 2,
        "recipe": {
            "implementation": "reef.recipe.base:Recipe",
            "runtime": {"type": "inference_proxy", "base-url": "http://localhost:8000"},
        },
    }
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    assert [process["name"] for process in validate_services(config, "test")] == ["reef"]
    recipe = _serving_recipe(raw["recipe"]["implementation"], service_config_from_mapping(config), {}, None)
    assert recipe.runtime.base_url == "http://localhost:8000"


def test_training_environment_defaults_belong_to_backend_and_honor_overrides():
    assert driver_environment({}) == {"CUDA_DEVICE_MAX_CONNECTIONS": "1", "NCCL_NVLS_ENABLE": "0"}
    assert driver_environment({"CUDA_DEVICE_MAX_CONNECTIONS": "8", "NCCL_NVLS_ENABLE": "1"}) == {
        "CUDA_DEVICE_MAX_CONNECTIONS": "8",
        "NCCL_NVLS_ENABLE": "1",
    }


@pytest.mark.parametrize("fail_driver", [False, True])
def test_reef_shutdown_and_startup_failure_leave_external_prm_running(tmp_path, monkeypatch, fail_driver):
    # Run real child processes and readiness checks; replace only GPU workloads/placement.
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    closed = []
    monkeypatch.setattr(
        orchestrator,
        "acquire_ray_runtime",
        lambda address: SimpleNamespace(address="127.0.0.1:6379", close=lambda: closed.append(True)),
    )
    external = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "from http.server import HTTPServer, SimpleHTTPRequestHandler; "
            "server=HTTPServer(('127.0.0.1',0),SimpleHTTPRequestHandler); "
            "print(server.server_port,flush=True); server.serve_forever()",
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        _exercise_reef_lifecycle(tmp_path, external, fail_driver)
        assert external.poll() is None
    finally:
        external.terminate()
        external.wait(timeout=10)
        external.stdout.close()
    assert closed == [True]


def _exercise_reef_lifecycle(tmp_path, external, fail_driver):
    port = int(external.stdout.readline())
    raw = method_config()
    endpoint = f"http://127.0.0.1:{port}"
    raw["recipe"]["config"]["prm-url"] = endpoint
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    assert [process["name"] for process in config["services"]] == ["slime-driver", "reef"]
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
        if fail_driver and name == "slime-driver":
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
        if fail_driver:
            with pytest.raises(RuntimeError, match=r"slime-driver.*exited"):
                stack.start()
            assert not (tmp_path / "slime-driver").exists()
            assert not (tmp_path / "reef").exists()
        else:
            stack.start()
            assert all((tmp_path / process["name"]).exists() for process in config["services"])
            assert stack.config["reef"]["prm_url"] == endpoint
    finally:
        stack.shutdown(grace=1)
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("GET", "/")
        assert connection.getresponse().status == 200
    finally:
        connection.close()
    for path in run_dir.glob("*.worker.json"):
        for pid in json.loads(path.read_text())["pids"].values():
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
