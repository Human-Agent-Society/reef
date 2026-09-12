"""Backend selection controls both deployment topology and HTTP runtime assembly."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest
import yaml
from reef_service._training_deployment import LocalDeployment

from reef.runtime.registry import RuntimeConfigError
from reef.service.assembly import _connect_training_runtime, _training_recipe
from reef.service.deploy import training_backend
from reef.service.deploy.config import DeployConfigError
from reef.service.deploy.orchestrator import resolve_deployment_config
from reef.service.deploy.provider import command_line_config
from reef.service.deploy.settings import ServiceSettings, service_settings_from_config
from reef.train.runtime_backend import RuntimeTrainingBackend

BACKEND = "reef_service._training_deployment:LocalDeployment"
RECIPE = "recipes.sao.recipe:SAORecipe"


def deployment():
    return {
        "schema-version": 2,
        "recipe": {"implementation": RECIPE},
        "inference": {"model-path": "/models/test"},
        "training": {"backend": BACKEND, "options": {"lora-rank": 8, "checkpoint-layers": True}},
    }


def test_cli_and_yaml_select_the_same_in_process_topology_and_runtime(tmp_path):
    overrides = {
        "recipe.implementation": RECIPE,
        "inference.model-path": "/models/test",
        "training.backend": BACKEND,
        "training.options.lora-rank": "16",
        "training.options.checkpoint-layers": "false",
        "recipe.config.max-staleness": "2",
    }
    file_config, _ = resolve_deployment_config(deployment(), overrides, tmp_path / "serve.yaml")
    cli_config, _ = resolve_deployment_config(command_line_config({}), overrides, tmp_path / "cli", standard=True)
    for config in (file_config, cli_config):
        assert [process["name"] for process in config["services"]] == ["reef"]
        assert "execution" not in config
        assert not {"ray_address", "ray_namespace", "inference_backend_factory"} & config["reef"].keys()
        settings = service_settings_from_config(config)
        runtime = _connect_training_runtime(settings, model_path=settings.model_path, max_staleness=2)
        assert runtime.received_model_path == "/models/test"
        assert runtime.config["lora_rank"] == 16
        assert runtime.config["checkpoint_layers"] is False
        assert runtime.config["label"] == "001"
        assert runtime.max_staleness == 2
        runtime.shutdown()
        assert runtime.closed


@pytest.mark.parametrize("options", [{"lora-rank": 0}, {"unknown": 1}, {"type": "ray_training"}, {"connect": "x"}])
def test_invalid_options_fail_before_constructing_runtime(tmp_path, options):
    raw = deployment()
    marker = tmp_path / "started"
    raw["training"]["options"] = {"marker": str(marker), **options}
    with pytest.raises((DeployConfigError, RuntimeConfigError)):
        resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    assert not marker.exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"training.ray-address": "auto"},
        {"execution.training.backend": "ray"},
        {"inference.backend": "sglang"},
        {"inference.tensor-parallel-size": "2"},
        {"recipe.runtime.type": "ray_training"},
    ],
)
def test_in_process_backend_rejects_other_topologies(tmp_path, overrides):
    with pytest.raises(DeployConfigError):
        resolve_deployment_config(deployment(), overrides, tmp_path / "serve.yaml")


def test_installed_backend_name_and_dotted_reference_resolve_the_same_definition(monkeypatch):
    point = EntryPoint(name="test-mlx", value=BACKEND, group="reef.training_backends")
    monkeypatch.setattr(training_backend, "entry_points", lambda **kwargs: (point,))
    assert isinstance(training_backend.training_deployment_for("test-mlx"), LocalDeployment)
    assert isinstance(training_backend.training_deployment_for(BACKEND), LocalDeployment)


@pytest.mark.parametrize("points", [(), (1, 2)])
def test_missing_or_ambiguous_backend_does_not_fall_back_to_slime(monkeypatch, points):
    monkeypatch.setattr(training_backend, "entry_points", lambda **kwargs: points)
    with pytest.raises(DeployConfigError, match=r"unknown|ambiguous"):
        training_backend.training_deployment_for("unavailable")


def test_runtime_type_is_checked_and_wrong_runtime_is_closed(monkeypatch):
    from reef_service._training_deployment import runtime_factory

    from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime

    runtime = InferenceProxyRuntime(base_url="http://unused")
    closed = []
    monkeypatch.setattr(runtime, "shutdown", lambda: closed.append(True))
    monkeypatch.setattr(type(runtime_factory), "__call__", lambda *args: runtime)
    with pytest.raises(TypeError, match="TrainingRuntime"):
        _connect_training_runtime(
            ServiceSettings(recipe=RECIPE, training_backend=BACKEND), model_path="demo", max_staleness=0
        )
    assert closed == [True]


def test_recipe_build_uses_generic_runtime_backend_and_preserves_cleanup(tmp_path):
    from recipes.sao.recipe import SAORecipe
    from reef.storage.sqlite import SQLiteRecordStore

    config, _ = resolve_deployment_config(deployment(), None, tmp_path / "serve.yaml")
    recipe = _training_recipe(SAORecipe, service_settings_from_config(config), {}, None)
    records = SQLiteRecordStore()
    try:
        trainer = recipe.build("scenario", records)
        assert isinstance(trainer.training_backend, RuntimeTrainingBackend)
        assert trainer.training_backend.runtime is recipe.runtime
        assert trainer.training_backend.experiment_config()["runtime"] == "LocalRuntime"
    finally:
        recipe.runtime.shutdown()
        records.close()


def test_in_process_serving_boots_on_cpu_without_loading_slime_or_ray(tmp_path):
    marker = tmp_path / "runtime"
    distribution = tmp_path / "reef_test_backend-0.0.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text("Name: reef-test-backend\nVersion: 0.0.0\n")
    (distribution / "entry_points.txt").write_text(f"[reef.training_backends]\ntest-mlx = {BACKEND}\n")
    raw = deployment()
    raw["training"]["backend"] = "test-mlx"
    raw["training"]["options"]["marker"] = str(marker)
    model = tmp_path / "model"
    model.mkdir()
    raw["inference"]["model-path"] = str(model)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    raw["reef"] = {"port": port, "ready-timeout": 10}
    config_path = tmp_path / "serve.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    script = """
import sys
class NoGPU:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','slime','ray','sglang','megatron','mlx'} or fullname.startswith('reef.train.slime_backend'):
            raise RuntimeError('unexpected execution dependency: ' + fullname)
sys.meta_path.insert(0, NoGPU())
from reef.cli import main
main(sys.argv[1:])
"""
    # The launched HTTP interpreter uses the same dependency guard.
    (tmp_path / "sitecustomize.py").write_text(script[: script.index("from reef.cli import main")])
    env = {key: value for key, value in os.environ.items() if not key.startswith("REEF_")}
    env["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", ""))
    )
    log_path = tmp_path / "serve.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", script, "serve", "-c", str(config_path)],
            cwd=tmp_path,
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 20
            while True:
                assert process.poll() is None, log_path.read_text()
                try:
                    with opener.open(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                        assert response.status == 200
                    break
                except OSError:
                    assert time.monotonic() < deadline, log_path.read_text()
                    time.sleep(0.05)
            assert json.loads(marker.read_text())["model_path"] == str(model)
        finally:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        assert process.returncode == 0, log_path.read_text()
    assert Path(str(marker) + ".closed").exists()
