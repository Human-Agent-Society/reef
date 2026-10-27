"""Tinker training in process, its exported adapters served by Reef's engines on an inference component (CPU contracts)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml

from reef.artifact.artifact import Artifact
from reef.inference.remote.runtime import RemoteServingRuntimeFactory
from reef.runtime.deployment import RuntimeConfigError
from reef.runtime.interfaces import InferenceBackend, InferenceHandler, TrainingMethod
from reef.runtime.jobs import ReleaseFormat, TrainingJob, job_id_for
from reef.runtime.serving import ADAPTER_DIR, AdapterServingRuntime
from reef.service.assembly import connect_training_component
from reef.service.deploy.config_utils import DeployConfigError
from reef.service.deploy.orchestrator import resolve_deployment_config
from reef.service.deploy.service_config import service_config_from_mapping
from reef.service.deploy.serving import serving_runtime_config
from reef.service.deploy.training import local_model_required
from reef.surface.adapter import adapter_name
from reef.tinker.checkpoint import TinkerCheckpoint
from reef.train.algos import TrainingSignal
from reef.train.algos.objective import TrainingObjective
from reef.train.algos.registry import register_objective, unregister_objective
from reef.train.tinker_backend.client import TINKER_ADAPTER_DIR
from reef.train.tinker_backend.component import TinkerTrainingComponent
from reef.train.tinker_backend.launch import TinkerDeployment
from reef.train.types import JobData, TrajectoryItem

TOKENS = ("REEF_REMOTE_CONTROL_TOKEN", "REEF_REMOTE_INFERENCE_TOKEN")
#: The release format of Tinker's exported adapters at the configs' LoRA rank.
EXPORTED_ADAPTERS = ReleaseFormat("adapter", 16, ("all",))


class RemoteClient:
    """The Tinker SDK boundary as an offline double: trains, and writes a fake PEFT adapter on download."""

    def __init__(self, *args):
        self.calls = []
        self.downloads = []
        self.closed = False

    def initialize(self):
        return TinkerCheckpoint("Qwen/Qwen3-8B", 16, "tinker://base/state", "tinker://base/sampler")

    def train(self, checkpoint, batches, loss, learning_rates, *, kl_coef, export_adapter):
        self.calls.append(checkpoint)
        number = len(self.calls)
        return TinkerCheckpoint(
            checkpoint.base_model, 16, f"tinker://update-{number}/state", f"tinker://update-{number}/sampler"
        ), {"loss": -1.0}

    def download(self, checkpoint, release):
        self.downloads.append(checkpoint)
        for name in (ADAPTER_DIR, TINKER_ADAPTER_DIR):
            (release / name).mkdir(parents=True)
            (release / name / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 16}))
            (release / name / "adapter_model.safetensors").write_bytes(checkpoint.sampler_path.encode())

    def close(self):
        self.closed = True


class UnusedSampler:
    """The sampler of the component's task jobs as an offline double; these jobs train trajectories, so none samples."""

    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass


class Engines(InferenceBackend):
    """The engines on the inference component, as the control plane drives them."""

    def __init__(self):
        self.loads = []
        self.versions = ["boot"]

    def load_weight_files(self, files, *, adapter_name, runtime_load_id):
        assert files.layout == "adapter"
        self.loads.append((adapter_name, files.path, runtime_load_id))
        self.versions = [runtime_load_id]

    def initialize_version(self, runtime_load_id):
        self.versions = [runtime_load_id]

    def inference_url(self):
        return "http://127.0.0.1:9000"

    def runtime_load_ids(self):
        return list(self.versions)

    def pause(self):
        return

    def resume(self):
        return

    def recover(self):
        return 0

    def abort(self):
        raise AssertionError("unused")

    def unload_adapter(self, name):
        self.loads.append(("unload", name))


class Handler(InferenceHandler):
    async def inference(self, artifact, path, payload):
        raise AssertionError("unused")


class SampleObjective(TrainingObjective):
    loss_family = "importance_sampling"
    name = "tinker-local-engine-objective"

    def prepare(self, batch, state):
        return TrainingSignal(
            "train", {"steps": state.get("steps", 0) + 1}, advantages=tuple(1.0 for _ in batch.items)
        )


@pytest.fixture
def objective():
    value = SampleObjective()
    register_objective(value)
    yield value
    unregister_objective(value.name)


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    for name in TOKENS:
        monkeypatch.delenv(name, raising=False)


def item(sample):
    # ``sample`` makes another rollout: a job is its rows, so a new step trains new rows.
    return TrajectoryItem(
        {
            "schema_version": "ATIF-v1.6",
            "agent": {"name": "test"},
            "steps": [{"step_id": 1}],
            "extra": {
                "reef": {
                    "training": {
                        "tokens": [10 + sample, 11, 20, 21],
                        "loss_mask": [1, 0],
                        "rollout_log_probs": [-0.2, -0.4],
                        "runtime_load_id": "boot",
                    }
                }
            },
        }
    )


def local_engine_config(tmp_path):
    return {
        "recipe": {"implementation": "recipes.tttd.recipe:TTTDRecipe"},
        "inference": {"model-path": "Qwen/Qwen3-8B", "backend": "sglang", "num-gpus": 2},
        "training": {
            "backend": "tinker",
            "options": {"state-dir": str(tmp_path / "state"), "lora-rank": 16, "export-adapter": True},
        },
    }


def remote_engine_config(tmp_path):
    raw = local_engine_config(tmp_path)
    raw["inference"]["remote"] = {"url": "https://gpu-host.example/", "inference-token-env": "GPU_HOST_TOKEN"}
    return raw


def serving_config(config, release_format=EXPORTED_ADAPTERS):
    """The serving runtime config that the HTTP service builds for the deployment ``config``."""
    return serving_runtime_config(service_config_from_mapping(config), TinkerDeployment(), release_format)


def build(config, monkeypatch, engines):
    """The component and serving runtime the HTTP service builds, over the SDK double and the engine double."""
    monkeypatch.setattr("reef.train.tinker_backend.client.TinkerSDKClient", RemoteClient)
    monkeypatch.setattr("reef.tinker.sampling.TinkerSDKSampler", UnusedSampler)
    seen = {}

    def serve(factory, config, model_path, recipe_config, environ):
        seen.update(config)
        return AdapterServingRuntime(engines, Handler(), adapter_capacity=config["adapter_capacity"])

    monkeypatch.setattr(RemoteServingRuntimeFactory, "__call__", serve)
    component, inference = connect_training_component(
        service_config_from_mapping(config), model_path="Qwen/Qwen3-8B", loss_family="tttd"
    )
    return component, inference, seen


def train(component, inference, objective, scenario, sample, start=None):
    """Run one job to its end, then serve its checkpoint for ``scenario``; return the checkpoint and its version."""
    data = JobData("b", (item(sample),))
    parameters = {"method": TrainingMethod(objective.name).to_dict()}
    job_id = job_id_for(component.name, start, data.to_dict()["items"], parameters)
    component.submit(TrainingJob(job_id, component.name, start, data, parameters))
    deadline = time.monotonic() + 10
    while (status := component.status(job_id)).state == "running":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert status.state == "finished", status.error
    (checkpoint,) = status.checkpoints
    return checkpoint, inference.activate_checkpoint(Artifact.local(Path(checkpoint.release_path)), scenario=scenario)


def test_local_engines_run_on_an_inference_component_that_reef_serve_starts(tmp_path):
    config, _ = resolve_deployment_config(local_engine_config(tmp_path), None, tmp_path / "serve.yaml")
    component, http = config["services"]
    assert (component["name"], http["name"], http["depends_on"]) == ("inference", "reef", ["inference"])
    url = config["reef"]["inference_remote"]["url"]
    port = url.rsplit(":", 1)[1]
    assert url.startswith("http://127.0.0.1:")
    assert component["command"][1:9] == [
        "-m",
        "reef.inference.remote",
        "--engine",
        "sglang",
        "--host",
        "127.0.0.1",
        "--port",
        port,
    ]
    assert component["ready"][-1] == port
    # The HTTP service starts the engines before it serves.
    assert config["ready_timeout"] >= component["ready_timeout"]
    # Both children inherit generated tokens; none is written into the configuration.
    import os

    tokens = [os.environ[name] for name in TOKENS]
    assert tokens[0] != tokens[1] and not any(token in yaml.safe_dump(config) for token in tokens)
    assert "test-key" not in yaml.safe_dump(config)
    assert not local_model_required(config)
    serving = serving_config(config)
    assert serving["type"] == "remote_serving" and serving["release_format"] == EXPORTED_ADAPTERS.to_dict()
    assert serving["remote"]["url"] == url and serving["engine"] == "sglang"
    assert (serving["num_gpus"], serving["tensor_parallel_size"]) == (2, 1)


def test_remote_engines_keep_the_section_and_start_no_local_component(tmp_path):
    config, _ = resolve_deployment_config(remote_engine_config(tmp_path), None, tmp_path / "serve.yaml")
    assert [service["name"] for service in config["services"]] == ["reef"]
    assert config["reef"]["inference_remote"] == {
        "url": "https://gpu-host.example",
        "control_token_env": "REEF_REMOTE_CONTROL_TOKEN",
        "inference_token_env": "GPU_HOST_TOKEN",
        "startup_timeout_s": 3600.0,
        "model_path": "Qwen/Qwen3-8B",
    }
    assert serving_config(config)["remote"] == config["reef"]["inference_remote"]


def test_the_factory_builds_the_component_and_serves_each_scenarios_adapter(tmp_path, monkeypatch, objective):
    raw = local_engine_config(tmp_path)
    raw["inference"]["max-loaded-adapters"] = 3
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    engines = Engines()
    component, inference, seen = build(config, monkeypatch, engines)
    try:
        assert isinstance(component, TinkerTrainingComponent) and isinstance(inference, AdapterServingRuntime)
        assert component.release_format == EXPORTED_ADAPTERS
        assert (seen["release_format"], seen["adapter_capacity"]) == (EXPORTED_ADAPTERS.to_dict(), 3)
        math, served = train(component, inference, objective, "math", 0)
        adapter = Path(math.release_path) / ADAPTER_DIR
        assert (adapter / "adapter_config.json").is_file()
        assert engines.loads == [(adapter_name("math", served), adapter, served)]
        assert inference.serving_adapter_runtime_load_id("math") == served
        # Another scenario serves its own adapter beside the first one.
        _, code = train(component, inference, objective, "code", 1)
        assert inference.serving_adapter_runtime_load_id("code") == code != served
        assert inference.serving_adapter_runtime_load_id("math") == served
        train(component, inference, objective, "math", 2, start=math.path)
        assert [checkpoint.state_path for checkpoint in component._client.calls] == [
            "tinker://base/state",
            "tinker://base/state",
            "tinker://update-1/state",
        ]
    finally:
        inference.shutdown()
        component.close()


def test_a_restart_serves_a_saved_release_and_trains_from_it(tmp_path, monkeypatch, objective):
    config, _ = resolve_deployment_config(local_engine_config(tmp_path), None, tmp_path / "serve.yaml")
    component, inference, _ = build(config, monkeypatch, Engines())
    math, _ = train(component, inference, objective, "math", 0)
    inference.shutdown()
    component.close()

    engines = Engines()
    component, inference, _ = build(config, monkeypatch, engines)
    try:
        served = inference.activate_checkpoint(Artifact.local(Path(math.release_path)), scenario="math")
        assert engines.loads == [(adapter_name("math", served), Path(math.release_path) / ADAPTER_DIR, served)]
        _, activated = train(component, inference, objective, "math", 1, start=math.path)
        assert component._client.calls[0].state_path == "tinker://update-1/state"
        # One adapter slot: generation is paused, so the scenario's new adapter replaces its served one.
        assert engines.loads[1:] == [
            ("unload", adapter_name("math", served)),
            (adapter_name("math", activated), engines.loads[2][1], activated),
        ]
    finally:
        inference.shutdown()
        component.close()


def test_engine_topologies_refuse_conflicting_inputs_before_starting(tmp_path, monkeypatch):
    for change, match in (
        ({"training": {"colocate": True}}, "training.colocate"),
        ({"inference": {"options": {"enable-lora": True}}}, "enable-lora is managed by Reef"),
        ({"training": {"options": {"lora-rank": 0}}}, "training.options"),
        ({"execution": {"training": "ray"}}, "in-process training owns its execution"),
    ):
        raw = local_engine_config(tmp_path)
        for section, values in change.items():
            current = raw.setdefault(section, {})
            for key, value in values.items():
                current[key] = {**current[key], **value} if isinstance(value, dict) and key in current else value
        with pytest.raises(DeployConfigError, match=match):
            resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    for change, match in (
        ({"remote": {"control-token-env": "GPU_HOST_CONTROL"}}, "inference.remote requires a url"),
        ({"remote": {"url": "https://gpu-host.example?key=1"}}, "without credentials, query"),
    ):
        raw = remote_engine_config(tmp_path)
        raw["inference"] = {**raw["inference"], **change}
        with pytest.raises(DeployConfigError, match=match):
            resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    monkeypatch.delenv("TINKER_API_KEY")
    with pytest.raises(DeployConfigError, match="TINKER_API_KEY"):
        resolve_deployment_config(local_engine_config(tmp_path), None, tmp_path / "serve.yaml")


def test_engines_serve_tinker_only_exported_adapters_that_their_engine_loads(tmp_path, monkeypatch):
    for name in TOKENS:
        monkeypatch.setenv(name, f"{name.lower()}-value")
    monkeypatch.setattr("reef.train.tinker_backend.client.TinkerSDKClient", RemoteClient)
    monkeypatch.setattr("reef.tinker.sampling.TinkerSDKSampler", UnusedSampler)
    raw = remote_engine_config(tmp_path)
    monkeypatch.setenv("GPU_HOST_TOKEN", "gpu-host-token")
    raw["inference"]["backend"] = "vllm"
    raw["inference"]["num-gpus"] = 1
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    with pytest.raises(RuntimeConfigError, match="vllm serves full weights only"):
        connect_training_component(service_config_from_mapping(config), model_path="Qwen/Qwen3-8B", loss_family="tttd")
    raw = remote_engine_config(tmp_path)
    raw["training"]["options"]["export-adapter"] = False
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    with pytest.raises(DeployConfigError, match="export-adapter"):
        connect_training_component(service_config_from_mapping(config), model_path="Qwen/Qwen3-8B", loss_family="tttd")


def test_tinker_samples_in_process_without_engines(tmp_path):
    raw = local_engine_config(tmp_path)
    raw["inference"] = {"model-path": "Qwen/Qwen3-8B"}
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    assert [service["name"] for service in config["services"]] == ["reef"]
    assert not local_model_required(config)
    assert not TinkerDeployment.requires_local_model
    assert serving_config(config)["type"] == "tinker"
