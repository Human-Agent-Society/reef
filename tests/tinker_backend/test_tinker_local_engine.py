"""Tinker training in process, served by Reef's engines on an inference component (CPU contracts)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from reef.artifact.artifact import Artifact
from reef.inference.remote.runtime import RemoteAdapterRuntimeFactory
from reef.runtime.deployment import RuntimeConfigError, RuntimeRegistry
from reef.runtime.interfaces import InferenceBackend, InferenceHandler, TrainingMethod
from reef.runtime.serving import ADAPTER_DIR, AdapterServingRuntime
from reef.service.deploy.config_utils import DeployConfigError
from reef.service.deploy.orchestrator import resolve_deployment_config
from reef.service.deploy.service_config import service_config_from_mapping
from reef.service.deploy.training import local_model_required, training_deployment_for
from reef.surface.adapter import adapter_name
from reef.train.algos import StepScheduling, StepSignal
from reef.train.algos.objective import TrainingObjective
from reef.train.algos.registry import register_objective, unregister_objective
from reef.train.job_runtime import JobTrainingRuntime
from reef.train.tinker_backend.checkpoint import TinkerCheckpoint
from reef.train.tinker_backend.launch import TinkerDeployment
from reef.train.types import TrainingBatch, TrajectoryItem

TOKENS = ("REEF_REMOTE_CONTROL_TOKEN", "REEF_REMOTE_INFERENCE_TOKEN")


class RemoteClient:
    """The Tinker SDK boundary as an offline double: trains, and writes a fake PEFT adapter on download."""

    def __init__(self, *args):
        self.calls = []
        self.downloads = []
        self.closed = False

    def initialize(self):
        return TinkerCheckpoint("Qwen/Qwen3-8B", 16, "tinker://base/state", "tinker://base/sampler")

    def train(self, checkpoint, batches, loss, learning_rates):
        self.calls.append(checkpoint)
        number = len(self.calls)
        return TinkerCheckpoint(
            checkpoint.base_model, 16, f"tinker://update-{number}/state", f"tinker://update-{number}/sampler"
        ), {"loss": -1.0}

    def download(self, checkpoint, directory):
        self.downloads.append(checkpoint)
        directory.mkdir(parents=True)
        (directory / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 16}))
        (directory / "adapter_model.safetensors").write_bytes(checkpoint.sampler_path.encode())

    def close(self):
        self.closed = True


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
        raise AssertionError("unused")

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
        return StepSignal("train", {"steps": state.get("steps", 0) + 1}, advantages=tuple(1.0 for _ in batch.items))


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
        "training": {"backend": "tinker", "options": {"state-dir": str(tmp_path / "state"), "lora-rank": 16}},
    }


def remote_engine_config(tmp_path):
    raw = local_engine_config(tmp_path)
    raw["inference"]["remote"] = {"url": "https://gpu-host.example/", "inference-token-env": "GPU_HOST_TOKEN"}
    return raw


def runtime_config(config):
    return training_deployment_for("tinker").runtime_config(service_config_from_mapping(config).__dict__)


def build(config, monkeypatch, engines):
    """The runtime pair the HTTP service builds, over the SDK double and the engine double."""
    monkeypatch.setattr("reef.train.tinker_backend.client.TinkerSDKClient", RemoteClient)
    seen = {}

    def serve(factory, config, model_path, recipe_config, environ):
        seen.update(config)
        return AdapterServingRuntime(engines, Handler(), adapter_capacity=config["adapter_capacity"])

    monkeypatch.setattr(RemoteAdapterRuntimeFactory, "__call__", serve)
    training, inference = RuntimeRegistry().build(
        runtime_config(config), model_path="Qwen/Qwen3-8B", environ={"TINKER_API_KEY": "test-key"}
    )
    return training, inference, seen


def train(training, inference, objective, scenario, sample):
    prepared = training.prepare_training_step(
        TrainingBatch("b", (item(sample),)),
        TrainingMethod(objective.name),
        {},
        StepScheduling(unit="sample"),
        sample,
        scenario=scenario,
    )
    candidate = training.train_candidate(prepared.payload)
    return candidate, inference.activate_candidate(candidate)


def commit(training, inference, candidate, scenario):
    inference.acknowledge_publication(candidate.training_job_id)
    training.commit_candidate(candidate.training_job_id, scenario=scenario)


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
    serving = runtime_config(config)["serving"]
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
    assert runtime_config(config)["serving"]["remote"] == config["reef"]["inference_remote"]


def test_each_scenario_trains_and_serves_its_own_adapter(tmp_path, monkeypatch, objective):
    raw = local_engine_config(tmp_path)
    raw["training"]["options"]["max-loaded-adapters"] = 3
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    engines = Engines()
    training, inference, seen = build(config, monkeypatch, engines)
    try:
        assert isinstance(training, JobTrainingRuntime) and training.concurrent_training_scenarios
        options = seen["engine_config"]["options"]
        assert options["enable_lora"] and options["max_lora_rank"] == 16 and options["lora_target_modules"] == ["all"]
        assert seen["engine_config"]["model_path"] == "Qwen/Qwen3-8B"
        math, activated = train(training, inference, objective, "math", 0)
        adapter = Path(math.checkpoint_path) / ADAPTER_DIR
        assert engines.loads == [(adapter_name("math", activated.runtime_load_id), adapter, activated.runtime_load_id)]
        commit(training, inference, math, "math")
        assert inference.serving_adapter_runtime_load_id("math") == activated.runtime_load_id
        # Another scenario starts from the base; the committed scenario continues from its own adapter.
        train(training, inference, objective, "code", 1)
        train(training, inference, objective, "math", 2)
        client = training.component._client
        assert [checkpoint.state_path for checkpoint in client.calls] == [
            "tinker://base/state",
            "tinker://base/state",
            "tinker://update-1/state",
        ]
    finally:
        inference.shutdown()
        training.shutdown()


def test_a_restart_serves_each_committed_release_and_trains_from_it(tmp_path, monkeypatch, objective):
    config, _ = resolve_deployment_config(local_engine_config(tmp_path), None, tmp_path / "serve.yaml")
    training, inference, _ = build(config, monkeypatch, Engines())
    math, _ = train(training, inference, objective, "math", 0)
    commit(training, inference, math, "math")
    inference.shutdown()
    training.shutdown()

    engines = Engines()
    training, inference, _ = build(config, monkeypatch, engines)
    try:
        served = inference.activate_checkpoint(Artifact.local(Path(math.checkpoint_path)), scenario="math")
        assert engines.loads == [(adapter_name("math", served), Path(math.checkpoint_path) / ADAPTER_DIR, served)]
        _, activated = train(training, inference, objective, "math", 1)
        assert training.component._client.calls[0].state_path == "tinker://update-1/state"
        # One adapter slot: generation is paused, so the scenario's new adapter replaces its served one.
        assert engines.loads[1:] == [
            ("unload", adapter_name("math", served)),
            (adapter_name("math", activated.runtime_load_id), engines.loads[2][1], activated.runtime_load_id),
        ]
    finally:
        inference.shutdown()
        training.shutdown()


def test_engine_topologies_refuse_conflicting_inputs_before_starting(tmp_path, monkeypatch):
    for change, match in (
        ({"training": {"colocate": True}}, "training.colocate"),
        ({"inference": {"backend": "vllm"}}, "inference.backend: sglang"),
        ({"inference": {"options": {"enable-lora": True}}}, "set from training.options"),
        ({"training": {"options": {"lora-rank": 0}}}, "training.options"),
        ({"execution": {"rollout": "ray"}}, "remove the execution roles"),
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


def test_training_options_cannot_select_the_serving_engines(tmp_path):
    raw = local_engine_config(tmp_path)
    raw["inference"] = {"model-path": "Qwen/Qwen3-8B"}
    raw["training"]["options"]["serving"] = {"remote": {"url": "http://elsewhere"}}
    with pytest.raises(DeployConfigError, match="cannot set serving"):
        resolve_deployment_config(raw, None, tmp_path / "serve.yaml")


def test_tinker_samples_in_process_without_engines(tmp_path):
    raw = local_engine_config(tmp_path)
    raw["inference"] = {"model-path": "Qwen/Qwen3-8B"}
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    assert [service["name"] for service in config["services"]] == ["reef"]
    assert not local_model_required(config)
    assert not TinkerDeployment.requires_local_model
    assert "serving" not in runtime_config(config)


def test_coordinator_runtime_kind_connects_through_an_injected_connector():
    seen = {}

    def connect(**kwargs):
        seen.update(kwargs)
        from tests.support.runtime_stubs import StubTrainingRuntime

        stub = StubTrainingRuntime()
        return stub, stub.inference

    built = RuntimeRegistry().build(
        {"type": "coordinator_training", "inference_runtime": "sglang", "actor_name": "bridge", "connect": connect},
        model_path="/models/demo",
    )
    assert seen["actor_name"] == "bridge" and seen["model_path"] == "/models/demo"
    assert built[0] is not None and built[1] is not None
    with pytest.raises(RuntimeConfigError, match="inference_runtime"):
        RuntimeRegistry().build({"type": "coordinator_training", "connect": connect}, model_path="/models/demo")
