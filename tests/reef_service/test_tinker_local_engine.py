"""Tinker training behind Reef's coordinator, served by a local SGLang engine (CPU contracts)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from reef.runtime.deployment import ADAPTER_FILES_PROTOCOL, RuntimeConfigError, RuntimeRegistry
from reef.runtime.executor.uniproc import UniProcExecutor
from reef.runtime.interfaces import InferenceBackend
from reef.runtime.recovery import marker_path, read_marker
from reef.runtime.scheduler import TrainingCoordinator
from reef.service.deploy.config_utils import DeployConfigError
from reef.service.deploy.orchestrator import resolve_deployment_config
from reef.service.deploy.service_config import service_config_from_mapping
from reef.service.deploy.training import local_model_required, training_deployment_for
from reef.surface.adapter import adapter_name
from reef.train.algos import StepScheduling, StepSignal
from reef.train.algos.objective import TrainingObjective
from reef.train.algos.registry import register_objective, unregister_objective
from reef.train.tinker_backend.backend import ADAPTER_DIR, TinkerTrainingBackend
from reef.train.tinker_backend.checkpoint import TinkerCheckpoint
from reef.train.tinker_backend.config import TinkerConfig
from reef.train.tinker_backend.launch import TinkerDeployment
from reef.train.tinker_backend.training import TinkerDeploymentResources, TinkerTrainingService
from reef.train.types import TrainingBatch, TrajectoryItem

pytestmark = pytest.mark.usefixtures("objective")


class RemoteClient:
    """The Tinker SDK boundary as an offline double: trains, and writes a fake PEFT adapter on download."""

    def __init__(self):
        self.calls = []
        self.downloads = []
        self.closed = False

    def initialize(self):
        return TinkerCheckpoint("Qwen/Qwen3-8B", 32, "tinker://base/state", "tinker://base/sampler")

    def train(self, checkpoint, batches, loss):
        self.calls.append((checkpoint, batches, loss))
        number = len(self.calls)
        return TinkerCheckpoint(
            checkpoint.base_model, 32, f"tinker://update-{number}/state", f"tinker://update-{number}/sampler"
        ), {"loss": -1.0}

    def render(self, messages, *, template_kwargs):
        raise AssertionError("the local engine samples; the SDK client never renders")

    def decode(self, tokens):
        raise AssertionError("the local engine samples; the SDK client never decodes")

    def sample(self, checkpoint, prompt, params):
        raise AssertionError("the local engine samples; the SDK client never samples")

    def download(self, checkpoint, directory):
        self.downloads.append((checkpoint, directory))
        directory.mkdir(parents=True)
        (directory / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 32}))
        (directory / "adapter_model.safetensors").write_bytes(checkpoint.sampler_path.encode())

    def close(self):
        self.closed = True


class Engines(InferenceBackend):
    """A receiver that loads adapter files, as Reef's coordinator drives it."""

    def __init__(self):
        self.loads = []
        self.versions = ["engine:0"]
        self.events = []

    def load_adapter_files(self, name, path, runtime_load_id):
        self.loads.append((name, str(path), runtime_load_id))
        if runtime_load_id is not None:
            self.versions = [runtime_load_id]

    def initialize_version(self, runtime_load_id):
        self.versions = [runtime_load_id]

    def inference_url(self):
        return "http://engine.example"

    def runtime_load_ids(self):
        return list(self.versions)

    def pause(self):
        self.events.append("pause")

    def resume(self):
        self.events.append("resume")

    def recover(self):
        self.events.append("recover")

    def abort(self):
        self.events.append("abort")

    def offload(self, tags):
        raise AssertionError("a hosted trainer never offloads the engines")

    def onload_weights(self):
        raise AssertionError("a hosted trainer never onloads")

    def onload_kv(self):
        raise AssertionError("a hosted trainer never onloads")

    def unload_adapter(self, name):
        self.events.append(("unload", name))


class SampleObjective(TrainingObjective):
    loss_family = "importance_sampling"
    name = "tinker-local-engine-objective"

    def prepare(self, batch, state):
        return StepSignal(
            "train",
            {"steps": state.get("steps", 0) + 1},
            advantages=tuple(1.0 for _ in batch.items),
        )


@pytest.fixture
def objective():
    value = SampleObjective()
    register_objective(value)
    yield value
    unregister_objective(value.name)


def item(version, sample=0):
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
                        "runtime_load_id": version,
                    }
                }
            },
        }
    )


class Stack:
    """A coordinator over the Tinker backend and the engine double, as the model driver builds it."""

    def __init__(self, tmp_path, client=None, engines=None):
        self.client = client or RemoteClient()
        self.engines = engines or Engines()
        self.backend = TinkerTrainingBackend(
            "Qwen/Qwen3-8B", TinkerConfig(state_dir=str(tmp_path / "tinker")), client=self.client
        )
        self.coordinator = TrainingCoordinator(self.backend, self.engines)

    def payload(self, objective, *, step):
        version = self.coordinator.serving_runtime_load_id()
        prepared = self.coordinator.prepare_training_step(
            TrainingBatch("b", (item(version, sample=step),)),
            objective.name,
            {},
            StepScheduling(unit="sample", batch_size="actual"),
        )
        return {**prepared.payload, "scenario": "math", "rollout_id": step, "expected_runtime_load_id": version}


def test_each_step_branches_from_the_published_checkpoint_and_loads_its_adapter(tmp_path, objective):
    stack = Stack(tmp_path)
    first = stack.coordinator.execute_training_job(stack.payload(objective, step=0))
    assert first.outcome == "checkpoint"
    checkpoint = Path(first.checkpoint_path)
    assert TinkerCheckpoint.read(checkpoint).state_path == "tinker://update-1/state"
    assert (checkpoint / ADAPTER_DIR / "adapter_config.json").is_file()
    assert stack.engines.loads == []
    published = stack.coordinator.update_serving_weights(first.training_job_id)
    assert published.outcome == "complete"
    assert stack.engines.loads == [
        (adapter_name("math", published.runtime_load_id), str(checkpoint / ADAPTER_DIR), published.runtime_load_id)
    ]
    assert stack.coordinator.serving_runtime_load_id() == published.runtime_load_id
    stack.coordinator.acknowledge_training_commit(first.training_job_id)
    assert stack.coordinator.health()["lora_adapters"]["math"]["adapter"] == adapter_name(
        "math", published.runtime_load_id
    )
    second = stack.coordinator.execute_training_job(stack.payload(objective, step=1))
    assert second.outcome == "checkpoint"
    assert stack.client.calls[1][0].state_path == "tinker://update-1/state"


def test_a_rejected_step_leaves_the_incumbent_and_the_engines_unchanged(tmp_path, objective):
    stack = Stack(tmp_path)
    first = stack.coordinator.execute_training_job(stack.payload(objective, step=0))
    stack.coordinator.reject_training_candidate(first.training_job_id)
    assert stack.engines.loads == []
    assert read_marker(marker_path(stack.backend.config.save_hf_template))["status"] == "REJECTED"
    second = stack.coordinator.execute_training_job(stack.payload(objective, step=1))
    assert stack.client.calls[1][0].state_path == "tinker://base/state"
    assert Path(second.checkpoint_path) != Path(first.checkpoint_path)


def test_restart_republishes_the_incumbent_from_disk_and_branches_from_it(tmp_path, objective):
    stack = Stack(tmp_path)
    first = stack.coordinator.execute_training_job(stack.payload(objective, step=0))
    published = stack.coordinator.update_serving_weights(first.training_job_id)
    stack.coordinator.acknowledge_training_commit(first.training_job_id)
    stack.coordinator.shutdown()
    assert stack.client.closed

    engines = Engines()
    restarted = Stack(tmp_path, engines=engines)
    # Recovery reloads the committed adapter under its recorded name and version, without training.
    assert engines.loads == [
        (
            adapter_name("math", published.runtime_load_id),
            str(Path(first.checkpoint_path) / ADAPTER_DIR),
            published.runtime_load_id,
        )
    ]
    assert restarted.coordinator.serving_runtime_load_id() == published.runtime_load_id
    assert restarted.client.calls == []
    result = restarted.coordinator.execute_training_job(restarted.payload(objective, step=1))
    assert restarted.client.calls[0][0].state_path == "tinker://update-1/state"
    assert Path(result.checkpoint_path).name == "rollout_1"


def test_the_trainer_delivers_files_and_never_sends(tmp_path, objective):
    stack = Stack(tmp_path)
    with pytest.raises(RuntimeError, match="no Tinker checkpoint"):
        stack.backend.adapter_files("math", "other:1")
    first = stack.coordinator.execute_training_job(stack.payload(objective, step=0))
    published = stack.coordinator.update_serving_weights(first.training_job_id)
    adapter = Path(first.checkpoint_path) / ADAPTER_DIR
    # Reef's residency re-activation asks for the published revision by version, not for a send.
    assert stack.backend.adapter_files("math", published.runtime_load_id) == adapter
    for send in (
        lambda: stack.backend.send_weights("x:1", force_full=False),
        lambda: stack.backend.prepare_weights("x:1", force_full=False),
        lambda: stack.backend.send_adapter("math", adapter_name("math", "x:1")),
    ):
        with pytest.raises(RuntimeError, match="delivers adapter files"):
            send()
    stack.coordinator.acknowledge_training_commit(first.training_job_id)
    with pytest.raises(ValueError, match="name their scenario"):
        stack.coordinator.execute_training_job({**stack.payload(objective, step=1), "scenario": ""})


def _local_engine_config(tmp_path):
    return {
        "schema-version": 2,
        "recipe": {"implementation": "recipes.tttd.recipe:TTTDRecipe"},
        "inference": {"model-path": str(tmp_path / "model"), "backend": "sglang", "num-gpus": 2},
        "training": {"backend": "tinker", "options": {"state-dir": str(tmp_path / "state"), "lora-rank": 16}},
    }


def test_local_engine_topology_runs_the_model_driver_and_connects_through_the_coordinator(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    config, _ = resolve_deployment_config(_local_engine_config(tmp_path), None, tmp_path / "serve.yaml")
    assert [service["name"] for service in config["services"]] == ["tinker-driver", "reef"]
    assert config["services"][0]["command"][-1] == "reef.service.training_driver"
    assert config["execution"]["rollout"] == "ray"
    reef = config["reef"]
    assert (reef["inference_backend"], reef["inference_num_gpus"], reef["tensor_parallel_size"]) == ("sglang", 2, 1)
    assert local_model_required(config)
    settings = service_config_from_mapping(config)
    runtime = training_deployment_for("tinker").runtime_config(
        {**settings.__dict__, "ray_address": "ray://head:10001"}, max_staleness=0
    )
    assert runtime["type"] == "coordinator_training" and runtime["inference_runtime"] == "sglang"
    assert "test-key" not in yaml.safe_dump(config)


def test_local_engine_topology_rejects_conflicting_inputs_before_starting(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    for change, match in (
        ({"training": {"colocate": True}}, "training.colocate"),
        ({"inference": {"backend": "vllm"}}, "inference.backend: sglang"),
        ({"inference": {"options": {"enable-lora": True}}}, "set from training.options"),
        ({"training": {"options": {"lora-rank": 0}}}, "training.options"),
    ):
        raw = _local_engine_config(tmp_path)
        for section, values in change.items():
            raw[section] = (
                {**raw[section], **values}
                if section != "training" or "options" not in values
                else {
                    **raw[section],
                    "options": {**raw[section]["options"], **values["options"]},
                }
            )
        with pytest.raises(DeployConfigError, match=match):
            resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    monkeypatch.delenv("TINKER_API_KEY")
    with pytest.raises(DeployConfigError, match="TINKER_API_KEY"):
        resolve_deployment_config(_local_engine_config(tmp_path), None, tmp_path / "serve.yaml")


def test_in_process_topology_is_unchanged_without_a_local_engine(tmp_path):
    raw = _local_engine_config(tmp_path)
    raw["inference"] = {"model-path": "Qwen/Qwen3-8B"}
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    assert [service["name"] for service in config["services"]] == ["reef"]
    assert not local_model_required(config)
    assert not TinkerDeployment.requires_local_model


def test_training_plan_pairs_a_gpu_less_trainer_with_lora_serving_engines(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    monkeypatch.setenv("RAY_ADDRESS", "ray://head:10001")
    monkeypatch.setenv("REEF_RAY_NAMESPACE", "reef-test")
    config, _ = resolve_deployment_config(_local_engine_config(tmp_path), None, tmp_path / "serve.yaml")
    plan = TinkerDeployment().create_training_plan(config, loss_family="tttd")
    assert isinstance(plan.resources, TinkerDeploymentResources)
    assert (plan.resources.layout.training_gpus, plan.resources.layout.inference_gpus) == (0, 2)
    assert isinstance(plan.training, TinkerTrainingService)
    assert plan.coordinator is not None and plan.coordinator.options["namespace"] == "reef-test"
    options = plan.inference_config["options"]
    assert options["enable_lora"] and options["max_lora_rank"] == 16 and options["lora_target_modules"] == ["all"]
    assert (plan.inference_config["num_gpus"], plan.inference_config["gpus_per_engine"]) == (2, 1)
    assert options["model_path"] == str(tmp_path / "model")
    monkeypatch.delenv("RAY_ADDRESS")
    with pytest.raises(RuntimeError, match="RAY_ADDRESS"):
        TinkerDeployment().create_training_plan(config, loss_family="tttd")


def test_training_service_hands_the_coordinator_a_backend_only_after_engines_attach(tmp_path):
    from reef.runtime.deployment import WeightTransferSession

    client = RemoteClient()
    control = UniProcExecutor.from_workers([Engines()])
    service = TinkerTrainingService("Qwen/Qwen3-8B", TinkerConfig(state_dir=str(tmp_path)), "key", client=client)
    assert service.weight_transfer_protocol == ADAPTER_FILES_PROTOCOL
    with pytest.raises(RuntimeError, match="before attaching"):
        service.attach_weight_transport(WeightTransferSession(ADAPTER_FILES_PROTOCOL, control, "s"))
    service.start(None)
    with pytest.raises(ValueError, match="delivers adapter files"):
        service.attach_weight_transport(WeightTransferSession("slime-sglang-control-v2", control, "s"))
    with pytest.raises(RuntimeError, match="attached engines"):
        service.backend()
    service.attach_weight_transport(WeightTransferSession(ADAPTER_FILES_PROTOCOL, control, "s"))
    assert isinstance(service.backend(), TinkerTrainingBackend)
    service.close()
    assert not client.closed  # the coordinator owns the backend, and closes the client with it


def test_coordinator_runtime_kind_connects_through_an_injected_connector():
    seen = {}

    def connect(**kwargs):
        seen.update(kwargs)
        from reef_service.runtime_stubs import StubTrainingRuntime

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
