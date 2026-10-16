"""The MLX training and serving runtimes and their factory, against a fake engine; no MLX needed."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from tests.support._trajectories import policy_trajectory
from tests.support.mlx_engine_stubs import FakeAdapterSnapshot, FakeMLXEngine

from reef.artifact.artifact import Artifact
from reef.core.batches import StepScheduling
from reef.runtime.deployment import RuntimeConfigError, RuntimeRegistry
from reef.runtime.interfaces import LearningRateSchedule, RuntimeContractError, RuntimeLoadId, TrainingMethod
from reef.runtime.scheduler import RuntimeScheduler
from reef.train.algos.objective import TrainingObjective
from reef.train.algos.registry import register_objective
from reef.train.algos.signals import StepSignal
from reef.train.mlx_backend.config import MLXConfig
from reef.train.mlx_backend.contracts import LossOptions
from reef.train.mlx_backend.runtime import MLXRuntime, mlx_runtimes
from reef.train.mlx_backend.serving import MLXServingRuntime
from reef.train.types import TrainingBatch, TrajectoryItem

pytestmark = pytest.mark.unit


@register_objective
class TwoSampleObjective(TrainingObjective):
    name = "mlx-test-tttd"
    loss_family = "tttd"

    def prepare(self, batch, state):
        return StepSignal("train", {"steps": int(state.get("steps", 0)) + 1}, {"prepared": True}, (1.0, -1.0))


@register_objective
class UnknownFamilyObjective(TrainingObjective):
    name = "mlx-test-unknown-family"
    loss_family = "fake-family"

    def prepare(self, batch, state):
        return StepSignal("train", {}, {}, (1.0, -1.0))


#: The recipe's step schedule for one batch; the runtime trains it as one step.
ONE_STEP = StepScheduling(unit="sample", batch_size="actual")


def sample(reward: float) -> TrajectoryItem:
    return policy_trajectory(f"rec-{reward}", (1, 2, 3, 4), (1, 1), (-0.5, -0.25), reward)


def batch() -> TrainingBatch:
    return TrainingBatch("batch-1", (sample(1.0), sample(0.0)))


def build(tmp_path: Path, engine: FakeMLXEngine | None = None, **options) -> MLXRuntime:
    training, _ = mlx_runtimes(
        MLXConfig(checkpoint_dir=str(tmp_path / "checkpoints"), **options), engine or FakeMLXEngine()
    )
    return training


def train(runtime: MLXRuntime, scenario_step: int = 0):
    prepared = runtime.prepare_training_step(batch(), TrainingMethod("mlx-test-tttd"), {}, ONE_STEP, scenario_step)
    return runtime.train_candidate(prepared.payload)


def test_the_first_request_is_served_under_a_runtime_load_id(tmp_path: Path) -> None:
    runtime = build(tmp_path)

    assert RuntimeLoadId.parse(runtime.serving.serving_runtime_load_id()).sequence == 1


def test_preparation_runs_the_objective_in_process(tmp_path: Path) -> None:
    runtime = build(tmp_path)
    prepared = runtime.prepare_training_step(batch(), TrainingMethod("mlx-test-tttd"), {"steps": 4}, ONE_STEP, 7)

    assert prepared.action == "train"
    assert prepared.next_algorithm_state == {"steps": 5}
    assert prepared.payload["advantages"] == (1.0, -1.0)
    assert prepared.payload["scenario_step"] == 7
    assert prepared.payload["loss_family"] == "tttd"
    assert prepared.payload["method"] == TrainingMethod("mlx-test-tttd").to_dict()
    assert len(prepared.payload["samples"]) == 2


@pytest.mark.parametrize(
    ("method", "scheduling", "message"),
    [
        (TrainingMethod("mlx-test-unknown-family"), ONE_STEP, "unsupported MLX loss family 'fake-family'"),
        (
            TrainingMethod("mlx-test-tttd", LearningRateSchedule(name="warmup", peak_learning_rate=1e-4)),
            ONE_STEP,
            "learning-rate schedule",
        ),
        (TrainingMethod("mlx-test-tttd"), StepScheduling(unit="sample", batch_size="actual", epochs=3), "epochs=3"),
        (TrainingMethod("mlx-test-tttd"), StepScheduling(unit="sample", batch_size=4), "batch_size=4"),
    ],
    ids=["loss-family", "learning-rate-schedule", "epochs", "sub-batching"],
)
def test_a_method_the_runtime_cannot_train_is_refused_before_training(
    tmp_path: Path, method: TrainingMethod, scheduling: StepScheduling, message: str
) -> None:
    runtime = build(tmp_path)

    with pytest.raises(RuntimeContractError, match=message):
        runtime.prepare_training_step(batch(), method, {}, scheduling, 0)
    assert runtime.engine.trained == []


def test_a_configured_batch_size_is_the_reserved_batch(tmp_path: Path) -> None:
    runtime = build(tmp_path)
    prepared = runtime.prepare_training_step(
        batch(), TrainingMethod("mlx-test-tttd"), {}, StepScheduling(unit="sample"), 0
    )

    assert prepared.action == "train"


def test_a_candidate_is_exported_and_serving_keeps_its_weights(tmp_path: Path) -> None:
    runtime = build(tmp_path)
    engine = runtime.engine
    served = runtime.serving.serving_runtime_load_id()

    candidate = train(runtime, scenario_step=3)

    [(destination, source)] = engine.saved
    assert candidate.checkpoint_path == str(destination)
    assert destination.parent == tmp_path / "checkpoints"
    assert json.loads((destination / "adapters.safetensors").read_text()) == {"layer.lora_a": 1.0, "layer.lora_b": 1.0}
    assert source == {
        "scenario_step": 3,
        "training_job_id": candidate.training_job_id,
        "objective": "mlx-test-tttd",
        "loss_family": "tttd",
        "source_runtime_load_id": None,
    }
    assert candidate.training_metrics["adapter_tensors_changed"] == 2
    assert engine.weights == {"layer.lora_a": 0.0, "layer.lora_b": 0.0}
    assert runtime.serving.serving_runtime_load_id() == served
    assert runtime.serving.inference_admission_status["open"] is True


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({"kl_coef": 0.05}, LossOptions(0.05, {})),
        ({"kl_coef": 0.05, "loss_options": {"kl_coef": 0.5}}, LossOptions(0.05, {"kl_coef": 0.5})),
    ],
    ids=["deployment-kl-coef", "loss-options-kl-coef"],
)
def test_the_deployment_loss_settings_reach_the_engine(tmp_path: Path, options: dict, expected: LossOptions) -> None:
    runtime = build(tmp_path, **options)

    train(runtime)

    [call] = runtime.engine.trained
    assert call.loss_family == "tttd"
    assert call.options == expected
    assert call.advantages == (1.0, -1.0)


def test_a_step_that_moves_nothing_is_refused(tmp_path: Path) -> None:
    runtime = build(tmp_path, FakeMLXEngine(moves_weights=False))

    with pytest.raises(RuntimeContractError, match="left every adapter tensor unchanged"):
        train(runtime)

    assert runtime.engine.saved == []
    assert runtime.serving.inference_admission_status["open"] is True


@dataclass
class FailingEngine(FakeMLXEngine):
    """Changes the weights and then fails, as a step that dies after the optimizer update."""

    def train(self, loss_family, samples, advantages, options):
        self.weights = {name: value + 99.0 for name, value in self.weights.items()}
        raise RuntimeError("metal failed")


def test_a_failed_step_restores_the_served_weights_and_reopens_admission(tmp_path: Path) -> None:
    runtime = build(tmp_path, FailingEngine())

    with pytest.raises(RuntimeError, match="metal failed"):
        train(runtime)

    assert runtime.engine.weights == {"layer.lora_a": 0.0, "layer.lora_b": 0.0}
    assert runtime.serving.inference_admission_status["open"] is True


@dataclass
class RefusingEngine(FakeMLXEngine):
    """Refuses the batch, as a loss family does for a sample without the inputs it trains on."""

    def train(self, loss_family, samples, advantages, options):
        raise ValueError("sample 0 carries no generation top-K")


def test_a_batch_the_loss_family_refuses_is_a_contract_error(tmp_path: Path) -> None:
    runtime = build(tmp_path, RefusingEngine())

    with pytest.raises(RuntimeContractError, match="refused its batch: sample 0 carries no generation top-K"):
        train(runtime)

    assert runtime.engine.saved == []
    assert runtime.serving.inference_admission_status["open"] is True


@dataclass
class AdmissionRecordingEngine(FakeMLXEngine):
    """Records whether serving admitted requests while the engine trained or generated."""

    serving: MLXServingRuntime | None = None
    admission_open: list[bool] = field(default_factory=list)

    def train(self, loss_family, samples, advantages, options):
        self.admission_open.append(self.serving.inference_admission_status["open"])
        return super().train(loss_family, samples, advantages, options)

    def generate(self, prompt_tokens, *, max_tokens=None, temperature=None, listener=None):
        self.admission_open.append(self.serving.inference_admission_status["open"])
        return super().generate(prompt_tokens, max_tokens=max_tokens, temperature=temperature, listener=listener)


def recording_runtime(tmp_path: Path) -> MLXRuntime:
    engine = AdmissionRecordingEngine()
    runtime = build(tmp_path, engine)
    engine.serving = runtime.serving
    return runtime


def test_requests_wait_while_the_adapter_trains(tmp_path: Path) -> None:
    runtime = recording_runtime(tmp_path)

    train(runtime)

    assert runtime.engine.admission_open == [False]
    assert runtime.serving.inference_admission_status["open"] is True


def test_the_scheduler_publishes_an_activated_candidate_only_after_the_commit(tmp_path: Path) -> None:
    runtime = build(tmp_path)
    serving = runtime.serving
    scheduler = RuntimeScheduler(runtime, serving)
    before = serving.serving_runtime_load_id()
    candidate = scheduler.train_candidate(
        runtime.prepare_training_step(batch(), TrainingMethod("mlx-test-tttd"), {}, ONE_STEP, 0).payload
    )

    activated = scheduler.activate_candidate(candidate)

    assert activated.runtime_load_id != before
    assert serving.serving_runtime_load_id() == activated.runtime_load_id
    assert runtime.engine.weights == {"layer.lora_a": 1.0, "layer.lora_b": 1.0}
    assert serving.inference_admission_status["open"] is False
    assert serving.current_runtime_load_id() == before

    scheduler.finish_committed_training_job(candidate.training_job_id)

    assert serving.inference_admission_status["open"] is True
    assert serving.current_runtime_load_id() == activated.runtime_load_id


def test_a_rejected_candidate_never_reaches_serving(tmp_path: Path) -> None:
    runtime = build(tmp_path)
    serving = runtime.serving
    scheduler = RuntimeScheduler(runtime, serving)
    before = serving.serving_runtime_load_id()
    candidate = train(runtime)

    scheduler.reject_candidate(candidate)

    assert serving.staged_candidate(candidate.candidate_id) is None
    assert serving.serving_runtime_load_id() == before
    assert serving.inference_admission_status["open"] is True
    with pytest.raises(RuntimeContractError, match="no pending mlx candidate"):
        serving.activate_candidate(candidate)


def published_adapter(tmp_path: Path, metadata: dict | None = None) -> Artifact:
    directory = tmp_path / "published"
    directory.mkdir()
    (directory / "adapters.safetensors").write_text(json.dumps({"layer.lora_a": 5.0, "layer.lora_b": 5.0}))
    return Artifact.local(directory, metadata=metadata)


@pytest.mark.parametrize("admission_open", [True, False], ids=["serving", "rollback-paused"])
def test_a_restore_loads_the_adapter_and_keeps_the_admission_state(tmp_path: Path, admission_open: bool) -> None:
    serving = build(tmp_path).serving
    artifact = published_adapter(tmp_path)
    if not admission_open:
        serving.pause_admission()

    restored = serving.restore_checkpoint(artifact)

    assert RuntimeLoadId.parse(restored).sequence == 2
    assert serving.serving_runtime_load_id() == restored
    assert serving.engine.loaded == [artifact.local_path]
    assert serving.inference_admission_status["open"] is admission_open


def test_recovery_reloads_a_checkpoint_published_under_another_engine(tmp_path: Path) -> None:
    serving = build(tmp_path).serving
    artifact = published_adapter(tmp_path, {"runtime_load_id": "previous-incarnation:4"})

    runtime_load_id = serving.activate_checkpoint(artifact)

    assert serving.engine.weights == {"layer.lora_a": 5.0, "layer.lora_b": 5.0}
    assert runtime_load_id == serving.serving_runtime_load_id()
    assert serving.current_runtime_load_id() == runtime_load_id
    assert serving.inference_admission_status["open"] is True


@pytest.mark.parametrize("recorded", ["served", None], ids=["served-version", "no-version"])
def test_activation_keeps_the_weights_the_engine_already_serves(tmp_path: Path, recorded: str | None) -> None:
    serving = build(tmp_path).serving
    served = serving.serving_runtime_load_id()
    artifact = published_adapter(tmp_path, None if recorded is None else {"runtime_load_id": served})

    assert serving.activate_checkpoint(artifact) == served
    assert serving.engine.loaded == []


def test_a_probe_generates_from_the_candidate_and_restores_the_served_weights(tmp_path: Path) -> None:
    runtime = recording_runtime(tmp_path)
    candidate = train(runtime)
    runtime.engine.admission_open.clear()

    replies = runtime.probe_candidate(candidate.candidate_id, [[1, 2], [3]], max_tokens=16)

    engine = runtime.engine
    assert replies == ["an answer", "an answer"]
    assert [call["weights"] for call in engine.generated] == [{"layer.lora_a": 1.0, "layer.lora_b": 1.0}] * 2
    assert [(call["max_tokens"], call["temperature"]) for call in engine.generated] == [(16, 0.0)] * 2
    assert engine.admission_open == [False, False]
    assert engine.weights == {"layer.lora_a": 0.0, "layer.lora_b": 0.0}
    assert runtime.serving.inference_admission_status["open"] is True


def test_a_probe_of_an_unknown_candidate_is_refused(tmp_path: Path) -> None:
    runtime = build(tmp_path)

    with pytest.raises(RuntimeContractError, match="no pending mlx candidate"):
        runtime.probe_candidate("missing", [[1]], max_tokens=4)


def test_serving_shutdown_closes_the_engine(tmp_path: Path) -> None:
    serving = build(tmp_path).serving

    serving.shutdown()

    assert serving.engine.closed is True


def test_a_staged_snapshot_is_the_trained_adapter(tmp_path: Path) -> None:
    runtime = build(tmp_path)
    candidate = train(runtime)

    assert runtime.serving.staged_candidate(candidate.candidate_id) == FakeAdapterSnapshot(
        {"layer.lora_a": 1.0, "layer.lora_b": 1.0}
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({}, "checkpoint_dir is required"),
        ({"chat_template_kwargs": "enable_thinking=false"}, "chat_template_kwargs"),
        ({"lora_keys": "self_attn.q_proj"}, "lora_keys"),
        ({"loss_option": {}}, "unknown config fields: loss_option"),
        ({"train_timeout_s": 60}, "no train timeout"),
        ({"recurrence_chunk_size": 48}, "power of two"),
    ],
    ids=[
        "no-checkpoint-dir",
        "template-kwargs",
        "lora-keys",
        "unknown-key",
        "train-timeout",
        "chunk",
    ],
)
def test_the_factory_refuses_invalid_options_before_it_loads_mlx(tmp_path: Path, options: dict, message: str) -> None:
    config = {"type": "mlx", **({"checkpoint_dir": str(tmp_path)} if options else {}), **options}

    with pytest.raises(RuntimeConfigError, match=message):
        RuntimeRegistry().build(config, model_path="fake/model", environ={})
