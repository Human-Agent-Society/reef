"""CPU contract tests for Tinker's training component, its sampler runtime and their deployment."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from recipes.tttd.tinker import TttdTinkerLoss
from reef.artifact.artifact import Artifact
from reef.cli import main
from reef.core.job_data import JobData, TrajectorySample
from reef.inference.tinker import TinkerInferenceRuntime
from reef.recipe.sft import SftObjective
from reef.runtime.deployment import RuntimeConfigError
from reef.runtime.interfaces import (
    LearningRateSchedule,
    LearningRateScheduleState,
    TrainingMethod,
    UpstreamStatusError,
)
from reef.runtime.jobs import ReleaseFormat, TrainingJob, job_id_for
from reef.runtime.serving import ADAPTER_DIR
from reef.service.deploy import orchestrator
from reef.service.deploy.orchestrator import resolve_deployment_config
from reef.service.deploy.training import local_model_required
from reef.tinker.checkpoint import MANIFEST, TinkerCheckpoint
from reef.tinker.sampling import SampleResult, TinkerSampler
from reef.train.algos import TrainingSignal
from reef.train.algos.objective import TrainingObjective
from reef.train.algos.registry import register_objective, unregister_objective
from reef.train.algos.schedule import StepScheduling
from reef.train.jobs import JobLoopState
from reef.train.local_jobs import CHECKPOINTS_FILE, RESULT_FILE
from reef.train.tinker_backend.client import TINKER_ADAPTER_DIR, TinkerClient
from reef.train.tinker_backend.component import TinkerJobParameters, TinkerTrainingComponent
from reef.train.tinker_backend.config import TinkerConfig
from reef.train.tinker_backend.launch import TinkerDeployment, component_factory
from reef.train.tinker_backend.losses import ImportanceSamplingLoss, TokenRow, resolve_tinker_loss
from reef.train.tinker_backend.preparation import prepare_tinker_job

MODEL = "Qwen/Qwen3-8B"


class RemoteClient(TinkerClient, TinkerSampler):
    """One offline double for both SDK boundaries: the trainer's and the sampler's."""

    def __init__(self):
        self.initializations = 0
        self.calls = []
        self.learning_rates = []
        self.sampled = []
        self.downloads = []
        self.training_options = []
        self.closed = False
        self.fail = False

    def initialize(self):
        self.initializations += 1
        return TinkerCheckpoint(MODEL, 32, "tinker://base/state", "tinker://base/sampler")

    def train(self, checkpoint, batches, loss, learning_rates, *, kl_coef, export_adapter):
        self.calls.append((checkpoint, batches, loss))
        self.learning_rates.append(tuple(learning_rates))
        self.training_options.append((kl_coef, export_adapter))
        if self.fail:
            raise TimeoutError("uncertain remote optimizer result")
        number = len(self.calls)
        return TinkerCheckpoint(
            checkpoint.base_model, 32, f"tinker://update-{number}/state", f"tinker://update-{number}/sampler"
        ), {"loss": -1.25}

    def render(self, messages, *, template_kwargs):
        self.rendered = (messages, template_kwargs)
        return [101, 102]

    def decode(self, tokens):
        return "hello"

    def sample(self, sampler_path, prompt, params):
        self.sampled.append((sampler_path, prompt, params))
        return SampleResult((201, 202), (-0.2, -0.3), "stop")

    def download(self, checkpoint, release):
        self.downloads.append((checkpoint, release))
        for name in (ADAPTER_DIR, TINKER_ADAPTER_DIR):
            (release / name).mkdir(parents=True)
            (release / name / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 32}))
            (release / name / "adapter_model.safetensors").write_bytes(checkpoint.sampler_path.encode())

    def close(self):
        self.closed = True


class SampleObjective(TrainingObjective):
    loss_family = "importance_sampling"
    name = "tinker-test-objective"

    def __init__(self):
        self.loss_family = "importance_sampling"
        self.action = "train"

    def prepare(self, batch, state):
        return TrainingSignal(
            self.action,
            {"steps": state.get("steps", 0) + 1},
            advantages=tuple(float(i + 1) for i in range(len(batch.samples))),
        )


class TttdLossObjective(SampleObjective):
    """The sample objective trained with TTTD's Tinker loss instead of importance sampling."""

    name = "tinker-test-tttd-objective"

    def __init__(self):
        super().__init__()
        self.loss_family = "tttd"


WARMUP = LearningRateSchedule("warmup", 1e-4, warmup_steps=4)
POLICY = LearningRateSchedule("policy", 1e-5, warmup_steps=2)


@pytest.fixture
def objective():
    value = SampleObjective()
    register_objective(value)
    yield value
    unregister_objective(value.name)


@pytest.fixture
def second_objective():
    value = TttdLossObjective()
    register_objective(value)
    yield value
    unregister_objective(value.name)


@pytest.fixture
def client():
    return RemoteClient()


@pytest.fixture
def config(tmp_path):
    return TinkerConfig(state_dir=str(tmp_path / "state"))


@pytest.fixture
def component(config, client):
    value = TinkerTrainingComponent(MODEL, config, client)
    yield value
    value.close()


@pytest.fixture
def inference(client):
    value = TinkerInferenceRuntime(client, base_model=MODEL)
    yield value
    value.shutdown()


def item(version="v", group="g", token=20):
    return TrajectorySample(
        {
            "schema_version": "ATIF-v1.6",
            "agent": {"name": "test"},
            "steps": [{"step_id": 1}],
            "extra": {
                "reef": {
                    "training": {
                        "tokens": [10, 11, token, 21],
                        "loss_mask": [1, 0],
                        "rollout_log_probs": [-0.2, -0.4],
                        "runtime_load_id": version,
                    }
                }
            },
        },
        group_id=group,
    )


def batch(rows=1, name="batch"):
    """``rows`` distinct samples; with one sample per optimizer step, a job of ``rows`` optimizer steps."""
    return JobData(name, tuple(item(group=str(index), token=20 + index) for index in range(rows)))


def wait(component, job_id):
    deadline = time.monotonic() + 10
    while (status := component.status(job_id)).state == "running":
        assert time.monotonic() < deadline, status
        time.sleep(0.01)
    return status


def submit(component, data, objective_name, *, start=None, schedule=None, **parameters):
    """Submit one job as a job loop does and return its ID."""
    parameters = {
        "method": TrainingMethod(objective_name, schedule).to_dict(),
        **parameters,
    }
    job_id = job_id_for(component.name, start, data.to_dict()["samples"], parameters)
    component.submit(TrainingJob(job_id, component.name, start, data, parameters))
    return job_id


def train(component, data, objective_name, **options):
    """Run one job to its end and return its status."""
    return wait(component, submit(component, data, objective_name, **options))


def empty_artifact(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir(exist_ok=True)
    return Artifact.local(directory)


# -- The training component


def test_a_job_saves_one_checkpoint_and_the_next_job_continues_from_it(component, client, objective):
    # The record-to-record stage decides which samples are too stale; the component trains what it receives.
    first = train(component, JobData("older", (item("old:0"),)), objective.name)
    assert first.state == "finished" and first.metrics["optimizer_steps"] == 1
    (checkpoint,) = first.checkpoints
    assert checkpoint.step == 1 and checkpoint.path == checkpoint.release_path
    assert Path(checkpoint.path) == component.state_dir / "jobs" / first.job_id / "1"
    assert checkpoint.metrics["loss"] == -1.25
    manifest = TinkerCheckpoint.read(Path(checkpoint.path))
    assert manifest.state_path == "tinker://update-1/state" and manifest.algorithm_state == {"steps": 1}
    assert manifest.objective == objective.name
    # Submitting the same job again trains nothing.
    assert submit(component, JobData("older", (item("old:0"),)), objective.name) == first.job_id
    second = train(component, batch(name="next"), objective.name, start=checkpoint.path)
    assert second.state == "finished"
    assert [call[0].state_path for call in client.calls] == ["tinker://base/state", "tinker://update-1/state"]
    assert TinkerCheckpoint.read(Path(second.checkpoints[0].path)).algorithm_state == {"steps": 2}


def test_releases_hold_an_adapter_that_engines_load_only_with_export_adapter(component, client, tmp_path):
    # Without the export, a release is the manifest that only Tinker's sampler serves.
    assert component.release_format is None
    settings = TinkerConfig(state_dir=str(tmp_path / "exporting"), export_adapter=True)
    exporting = TinkerTrainingComponent(MODEL, settings, client)
    try:
        assert exporting.release_format == ReleaseFormat("adapter", settings.lora_rank, ("all",))
    finally:
        exporting.close()


def test_a_failed_job_reports_its_error_and_saves_no_checkpoint(component, client, objective):
    client.fail = True
    failed = train(component, batch(), objective.name)
    assert failed.state == "failed" and "uncertain remote optimizer result" in failed.error
    assert failed.checkpoints == ()
    client.fail = False
    retried = train(component, batch(name="retry", rows=2), objective.name)
    assert retried.state == "finished"
    assert all(call[0].state_path == "tinker://base/state" for call in client.calls)


def test_save_interval_saves_a_checkpoint_every_interval_and_at_the_end(component, client, objective):
    status = train(component, batch(5), objective.name, schedule=WARMUP, save_interval=2)
    assert [checkpoint.step for checkpoint in status.checkpoints] == [2, 4, 5]
    assert [len(call[1]) for call in client.calls] == [2, 2, 1]
    # Each chunk continues from the previous chunk's checkpoint with the schedule's next rates.
    assert [call[0].state_path for call in client.calls] == [
        "tinker://base/state",
        "tinker://update-1/state",
        "tinker://update-2/state",
    ]
    rates = [rate for chunk in client.learning_rates for rate in chunk]
    assert rates == pytest.approx([0.0, 2.5e-5, 5e-5, 7.5e-5, 1e-4])
    assert [row.tokens[2] for chunk in client.calls for rows in chunk[1] for row in rows] == [20, 21, 22, 23, 24]
    assert [checkpoint.metrics["learning_rate_schedule"]["completed_steps"] for checkpoint in status.checkpoints] == [
        2,
        4,
        5,
    ]
    assert TinkerCheckpoint.read(Path(status.checkpoints[-1].path)).learning_rate_schedule_state == (
        LearningRateScheduleState(WARMUP, 5)
    )
    # Only the last checkpoint has trained all the data that the objective's next state is for.
    manifests = [TinkerCheckpoint.read(Path(checkpoint.path)) for checkpoint in status.checkpoints]
    assert [manifest.algorithm_state for manifest in manifests] == [{}, {}, {"steps": 1}]
    assert [checkpoint.metrics["checkpoint_optimizer_steps"] for checkpoint in status.checkpoints] == [2, 2, 1]
    assert status.metrics["optimizer_steps"] == 5


def test_the_jobs_kl_coefficient_and_the_adapter_export_reach_training(component, client, second_objective):
    # The job's coefficient, not the component's default of 0, weighs the TTTD loss's KL term.
    status = train(component, batch(), second_objective.name, kl_coef=0.2)
    assert status.state == "finished", status.error
    assert client.training_options == [(0.2, False)]


def test_job_parameters_fill_the_configured_defaults_and_jobs_of_several_scenarios_share_the_component(
    component, config, objective
):
    method = {"method": TrainingMethod(objective.name).to_dict()}
    parameters = component.job_parameters({**method, "kl_coef": 0.2})
    assert parameters == TinkerJobParameters(
        TrainingMethod(objective.name), StepScheduling(), config.batch_size, config.learning_rate, 0.2, None, None
    )
    assert component.shares_scenarios is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"lora_rank": 8}, r"accept no parameters \['lora_rank'\]"),
        ({"method": "sft"}, "'method' parameter as an object"),
        ({"scheduling": None}, "scheduling parameter must be an object"),
        ({"scheduling": {"epochs": 2}}, "does not support 2 epochs per job"),
        ({"save_interval": 0}, "save_interval must be a positive int"),
        ({"save_interval": True}, "save_interval must be a positive int"),
        ({"save_interval": 1.5}, "save_interval must be a positive int"),
        ({"batch_size": 0}, "batch_size must be a positive int"),
        ({"learning_rate": "fast"}, "learning_rate must be a finite number"),
        ({"learning_rate": -1e-4}, "learning_rate must be positive"),
        ({"kl_coef": -0.1}, "kl_coef non-negative"),
    ],
)
def test_a_job_with_parameters_tinker_does_not_take_is_refused_and_records_nothing(
    component, client, objective, overrides, message
):
    parameters = {"method": TrainingMethod(objective.name).to_dict(), **overrides}
    with pytest.raises(ValueError, match=message):
        component.submit(TrainingJob("refused", component.name, None, batch(), parameters))
    with pytest.raises(KeyError):
        component.status("refused")
    assert client.calls == []


def test_a_restarted_component_resumes_a_job_after_its_last_checkpoint(config, client, objective):
    component = TinkerTrainingComponent(MODEL, config, client)
    job_id = submit(component, batch(3), objective.name, save_interval=1)
    first = wait(component, job_id).checkpoints[0]
    component.close()
    # As a process that stopped after the first checkpoint left the job: one checkpoint and no end.
    directory = component.state_dir / "jobs" / job_id
    (directory / RESULT_FILE).unlink()
    saved = json.loads((directory / CHECKPOINTS_FILE).read_text())
    (directory / CHECKPOINTS_FILE).write_text(json.dumps({"checkpoints": saved["checkpoints"][:1]}))
    trained = len(client.calls)

    restarted = TinkerTrainingComponent(MODEL, config, client)
    try:
        status = wait(restarted, job_id)
        assert status.state == "finished"
        assert [checkpoint.step for checkpoint in status.checkpoints] == [1, 2, 3]
        resumed = client.calls[trained:]
        assert [len(call[1]) for call in resumed] == [1, 1]
        assert resumed[0][0].state_path == TinkerCheckpoint.read(Path(first.path)).state_path
        assert [row.tokens[2] for call in resumed for rows in call[1] for row in rows] == [21, 22]
        assert client.initializations == 1
    finally:
        restarted.close()


def test_a_component_refuses_checkpoints_that_another_tinker_service_holds(config, client, tmp_path):
    TinkerTrainingComponent(MODEL, config, client).close()
    spindle = replace(config, base_url="https://spindle.example")
    with pytest.raises(ValueError, match="held by hosted Tinker, not by the configured service https://spindle"):
        TinkerTrainingComponent(MODEL, spindle, client)

    held = TinkerCheckpoint(MODEL, 32, "tinker://s/state", "tinker://s/sampler", service="https://spindle.example")
    assert TinkerCheckpoint.from_dict(held.to_dict()) == held
    assert "service" not in TinkerCheckpoint(MODEL, 32, "tinker://s/state", "tinker://s/sampler").to_dict()
    held.check_compatible(MODEL, service="https://spindle.example", lora_rank=32)
    with pytest.raises(ValueError, match="not by the configured service hosted Tinker"):
        held.check_compatible(MODEL, service=None, lora_rank=32)


def test_import_start_copies_a_release_manifest_and_takes_the_base_tree_as_the_base(component, objective, tmp_path):
    status = train(component, batch(), objective.name)
    release = tmp_path / "release"
    release.mkdir()
    (release / ".gitattributes").write_text("")
    assert component.import_start(release) is None
    (release / MANIFEST).write_text((Path(status.checkpoints[0].release_path) / MANIFEST).read_text())
    start = component.import_start(release)
    assert Path(start).parent == component.state_dir / "imported"
    assert TinkerCheckpoint.read(Path(start)) == TinkerCheckpoint.read(release)
    assert component.import_start(release) == start


def test_exclusive_state_dir_and_model_validation(component, config, inference, tmp_path):
    assert inference.served_weight_kind == "model_adapter", "a Tinker sampler serves a LoRA adapter"
    with pytest.raises(ValueError, match="already owned"):
        TinkerTrainingComponent(MODEL, config, RemoteClient())
    other = TinkerCheckpoint("other/model", 32, "tinker://other/state", "tinker://other/sampler")
    other.write(tmp_path / "other")
    with pytest.raises(ValueError, match="base model"):
        inference.activate_checkpoint(Artifact.local(tmp_path / "other"))
    with pytest.raises(ValueError, match="LoRA of base model"):
        component.import_start(tmp_path / "other")
    (tmp_path / "other" / MANIFEST).unlink()
    (tmp_path / "other" / "weights.bin").write_bytes(b"not a manifest")
    with pytest.raises(ValueError, match="missing"):
        inference.snapshot(Artifact.local(tmp_path / "other"))
    with pytest.raises(ValueError, match="missing"):
        component.import_start(tmp_path / "other")


def test_skip_and_unsupported_loss_do_not_mutate_remote(component, client, objective):
    objective.action = "skip"
    skipped = train(component, batch(), objective.name)
    assert skipped.state == "finished" and skipped.checkpoints == ()
    objective.action = "train"
    objective.loss_family = "sao"
    refused = train(component, batch(2), objective.name)
    assert refused.state == "failed" and "unsupported Tinker loss" in refused.error
    assert not client.calls


def test_a_method_without_a_tinker_loss_is_refused_before_training(component, client):
    status = train(component, batch(), "sdpo")
    assert status.state == "failed" and "unsupported Tinker loss family 'sdpo'" in status.error
    assert client.calls == []


def test_a_method_whose_loss_has_no_kl_penalty_is_refused_before_training(
    component, client, objective, second_objective
):
    # Importance sampling has no KL penalty to train, so the job is refused before it reaches Tinker.
    refused = train(component, batch(), objective.name, kl_coef=0.1)
    assert refused.state == "failed" and "importance_sampling does not implement a KL penalty" in refused.error
    assert client.calls == []
    # TTTD's loss trains the configured KL penalty.
    assert train(component, batch(), second_objective.name, kl_coef=0.1).state == "finished"
    assert client.calls[-1][2] is resolve_tinker_loss("tttd")


def test_a_schedule_counts_optimizer_steps_across_jobs_and_restarts(config, client, objective):
    component = TinkerTrainingComponent(MODEL, config, client)
    first = train(component, batch(3), objective.name, schedule=WARMUP).checkpoints[-1]
    assert client.learning_rates[-1] == pytest.approx((0.0, 2.5e-5, 5e-5))
    assert first.metrics["learning_rate"] == pytest.approx(5e-5)
    assert first.metrics["learning_rate_schedule"] == {"name": "warmup", "completed_steps": 3}
    assert TinkerCheckpoint.read(Path(first.path)).learning_rate_schedule_state == LearningRateScheduleState(WARMUP, 3)
    component.close()

    # A restart continues from the last checkpoint: the warmup continues instead of starting over.
    restarted = TinkerTrainingComponent(MODEL, config, client)
    try:
        continued = train(restarted, batch(3, "b1"), objective.name, start=first.path, schedule=WARMUP)
        assert client.learning_rates[-1] == pytest.approx((7.5e-5, 1e-4, 1e-4))
        # A job that selects no schedule keeps the active one.
        start = continued.checkpoints[-1].path
        kept = train(restarted, batch(3, "b2"), objective.name, start=start).checkpoints[-1]
        assert client.learning_rates[-1] == pytest.approx((1e-4, 1e-4, 1e-4))
        assert kept.metrics["learning_rate_schedule"]["completed_steps"] == 9
        # Another schedule starts at its step 0 from the same optimizer state.
        switched = train(restarted, batch(3, "b3"), objective.name, start=kept.path, schedule=POLICY)
        assert client.learning_rates[-1] == pytest.approx((0.0, 5e-6, 1e-5))
        assert client.calls[-1][0].state_path == TinkerCheckpoint.read(Path(kept.path)).state_path
        assert switched.checkpoints[-1].metrics["learning_rate_schedule"] == {"name": "policy", "completed_steps": 3}
        assert client.initializations == 1
    finally:
        restarted.close()


def test_without_a_selected_schedule_every_step_uses_the_configured_rate(component, client, config, objective):
    (checkpoint,) = train(component, batch(3), objective.name).checkpoints
    assert client.learning_rates[-1] == (config.learning_rate,) * 3
    assert checkpoint.metrics["learning_rate"] == config.learning_rate
    assert "learning_rate_schedule" not in checkpoint.metrics
    # A checkpoint without a schedule keeps its manifest in the earlier shape.
    assert "learning_rate_schedule_state" not in json.loads((Path(checkpoint.path) / MANIFEST).read_text())
    train(component, batch(name="rate"), objective.name, learning_rate=3e-5)
    assert client.learning_rates[-1] == (3e-5,)


def test_each_job_trains_the_loss_of_its_selected_objective(component, client, objective, second_objective):
    first = train(component, batch(), objective.name)
    second = train(component, batch(), second_objective.name, start=first.checkpoints[0].path)
    assert isinstance(client.calls[0][2], ImportanceSamplingLoss)
    assert isinstance(client.calls[1][2], TttdTinkerLoss)
    # The second objective continues the first one's weights and optimizer state, not its objective state.
    assert client.calls[1][0].state_path == "tinker://update-1/state"
    assert second.job_id != first.job_id
    manifest = TinkerCheckpoint.read(Path(second.checkpoints[0].path))
    assert (manifest.objective, manifest.algorithm_state) == (second_objective.name, {"steps": 1})


def test_tinker_cannot_restart_training_and_names_the_objective_of_a_saved_state(component, objective):
    with pytest.raises(ValueError, match="the tinker training component cannot restart training"):
        component.fresh_start(None)
    with pytest.raises(ValueError, match="objective of a non-empty state, and only of one"):
        TinkerCheckpoint(MODEL, 32, "tinker://s", "tinker://p", algorithm_state={"steps": 1})
    with pytest.raises(ValueError, match="objective of a non-empty state, and only of one"):
        TinkerCheckpoint(MODEL, 32, "tinker://s", "tinker://p", objective=objective.name)


# -- Preparation and losses


def test_schedule_keeps_comparison_sets_and_handles_epochs(objective):
    objective.supports_multiple_epochs = True
    objective.rollout_unit = "comparison_set"
    scheduling = StepScheduling(epochs=2, remainder="partial")
    data = JobData("schedule", tuple(item("v", group) for group in ("a", "a", "b", "c")))
    step = prepare_tinker_job(data, TrainingMethod(objective.name), {}, scheduling, batch_size=2, kl_coef=0.0)
    assert [len(rows) for rows in step.steps] == [3, 1, 3, 1]
    assert step.metrics["optimizer_steps"] == 4
    assert step.next_algorithm_state == {"steps": 1}


def test_a_packed_tinker_step_trains_batch_size_packs(objective):
    # Each sample has 4 tokens, so packs of 8 tokens hold two samples, and one step trains one pack.
    scheduling = StepScheduling(packing_length=8)
    step = prepare_tinker_job(batch(3), TrainingMethod(objective.name), {}, scheduling, batch_size=1, kl_coef=0.0)
    assert [len(rows) for rows in step.steps] == [2, 1]


def test_step_size_respects_error_remainder(objective):
    with pytest.raises(ValueError, match="do not form complete optimizer steps of 2"):
        prepare_tinker_job(
            JobData("batch", (item("v"),)),
            TrainingMethod(objective.name),
            {},
            StepScheduling(remainder="error"),
            batch_size=2,
            kl_coef=0.0,
        )


def test_loss_alignment_mask_and_ttdd_centered_kl():
    rows = [TokenRow((10, 11, 20, 21), (1, 0), (-0.2, -0.4), 2.0), TokenRow((10, 30), (1,), (-0.6,), -1.0)]
    inputs = ImportanceSamplingLoss().inputs(rows, [], kl_coef=0)
    assert inputs[0] == {"target_tokens": [11, 20, 21], "logprobs": [0, -0.2, -0.4], "advantages": [0, 2, 0]}
    assert inputs[1]["advantages"] == [-1]
    with pytest.raises(ValueError, match="every response token"):
        rows[0].inputs([2.0])
    # Differences are .3 and .1 on selected tokens, mean .2.
    inputs = TttdTinkerLoss().inputs(rows, [[-0.5, -0.8], [-0.7]], kl_coef=0.1)
    assert inputs[0]["advantages"] == pytest.approx([0, 1.99, 2.0])
    assert inputs[1]["advantages"] == pytest.approx([-0.99])
    assert resolve_tinker_loss("tttd").needs_base_logprobs
    assert isinstance(resolve_tinker_loss("tttd"), TttdTinkerLoss)


@pytest.mark.parametrize(
    "field,value",
    [("tokens", [1, 2]), ("loss_mask", [0, 0]), ("rollout_log_probs", [float("nan"), 0]), ("tokens", [True, 1, 2, 3])],
)
def test_malformed_training_rows_are_rejected(field, value):
    with pytest.raises(ValueError):
        TokenRow.from_item(item("v").with_training(**{field: value}), 1.0)


# -- The sampler runtime


def test_exact_chat_tokens_and_buffered_sse(inference, client, tmp_path):
    artifact = empty_artifact(tmp_path)
    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    handler = inference.inference_handler
    response = asyncio.run(handler.inference(artifact, "/v1/chat/completions", request))
    assert response["training"]["tokens"] == [101, 102, 201, 202]
    assert response["training"]["rollout_log_probs"] == [-0.2, -0.3]
    assert response["training"]["runtime_load_id"] == inference.current_runtime_load_id()
    assert response["usage"]["total_tokens"] == 4
    assert client.rendered[1] == {"enable_thinking": False}

    async def read_stream():
        stream = await handler.inference_stream(artifact, "/v1/chat/completions", {**request, "stream": True})
        body = b"".join([chunk async for chunk in stream.chunks])
        assert stream.record_response["training"]["tokens"] == [101, 102, 201, 202]
        assert b"training" not in body
        assert b"data: [DONE]" in body
        await stream.close()

    asyncio.run(read_stream())


@pytest.mark.parametrize(
    "extra",
    [
        {"tools": []},
        {"n": 2},
        {"temperature": float("nan")},
        {"top_p": 0},
        {"top_k": 0},
        {"max_tokens": 0},
        {"stream_options": []},
        {"messages": [{"role": "tool", "content": "x"}]},
    ],
)
def test_unsupported_chat_options_fail_explicitly(inference, client, tmp_path, extra):
    with pytest.raises(UpstreamStatusError):
        asyncio.run(
            inference.inference_handler.inference(
                empty_artifact(tmp_path),
                "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hi"}], **extra},
            )
        )
    assert not client.sampled


def test_activation_serves_the_new_sampler_and_freezes_older_releases(component, inference, objective, tmp_path):
    base = empty_artifact(tmp_path)
    first = inference.activate_checkpoint(base)
    (checkpoint,) = train(component, batch(), objective.name).checkpoints
    published = Artifact.local(Path(checkpoint.release_path))
    selected = inference.activate_checkpoint(published)
    assert inference.current_runtime_load_id() == selected != first
    # An in-flight request on the older release keeps its sampler and load.
    assert inference.snapshot(base) == (None, first)
    assert inference.snapshot(published) == ("tinker://update-1/sampler", selected)
    rollback = Artifact.local(base.local_path)
    latest = inference.activate_checkpoint(rollback)
    assert int(first.split(":")[-1]) < int(selected.split(":")[-1]) < int(latest.split(":")[-1])
    assert inference.snapshot(base)[1] == first
    assert inference.snapshot(published)[1] == selected
    assert inference.snapshot(rollback)[1] == latest
    assert inference.activate_checkpoint(rollback) == latest
    # A release another component minted with the same weights is never activated; it serves the active load.
    carried = Artifact.local(base.local_path)
    assert inference.snapshot(carried)[1] == latest == inference.current_runtime_load_id()


# -- Deployment


def test_deployment_preserves_remote_model_and_discovers_without_sdk(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    raw = {
        "recipe": {"implementation": "recipes.tttd.recipe:TTTDRecipe"},
        "inference": {"model-path": MODEL},
        "training": {"backend": "tinker", "options": {"state-dir": str(tmp_path / "state")}},
    }
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    assert not local_model_required(config)
    assert config["reef"]["model_path"] == MODEL
    assert [service["name"] for service in config["services"]] == ["reef"]
    assert not TinkerDeployment.requires_local_model
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import reef.train.tinker_backend.launch; import recipes.tttd; "
                "assert 'tinker' not in sys.modules; assert 'torch' not in sys.modules; "
                "assert 'reef.train.tinker_backend.losses' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_serve_skips_the_model_download_for_a_hosted_backend(tmp_path, monkeypatch):
    path = tmp_path / "serve.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "reef": {"run-dir": str(tmp_path / "stack")},
                "recipe": {"implementation": "recipes.tttd.recipe:TTTDRecipe"},
                "inference": {"model-path": MODEL},
                "training": {"backend": "tinker", "options": {"state-dir": str(tmp_path / "state")}},
            }
        )
    )
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    started = []

    class Stack:
        def __init__(self, config, *args, **kwargs):
            started.append(config)
            self._stopping = threading.Event()
            self.exit_code = 0

        def start(self):
            return None

        def block(self):
            return None

        def shutdown(self):
            return None

    monkeypatch.setattr(orchestrator, "resolve_model_paths", lambda config: pytest.fail("downloaded remote model"))
    monkeypatch.setattr(orchestrator, "_Stack", Stack)
    with pytest.raises(SystemExit) as result:
        main(["serve", "-c", str(path)])
    assert result.value.code == 0
    assert started[0]["reef"]["model_path"] == MODEL
    assert local_model_required({"reef": {"recipe": "recipes.tttd.recipe:TTTDRecipe", "training_backend": "slime"}})
    assert local_model_required({"reef": {}})


@pytest.mark.parametrize(
    "options",
    [
        {"max_staleness": 1},
        {"lora_rank": 0},
        {"learning_rate": 0},
        {"kl_coef": -1},
        {"base_url": "spindle.example"},
        {"unknown": True},
    ],
)
def test_bad_runtime_configuration_fails_without_api_key(tmp_path, options):
    with pytest.raises(RuntimeConfigError):
        component_factory.parse_config({"state_dir": str(tmp_path), **options}, {})


# -- Through the dispatcher and the HTTP service


def test_tinker_serves_its_releases_with_the_training_options_as_cli_text_gives_them():
    options = {
        "state-dir": "state",
        "batch-size": "4",
        "inference-timeout-s": "30",
        "project-id": "p",
        "base-url": "https://spindle.example",
    }
    assert TinkerDeployment.default_serving == "trainer"
    assert TinkerDeployment().trainer_serving_config({"training_backend_options": options}) == {
        "type": "tinker",
        "api_key_env": "TINKER_API_KEY",
        "project_id": "p",
        "base_url": "https://spindle.example",
        "timeout_s": 30.0,
    }


def test_scenario_commits_recovers_and_rolls_back_remote_checkpoint(tmp_path, objective):
    from dataclasses import dataclass

    from tests.support._threshold_processor import ThresholdJobData, ThresholdSamples
    from tests.support.runtime_stubs import job_sized_stages

    from reef.artifact import InMemoryRepositoryBackend
    from reef.core import AgentRecord, RequestType
    from reef.dispatcher import Dispatcher
    from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
    from reef.record2record import ReportRecord2Record
    from reef.storage.sqlite import SQLiteScenarioStorage
    from reef.train.pipeline import PipelineSpec

    @dataclass(frozen=True)
    class PolicyRecipe(WeightTrainingRecipe):
        @classmethod
        def training_spec(cls):
            return WeightTrainingSpec(objective.name)

        @classmethod
        def pipeline_spec(cls):
            return PipelineSpec(
                record2record=ReportRecord2Record, record2sample=ThresholdSamples, sample2job=ThresholdJobData
            )

    initial = tmp_path / "initial"
    initial.mkdir()
    factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    settings = TinkerConfig(state_dir=str(tmp_path / "runtime"))

    def build(client):
        component = TinkerTrainingComponent(MODEL, settings, client)
        inference = TinkerInferenceRuntime(client, base_model=MODEL)
        dispatcher = Dispatcher(
            PolicyRecipe(component, runtime=inference, pipeline_stages=job_sized_stages(PolicyRecipe.pipeline_spec())),
            factory,
            local_artifact_dir=tmp_path / "staged",
            scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        )
        return dispatcher, component, inference

    def loop_state(scenario):
        return JobLoopState.from_dict(scenario.trainer.state)

    first, component, inference = build(RemoteClient())
    try:
        scenario = first.get_or_create_scenario("math")
        base = scenario.current_artifact_ref().release_id
        response = asyncio.run(
            inference.inference_handler.inference(
                scenario.release_queries.artifact_for_version(base),
                "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hello"}]},
            )
        )
        first.accept_record(
            AgentRecord.create(
                scenario="math",
                request_type=RequestType.INFERENCE,
                payload={"response": response},
                agent_record_id="i1",
            )
        )
        first.accept_record(
            AgentRecord.create(
                scenario="math",
                request_type=RequestType.REPORT,
                payload={"score": 1.0, "references": ["i1"]},
                references=("i1",),
                agent_record_id="r1",
            )
        )
        deadline = time.monotonic() + 10
        # Submission, the checkpoint's publication and the job's end each commit a step.
        while (loop_state(scenario).finished_jobs < 1) and time.monotonic() < deadline:
            time.sleep(0.01)
        state = loop_state(scenario)
        assert state.finished_jobs == 1 and state.job is None, first.build_training_status()
        published = scenario.current_artifact_ref()
        assert published.release_id != base
        artifact = scenario.release_queries.artifact_for_version(published.release_id)
        assert inference.snapshot(artifact)[0] == "tinker://update-1/sampler"
        assert inference.current_runtime_load_id() == inference.snapshot(artifact)[1]
        assert TinkerCheckpoint.read(Path(state.start)).state_path == "tinker://update-1/state"
        step = scenario.scenario_step
    finally:
        first.close()
        component.close()
        inference.shutdown()

    second, component, inference = build(RemoteClient())
    try:
        scenario = second.get_or_create_scenario("math")
        assert scenario.scenario_step == step
        assert loop_state(scenario) == state
        artifact = scenario.release_queries.artifact_for_version(published.release_id)
        assert inference.snapshot(artifact)[1] == inference.current_runtime_load_id()
        scenario.rollback(base)
        assert scenario.scenario_step == step + 1
        current = scenario.release_queries.artifact_for_version(scenario.current_artifact_ref().release_id)
        assert inference.snapshot(current)[0] is None
        # The base tree imports as the base model: the next job starts from the base checkpoint.
        assert loop_state(scenario).start is None
    finally:
        second.close()
        component.close()
        inference.shutdown()


def test_documented_smoke_runs_through_http_and_ttdd_recipe(tmp_path, monkeypatch):
    import runpy

    from aiohttp.test_utils import TestClient, TestServer

    from recipes.tttd.recipe import TTTDRecipe
    from reef.artifact import InMemoryRepositoryBackend
    from reef.dispatcher import Dispatcher
    from reef.service.app import create_app
    from reef.storage.sqlite import SQLiteScenarioStorage
    from reef.train.pipeline import PipelineConfig

    initial = tmp_path / "initial"
    initial.mkdir()
    remote = RemoteClient()
    # A job and a batch size of the whole grid train it as one optimizer step.
    component = TinkerTrainingComponent(MODEL, TinkerConfig(state_dir=str(tmp_path / "runtime"), batch_size=4), remote)
    inference = TinkerInferenceRuntime(remote, base_model=MODEL)
    step_grid = PipelineConfig.from_setting(
        {
            "record2record": {"config": {"groups_per_step": 1, "rollouts_per_group": 4}},
            "sample2job": {"config": {"job_samples": 4}},
        },
        spec=TTTDRecipe.pipeline_spec(),
        environ={},
    )
    dispatcher = Dispatcher(
        TTTDRecipe(component, runtime=inference, pipeline_stages=step_grid),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(),
    )
    script = Path(__file__).resolve().parents[2] / "tutorials" / "tinker" / "smoke.py"
    monkeypatch.setenv("REEF_TOKEN", "test-token")

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens="test-token"))) as client:
            monkeypatch.setattr(sys, "argv", [str(script), "--url", str(client.make_url("")), "--timeout", "10"])
            await asyncio.to_thread(runpy.run_path, str(script), run_name="__main__")
        deadline = time.monotonic() + 10
        while not remote.calls and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert len(remote.calls) == 1
        assert len(remote.calls[0][1][0]) == 4
        assert max(abs(row.advantage) for row in remote.calls[0][1][0]) < 100
        assert isinstance(remote.calls[0][2], TttdTinkerLoss)

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
        component.close()
        inference.shutdown()


def sft_item(name: str) -> TrajectorySample:
    """A supervised sample: three prompt tokens, two target tokens and no rollout log probabilities."""
    return TrajectorySample(
        {
            "schema_version": "ATIF-v1.6",
            "agent": {"name": name},
            "steps": [{"step_id": 1}],
            "extra": {"reef": {"training": {"tokens": [10, 11, 12, 30, 31], "loss_mask": [1, 1]}}},
        }
    )


def test_sft_trains_tinker_cross_entropy_on_the_target_tokens_without_rollout_log_probs():
    data = JobData("sft", (sft_item("a"), sft_item("b")))
    step = prepare_tinker_job(data, TrainingMethod(SftObjective.name), {}, StepScheduling(), batch_size=2, kl_coef=0.0)

    loss = resolve_tinker_loss("sft")
    (rows,) = step.steps
    assert loss.loss_fn == "cross_entropy"
    # Position i predicts token i+1: the two prompt positions before the target weigh 0.
    assert loss.inputs(rows, [], kl_coef=0.0)[0] == {
        "target_tokens": [11, 12, 30, 31],
        "weights": [0.0, 0.0, 1.0, 1.0],
    }
    with pytest.raises(ValueError, match="sft trains no KL penalty"):
        prepare_tinker_job(data, TrainingMethod(SftObjective.name), {}, StepScheduling(), batch_size=2, kl_coef=0.1)


def test_an_sft_job_trains_cross_entropy_on_the_component_and_saves_its_checkpoint(component, client):
    status = train(component, JobData("sft", (sft_item("a"), sft_item("b"))), SftObjective.name)
    assert status.state == "finished" and len(status.checkpoints) == 1, status.error
    ((_, batches, loss),) = client.calls
    assert loss.loss_fn == "cross_entropy" and sum(len(rows) for rows in batches) == 2
