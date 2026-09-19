"""CPU contract tests for Tinker's snapshot and publication semantics."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import yaml

from recipes.tttd.tinker import TttdTinkerLoss
from reef.artifact.artifact import Artifact, LiveWeightArtifactRef
from reef.cli import main
from reef.core.batches import TrainingBatch, TrajectoryItem
from reef.core.evaluation import EvaluationResult, SelectionDecision
from reef.inference.tinker import SampleResult, TinkerInferenceRuntime, TinkerSampler
from reef.runtime.deployment import RuntimeConfigError
from reef.runtime.interfaces import TrainingRuntimeError, UpstreamStatusError
from reef.service.deploy import orchestrator
from reef.service.deploy.orchestrator import resolve_deployment_config
from reef.service.deploy.training import local_model_required
from reef.surface.weights import WeightLoader
from reef.train.algos import StepScheduling, StepSignal
from reef.train.algos.objective import TrainingObjective
from reef.train.algos.registry import register_objective, unregister_objective
from reef.train.runtime_backend import RuntimeCandidateBackend
from reef.train.tinker_backend.checkpoint import TinkerCheckpoint
from reef.train.tinker_backend.client import TinkerClient
from reef.train.tinker_backend.config import TinkerConfig
from reef.train.tinker_backend.launch import TinkerDeployment, runtime_factory
from reef.train.tinker_backend.losses import ImportanceSamplingLoss, TokenRow, resolve_tinker_loss
from reef.train.tinker_backend.preparation import prepare_tinker_step
from reef.train.tinker_backend.runtime import TinkerTrainingRuntime


class RemoteClient(TinkerClient, TinkerSampler):
    """One offline double for both SDK boundaries: the trainer's and the sampler's."""

    def __init__(self):
        self.initializations = 0
        self.calls = []
        self.sampled = []
        self.downloads = []
        self.closed = False
        self.fail = False

    def initialize(self):
        self.initializations += 1
        return TinkerCheckpoint("Qwen/Qwen3-8B", 32, "tinker://base/state", "tinker://base/sampler")

    def train(self, checkpoint, batches, loss):
        self.calls.append((checkpoint, batches, loss))
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

    def download(self, checkpoint, directory):
        self.downloads.append((checkpoint, directory))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 32}))
        (directory / "adapter_model.safetensors").write_bytes(checkpoint.sampler_path.encode())

    def close(self):
        self.closed = True


class SampleObjective(TrainingObjective):
    loss_family = "importance_sampling"
    name = "tinker-test-objective"

    def __init__(self):
        self.loss_family = "importance_sampling"
        self.action = "train"

    def prepare(self, batch, state):
        return StepSignal(
            self.action,
            {"steps": state.get("steps", 0) + 1},
            advantages=tuple(float(i + 1) for i in range(len(batch.items))),
        )


# The schedule a recipe over this objective binds: one step over every sample.
SCHEDULING = StepScheduling(unit="sample", batch_size="actual")


@pytest.fixture
def objective():
    value = SampleObjective()
    register_objective(value)
    yield value
    unregister_objective(value.name)


class Deployment:
    """The training and inference runtimes as the factory builds them: two SDK boundaries, no shared object."""

    def __init__(self, config, client):
        self.config = config
        self.training = TinkerTrainingRuntime("Qwen/Qwen3-8B", config, client)
        self.inference = TinkerInferenceRuntime(client, base_model="Qwen/Qwen3-8B")

    def backend(self, objective):
        return RuntimeCandidateBackend(self.training, objective.name, SCHEDULING, inference_runtime=self.inference)

    def shutdown(self):
        self.training.shutdown()
        self.inference.shutdown()


@pytest.fixture
def runtime(tmp_path):
    client = RemoteClient()
    value = Deployment(TinkerConfig(state_dir=str(tmp_path / "state")), client)
    yield value, client
    value.shutdown()


def item(version, group="g"):
    return TrajectoryItem(
        {
            "schema_version": "ATIF-v1.6",
            "agent": {"name": "test"},
            "steps": [{"step_id": 1}],
            "extra": {
                "reef": {
                    "training": {
                        "tokens": [10, 11, 20, 21],
                        "loss_mask": [1, 0],
                        "rollout_log_probs": [-0.2, -0.4],
                        "runtime_load_id": version,
                    }
                }
            },
        },
        group_id=group,
    )


def prepared(runtime, objective, *, step=0):
    batch = TrainingBatch("batch", (item(runtime.inference.serving_runtime_load_id()),))
    return runtime.training.prepare_training_step(batch, objective.name, {}, SCHEDULING, step)


def decision(selected):
    return SelectionDecision(
        "select" if selected else "reject", "test", "1", "test decision", EvaluationResult("test", "1", {})
    )


def empty_artifact(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir(exist_ok=True)
    return Artifact.local(directory)


def test_selection_waits_for_matching_commit_and_freezes_old_sampler(runtime, objective, tmp_path):
    value, client = runtime
    backend = value.backend(objective)
    base = empty_artifact(tmp_path)
    original = value.inference.serving_runtime_load_id()
    batch = TrainingBatch("batch", (item(original),))
    step = backend.prepare_step(batch, {}, 0)
    assert value.inference.current_runtime_load_id() == original
    assert value.inference.snapshot(base)[1] == original
    result = backend.settle_step(step, decision(True))
    assert value.inference.current_runtime_load_id() == original
    assert value.inference.serving_runtime_load_id() == result.runtime_load_id != original
    assert not value.inference.inference_admission_status["open"]
    assert value.inference.snapshot(base)[1] == original
    published = Artifact.local(Path(result.checkpoint_path))
    WeightLoader().activate(published, value.inference)
    assert not value.inference.inference_admission_status["open"]
    with pytest.raises(TrainingRuntimeError, match="does not match"):
        backend.acknowledge_commit(1, "another")
    backend.acknowledge_commit(1, result.training_job_id)
    assert value.inference.current_runtime_load_id() == result.runtime_load_id
    assert value.inference.inference_admission_status["open"]
    assert value.training.incumbent.state_path == "tinker://update-1/state"
    assert len(client.calls) == 1


def test_rejected_and_uncertain_candidates_leave_incumbent_optimizer_unchanged(runtime, objective):
    value, client = runtime
    payload = prepared(value, objective).payload
    original = value.inference.current_runtime_load_id()
    candidate = value.training.train_candidate(payload)
    assert value.training.train_candidate(payload) == candidate
    assert len(client.calls) == 1
    value.training.reject_candidate(candidate, decision(False))
    client.fail = True
    with pytest.raises(TimeoutError):
        value.training.train_candidate(payload)
    client.fail = False
    value.training.train_candidate(payload)
    assert len(client.calls) == 3
    assert all(call[0].state_path == "tinker://base/state" for call in client.calls)
    assert value.inference.current_runtime_load_id() == original


def test_reload_before_and_after_commit_restores_authoritative_artifact(runtime, objective, tmp_path):
    value, client = runtime
    candidate = value.training.train_candidate(prepared(value, objective).payload)
    value.inference.activate_candidate(candidate)
    assert value.inference.pending_training_job_id == candidate.training_job_id
    # Reloading a different head after a failed publication drops the pending candidate.
    value.inference.activate_checkpoint(empty_artifact(tmp_path))
    assert value.inference.pending_training_job_id is None
    value.shutdown()
    restored = Deployment(value.config, client)
    try:
        restored.inference.activate_checkpoint(Artifact.local(Path(candidate.checkpoint_path)))
        restored.backend(objective).acknowledge_commit(1, candidate.training_job_id)
        assert restored.inference.current_runtime_load_id() != value.inference.serving_runtime_load_id()
        restored.training.train_candidate(prepared(restored, objective, step=1).payload)
        assert client.calls[-1][0].state_path == "tinker://update-1/state"
        assert client.initializations == 1
    finally:
        restored.shutdown()


def test_live_versions_and_rollback_bind_exact_snapshots(runtime, objective, tmp_path):
    value, _ = runtime
    candidate = value.training.train_candidate(prepared(value, objective).payload)
    activated = value.inference.activate_candidate(candidate)
    value.backend(objective).acknowledge_commit(1, candidate.training_job_id)
    live = Artifact(LiveWeightArtifactRef("live", "release", "parent", activated.runtime_load_id), None)
    assert value.inference.snapshot(live)[0] == "tinker://update-1/sampler"
    assert value.training.incumbent.state_path == "tinker://update-1/state"
    base = empty_artifact(tmp_path)
    value.training.restore_checkpoint(base)
    WeightLoader().load(base, value.inference)
    WeightLoader().activate(base, value.inference, source=base)
    assert value.training.incumbent.state_path == "tinker://base/state"
    assert value.inference.snapshot(live)[1] == activated.runtime_load_id
    assert value.inference.current_runtime_load_id() != activated.runtime_load_id
    with pytest.raises(ValueError, match="incarnation"):
        value.inference.snapshot(Artifact(LiveWeightArtifactRef("live", "x", None, "old:0"), None))


def test_exclusive_state_dir_and_model_validation(runtime, tmp_path):
    value, _ = runtime
    with pytest.raises(ValueError, match="already owned"):
        TinkerTrainingRuntime("Qwen/Qwen3-8B", value.config, RemoteClient())
    other = TinkerCheckpoint("other/model", 32, "tinker://other/state", "tinker://other/sampler")
    other.write(tmp_path / "other")
    with pytest.raises(ValueError, match="base model"):
        value.inference.activate_checkpoint(Artifact.local(tmp_path / "other"))
    with pytest.raises(ValueError, match="model/rank"):
        value.training.restore_checkpoint(Artifact.local(tmp_path / "other"))
    (tmp_path / "other" / "tinker-checkpoint.json").unlink()
    (tmp_path / "other" / "weights.bin").write_bytes(b"not a manifest")
    with pytest.raises(ValueError, match="missing"):
        value.inference.snapshot(Artifact.local(tmp_path / "other"))
    with pytest.raises(ValueError, match="missing"):
        value.training.restore_checkpoint(Artifact.local(tmp_path / "other"))


def test_exact_chat_tokens_and_buffered_sse(runtime, tmp_path):
    value, client = runtime
    artifact = empty_artifact(tmp_path)
    request = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    handler = value.inference.inference_handler
    response = asyncio.run(handler.inference(artifact, "/v1/chat/completions", request))
    assert response["training"]["tokens"] == [101, 102, 201, 202]
    assert response["training"]["rollout_log_probs"] == [-0.2, -0.3]
    assert response["training"]["runtime_load_id"] == value.inference.current_runtime_load_id()
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
def test_unsupported_chat_options_fail_explicitly(runtime, tmp_path, extra):
    value, client = runtime
    with pytest.raises(UpstreamStatusError):
        asyncio.run(
            value.inference.inference_handler.inference(
                empty_artifact(tmp_path),
                "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hi"}], **extra},
            )
        )
    assert not client.sampled


def test_schedule_keeps_comparison_sets_and_handles_epochs(objective):
    objective.supports_multiple_epochs = True
    scheduling = StepScheduling(unit="comparison_set", batch_size=2, epochs=2, remainder="partial")
    batch = TrainingBatch("schedule", tuple(item("v", group) for group in ("a", "a", "b", "c")))
    step = prepare_tinker_step(batch, objective.name, {}, scheduling, runtime_load_id="v", batch_size=1)
    assert [len(rows) for rows in step.payload["batches"]] == [3, 1, 3, 1]
    assert step.metrics["optimizer_steps"] == 4
    assert step.next_algorithm_state == {"steps": 1}


def test_stale_policy_is_dropped_through_generic_backend(runtime, objective):
    value, client = runtime
    backend = value.backend(objective)
    result = backend.prepare_step(TrainingBatch("stale", (item("old:0"),)), {}, 0)
    assert result.outcome == "drop"
    assert not client.calls


def test_skip_and_unsupported_loss_do_not_mutate_remote(runtime, objective):
    value, client = runtime
    objective.action = "skip"
    assert prepared(value, objective).action == "skip"
    objective.action = "train"
    objective.loss_family = "sao"
    with pytest.raises(ValueError, match="unsupported Tinker loss"):
        prepared(value, objective)
    assert not client.calls


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


def test_deployment_preserves_remote_model_and_discovers_without_sdk(tmp_path):
    raw = {
        "schema-version": 2,
        "recipe": {"implementation": "recipes.tttd.recipe:TTTDRecipe"},
        "inference": {"model-path": "Qwen/Qwen3-8B"},
        "training": {"backend": "tinker", "options": {"state-dir": str(tmp_path / "state")}},
    }
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    assert not local_model_required(config)
    assert config["reef"]["model_path"] == "Qwen/Qwen3-8B"
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
                "schema-version": 2,
                "reef": {"run-dir": str(tmp_path / "stack")},
                "recipe": {"implementation": "recipes.tttd.recipe:TTTDRecipe"},
                "inference": {"model-path": "Qwen/Qwen3-8B"},
                "training": {"backend": "tinker", "options": {"state-dir": str(tmp_path / "state")}},
            }
        )
    )
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
    assert started[0]["reef"]["model_path"] == "Qwen/Qwen3-8B"
    assert local_model_required({"reef": {"training_backend": "slime"}})
    assert local_model_required({"reef": {}})


@pytest.mark.parametrize(
    "options", [{"max_staleness": 1}, {"lora_rank": 0}, {"learning_rate": 0}, {"kl_coef": -1}, {"unknown": True}]
)
def test_bad_runtime_configuration_fails_without_api_key(tmp_path, options):
    with pytest.raises(RuntimeConfigError):
        runtime_factory.parse_config({"state_dir": str(tmp_path), **options}, {})


def test_scenario_commits_recovers_and_rolls_back_remote_checkpoint(tmp_path, objective):
    import time
    from dataclasses import dataclass

    from reef_service._threshold_processor import ThresholdProcessor

    from reef.artifact import InMemoryRepositoryBackend
    from reef.core import AgentRecord, RequestType
    from reef.dispatcher import Dispatcher
    from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
    from reef.recipe.config_fields import config_field
    from reef.storage.sqlite import SQLiteScenarioStorage

    @dataclass(frozen=True)
    class PolicyRecipe(WeightTrainingRecipe):
        batch_size: int = config_field(1)

        @classmethod
        def training_spec(cls):
            return WeightTrainingSpec(objective.name, ThresholdProcessor)

    initial = tmp_path / "initial"
    initial.mkdir()
    factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    settings = TinkerConfig(state_dir=str(tmp_path / "runtime"))
    client = RemoteClient()
    runtime = Deployment(settings, client)

    def build(value):
        return Dispatcher(
            PolicyRecipe(value.training, runtime=value.inference),
            factory,
            local_artifact_dir=tmp_path / "staged",
            agent_record_dir=tmp_path / "records",
            scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        )

    first = build(runtime)
    try:
        scenario = first.get_or_create_scenario("math")
        base = scenario.current_artifact_ref().release_id
        response = asyncio.run(
            runtime.inference.inference_handler.inference(
                scenario.artifact_for_version(base),
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
        deadline = time.monotonic() + 5
        # The commit advances the step first; the trainer thread reopens admission right after.
        while (
            scenario.scenario_step < 1 or not runtime.inference.inference_admission_status["open"]
        ) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert scenario.scenario_step == 1, first.build_training_status()
        published = scenario.current_artifact_ref()
        assert scenario.committed_training_job_id
        assert runtime.inference.inference_admission_status["open"]
        assert (
            runtime.inference.snapshot(scenario.artifact_for_version(published.release_id))[0]
            == "tinker://update-1/sampler"
        )
        assert runtime.training.incumbent.state_path == "tinker://update-1/state"
    finally:
        first.close()

    restored = Deployment(settings, RemoteClient())
    second = build(restored)
    try:
        scenario = second.get_or_create_scenario("math")
        assert scenario.scenario_step == 1
        assert scenario.trainer.state == {"steps": 1}
        assert (
            restored.inference.snapshot(scenario.artifact_for_version(published.release_id))[1]
            == restored.inference.current_runtime_load_id()
        )
        assert restored.training.incumbent.state_path == "tinker://update-1/state"
        scenario.rollback(base)
        assert scenario.scenario_step == 2
        current = scenario.artifact_for_version(scenario.current_artifact_ref().release_id)
        assert restored.inference.snapshot(current)[0] is None
        assert restored.training.incumbent.state_path == "tinker://base/state"
    finally:
        second.close()


def test_documented_smoke_runs_through_http_and_ttdd_recipe(tmp_path, monkeypatch):
    import runpy

    from aiohttp.test_utils import TestClient, TestServer

    from recipes.tttd.recipe import TTTDRecipe
    from reef.artifact import InMemoryRepositoryBackend
    from reef.dispatcher import Dispatcher
    from reef.service.app import create_app
    from reef.storage.sqlite import SQLiteScenarioStorage

    initial = tmp_path / "initial"
    initial.mkdir()
    remote = RemoteClient()
    runtime = Deployment(TinkerConfig(state_dir=str(tmp_path / "runtime")), remote)
    dispatcher = Dispatcher(
        TTTDRecipe(runtime.training, runtime=runtime.inference, groups_per_step=1, rollouts_per_group=4),
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
        assert len(remote.calls) == 1
        assert len(remote.calls[0][1][0]) == 4
        assert max(abs(row.advantage) for row in remote.calls[0][1][0]) < 100
        assert isinstance(remote.calls[0][2], TttdTinkerLoss)

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_rollback_mints_a_new_load_without_changing_an_older_release(runtime, objective, tmp_path):
    value, _ = runtime
    inference = value.inference
    base = empty_artifact(tmp_path)
    first = inference.activate_checkpoint(base)
    candidate = value.training.train_candidate(prepared(value, objective).payload)
    selected = inference.activate_candidate(candidate)
    published = Artifact.local(Path(candidate.checkpoint_path))
    inference.activate_checkpoint(published)
    inference.acknowledge_publication(candidate.training_job_id)
    rollback = Artifact.local(base.local_path)
    latest = inference.activate_checkpoint(rollback)
    assert int(first.split(":")[-1]) < int(selected.runtime_load_id.split(":")[-1]) < int(latest.split(":")[-1])
    assert inference.snapshot(base)[1] == first
    assert inference.snapshot(published)[1] == selected.runtime_load_id
    assert inference.snapshot(rollback)[1] == latest
    assert inference.activate_checkpoint(rollback) == latest


def test_configured_batch_size_respects_error_remainder(objective):
    scheduling = StepScheduling(batch_size="configured", remainder="error")
    with pytest.raises(ValueError, match="configured batch_size"):
        prepare_tinker_step(
            TrainingBatch("batch", (item("v"),)), objective.name, {}, scheduling, runtime_load_id="v", batch_size=2
        )
