from __future__ import annotations

import asyncio
import time
from pathlib import Path
from threading import Event

import pytest

from reef.artifact import ArtifactPublicationError, InMemoryRepositoryBackend
from reef.core import AgentRecord, ReefError, RequestType
from reef.dispatcher import Dispatcher
from reef.recipe import RecipeRegistry
from reef.runtime import ActivatedModel, ModelCandidate, PreparedTrainingStep, TrainingJobResult, TrainingRuntime
from reef.runtime.candidates import CandidateTrainingDeferred, StaleCandidate
from reef.runtime.inference import InferenceBackend
from reef.service.app import RequestService

from ._policy_recipe import TestPolicyRecipe

_ASYNC_WAIT_TIMEOUT_S = 5.0
_ASYNC_WAIT_POLL_S = 0.01


class DurableRuntime(TrainingRuntime):
    def __init__(
        self,
        checkpoint_root: Path,
        *,
        fail_once: bool = False,
        serving_version: str = "v0",
        block: bool = False,
        block_storage: bool = False,
        stale_metrics: dict | None = None,
        complete_metrics: dict | None = None,
    ) -> None:
        super().__init__(base_url="http://inference")
        self.checkpoint_root = checkpoint_root
        self.fail_once = fail_once
        self.serving_version = serving_version
        self.block = block
        self.block_storage = block_storage
        self.stale_metrics = stale_metrics
        self.complete_metrics = complete_metrics
        self.started = Event()
        self.release = Event()
        self.calls: list[dict] = []
        self.completed: dict[int, ModelCandidate] = {}
        self.candidate_versions: dict[str, str] = {}

    @property
    def inference_backend(self):
        return None

    def serving_weight_version(self):
        return self.serving_version

    def prepare_training_step(self, batch, step_preparer, algorithm_state, scenario_step):
        sample = batch.samples[0]
        payload = {
            "rollout_id": scenario_step,
            "loss": step_preparer,
            "source": sample.source_agent_record_id,
            "expected_weight_version": sample.weight_version,
        }
        return PreparedTrainingStep(
            action="train",
            payload=payload,
            next_algorithm_state={"steps": int(algorithm_state.get("steps", 0)) + 1},
            metrics={},
        )

    def train_candidate(self, payload):
        rollout_id = payload["rollout_id"]
        existing = self.completed.get(rollout_id)
        if existing is not None:
            return existing
        if payload["expected_weight_version"] != self.serving_version:
            raise StaleCandidate(self.stale_metrics)
        if self.block_storage:
            self.started.set()
            raise CandidateTrainingDeferred({"blocked": True, "reasons": ["test cap"], "delete": []})
        self.calls.append(dict(payload))
        self.started.set()
        if self.block and not self.release.wait(5):
            raise TimeoutError("test did not release blocked training")
        job_id = f"job-{rollout_id}"
        checkpoint = self.checkpoint_root / job_id
        checkpoint.mkdir(parents=True)
        candidate = ModelCandidate(
            candidate_id=job_id,
            training_job_id=job_id,
            checkpoint_path=str(checkpoint),
            current_version=self.serving_version,
            current_weight_version=self.serving_version,
            training_metrics=self.complete_metrics or {},
        )
        self.completed[rollout_id] = candidate
        self.candidate_versions[job_id] = f"job:{job_id}"
        if self.fail_once:
            self.fail_once = False
            raise ConnectionError("lost remote acknowledgement")
        return candidate

    def activate_candidate(self, candidate):
        weight_version = self.candidate_versions[candidate.candidate_id]
        self.serving_version = weight_version
        return ActivatedModel(candidate.candidate_id, weight_version)

    def reject_candidate(self, candidate, decision):
        del candidate, decision


class BlockingLostAckBackend(InMemoryRepositoryBackend):
    started = Event()
    release = Event()
    failed = False

    def publish(self, artifact, *, expected_parent):
        type(self).started.set()
        if not type(self).release.wait(5):
            raise TimeoutError("test did not release blocked publication")
        ref = super().publish(artifact, expected_parent=expected_parent)
        if not type(self).failed:
            type(self).failed = True
            raise ArtifactPublicationError("lost publication acknowledgement")
        return ref


def _training(weight_version: str) -> dict:
    return {"tokens": [1, 2], "loss_mask": [1], "rollout_log_probs": [-0.2], "weight_version": weight_version}


class ImmediateBackend(InferenceBackend):
    async def inference(self, artifact, path, payload):
        assert payload["return_meta_info"] is True
        return {"metadata": {"weight_version": "v0"}}


class RecordingExperimentTracker:
    def __init__(self) -> None:
        self.contexts = []
        self.events = []
        self.loggers = {}
        self.closed = False

    def bind_scenario(self, *, scenario, recipe, source_artifact_ref, run_segment):
        del recipe, source_artifact_ref, run_segment
        return self.loggers.setdefault(scenario, RecordingExperimentLogger())

    def correlation_metrics(self, context):
        self.contexts.append(context)
        return {
            "experiment/provider": "test",
            "experiment/run_id": f"run-{context.scenario}",
        }

    def record(self, event):
        self.events.append(event)

    def close(self):
        self.closed = True


class RecordingExperimentLogger:
    def __init__(self) -> None:
        self.logged = []

    def log(self, metrics, *, namespace):
        self.logged.append((namespace, dict(metrics)))


@pytest.fixture
def start_dispatcher(tmp_path: Path):
    opened = []

    def start(backend_type=None, *, experiment_tracker=None, **runtime_options):
        initial = tmp_path / "initial"
        initial.mkdir(exist_ok=True)
        runtime = DurableRuntime(tmp_path / "checkpoints", **runtime_options)
        factory = (backend_type or InMemoryRepositoryBackend).factory(initial, root=tmp_path / "repository")
        dispatcher = Dispatcher(
            RecipeRegistry({"test_policy": TestPolicyRecipe(runtime, batch_size=1)}),
            factory,
            local_artifact_dir=tmp_path / "staged",
            agent_record_dir=tmp_path / "agent-record",
            experiment_tracker=experiment_tracker,
        )
        opened.append((runtime, dispatcher))
        return runtime, dispatcher

    yield start
    for runtime, dispatcher in opened:
        runtime.release.set()
        dispatcher.close()


def _submit_pair(dispatcher: Dispatcher, suffix: str = "1", weight_version: str = "v0") -> None:
    dispatcher.get_or_create_scenario("math", "test_policy")
    inference_id = f"inference-{suffix}"
    records = (
        AgentRecord.create(
            scenario="math",
            request_type=RequestType.INFERENCE,
            agent_record_id=inference_id,
            payload={"response": {"training": _training(weight_version)}},
        ),
        AgentRecord.create(
            scenario="math",
            request_type=RequestType.REPORT,
            agent_record_id=f"report-{suffix}",
            payload={"score": 1.0, "references": [inference_id]},
            references=(inference_id,),
        ),
    )
    for record in records:
        dispatcher.accept_record(record, recipe="test_policy")


def _wait_for_step(dispatcher: Dispatcher, step: int) -> None:
    deadline = time.monotonic() + _ASYNC_WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        if dispatcher.get_or_create_scenario("math").scenario_step == step:
            return
        time.sleep(_ASYNC_WAIT_POLL_S)
    pytest.fail(f"scenario did not reach step {step}: {dispatcher.training_status}")


def _wait_for_error(dispatcher: Dispatcher) -> str:
    deadline = time.monotonic() + _ASYNC_WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        error = dispatcher.training_status["error"]
        if isinstance(error, str):
            return error
        time.sleep(_ASYNC_WAIT_POLL_S)
    pytest.fail(f"training error was not reported: {dispatcher.training_status}")


@pytest.mark.unit
def test_empty_checkpoint_result_fails_closed() -> None:
    # The invariant lives on the result type, so a completed job that cannot
    # name its exported checkpoint cannot be constructed at all -- it can never
    # reach the commit protocol and be published as a durable version.
    with pytest.raises(ValueError, match="must report the checkpoint path"):
        TrainingJobResult(outcome="complete", weight_version="v1", checkpoint_path="")


@pytest.mark.unit
def test_report_returns_and_inference_resolves_while_remote_job_is_blocked(start_dispatcher) -> None:
    runtime, dispatcher = start_dispatcher(fail_once=True, block=True)
    scenario = dispatcher.get_or_create_scenario("math", "test_policy")
    expected = scenario.current_artifact_ref()
    _submit_pair(dispatcher)
    assert runtime.started.wait(1)
    assert scenario.current_artifact_ref() == expected
    runtime.release.set()
    _wait_for_step(dispatcher, 1)
    assert len(runtime.calls) == 1


@pytest.mark.unit
def test_backend_failure_reaches_reef_status(start_dispatcher) -> None:
    runtime, dispatcher = start_dispatcher()

    def fail(payload) -> ModelCandidate:
        del payload
        raise RuntimeError("backend submission failed")

    runtime.train_candidate = fail
    _submit_pair(dispatcher)

    assert _wait_for_error(dispatcher) == "math: RuntimeError: backend submission failed"


@pytest.mark.unit
def test_inference_resolves_while_lost_ack_publication_blocks(start_dispatcher) -> None:
    BlockingLostAckBackend.started = Event()
    BlockingLostAckBackend.release = Event()
    BlockingLostAckBackend.failed = False
    runtime, dispatcher = start_dispatcher(BlockingLostAckBackend)
    _submit_pair(dispatcher)
    assert BlockingLostAckBackend.started.wait(1)
    response, item = asyncio.run(
        asyncio.wait_for(
            RequestService(dispatcher).infer_with_data(
                {"x-reef-scenario": "math"},
                {"messages": [{"role": "user", "content": "hi"}]},
                "/v1/chat/completions",
                ImmediateBackend(),
            ),
            1,
        )
    )
    assert response["metadata"]["weight_version"] == item.payload["weight_version"] == "v0"
    BlockingLostAckBackend.release.set()
    _wait_for_step(dispatcher, 1)
    assert len(runtime.calls) == 1


@pytest.mark.unit
def test_weight_inference_does_not_materialize_its_checkpoint(start_dispatcher, monkeypatch) -> None:
    _, dispatcher = start_dispatcher()
    scenario = dispatcher.get_or_create_scenario("math", "test_policy")
    monkeypatch.setattr(
        scenario.repository,
        "resolve",
        lambda ref: pytest.fail(f"unexpected inference materialization: {ref.version}"),
    )

    response = asyncio.run(
        RequestService(dispatcher).infer(
            {"x-reef-scenario": "math"},
            {"messages": [{"role": "user", "content": "hi"}]},
            "/v1/chat/completions",
            ImmediateBackend(),
        )
    )

    assert response["metadata"]["weight_version"] == "v0"


@pytest.mark.unit
def test_two_jobs_form_one_deterministic_version_chain(start_dispatcher) -> None:
    runtime, dispatcher = start_dispatcher()
    _submit_pair(dispatcher)
    _wait_for_step(dispatcher, 1)
    scenario = dispatcher.get_or_create_scenario("math")
    first = scenario.current_artifact_ref()
    _submit_pair(dispatcher, "2", runtime.serving_version)
    _wait_for_step(dispatcher, 2)

    assert [call["rollout_id"] for call in runtime.calls] == [0, 1]
    assert [call["source"] for call in runtime.calls] == ["inference-1", "inference-2"]
    assert scenario.current_artifact_ref().parent_version == first.version
    with pytest.raises(ReefError, match="already bound"):
        dispatcher.get_or_create_scenario("other", "test_policy")


@pytest.mark.unit
def test_training_result_metrics_reach_the_durable_commit(start_dispatcher) -> None:
    metrics = {"staleness/samples_fresh": 1, "staleness/samples_admitted_stale": 0}
    _, dispatcher = start_dispatcher(complete_metrics=metrics)

    _submit_pair(dispatcher)
    _wait_for_step(dispatcher, 1)
    scenario = dispatcher.get_or_create_scenario("math")

    committed = scenario.metrics_for_version(scenario.current_artifact_ref().version)
    assert committed is not None
    assert {key: committed[key] for key in metrics} == metrics
    assert committed["selection"]["outcome"] == "select"


@pytest.mark.unit
def test_experiment_provider_observes_the_generic_commit_boundary(start_dispatcher) -> None:
    tracker = RecordingExperimentTracker()
    _, dispatcher = start_dispatcher(
        experiment_tracker=tracker,
        complete_metrics={"train/loss": 0.25},
    )

    _submit_pair(dispatcher)
    _wait_for_step(dispatcher, 1)
    scenario = dispatcher.get_or_create_scenario("math")
    produced = scenario.current_artifact_ref()
    committed = scenario.metrics_for_version(produced.version)

    assert committed is not None
    assert committed["experiment/provider"] == "test"
    assert committed["experiment/run_id"] == "run-math"
    assert len(tracker.events) == 1
    event = tracker.events[0]
    assert event.context.scenario == "math"
    assert event.context.recipe == "test_policy"
    assert event.context.backend == "SlimeTrainingBackend"
    assert event.context.backend_config == {"runtime": "slime", "step_preparer": "sft"}
    assert event.context.source_artifact_ref.version == produced.parent_version
    assert event.produced_artifact_ref == produced
    assert event.metrics["train/loss"] == pytest.approx(0.25)
    assert event.training_job_id == "job-0"
    assert event.source_weight_version == "v0"
    assert event.produced_weight_version == "job:job-0"
    assert scenario.trainer.processor.experiment_logger is tracker.loggers["math"]
    scenario.trainer.processor.experiment_logger.log({"accepted": 1}, namespace="processor")
    assert tracker.loggers["math"].logged == [
        ("recipe", {"batch_size": 1}),
        ("processor", {"accepted": 1}),
    ]


@pytest.mark.unit
def test_stale_batch_is_discarded_and_next_valid_job_runs(start_dispatcher) -> None:
    stale_metrics = {
        "staleness/samples_dropped": 1,
        "staleness/drop_reason": "policy_lag_exceeded",
        "staleness/source_agent_record_ids": ["inference-1"],
        "staleness/producing_weight_versions": ["v0"],
        "staleness/serving_weight_version": "v2",
        "staleness/drop_policy_lags": [2],
    }
    runtime, dispatcher = start_dispatcher(serving_version="v2", stale_metrics=stale_metrics)
    _submit_pair(dispatcher)
    _submit_pair(dispatcher, "2", "v2")
    _wait_for_step(dispatcher, 1)
    scenario = dispatcher.get_or_create_scenario("math")

    assert [call["source"] for call in runtime.calls] == ["inference-2"]
    # Rejecting the first batch consumes neither side's step counter, so the
    # next valid batch reuses rollout 0 rather than wedging the bridge at 1.
    assert runtime.calls[0]["rollout_id"] == 0
    assert scenario.scenario_step == 1
    assert scenario.records.count("math") == 0
    receipts = scenario.records.compaction_receipts("math")
    assert len(receipts) == 1
    assert receipts[0]["metadata"] == {"outcome": "stale", "metrics": stale_metrics}
    assert set(receipts[0]["compacted_ids"]) == {"inference-1", "report-1"}


@pytest.mark.unit
def test_storage_block_preserves_pending_batch_and_retries(start_dispatcher, monkeypatch) -> None:
    monkeypatch.setattr("reef.dispatcher._STORAGE_RETRY_SECONDS", 0.01)
    runtime, dispatcher = start_dispatcher(block_storage=True)
    _submit_pair(dispatcher)
    assert runtime.started.wait(1)
    scenario = dispatcher.get_or_create_scenario("math")

    assert scenario.scenario_step == 0
    assert scenario.trainer.pending_batch is not None
    assert scenario.records.count("math") == 2
    assert dispatcher.training_status["scenarios"]["math"]["checkpoint_storage"]["reasons"] == ["test cap"]
    runtime.block_storage = False
    _wait_for_step(dispatcher, 1)
    assert runtime.calls[0]["rollout_id"] == 0
    assert runtime.calls[0]["source"] == "inference-1"
    assert dispatcher.training_status["scenarios"]["math"]["checkpoint_storage"] is None
