from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from pathlib import Path
from threading import Event

import pytest
from reef_service.runtime_stubs import StubTrainingRuntime, runtime_bindings

from reef.artifact import Artifact, ArtifactPublicationError, InMemoryRepositoryBackend
from reef.core import AgentRecord, ReefError, RequestType
from reef.core.trajectories import source_record_id
from reef.dispatcher import Dispatcher
from reef.observability import ExperimentLogger, ExperimentTracker
from reef.recipe import Recipe
from reef.runtime.interfaces import (
    ActivatedModel,
    CandidateTrainingDeferred,
    InferenceHandler,
    ModelCandidate,
    PreparedTrainingStep,
    StaleCandidate,
    TrainingJobResult,
)
from reef.service.app import RequestService
from reef.storage.sqlite import SQLiteScenarioStorage

from ._policy_recipe import TestPolicyRecipe

_ASYNC_WAIT_TIMEOUT_S = 5.0
_ASYNC_WAIT_POLL_S = 0.01


class DurableRuntime(StubTrainingRuntime):
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
    def inference_handler(self):
        return None

    def serving_runtime_load_id(self):
        return self.serving_version

    def prepare_training_step(
        self, batch, objective, algorithm_state, scheduling, scenario_step, *, serving_runtime_load_id=None
    ):
        sample = batch.items[0]
        payload = {
            "rollout_id": scenario_step,
            "loss": objective,
            "source": source_record_id(sample),
            "expected_runtime_load_id": sample.training.get("runtime_load_id", None),
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
        if payload["expected_runtime_load_id"] != self.serving_version:
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
            current_runtime_load_id=self.serving_version,
            training_metrics=self.complete_metrics or {},
        )
        self.completed[rollout_id] = candidate
        self.candidate_versions[job_id] = f"job:{job_id}"
        if self.fail_once:
            self.fail_once = False
            raise ConnectionError("lost remote acknowledgement")
        return candidate

    def activate_candidate(self, candidate):
        runtime_load_id = self.candidate_versions[candidate.candidate_id]
        self.serving_version = runtime_load_id
        return ActivatedModel(candidate.candidate_id, runtime_load_id)

    def reject_candidate(self, candidate, decision):
        del candidate, decision


class BlockingLostAckBackend(InMemoryRepositoryBackend):
    started = Event()
    release = Event()
    failed = False

    def publish(self, artifact, *, expected_parent, advance_head=True):
        type(self).started.set()
        if not type(self).release.wait(5):
            raise TimeoutError("test did not release blocked publication")
        ref = super().publish(artifact, expected_parent=expected_parent, advance_head=advance_head)
        if not type(self).failed:
            type(self).failed = True
            raise ArtifactPublicationError("lost publication acknowledgement")
        return ref


def _training(runtime_load_id: str) -> dict:
    return {"tokens": [1, 2], "loss_mask": [1], "rollout_log_probs": [-0.2], "runtime_load_id": runtime_load_id}


class ImmediateBackend(InferenceHandler):
    async def inference(self, artifact, path, payload):
        assert payload["return_meta_info"] is True
        return {"metadata": {"runtime_load_id": "v0"}}


class RecordingExperimentTracker(ExperimentTracker):
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


class RecordingExperimentLogger(ExperimentLogger):
    def __init__(self) -> None:
        self.logged = []

    def log(self, metrics, *, namespace):
        self.logged.append((namespace, dict(metrics)))


class OperationalExperimentTracker(RecordingExperimentTracker):
    @property
    def operational_metrics_enabled(self) -> bool:
        return True


def wait_for_operational_sample(
    tracker: RecordingExperimentTracker, expected: Mapping[str, int]
) -> dict[str, float | int]:
    deadline = time.monotonic() + _ASYNC_WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        for namespace, metrics in tracker.loggers["math"].logged:
            if namespace == "operations" and all(metrics.get(key) == value for key, value in expected.items()):
                return metrics
        time.sleep(_ASYNC_WAIT_POLL_S)
    pytest.fail(f"no operational sample matching {expected}")


@pytest.fixture
def start_dispatcher(tmp_path: Path):
    opened = []

    def start(backend_type=None, *, experiment_tracker=None, **runtime_options):
        initial = tmp_path / "initial"
        initial.mkdir(exist_ok=True)
        runtime = DurableRuntime(tmp_path / "checkpoints", **runtime_options)
        factory = (backend_type or InMemoryRepositoryBackend).factory(initial, root=tmp_path / "repository")
        dispatcher = Dispatcher(
            TestPolicyRecipe(**runtime_bindings(runtime), batch_size=1),
            factory,
            local_artifact_dir=tmp_path / "staged",
            agent_record_dir=tmp_path / "agent-record",
            experiment_tracker=experiment_tracker,
            scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record"),
        )
        opened.append((runtime, dispatcher))
        return runtime, dispatcher

    yield start
    for runtime, dispatcher in opened:
        runtime.release.set()
        dispatcher.close()


def _submit_pair(dispatcher: Dispatcher, suffix: str = "1", runtime_load_id: str = "v0") -> None:
    dispatcher.get_or_create_scenario("math")
    inference_id = f"inference-{suffix}"
    records = (
        AgentRecord.create(
            scenario="math",
            request_type=RequestType.INFERENCE,
            agent_record_id=inference_id,
            payload={"response": {"training": _training(runtime_load_id)}},
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
        dispatcher.accept_record(record)


def _wait_for_step(dispatcher: Dispatcher, step: int) -> None:
    deadline = time.monotonic() + _ASYNC_WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        if dispatcher.get_or_create_scenario("math").scenario_step == step:
            return
        time.sleep(_ASYNC_WAIT_POLL_S)
    pytest.fail(f"scenario did not reach step {step}: {dispatcher.build_training_status()}")


def _wait_for_error(dispatcher: Dispatcher) -> str:
    deadline = time.monotonic() + _ASYNC_WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        error = dispatcher.build_training_status()["error"]
        if isinstance(error, str):
            return error
        time.sleep(_ASYNC_WAIT_POLL_S)
    pytest.fail(f"training error was not reported: {dispatcher.build_training_status()}")


@pytest.mark.unit
def test_empty_checkpoint_result_fails_closed() -> None:
    # The invariant lives on the result type, so a completed job that cannot
    # name its exported checkpoint cannot be constructed at all -- it can never
    # reach the committer and be published as a durable version.
    with pytest.raises(ValueError, match="must report the checkpoint path"):
        TrainingJobResult(outcome="complete", runtime_load_id="v1", checkpoint_path="")


@pytest.mark.unit
def test_report_returns_and_inference_resolves_while_remote_job_is_blocked(start_dispatcher) -> None:
    runtime, dispatcher = start_dispatcher(fail_once=True, block=True)
    scenario = dispatcher.get_or_create_scenario("math")
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
def test_inference_waits_until_lost_ack_publication_is_committed(start_dispatcher) -> None:
    BlockingLostAckBackend.started = Event()
    BlockingLostAckBackend.release = Event()
    BlockingLostAckBackend.failed = False
    runtime, dispatcher = start_dispatcher(BlockingLostAckBackend)
    _submit_pair(dispatcher)
    assert BlockingLostAckBackend.started.wait(1)
    assert not runtime.inference.inference_admission_status["open"]

    async def request_during_publication():
        admission = asyncio.create_task(runtime.inference.acquire_inference())
        await asyncio.sleep(0.05)
        assert not admission.done()
        BlockingLostAckBackend.release.set()
        handle = await asyncio.wait_for(admission, 5)
        handle.release()

    asyncio.run(request_during_publication())
    _wait_for_step(dispatcher, 1)
    assert len(runtime.calls) == 1
    assert runtime.inference.current_runtime_load_id() == "job:job-0"


@pytest.mark.unit
def test_weight_inference_does_not_materialize_its_checkpoint(start_dispatcher, monkeypatch) -> None:
    _, dispatcher = start_dispatcher()
    scenario = dispatcher.get_or_create_scenario("math")
    monkeypatch.setattr(
        scenario.repository,
        "resolve",
        lambda ref: pytest.fail(f"unexpected inference materialization: {ref.release_id}"),
    )

    response = asyncio.run(
        RequestService(dispatcher).infer(
            {"x-reef-scenario": "math"},
            {"messages": [{"role": "user", "content": "hi"}]},
            "/v1/chat/completions",
            ImmediateBackend(),
        )
    )

    assert response["metadata"]["runtime_load_id"] == "v0"


@pytest.mark.unit
def test_two_jobs_form_one_deterministic_release_chain(start_dispatcher) -> None:
    runtime, dispatcher = start_dispatcher()
    _submit_pair(dispatcher)
    _wait_for_step(dispatcher, 1)
    scenario = dispatcher.get_or_create_scenario("math")
    first = scenario.current_artifact_ref()
    _submit_pair(dispatcher, "2", runtime.serving_version)
    _wait_for_step(dispatcher, 2)

    assert [call["rollout_id"] for call in runtime.calls] == [0, 1]
    assert [call["source"] for call in runtime.calls] == ["inference-1", "inference-2"]
    assert scenario.current_artifact_ref().parent_release_id == first.release_id
    with pytest.raises(ReefError, match="already bound"):
        dispatcher.get_or_create_scenario("other")


@pytest.mark.unit
def test_training_result_metrics_reach_the_durable_commit(start_dispatcher) -> None:
    metrics = {"staleness/samples_fresh": 1, "staleness/samples_admitted_stale": 0}
    _, dispatcher = start_dispatcher(complete_metrics=metrics)
    dispatcher.get_or_create_scenario("math")

    assert dispatcher.build_training_status()["scenarios"]["math"]["last_committed_step"] is None

    _submit_pair(dispatcher)
    _wait_for_step(dispatcher, 1)
    scenario = dispatcher.get_or_create_scenario("math")

    committed = scenario.metrics_for_version(scenario.current_artifact_ref().release_id)
    assert committed is not None
    assert {key: committed[key] for key in metrics} == metrics
    assert committed["selection"]["outcome"] == "select"
    status = dispatcher.build_training_status()["scenarios"]["math"]["last_committed_step"]
    assert status["step"] == 1
    assert isinstance(status["recorded_at"], float)
    assert status["metrics"] == committed
    status["metrics"]["selection"]["outcome"] = "mutated by caller"
    refreshed = dispatcher.build_training_status()["scenarios"]["math"]["last_committed_step"]
    assert refreshed["metrics"]["selection"]["outcome"] == "select"


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
    committed = scenario.metrics_for_version(produced.release_id)

    assert committed is not None
    assert committed["experiment/provider"] == "test"
    assert committed["experiment/run_id"] == "run-math"
    assert len(tracker.events) == 1
    event = tracker.events[0]
    assert event.context.scenario == "math"
    assert event.context.recipe == "test_policy"
    assert event.context.backend == "RuntimeCandidateBackend"
    assert event.context.backend_config == {
        "runtime": "DurableRuntime",
        "objective": "sft",
        "scheduling": {
            "unit": "comparison_set",
            "batch_size": "configured",
            "epochs": 1,
            "shuffle": False,
            "remainder": "partial",
        },
    }
    assert event.context.source_artifact_ref.release_id == produced.parent_release_id
    assert event.produced_artifact_ref == produced
    assert event.metrics["train/loss"] == pytest.approx(0.25)
    assert event.training_job_id == "job-0"
    assert event.source_runtime_load_id == "v0"
    assert event.produced_runtime_load_id == "job:job-0"
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
        "staleness/producing_runtime_load_ids": ["v0"],
        "staleness/serving_runtime_load_id": "v2",
        "staleness/drop_policy_lags": [2],
    }
    runtime, dispatcher = start_dispatcher(serving_version="v2", stale_metrics=stale_metrics)
    _submit_pair(dispatcher)
    _submit_pair(dispatcher, "2", "v2")
    _wait_for_step(dispatcher, 1)
    scenario = dispatcher.get_or_create_scenario("math")

    assert scenario.trainer.operational_metrics()["runtime/stale_batches_total"] == 1
    assert [call["source"] for call in runtime.calls] == ["inference-2"]
    # Rejecting the first batch consumes neither side's step counter, so the
    # next valid batch reuses rollout 0 rather than wedging the bridge at 1.
    assert runtime.calls[0]["rollout_id"] == 0
    assert scenario.scenario_step == 1
    assert scenario.records.count("math") == 4
    receipts = scenario.records.consumption_receipts("math")
    assert len(receipts) == 1
    assert receipts[0]["metadata"] == {
        "outcome": "stale",
        "metrics": stale_metrics,
    }
    assert receipts[0]["consumed_ids"] == ("inference-1", "report-1")


@pytest.mark.unit
def test_storage_block_preserves_pending_batch_and_retries(start_dispatcher, monkeypatch) -> None:
    monkeypatch.setattr("reef.dispatcher.Dispatcher.storage_retry_seconds", 0.01)
    runtime, dispatcher = start_dispatcher(block_storage=True)
    _submit_pair(dispatcher)
    assert runtime.started.wait(1)
    scenario = dispatcher.get_or_create_scenario("math")

    assert scenario.scenario_step == 0
    assert scenario.trainer.pending_batch is not None
    assert scenario.records.count("math") == 2
    assert dispatcher.build_training_status()["scenarios"]["math"]["checkpoint_storage"]["reasons"] == ["test cap"]
    runtime.block_storage = False
    _wait_for_step(dispatcher, 1)
    assert runtime.calls[0]["rollout_id"] == 0
    assert runtime.calls[0]["source"] == "inference-1"
    assert dispatcher.build_training_status()["scenarios"]["math"]["checkpoint_storage"] is None


def test_operational_sample_wait_ignores_a_matching_backlog_before_training() -> None:
    tracker = OperationalExperimentTracker()
    logger = RecordingExperimentLogger()
    tracker.loggers["math"] = logger
    logger.log({"records/unread_count": 2, "training/execution/active": 0}, namespace="operations")
    running = {"records/unread_count": 2, "training/execution/active": 1}
    logger.log(running, namespace="operations")

    assert wait_for_operational_sample(tracker, running) == running


def test_periodic_metrics_report_backlog_during_training_without_status_reads(start_dispatcher, monkeypatch) -> None:
    monkeypatch.setattr(Dispatcher, "operational_metrics_interval_seconds", 0.01)
    tracker = OperationalExperimentTracker()
    runtime, dispatcher = start_dispatcher(block=True, experiment_tracker=tracker)
    _submit_pair(dispatcher)
    assert runtime.started.wait(1)
    _submit_pair(dispatcher, "2", "job:job-0")
    # The first pair can also produce a backlog of two before training starts.
    sample = wait_for_operational_sample(tracker, {"records/unread_count": 2, "training/execution/active": 1})
    assert sample["training/execution/active"] == 1
    assert sample["training/execution/elapsed_seconds"] >= 0
    assert sample["records/oldest_unread_age_seconds"] >= 0
    assert sample["training/reserved_batches"] == 1
    assert sample["processor/unreserved_reports"] == 0
    assert sample["processor/reserved_reports"] == 1
    assert not tracker.events
    runtime.release.set()
    _wait_for_step(dispatcher, 2)
    dispatcher.close()
    final = [metrics for namespace, metrics in tracker.loggers["math"].logged if namespace == "operations"][-1]
    assert final["records/unread_count"] == 0
    assert final["training/execution/active"] == 0
    assert final["runtime/weight_sync/completed_total"] == 2
    assert final["training/failed_attempts_total"] == 0
    assert tracker.closed
    assert not dispatcher._lifecycle.metrics_thread.is_alive()


def test_periodic_metrics_keep_failures_after_training_recovery(start_dispatcher, monkeypatch) -> None:
    monkeypatch.setattr(Dispatcher, "operational_metrics_interval_seconds", 0.01)
    tracker = OperationalExperimentTracker()
    runtime, dispatcher = start_dispatcher(experiment_tracker=tracker)
    train_candidate = runtime.train_candidate
    failures: list[Mapping[str, object]] = []

    def fail(payload: Mapping[str, object]) -> ModelCandidate:
        failures.append(payload)
        raise ConnectionError("training submission unavailable")

    monkeypatch.setattr(runtime, "train_candidate", fail)
    _submit_pair(dispatcher)
    sample = wait_for_operational_sample(tracker, {"training/error": 1})
    assert sample["training/error"] == 1
    assert not tracker.events
    monkeypatch.setattr(runtime, "train_candidate", train_candidate)
    dispatcher._training.ready.set()
    _wait_for_step(dispatcher, 1)
    dispatcher.record_operational_metrics()
    final = [metrics for namespace, metrics in tracker.loggers["math"].logged if namespace == "operations"][-1]
    assert final["training/error"] == 0
    assert final["training/failed_attempts_total"] == len(failures)
    assert final["training/failed_attempts_total"] >= sample["training/failed_attempts_total"]


def test_disabled_tracking_does_not_start_periodic_sampling(start_dispatcher) -> None:
    _, dispatcher = start_dispatcher()
    assert dispatcher._lifecycle.metrics_thread is None


def test_periodic_metrics_observe_incomplete_weight_sync(start_dispatcher, monkeypatch) -> None:
    monkeypatch.setattr(Dispatcher, "operational_metrics_interval_seconds", 0.01)
    tracker = OperationalExperimentTracker()
    runtime, dispatcher = start_dispatcher(experiment_tracker=tracker)
    activate = runtime.activate_candidate
    transferring = Event()
    release = Event()

    def block_transfer(candidate: ModelCandidate) -> ActivatedModel:
        transferring.set()
        if not release.wait(5):
            raise TimeoutError("test did not release weight transfer")
        return activate(candidate)

    monkeypatch.setattr(runtime, "activate_candidate", block_transfer)
    try:
        _submit_pair(dispatcher)
        assert transferring.wait(1)
        sample = wait_for_operational_sample(tracker, {"runtime/weight_sync/active": 1})
        assert sample["runtime/weight_sync/elapsed_seconds"] >= 0
        assert sample["runtime/weight_sync/completed_total"] == 0
        assert not tracker.events
    finally:
        release.set()
    _wait_for_step(dispatcher, 1)


def test_processor_lock_does_not_hide_execution_metrics(start_dispatcher) -> None:
    runtime, dispatcher = start_dispatcher(block=True)
    _submit_pair(dispatcher)
    assert runtime.started.wait(1)
    trainer = dispatcher.get_or_create_scenario("math").trainer
    with trainer._lock:
        sample = trainer.operational_metrics()
    assert sample["training/execution/active"] == 1
    assert "records/unread_count" not in sample


def test_serving_only_scenarios_upload_requests_during_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Dispatcher, "operational_metrics_interval_seconds", 0.01)
    tracker = OperationalExperimentTracker()
    initial = tmp_path / "initial"
    initial.mkdir()
    dispatcher = Dispatcher(
        Recipe(),
        InMemoryRepositoryBackend.factory(initial),
        scenario_storage=SQLiteScenarioStorage(),
        experiment_tracker=tracker,
    )

    class BlockingHandler(InferenceHandler):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def inference(self, artifact: Artifact, path: str, payload: dict[str, object]) -> dict[str, object]:
            self.started.set()
            await self.release.wait()
            return {"choices": []}

    async def run() -> None:
        handler = BlockingHandler()
        service = RequestService(dispatcher)
        request = asyncio.create_task(service.infer({"x-reef-scenario": "math"}, {}, "/v1/chat/completions", handler))
        try:
            await asyncio.wait_for(handler.started.wait(), 2)
            sample = await asyncio.to_thread(wait_for_operational_sample, tracker, {"serve/request/active": 1})
            assert sample["serve/request/completed_total"] == 0
            assert sample["serve/request/elapsed_seconds"] >= 0
            assert not tracker.events
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            sample = await asyncio.to_thread(wait_for_operational_sample, tracker, {"serve/request/failed_total": 1})
            assert sample["serve/request/active"] == 0
            assert sample["ingest/accepted_total"] == 0
            handler.release.set()
            await service.infer({"x-reef-scenario": "math"}, {}, "/v1/chat/completions", handler)
            sample = await asyncio.to_thread(
                wait_for_operational_sample, tracker, {"serve/request/completed_total": 1}
            )
            assert sample["ingest/accepted_total"] == 1
        finally:
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
