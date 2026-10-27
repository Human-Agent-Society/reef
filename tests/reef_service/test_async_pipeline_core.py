from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from pathlib import Path
from threading import Event

import pytest
from tests.support._policy_recipe import TestPolicyRecipe
from tests.support.runtime_stubs import StubInferenceRuntime, StubTrainingComponent, runtime_bindings

from reef.artifact import Artifact, ArtifactPublicationError, InMemoryRepositoryBackend
from reef.core import AgentRecord, ReefError, RequestType
from reef.core.trajectories import source_record_id
from reef.dispatcher import Dispatcher
from reef.observability import ExperimentLogger, ExperimentTracker
from reef.recipe import Recipe
from reef.runtime.interfaces import InferenceHandler
from reef.runtime.jobs import SavedCheckpoint, TrainingJob
from reef.service.app import RequestService
from reef.storage.sqlite import SQLiteScenarioStorage

_ASYNC_WAIT_TIMEOUT_S = 5.0
_ASYNC_WAIT_POLL_S = 0.01


class MaterializingInference(StubInferenceRuntime):
    """Serve each release under the version its ``model.txt`` names, so the initial release serves as ``w0``."""

    def serve(self, artifact: Artifact) -> str:
        return super().serve(artifact.materialize())


class DurableComponent(StubTrainingComponent):
    """A component whose jobs run in ``submit``; ``block`` holds a job until the test sets ``release``.

    ``calls`` lists the jobs that trained, and each saved checkpoint carries ``complete_metrics``.
    """

    def __init__(self, checkpoint_root: Path, *, block: bool = False, complete_metrics: dict | None = None) -> None:
        super().__init__(checkpoint_root)
        self.block = block
        self.complete_metrics = complete_metrics or {}
        self.started = Event()
        self.release = Event()
        self.calls: list[TrainingJob] = []
        self.inference = MaterializingInference(base_url="http://inference")

    def train(self, job: TrainingJob) -> tuple[SavedCheckpoint, ...]:
        self.calls.append(job)
        self.started.set()
        if self.block and not self.release.wait(5):
            raise TimeoutError("test did not release blocked training")
        (checkpoint,) = super().train(job)
        return (SavedCheckpoint(checkpoint.step, checkpoint.path, checkpoint.release_path, self.complete_metrics),)


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
        return {"metadata": {"runtime_load_id": "w0"}}


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

    def start(backend_type=None, *, experiment_tracker=None, **component_options):
        initial = tmp_path / "initial"
        initial.mkdir(exist_ok=True)
        (initial / "model.txt").write_text("w0")
        component = DurableComponent(tmp_path / "checkpoints", **component_options)
        factory = (backend_type or InMemoryRepositoryBackend).factory(initial, root=tmp_path / "repository")
        dispatcher = Dispatcher(
            TestPolicyRecipe(**runtime_bindings(component, component.inference)),
            factory,
            local_artifact_dir=tmp_path / "staged",
            experiment_tracker=experiment_tracker,
            scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record"),
        )
        opened.append((component, dispatcher))
        return component, dispatcher

    yield start
    for component, dispatcher in opened:
        component.release.set()
        dispatcher.close()


def _submit_pair(dispatcher: Dispatcher, suffix: str = "1", runtime_load_id: str = "w0") -> None:
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


def _wait_for_jobs(dispatcher: Dispatcher, finished_jobs: int) -> None:
    """Wait until ``finished_jobs`` jobs ended; each job commits its submission, its checkpoint and its end."""
    deadline = time.monotonic() + _ASYNC_WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        status = dispatcher.build_training_status()["scenarios"]["math"]
        if status["finished_jobs"] == finished_jobs and status["training_job"] is None:
            return
        time.sleep(_ASYNC_WAIT_POLL_S)
    pytest.fail(f"scenario did not finish {finished_jobs} jobs: {dispatcher.build_training_status()}")


def _wait_for_error(dispatcher: Dispatcher) -> str:
    deadline = time.monotonic() + _ASYNC_WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        error = dispatcher.build_training_status()["error"]
        if isinstance(error, str):
            return error
        time.sleep(_ASYNC_WAIT_POLL_S)
    pytest.fail(f"training error was not reported: {dispatcher.build_training_status()}")


@pytest.mark.unit
def test_report_returns_and_inference_resolves_while_training_job_is_blocked(start_dispatcher) -> None:
    component, dispatcher = start_dispatcher(block=True)
    scenario = dispatcher.get_or_create_scenario("math")
    expected = scenario.current_artifact_ref()
    _submit_pair(dispatcher)
    assert component.started.wait(1)
    assert scenario.current_artifact_ref() == expected
    component.release.set()
    _wait_for_jobs(dispatcher, 1)
    assert len(component.calls) == 1
    assert scenario.scenario_step == 3


@pytest.mark.unit
def test_lost_submission_acknowledgement_resubmits_the_same_job(start_dispatcher, monkeypatch) -> None:
    component, dispatcher = start_dispatcher()
    submit = component.submit
    submissions: list[str] = []

    def lose_first_acknowledgement(job: TrainingJob) -> None:
        submissions.append(job.job_id)
        submit(job)
        if len(submissions) == 1:
            raise ConnectionError("lost submission acknowledgement")

    monkeypatch.setattr(component, "submit", lose_first_acknowledgement)
    _submit_pair(dispatcher)
    # Each record wakes the worker, so a retry may clear the error before a poll sees it: wait for the count.
    deadline = time.monotonic() + _ASYNC_WAIT_TIMEOUT_S
    while dispatcher.errors.failure_count("math") == 0 and time.monotonic() < deadline:
        time.sleep(_ASYNC_WAIT_POLL_S)
    assert dispatcher.errors.failure_count("math") == 1
    dispatcher.local_workers.wake(dispatcher.get_or_create_scenario("math"))
    _wait_for_jobs(dispatcher, 1)

    # The retry names the same job, so the component trains it once.
    assert submissions[0] == submissions[1]
    assert len(component.calls) == 1


@pytest.mark.unit
def test_backend_failure_reaches_reef_status(start_dispatcher, monkeypatch) -> None:
    component, dispatcher = start_dispatcher()

    def fail(job: TrainingJob) -> None:
        del job
        raise RuntimeError("backend submission failed")

    monkeypatch.setattr(component, "submit", fail)
    _submit_pair(dispatcher)

    assert _wait_for_error(dispatcher) == "math: RuntimeError: backend submission failed"


@pytest.mark.unit
def test_lost_ack_publication_commits_the_checkpoint_once(start_dispatcher) -> None:
    BlockingLostAckBackend.started = Event()
    BlockingLostAckBackend.release = Event()
    BlockingLostAckBackend.failed = False
    component, dispatcher = start_dispatcher(BlockingLostAckBackend)
    _submit_pair(dispatcher)
    assert BlockingLostAckBackend.started.wait(1)
    BlockingLostAckBackend.release.set()
    _wait_for_jobs(dispatcher, 1)

    assert len(component.calls) == 1
    published = [
        row for row in dispatcher.list_releases("math") if row["checkpoint"] and row["operation"] == "training"
    ]
    assert len(published) == 1
    assert component.inference.current_runtime_load_id() == "w1"


@pytest.mark.unit
def test_weight_inference_does_not_materialize_its_checkpoint(start_dispatcher, monkeypatch) -> None:
    _, dispatcher = start_dispatcher()
    scenario = dispatcher.get_or_create_scenario("math")
    monkeypatch.setattr(
        scenario.repository,
        "resolve",
        lambda ref: pytest.fail(f"unexpected inference materialization: {ref.release_id}"),
    )

    response, _ = asyncio.run(
        RequestService(dispatcher).infer_with_data(
            {"x-reef-scenario": "math"},
            {"messages": [{"role": "user", "content": "hi"}]},
            "/v1/chat/completions",
            ImmediateBackend(),
        )
    )

    assert response["metadata"]["runtime_load_id"] == "w0"


@pytest.mark.unit
def test_two_jobs_form_one_deterministic_release_chain(start_dispatcher) -> None:
    component, dispatcher = start_dispatcher()
    _submit_pair(dispatcher)
    _wait_for_jobs(dispatcher, 1)
    scenario = dispatcher.get_or_create_scenario("math")
    first = scenario.current_artifact_ref()
    _submit_pair(dispatcher, "2", "w1")
    _wait_for_jobs(dispatcher, 2)

    assert [source_record_id(job.data.samples[0]) for job in component.calls] == ["inference-1", "inference-2"]
    # Training is linear: the second job starts from the first job's checkpoint.
    assert component.calls[0].start is None
    assert component.calls[1].start == str(component.checkpoint_root / "1")
    assert scenario.current_artifact_ref().parent_release_id == first.release_id
    with pytest.raises(ReefError, match="already bound"):
        dispatcher.get_or_create_scenario("other")


@pytest.mark.unit
def test_training_result_metrics_reach_the_durable_commit(start_dispatcher) -> None:
    metrics = {"train/loss": 0.25, "train/grad_norm": 1.5}
    _, dispatcher = start_dispatcher(complete_metrics=metrics)
    dispatcher.get_or_create_scenario("math")

    assert dispatcher.build_training_status()["scenarios"]["math"]["last_committed_step"] is None

    _submit_pair(dispatcher)
    _wait_for_jobs(dispatcher, 1)
    scenario = dispatcher.get_or_create_scenario("math")

    # The checkpoint's commit publishes the release and carries the job's training metrics.
    _, committed = scenario.release_queries.artifact_with_metrics()
    assert committed is not None
    assert {key: committed[key] for key in metrics} == metrics
    assert committed["selection"]["selected"] is True
    assert committed["training_job"]["event"] == "checkpoint"
    # The last commit is the job's end.
    status = dispatcher.build_training_status()["scenarios"]["math"]["last_committed_step"]
    assert status["step"] == 3
    assert isinstance(status["recorded_at"], float)
    assert status["metrics"]["training_job"]["event"] == "finished"
    status["metrics"]["training_job"]["event"] = "failed"
    refreshed = dispatcher.build_training_status()["scenarios"]["math"]["last_committed_step"]
    assert refreshed["metrics"]["training_job"]["event"] == "finished"


@pytest.mark.unit
def test_experiment_provider_observes_the_generic_commit_boundary(start_dispatcher) -> None:
    tracker = RecordingExperimentTracker()
    _, dispatcher = start_dispatcher(
        experiment_tracker=tracker,
        complete_metrics={"train/loss": 0.25},
    )

    _submit_pair(dispatcher)
    _wait_for_jobs(dispatcher, 1)
    scenario = dispatcher.get_or_create_scenario("math")
    produced = scenario.current_artifact_ref()
    _, committed = scenario.release_queries.artifact_with_metrics(produced.release_id)

    assert committed is not None
    assert committed["experiment/provider"] == "test"
    assert committed["experiment/run_id"] == "run-math"
    # One event per commit: the job's submission, its checkpoint and its end.
    assert [event.metrics["training_job"]["event"] for event in tracker.events] == [
        "submitted",
        "checkpoint",
        "finished",
    ]
    event = tracker.events[1]
    assert event.context.scenario == "math"
    assert event.context.recipe == "test_policy"
    assert event.context.backend == "stub"
    assert event.context.backend_config == {
        "job_plan": "fixed",
        "parameters": {"method": {"objective": "sft"}},
    }
    assert event.context.source_artifact_ref.release_id == produced.parent_release_id
    assert event.produced_artifact_ref == produced
    assert event.metrics["train/loss"] == pytest.approx(0.25)
    assert event.training_job_id == tracker.events[0].training_job_id
    assert scenario.trainer.pipeline.experiment_logger is tracker.loggers["math"]
    scenario.trainer.pipeline.experiment_logger.log({"accepted": 1}, namespace="processor")
    assert tracker.loggers["math"].logged == [("processor", {"accepted": 1})]


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
    component, dispatcher = start_dispatcher(block=True, experiment_tracker=tracker)
    _submit_pair(dispatcher)
    assert component.started.wait(1)
    _submit_pair(dispatcher, "2", "w1")
    # The first pair can also produce a backlog of two before training starts.
    sample = wait_for_operational_sample(tracker, {"records/unread_count": 2, "training/execution/active": 1})
    assert sample["training/execution/active"] == 1
    assert sample["training/execution/elapsed_seconds"] >= 0
    assert sample["records/oldest_unread_age_seconds"] >= 0
    assert sample["training/reserved_job_data"] == 1
    assert sample["processor/unreserved_reports"] == 0
    assert sample["processor/reserved_reports"] == 1
    # The pipeline status is exported too; the default buffer holds the data of four jobs.
    assert "processor/ready_samples" in sample
    assert (sample["processor/job_samples"], sample["processor/buffer_size"]) == (1, 4)
    assert not tracker.events
    component.release.set()
    _wait_for_jobs(dispatcher, 2)
    dispatcher.close()
    final = [metrics for namespace, metrics in tracker.loggers["math"].logged if namespace == "operations"][-1]
    assert final["records/unread_count"] == 0
    assert final["training/execution/active"] == 0
    assert final["training/failed_attempts_total"] == 0
    assert tracker.closed
    assert not dispatcher.background_threads.metrics_thread.is_alive()


def test_periodic_metrics_keep_failures_after_training_recovery(start_dispatcher, monkeypatch) -> None:
    monkeypatch.setattr(Dispatcher, "operational_metrics_interval_seconds", 0.01)
    tracker = OperationalExperimentTracker()
    component, dispatcher = start_dispatcher(experiment_tracker=tracker)
    submit = component.submit
    failures: list[TrainingJob] = []

    def fail(job: TrainingJob) -> None:
        failures.append(job)
        raise ConnectionError("training submission unavailable")

    monkeypatch.setattr(component, "submit", fail)
    _submit_pair(dispatcher)
    sample = wait_for_operational_sample(tracker, {"training/error": 1})
    assert sample["training/error"] == 1
    assert not tracker.events
    monkeypatch.setattr(component, "submit", submit)
    dispatcher.local_workers.wake(dispatcher.get_or_create_scenario("math"))
    _wait_for_jobs(dispatcher, 1)
    dispatcher.record_operational_metrics()
    final = [metrics for namespace, metrics in tracker.loggers["math"].logged if namespace == "operations"][-1]
    assert final["training/error"] == 0
    assert final["training/failed_attempts_total"] == len(failures)
    assert final["training/failed_attempts_total"] >= sample["training/failed_attempts_total"]


def test_disabled_tracking_does_not_start_periodic_sampling(start_dispatcher) -> None:
    _, dispatcher = start_dispatcher()
    assert dispatcher.background_threads.metrics_thread is None


def test_processor_lock_does_not_hide_execution_metrics(start_dispatcher) -> None:
    component, dispatcher = start_dispatcher(block=True)
    _submit_pair(dispatcher)
    assert component.started.wait(1)
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
        request = asyncio.create_task(
            service.infer_with_data({"x-reef-scenario": "math"}, {}, "/v1/chat/completions", handler)
        )
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
            await service.infer_with_data({"x-reef-scenario": "math"}, {}, "/v1/chat/completions", handler)
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
