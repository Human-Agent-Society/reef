"""The training status that ``GET /reef/status`` reports, and when a local training worker may sleep.

The status tests bind a scenario double to the dispatcher's registry, so each
one checks one status field. The wait tests pin when a worker looks again
without a new record: a running training job saves checkpoints, and a
background job, such as a judgment, can make a batch ready.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from tests.support.runtime_stubs import StubTrainingComponent

from reef.artifact.memory import InMemoryRepositoryBackend
from reef.core.job_data import JobOutcome
from reef.dispatcher import Dispatcher
from reef.dispatcher.errors import TrainingErrors, local_error_source
from reef.dispatcher.local_workers import POLL_SECONDS, LocalBackendWorker, LocalTrainingWorkers
from reef.evaluation.evaluators import AlwaysSelectCheckpointPlugin
from reef.recipe import Recipe
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.jobs import FixedJobPlan, JobLoop, JobLoopState, JobRequest, RunningJob

pytestmark = pytest.mark.unit


def scenario_double(
    *,
    background_jobs_running: bool = False,
    job_running: bool = False,
    job_data_ready: bool = False,
    pipeline_status: dict | None = None,
    runtime=None,
    last_committed_step: dict | None = None,
    job_loop=None,
    state: dict | None = None,
) -> SimpleNamespace:
    """A scenario with one trainer; it carries what the status and the local workers read."""
    pipeline = SimpleNamespace(background_jobs_running=lambda: background_jobs_running, training_mode="auto")
    trainer = SimpleNamespace(
        pipeline=pipeline,
        trains=True,
        job_running=job_running,
        needs_polling=lambda: job_running or background_jobs_running,
        job_loop=job_loop,
        state={} if state is None else state,
        job_data_ready=lambda: job_data_ready,
        pipeline_status=lambda: dict(pipeline_status or {}),
    )
    return SimpleNamespace(
        trainer=trainer,
        trainers=(SimpleNamespace(component=None, trainer=trainer),),
        trainer_for=lambda component: trainer,
        runtime=runtime,
        commit_status={"scenario_step": 0, "last_committed_step": last_committed_step},
    )


def registry_double(scenario: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        training_status_scenario_names=("s",),
        get_optional=lambda name: scenario if name == "s" else None,
        loaded_scenarios=lambda: (scenario,),
        preload_errors={},
    )


def bound_dispatcher(scenario: SimpleNamespace) -> Dispatcher:
    root = Path(tempfile.mkdtemp(prefix="reef-artifacts-"))
    initial = root / "initial"
    initial.mkdir()
    dispatcher = Dispatcher(
        Recipe(),
        InMemoryRepositoryBackend.factory(initial, root=root / "repository"),
        scenario_storage=SQLiteScenarioStorage(),
    )
    dispatcher.registry = registry_double(scenario)
    dispatcher.local_workers.registry = dispatcher.registry
    dispatcher.training_status.registry = dispatcher.registry
    return dispatcher


def test_status_exposes_the_ready_state() -> None:
    dispatcher = bound_dispatcher(scenario_double(job_data_ready=True))

    assert dispatcher.build_training_status()["scenarios"]["s"]["job_data_ready"] is True


def test_status_exposes_the_last_committed_step_outcome() -> None:
    committed = {
        "step": 3,
        "recorded_at": 1_756_400_000.0,
        "metrics": {"skipped": "no proposal", "selection": {"reason": "candidate lost"}},
    }
    dispatcher = bound_dispatcher(scenario_double(last_committed_step=committed))

    assert dispatcher.build_training_status()["scenarios"]["s"]["last_committed_step"] == committed


def test_status_keeps_the_published_version_until_inference_reopens() -> None:
    class Runtime:
        open = False
        current = "engine:old"
        per_scenario_adapters = False

        @property
        def inference_admission_status(self):
            return {"open": self.open, "active": 0}

        def current_runtime_load_id(self):
            return self.current

    runtime = Runtime()
    dispatcher = bound_dispatcher(scenario_double(runtime=runtime))

    status = dispatcher.build_training_status()["scenarios"]["s"]
    assert status["current_runtime_load_id"] == "engine:old"
    assert status["inference_admission"] == {"open": False, "active": 0}

    runtime.current = "engine:new"
    runtime.open = True
    status = dispatcher.build_training_status()["scenarios"]["s"]
    assert status["current_runtime_load_id"] == "engine:new"
    assert status["inference_admission"] == {"open": True, "active": 0}


def test_status_exposes_terminal_processor_outcomes() -> None:
    outcome = {"failed_steps": [{"step": 1, "reason": "mixed_release_ids", "release_ids": ["old", "new"]}]}
    dispatcher = bound_dispatcher(scenario_double(pipeline_status=outcome))

    assert dispatcher.build_training_status()["scenarios"]["s"]["processor"] == outcome


def test_status_reports_the_running_job_and_the_finished_jobs_of_a_job_trainer() -> None:
    finished = JobOutcome("job-2", "stub", "finished")
    running = JobLoopState(job=RunningJob("job-3", checkpoints_handled=1), finished_jobs=2, last_job=finished)
    loop = JobLoop(StubTrainingComponent(), FixedJobPlan(JobRequest()), AlwaysSelectCheckpointPlugin(), scenario="s")
    dispatcher = bound_dispatcher(scenario_double(job_loop=loop, state=running.to_dict()))

    status = dispatcher.build_training_status()["scenarios"]["s"]
    assert status["training_job"] == {"job_id": "job-3", "backend": "stub", "checkpoints_handled": 1}
    assert status["finished_jobs"] == 2

    dispatcher.registry.get_optional("s").trainer.state = JobLoopState(finished_jobs=3, last_job=finished).to_dict()
    status = dispatcher.build_training_status()["scenarios"]["s"]
    assert status["training_job"] is None
    assert status["finished_jobs"] == 3


def test_status_of_a_trainer_without_a_job_loop_has_no_job_fields() -> None:
    status = bound_dispatcher(scenario_double()).build_training_status()["scenarios"]["s"]

    assert "training_job" not in status
    assert "finished_jobs" not in status


def test_status_reports_build_failure_instead_of_raising() -> None:
    dispatcher = bound_dispatcher(scenario_double())

    def fail() -> bool:
        raise RuntimeError("processor status is unavailable")

    dispatcher.registry.get_optional("s").trainer.job_data_ready = fail

    status = dispatcher.build_training_status()

    assert status["scenarios"] == {}
    assert status["error"] == "s: RuntimeError: processor status is unavailable"

    dispatcher.registry.get_optional("s").trainer.job_data_ready = lambda: False
    recovered = dispatcher.build_training_status()
    assert recovered["error"] is None
    assert "s" in recovered["scenarios"]


def test_status_reports_training_and_build_failures() -> None:
    dispatcher = bound_dispatcher(scenario_double())
    dispatcher.errors.record("s", "RuntimeError: backend failed", local_error_source(None))

    def fail() -> bool:
        raise RuntimeError("processor status is unavailable")

    dispatcher.registry.get_optional("s").trainer.job_data_ready = fail

    assert dispatcher.build_training_status()["error"] == (
        "s: RuntimeError: backend failed\ns: RuntimeError: processor status is unavailable"
    )


class RecordingEvent(Event):
    """An event whose ``wait`` returns at once and records the timeout it was given."""

    def __init__(self) -> None:
        super().__init__()
        self.timeouts: list[float | None] = []

    def wait(self, timeout: float | None = None) -> bool:
        self.timeouts.append(timeout)
        return True


def worker_wait_timeout(scenario: SimpleNamespace) -> float | None:
    """The timeout that the worker of ``scenario``'s trainer waits with."""
    workers = LocalTrainingWorkers(registry_double(scenario), TrainingErrors(), held=False)
    ready = RecordingEvent()
    workers.workers[("s", None)] = LocalBackendWorker(ready, Thread(target=print))
    assert workers.wait("s", None, ready) is True
    return ready.timeouts[-1]


def test_an_idle_worker_sleeps_until_the_next_record() -> None:
    assert worker_wait_timeout(scenario_double()) is None


def test_a_worker_with_a_running_job_polls_for_its_checkpoints() -> None:
    assert worker_wait_timeout(scenario_double(job_running=True)) == POLL_SECONDS


def test_a_worker_with_running_background_jobs_polls_on_a_bounded_interval() -> None:
    # A judgment that lands makes a batch ready without a new record waking the worker:
    # sleeping without a timeout stalls training on a quiet workload.
    timeout = worker_wait_timeout(scenario_double(background_jobs_running=True))

    assert timeout is not None and timeout > 0
