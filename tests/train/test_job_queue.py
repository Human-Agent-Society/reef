"""Training jobs through the jobs queue: control adds them, a training worker runs them on the component."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from tests.support._trajectories import policy_trajectory

from reef.artifact import BaseArtifactSource
from reef.core.artifact_ref import ArtifactRef
from reef.core.job_data import JobData, TaskSample
from reef.core.training_request import TrainingRequest
from reef.observability.base import (
    ExperimentRun,
    ExperimentTracker,
    TrainingExperimentContext,
    TrainingExperimentEvent,
)
from reef.runtime.jobs import JobStatus, ReleaseFormat, SavedCheckpoint, TrainingJob
from reef.storage.sql_coordination import memory_coordination
from reef.storage.work_queue import NewWork
from reef.train.job_queue import (
    JOBS_QUEUE,
    JobExperimentRecorder,
    JobQueueComponent,
    TrainingJobWorker,
    checkpoint_json,
    job_json,
    job_kind,
    job_payload,
    scenario_jobs,
)
from reef.train.local_jobs import LocalJobComponent

WAIT_SECONDS = 10.0


class StepComponent(LocalJobComponent):
    """Saves two checkpoints per job; a job with ``{"fail": true}`` raises, ``{"invalid": true}`` is refused."""

    def __init__(self, jobs_root: Path) -> None:
        super().__init__(jobs_root)
        self.closed_count = 0
        self.start_jobs()

    @property
    def name(self) -> str:
        return "steps"

    @property
    def release_format(self) -> ReleaseFormat:
        return ReleaseFormat("full")

    def base_artifact_source(self) -> BaseArtifactSource:
        return BaseArtifactSource()

    @property
    def base_start(self) -> str:
        return "steps:base"

    def import_start(self, release: Path) -> str:
        return str(release)

    def check_job(self, job: TrainingJob) -> None:
        if job.parameters.get("invalid"):
            raise ValueError("this component cannot train the job")

    def run(self, job: TrainingJob, directory: Path) -> Mapping[str, Any]:
        if job.parameters.get("fail"):
            raise RuntimeError("boom")
        for step in range(len(self.saved(directory)) + 1, 3):
            path = str(directory / f"step-{step}")
            self.save_checkpoint(directory, SavedCheckpoint(step, path, path, {"loss": 1.0 / step}))
        return {"steps": 2}

    def close_resources(self) -> None:
        self.closed_count += 1


def job(job_id: str, **parameters: Any) -> TrainingJob:
    data = JobData(job_id, (policy_trajectory(f"s-{job_id}", (1, 2, 3), (1, 1), (-0.1, -0.2), 1.0, artifact_id="v1"),))
    return TrainingJob(job_id, "steps", "steps:base", data, parameters, scenario="s")


def wait_for(status: Callable[[], JobStatus], state: str) -> JobStatus:
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        current = status()
        if current.state == state:
            return current
        time.sleep(0.02)
    raise AssertionError(f"the job never reached {state}")


@pytest.fixture
def worked(tmp_path):
    queue = memory_coordination(lease_seconds=30).work_queue
    component = StepComponent(tmp_path / "jobs")
    worker = TrainingJobWorker(component, queue, worker_id="training-1", poll_seconds=0.02)
    worker.start()
    yield queue, component, worker
    worker.stop()


@pytest.mark.unit
def test_a_queued_job_runs_on_the_worker_and_reports_its_checkpoints_through_the_queue(worked):
    queue, component, _worker = worked
    view = JobQueueComponent(component, queue)
    assert (view.name, view.base_start, view.release_format) == ("steps", "steps:base", ReleaseFormat("full"))

    view.submit(job("j1"))
    view.submit(job("j1"))  # a second submit of one ID does nothing
    item = queue.get(JOBS_QUEUE, "j1")
    assert (item.scenario, item.kind, item.serial_key) == ("s", job_kind("steps"), "s")

    status = wait_for(lambda: view.status("j1"), "finished")
    assert [checkpoint.step for checkpoint in status.checkpoints] == [1, 2]
    assert status.checkpoints[0].metrics == {"loss": 1.0}
    with pytest.raises(KeyError):
        view.status("unknown")


@pytest.mark.unit
def test_a_refused_job_never_enters_the_queue_and_a_failed_job_reports_its_error(worked):
    queue, component, _worker = worked
    view = JobQueueComponent(component, queue)
    with pytest.raises(ValueError, match="cannot train"):
        view.submit(job("bad", invalid=True))
    assert queue.get(JOBS_QUEUE, "bad") is None

    view.submit(job("boom", fail=True))
    failed = wait_for(lambda: view.status("boom"), "failed")
    assert "boom" in failed.error


@pytest.mark.unit
def test_a_cancelled_job_fails_and_the_worker_owns_the_component(tmp_path):
    queue = memory_coordination(lease_seconds=30).work_queue
    component = StepComponent(tmp_path / "jobs")
    view = JobQueueComponent(component, queue)
    view.submit(job("waiting"))
    view.cancel("waiting")
    assert view.status("waiting").state == "failed" and view.status("waiting").error == "cancelled"

    worker = TrainingJobWorker(component, queue, worker_id="training-1", poll_seconds=0.02)
    assert worker.run_once() is False  # the cancelled job is final, so nothing is left to take
    worker.stop()
    assert component.closed_count == 1


@pytest.mark.unit
def test_a_control_only_component_closes_what_it_owns(tmp_path):
    queue = memory_coordination().work_queue
    component = StepComponent(tmp_path / "jobs")
    JobQueueComponent(component, queue).close()
    assert component.closed_count == 0
    JobQueueComponent(component, queue, owns_component=True).close()
    assert component.closed_count == 1


@pytest.mark.unit
def test_a_worker_that_continues_a_job_reports_each_checkpoint_step_once(tmp_path):
    queue = memory_coordination(lease_seconds=30).work_queue
    queue.add(NewWork(JOBS_QUEUE, "j1", "s", job_kind("steps"), job_payload(job("j1")), serial_key="s"))
    # The first worker reported step 1, then stopped without a word; its lease expires.
    queue.take(JOBS_QUEUE, "training-0", [job_kind("steps")])
    queue.start(JOBS_QUEUE, "j1", "training-0")
    first = SavedCheckpoint(1, "/old/step-1", "/old/step-1", {"loss": 1.0})
    queue.add_progress(JOBS_QUEUE, "j1", "training-0", checkpoint_json(first), number=0)
    queue.requeue_expired(now=time.time() + 60)

    worker = TrainingJobWorker(StepComponent(tmp_path / "jobs"), queue, worker_id="training-1", poll_seconds=0.02)
    assert worker.run_once()
    item = queue.get(JOBS_QUEUE, "j1")
    assert item.state == "finished"
    assert [entry["step"] for entry in item.progress] == [1, 2]
    worker.stop()


class ReleaseRefusingComponent(StepComponent):
    """A component that continues only from its own checkpoints, as Slime does."""

    @property
    def starts_from_releases(self) -> bool:
        return False

    def import_start(self, release: Path) -> str:
        raise ValueError("this component starts only from its own checkpoints")


@pytest.mark.unit
def test_a_queue_component_reports_whether_its_component_starts_from_releases(tmp_path):
    queue = memory_coordination().work_queue
    assert JobQueueComponent(StepComponent(tmp_path / "a"), queue).starts_from_releases is True
    refusing = JobQueueComponent(ReleaseRefusingComponent(tmp_path / "b"), queue)
    assert refusing.starts_from_releases is False


@pytest.mark.unit
def test_a_listed_job_carries_its_parameters_instruction_data_and_times(tmp_path):
    queue = memory_coordination(lease_seconds=300).work_queue
    request = TrainingRequest("prefer tests", "session-1", "release-1", id="r1", parameters={"learning_rate": 1e-5})
    sample = policy_trajectory("s-j1", (1, 2, 3), (1, 1), (-0.1, -0.2), 1.0, artifact_id="v1")
    data = JobData("s:instruction:r1", (sample, sample), request=request)
    parameters = {"method": {"objective": "sft"}, "learning_rate": 1e-5}
    instructed = TrainingJob("j1", "steps", "steps:base", data, parameters, scenario="s")
    queue.add(NewWork(JOBS_QUEUE, "j1", "s", job_kind("steps"), job_payload(instructed), serial_key="s"), now=0)

    queued = job_json(queue.get(JOBS_QUEUE, "j1"))
    assert queued["parameters"] == parameters
    assert (queued["sample_count"], queued["sample_kind"]) == (2, "trajectory")
    assert queued["instruction"] == {"id": "r1", "text": "prefer tests", "parameters": {"learning_rate": 1e-5}}
    assert (queued["created_at"], queued["started_at"], queued["ended_at"]) == (
        "1970-01-01T00:00:00+00:00",
        None,
        None,
    )
    assert queued["experiment"] is None

    queue.take(JOBS_QUEUE, "training-1", [job_kind("steps")], now=60)
    running = job_json(queue.start(JOBS_QUEUE, "j1", "training-1", now=61))
    assert (running["started_at"], running["ended_at"]) == ("1970-01-01T00:01:00+00:00", None)
    queue.finish(JOBS_QUEUE, "j1", "training-1", {"state": "finished", "metrics": {}}, now=90.5)
    (ended,) = scenario_jobs(queue, "s")
    assert (ended["state"], ended["ended_at"]) == ("finished", "1970-01-01T00:01:30.500000+00:00")

    # A job of the auto mode has no instruction; a task job names its kind, and an empty job has none.
    automatic = TrainingJob(
        "j2", "steps", "steps:base", JobData("s:data:1", (TaskSample(tmp_path),)), {}, scenario="s"
    )
    empty = TrainingJob("j3", "steps", "steps:base", JobData("s:data:2"), {}, scenario="s")
    for value in (automatic, empty):
        queue.add(NewWork(JOBS_QUEUE, value.job_id, "s", job_kind("steps"), job_payload(value), serial_key="s"))
    listed = {listed_job["id"]: listed_job for listed_job in scenario_jobs(queue, "s")}
    assert (listed["j2"]["instruction"], listed["j2"]["sample_kind"], listed["j2"]["parameters"]) == (None, "task", {})
    assert (listed["j3"]["sample_count"], listed["j3"]["sample_kind"]) == (0, None)


class RunTracker(ExperimentTracker):
    """Records events and has the run ``run-<scenario>`` open for every context, or no run with ``open_run`` false."""

    def __init__(self, *, open_run: bool = True) -> None:
        self.open_run = open_run
        self.events: list[TrainingExperimentEvent] = []

    def experiment_run(self, context):
        if not self.open_run:
            return None
        run_id = f"run-{context.scenario}"
        return ExperimentRun("wandb", "reef", run_id, "team", f"https://wandb.ai/team/reef/runs/{run_id}")

    def record(self, event):
        self.events.append(event)


def job_event(job_id: str | None) -> TrainingExperimentEvent:
    ref = ArtifactRef(content_id="s", release_id="r1", parent_release_id=None)
    context = TrainingExperimentContext(scenario="s", recipe="recipe", step=1, source_artifact_ref=ref)
    return TrainingExperimentEvent(context, ref, {}, "committed", training_job_id=job_id)


@pytest.mark.unit
def test_control_writes_the_run_that_records_a_job_into_the_jobs_item():
    queue = memory_coordination().work_queue
    queue.add(NewWork(JOBS_QUEUE, "j1", "s", job_kind("steps"), job_payload(job("j1")), serial_key="s"))
    tracker = RunTracker()
    recorder = JobExperimentRecorder(tracker, queue)
    recorder.record(job_event("j1"))
    recorder.record(job_event(None))  # a step without a job writes nothing
    recorder.record(job_event("not-a-queued-job"))  # nor does a job that the queue does not hold
    assert len(tracker.events) == 3

    (listed,) = scenario_jobs(queue, "s")
    assert listed["experiment"] == {
        "tracker": "wandb",
        "project": "reef",
        "entity": "team",
        "run_id": "run-s",
        "url": "https://wandb.ai/team/reef/runs/run-s",
    }
    # The run of every later commit of the job is the same, so the item does not change again.
    sequence = queue.get(JOBS_QUEUE, "j1").changed_sequence
    recorder.record(job_event("j1"))
    assert queue.get(JOBS_QUEUE, "j1").changed_sequence == sequence

    with pytest.raises(TypeError):
        JobExperimentRecorder(recorder, queue)


@pytest.mark.unit
def test_a_job_has_no_experiment_when_the_tracker_has_no_run():
    queue = memory_coordination().work_queue
    queue.add(NewWork(JOBS_QUEUE, "j1", "s", job_kind("steps"), job_payload(job("j1")), serial_key="s"))
    JobExperimentRecorder(RunTracker(open_run=False), queue).record(job_event("j1"))
    JobExperimentRecorder(ExperimentTracker(), queue).record(job_event("j1"))
    assert scenario_jobs(queue, "s")[0]["experiment"] is None
