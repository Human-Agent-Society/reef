"""Training jobs through the jobs queue: control adds them, a training worker runs them on the component."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from tests.support._trajectories import policy_trajectory

from reef.artifact import BaseArtifactSource
from reef.core.job_data import JobData
from reef.runtime.jobs import JobStatus, ReleaseFormat, SavedCheckpoint, TrainingJob
from reef.storage.sql_coordination import memory_coordination
from reef.storage.work_queue import NewWork
from reef.train.job_queue import (
    JOBS_QUEUE,
    JobQueueComponent,
    TrainingJobWorker,
    checkpoint_json,
    job_kind,
    job_payload,
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
