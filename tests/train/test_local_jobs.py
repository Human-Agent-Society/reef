"""LocalJobComponent: durable submission, one job at a time, cancellation, close and resume after a restart."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from tests.support._trajectories import policy_trajectory

from reef.artifact import BaseArtifactSource
from reef.core.job_data import JobData
from reef.runtime.durable_files import write_json
from reef.runtime.jobs import ReleaseFormat, SavedCheckpoint, TrainingJob
from reef.train import local_jobs
from reef.train.local_jobs import CHECKPOINTS_FILE, JOB_FILE, RESULT_FILE, LocalJobComponent

WAIT_SECONDS = 5.0


class GatedComponent(LocalJobComponent):
    """Saves ``checkpoint_count`` checkpoints per job; while ``gate`` is clear, each save waits for it.

    ``started`` is set once a run begins. A job with ``{"fail": true}`` raises in
    ``run``; a job with ``{"invalid": true}`` fails ``check_job``. ``runs`` lists
    the run job IDs in order, and ``resumed_from`` the checkpoints each run found.
    """

    def __init__(self, jobs_root: Path, *, checkpoint_count: int = 1, gated: bool = False) -> None:
        super().__init__(jobs_root)
        self.checkpoint_count = checkpoint_count
        self.gate = threading.Event()
        if not gated:
            self.gate.set()
        self.started = threading.Event()
        self.runs: list[str] = []
        self.resumed_from: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "gated"

    @property
    def release_format(self) -> ReleaseFormat:
        return ReleaseFormat("full")

    def base_artifact_source(self) -> BaseArtifactSource:
        return BaseArtifactSource()

    @property
    def base_start(self) -> str:
        return "gated:base"

    def import_start(self, release: Path) -> str:
        return str(release)

    def check_job(self, job: TrainingJob) -> None:
        if job.parameters.get("invalid"):
            raise ValueError("this component cannot train the job")

    def run(self, job: TrainingJob, directory: Path) -> Mapping[str, Any]:
        self.runs.append(job.job_id)
        self.started.set()
        earlier = self.saved(directory)
        self.resumed_from[job.job_id] = len(earlier)
        if job.parameters.get("fail"):
            raise RuntimeError("boom")
        for step in range(len(earlier) + 1, self.checkpoint_count + 1):
            if not self.gate.wait(WAIT_SECONDS):
                raise TimeoutError("the test never opened the gate")
            path = str(directory / f"step-{step}")
            self.save_checkpoint(directory, SavedCheckpoint(step, path, path, {"loss": 1.0 / step}))
            if self.stopping(directory):
                break
        return {"steps": len(self.saved(directory))}


def training_job(job_id: str, **parameters: bool) -> TrainingJob:
    data = JobData(job_id, (policy_trajectory(f"s-{job_id}", (1, 2, 3), (1, 1), (-0.1, -0.2), 1.0, "v1"),))
    return TrainingJob(job_id, "gated", "gated:base", data, parameters)


def wait_until(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + WAIT_SECONDS
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("the condition did not hold in time")
        time.sleep(0.01)


def wait_ended(component: LocalJobComponent, job_id: str) -> None:
    wait_until(lambda: component.status(job_id).state != "running")


@pytest.fixture
def make_component() -> Iterator[Callable[..., GatedComponent]]:
    made: list[GatedComponent] = []

    def build(jobs_root: Path, **options: Any) -> GatedComponent:
        component = GatedComponent(jobs_root, **options)
        made.append(component)
        return component

    yield build
    for component in made:
        component.gate.set()
        component.close()


@pytest.mark.unit
def test_submitted_job_is_durable_runs_and_finishes_with_checkpoints(tmp_path, make_component):
    component = make_component(tmp_path, checkpoint_count=2, gated=True)
    component.start_jobs()
    component.submit(training_job("a"))

    assert (component.job_directory("a") / JOB_FILE).is_file()
    assert component.started.wait(WAIT_SECONDS)
    assert component.status("a").state == "running"
    component.gate.set()
    wait_ended(component, "a")

    status = component.status("a")
    assert status.state == "finished"
    assert status.error is None
    assert [checkpoint.step for checkpoint in status.checkpoints] == [1, 2]
    assert status.checkpoints[1].path == str(component.job_directory("a") / "step-2")
    assert status.checkpoints[0].metrics == {"loss": 1.0}
    assert status.metrics == {"steps": 2}


@pytest.mark.unit
def test_failing_run_ends_failed_with_the_error_text(tmp_path, make_component):
    component = make_component(tmp_path)
    component.start_jobs()
    component.submit(training_job("a", fail=True))
    wait_ended(component, "a")

    status = component.status("a")
    assert status.state == "failed"
    assert status.error == "RuntimeError: boom"
    assert status.metrics == {}


@pytest.mark.unit
def test_resubmitting_a_job_id_runs_it_once(tmp_path, make_component):
    component = make_component(tmp_path, gated=True)
    component.start_jobs()
    component.submit(training_job("a"))
    component.submit(training_job("a"))
    component.gate.set()
    wait_ended(component, "a")
    component.submit(training_job("a"))
    component.submit(training_job("b"))
    wait_ended(component, "b")

    assert component.runs == ["a", "b"]


@pytest.mark.unit
def test_job_submitted_before_start_jobs_runs_once(tmp_path, make_component):
    component = make_component(tmp_path)
    component.submit(training_job("a"))
    component.start_jobs()
    wait_ended(component, "a")
    component.submit(training_job("b"))
    wait_ended(component, "b")

    assert component.runs == ["a", "b"]


@pytest.mark.unit
def test_status_of_an_unknown_job_raises_key_error(tmp_path, make_component):
    component = make_component(tmp_path)

    with pytest.raises(KeyError):
        component.status("missing")


@pytest.mark.unit
def test_rejected_job_raises_and_records_nothing(tmp_path, make_component):
    component = make_component(tmp_path)
    component.start_jobs()

    with pytest.raises(ValueError, match="cannot train"):
        component.submit(training_job("a", invalid=True))

    assert not (component.job_directory("a") / JOB_FILE).exists()
    assert list(component.queue) == []
    with pytest.raises(KeyError):
        component.status("a")
    # The refused job left nothing to run: the next job is the only run.
    component.submit(training_job("b"))
    wait_ended(component, "b")
    assert component.runs == ["b"]


@pytest.mark.unit
def test_cancelled_queued_job_never_runs(tmp_path, make_component):
    component = make_component(tmp_path, gated=True)
    component.start_jobs()
    component.submit(training_job("first"))
    assert component.started.wait(WAIT_SECONDS)
    component.submit(training_job("second"))

    component.cancel("second")
    component.gate.set()
    wait_ended(component, "first")

    status = component.status("second")
    assert (status.state, status.error) == ("failed", "cancelled")
    assert component.status("first").state == "finished"
    assert component.runs == ["first"]


@pytest.mark.unit
def test_a_job_cancelled_at_the_head_of_the_queue_never_runs(tmp_path, make_component):
    component = make_component(tmp_path)
    component.submit(training_job("a"))
    component.submit(training_job("b"))
    component.cancel("a")
    assert list(component.queue) == ["a", "b"]
    # Run the job thread on the queue as it is, with the cancelled job at its head.
    component.job_thread = threading.Thread(target=component.run_jobs, daemon=True)
    component.job_thread.start()
    wait_ended(component, "b")

    assert component.runs == ["b"]
    status = component.status("a")
    assert (status.state, status.error, status.checkpoints) == ("failed", "cancelled", ())


@pytest.mark.unit
def test_status_of_an_ended_job_lists_its_last_checkpoint_when_the_job_ends_during_the_read(
    tmp_path, make_component, monkeypatch
):
    component = make_component(tmp_path)
    component.submit(training_job("a"))
    directory = component.job_directory("a")
    first = str(directory / "step-1")
    last = str(directory / "step-2")
    component.save_checkpoint(directory, SavedCheckpoint(1, first, first))
    original_read = local_jobs.read_json
    job_ended = False

    def read_then_end_the_job(path: Path) -> Any:
        nonlocal job_ended
        value = original_read(path)
        if not job_ended:
            # The job thread saves its last checkpoint, then its result, between the two reads of status.
            job_ended = True
            component.save_checkpoint(directory, SavedCheckpoint(2, last, last))
            write_json(directory / RESULT_FILE, {"state": "finished", "metrics": {}})
        return value

    monkeypatch.setattr(local_jobs, "read_json", read_then_end_the_job)
    statuses = [component.status("a"), component.status("a")]

    assert statuses[-1].state == "finished"
    for status in statuses:
        assert status.state == "running" or status.checkpoints[-1].step == 2


@pytest.mark.unit
def test_cancelled_running_job_stops_and_keeps_the_cancelled_result(tmp_path, make_component):
    component = make_component(tmp_path, checkpoint_count=3, gated=True)
    component.start_jobs()
    component.submit(training_job("a"))
    assert component.started.wait(WAIT_SECONDS)
    directory = component.job_directory("a")
    assert not component.stopping(directory)

    component.cancel("a")

    assert component.stopping(directory)
    status = component.status("a")
    assert (status.state, status.error) == ("failed", "cancelled")
    component.gate.set()
    # The next job runs only after the cancelled run returned and its result was handled.
    component.submit(training_job("b"))
    wait_ended(component, "b")
    status = component.status("a")
    assert (status.state, status.error, status.metrics) == ("failed", "cancelled", {})
    # The save after the cancel is dropped: the checkpoint list stays as the cancel's result found it.
    assert status.checkpoints == ()


@pytest.mark.unit
def test_restarted_component_resumes_unfinished_jobs_in_submission_order(tmp_path, make_component):
    previous = make_component(tmp_path)
    for job_id in ("a", "b", "c", "done"):
        previous.submit(training_job(job_id))
    (previous.job_directory("done") / RESULT_FILE).write_text('{"state": "finished", "metrics": {}}')
    earlier_path = str(previous.job_directory("b") / "step-1")
    previous.save_checkpoint(previous.job_directory("b"), SavedCheckpoint(1, earlier_path, earlier_path))
    # Submission order c, a, b, by the job files' modification times.
    for order, job_id in enumerate(("c", "a", "b")):
        os.utime(previous.job_directory(job_id) / JOB_FILE, (1_000_000 + order, 1_000_000 + order))
    previous.close()

    restarted = make_component(tmp_path, checkpoint_count=2)
    restarted.start_jobs()
    for job_id in ("a", "b", "c"):
        wait_ended(restarted, job_id)

    assert restarted.runs == ["c", "a", "b"]
    assert restarted.resumed_from == {"c": 0, "a": 0, "b": 1}
    assert [checkpoint.step for checkpoint in restarted.status("b").checkpoints] == [1, 2]
    assert restarted.status("done").state == "finished"


@pytest.mark.unit
def test_close_mid_job_leaves_it_to_resume_and_refuses_new_jobs(tmp_path, make_component):
    component = make_component(tmp_path, checkpoint_count=2, gated=True)
    component.start_jobs()
    component.submit(training_job("a"))
    assert component.started.wait(WAIT_SECONDS)
    directory = component.job_directory("a")

    closer = threading.Thread(target=component.close)
    closer.start()
    wait_until(lambda: component.stopping(directory))
    component.gate.set()
    closer.join(WAIT_SECONDS)

    assert not closer.is_alive()
    assert not (directory / RESULT_FILE).exists()
    assert (directory / CHECKPOINTS_FILE).is_file()
    assert component.status("a").state == "running"
    with pytest.raises(RuntimeError, match="closed"):
        component.submit(training_job("b"))

    restarted = make_component(tmp_path, checkpoint_count=2)
    restarted.start_jobs()
    wait_ended(restarted, "a")
    assert restarted.resumed_from == {"a": 1}
    assert restarted.status("a").state == "finished"
