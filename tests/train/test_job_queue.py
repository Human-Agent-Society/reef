"""Training jobs through the jobs queue: control adds them, a training worker runs them on the component."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from tests.support.job_data import job_data_ref
from tests.support.training_deployment import LocalFacts
from tests.support.trajectories import policy_trajectory

from reef.artifact import (
    Artifact,
    ArtifactError,
    GitLFSRepositoryBackend,
    InMemoryRepositoryBackend,
    RepositoryBackendFactory,
)
from reef.artifact.components import ComponentChecks
from reef.core.artifact_selection import ArtifactSelection
from reef.core.errors import InternalError
from reef.core.job_data import JobData, TaskSample
from reef.core.snapshot import RootSource, SnapshotRef
from reef.core.training_request import TrainingRequest
from reef.core.weight_layout import ArtifactFormat
from reef.dispatcher.work_retention import EndedWorkSweeper
from reef.observability.base import (
    ExperimentRun,
    ExperimentTracker,
    TrainingExperimentContext,
    TrainingExperimentEvent,
)
from reef.recipe import Recipe
from reef.scenario.live_objects import RecipeChecks
from reef.storage.data import MemoryDataStore
from reef.storage.sql_coordination import memory_coordination
from reef.storage.tasks import LocalTaskFileStore
from reef.storage.work_queue import NewWork, WorkLeaseLostError
from reef.train.component import (
    JobCheckPendingError,
    JobStatus,
    SavedCheckpoint,
    ScenarioRoot,
    SnapshotStart,
    TrainingJob,
)
from reef.train.job_queue import (
    CHECKS_QUEUE,
    JOB_CANCELLED_ERROR,
    JOBS_QUEUE,
    SCENARIO_DELETED_ERROR,
    STORED_CONTENT_KEY,
    UNASKED_CHECK_ERROR,
    UNASKED_CHECK_SECONDS,
    JobExperimentRecorder,
    JobQueueComponent,
    TrainingCheckWorker,
    TrainingJobWorker,
    cancel_request_jobs,
    checkpoint_json,
    discard_ended_content,
    job_from_item,
    job_json,
    job_kind,
    job_payload,
    job_records_content,
    resolved_job,
    scenario_jobs,
)
from reef.train.jobs import JobLoop, JobLoopState, RunningJob
from reef.train.local_jobs import LocalJobComponent
from reef.train.training_requests import REQUEST_CANCELLED_ERROR, ScenarioTrainingRequests

WAIT_SECONDS = 10.0
#: The checks of a recipe that declares no component: they admit every checkpoint.
ACCEPT_ALL = RecipeChecks(Recipe())


class StepComponent(LocalJobComponent):
    """Saves two checkpoints per job; a job with ``{"fail": true}`` raises and reports what it reached, and
    ``{"invalid": true}`` is refused."""

    def __init__(self, jobs_root: Path) -> None:
        super().__init__(jobs_root)
        self.closed_count = 0
        self.checked: list[str] = []
        #: What ``job_progress`` reports for each job.
        self.progress: dict[str, Mapping[str, Any]] = {}
        self.archived: list[str] = []
        self.start_jobs()

    @property
    def name(self) -> str:
        return "steps"

    @property
    def artifact_format(self) -> ArtifactFormat:
        return ArtifactFormat("full")

    def root_start(self, root: ScenarioRoot) -> str:
        return "steps:base"

    def import_start(self, artifact_dir: Path) -> str:
        return str(artifact_dir)

    def check_job(self, job: TrainingJob) -> None:
        self.checked.append(job.job_id)
        if job.parameters.get("invalid"):
            raise ValueError("this component cannot train the job")
        if job.parameters.get("unreachable"):
            raise ConnectionError("the artifact store did not answer")

    def check_remote_artifact(self, names: Mapping[str, Any]) -> None:
        if names != {"path": "known"}:
            raise ValueError(f"the service holds no {names['path']}")

    def changed_parts(self, start: str, checkpoint: SavedCheckpoint) -> frozenset[str]:
        return frozenset({f"step-{checkpoint.step}"})

    def status(self, job_id: str) -> JobStatus:
        status = super().status(job_id)
        if status.state != "failed" or status.error == "cancelled":
            return status
        return JobStatus(job_id, "failed", status.checkpoints, metrics={"reached": "start"}, error=status.error)

    def run(self, job: TrainingJob, directory: Path) -> Mapping[str, Any]:
        if job.parameters.get("fail"):
            raise RuntimeError("boom")
        for step in range(len(self.saved(directory)) + 1, 3):
            path = directory / f"step-{step}"
            path.mkdir(exist_ok=True)
            (path / "weights.txt").write_text(f"step {step}")
            self.save_checkpoint(directory, SavedCheckpoint(step, str(path), str(path), {"loss": 1.0 / step}))
        return {"steps": 2}

    def job_progress(self, job_id: str) -> Mapping[str, Any] | None:
        return self.progress.get(job_id)

    def job_records(self, job_id: str) -> Path | None:
        records = self.job_directory(job_id) / "step-1"
        return records if records.is_dir() else None

    def archive_scenario(self, scenario: str) -> None:
        self.archived.append(scenario)

    def close_resources(self) -> None:
        self.closed_count += 1


def job(job_id: str, **parameters: Any) -> TrainingJob:
    data = JobData(job_id, (policy_trajectory(f"s-{job_id}", (1, 2, 3), (1, 1), (-0.1, -0.2), 1.0, artifact_id="v1"),))
    return TrainingJob(job_id, "steps", "steps:base", job_data_ref(data), parameters, scenario="s")


class StepFacts(LocalFacts):
    """The facts of ``StepComponent``; ``check_import_start`` refuses a directory named ``refused``."""

    @property
    def name(self) -> str:
        return "steps"

    def check_import_start(self, artifact_dir: Path) -> None:
        if artifact_dir.name == "refused":
            raise ValueError("the steps component cannot start from this snapshot")


def submit(view: JobQueueComponent, submitted: TrainingJob) -> None:
    """Submit ``submitted`` as a trainer does: again while the training service has not answered its check."""
    deadline = time.monotonic() + WAIT_SECONDS
    while True:
        try:
            view.submit(submitted)
            return
        except JobCheckPendingError:
            assert time.monotonic() < deadline, "the training service never answered the job check"
            time.sleep(0.02)


def wait_for(status: Callable[[], JobStatus], state: str) -> JobStatus:
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        current = status()
        if current.state == state:
            return current
        time.sleep(0.02)
    raise AssertionError(f"the job never reached {state}")


def registered_backends(directory: Path) -> RepositoryBackendFactory:
    """In-memory artifact stores under ``directory`` in which the jobs' scenario ``s`` exists: the training worker
    runs no job of a scenario that has no registration."""
    backends = InMemoryRepositoryBackend.factory(directory=directory)
    root = directory.with_name(f"{directory.name}-root")
    root.mkdir()
    backends("s").register(Artifact.local(root), metadata={})
    return backends


@pytest.fixture
def worked(tmp_path):
    queue = memory_coordination(lease_seconds=30).work_queue
    component = StepComponent(tmp_path / "jobs")
    data_store = MemoryDataStore(LocalTaskFileStore(tmp_path / "task-files"))
    backends = registered_backends(tmp_path / "artifacts")
    worker = TrainingJobWorker(
        component, queue, data_store, backends, checks=ACCEPT_ALL, worker_id="training-1", poll_seconds=0.02
    )
    checks = TrainingCheckWorker(component, queue, data_store, None, worker_id="checks-1", poll_seconds=0.02)
    worker.start()
    checks.start()
    yield queue, component, worker
    checks.stop()
    worker.stop()


@pytest.mark.unit
def test_a_queued_job_keeps_its_component_and_previous_job():
    queue = memory_coordination(lease_seconds=300).work_queue
    job = TrainingJob(
        "j1",
        "steps",
        "steps:base",
        job_data_ref(JobData("s:data:1")),
        {},
        scenario="s",
        component="harness",
        previous_job_id="j0",
    )
    queue.append(NewWork(JOBS_QUEUE, "j1", "s", job_kind("steps"), job_payload(job), serial_key="s"))
    taken = job_from_item(queue.read(JOBS_QUEUE, "j1"), MemoryDataStore(LocalTaskFileStore()))
    assert (taken.component, taken.previous_job_id) == ("harness", "j0")


@pytest.mark.unit
def test_a_queued_job_runs_on_the_worker_and_reports_its_checkpoints_through_the_queue(worked):
    queue, component, worker = worked
    view = JobQueueComponent(StepFacts(), queue)
    assert (view.name, view.artifact_format) == ("steps", ArtifactFormat("full"))

    submit(view, job("j1"))
    view.submit(job("j1"))  # a second submit of one ID does nothing
    item = queue.read(JOBS_QUEUE, "j1")
    # The item has no serial key: a free worker takes it beside another job of its scenario.
    assert (item.scenario, item.kind, item.serial_key) == ("s", job_kind("steps"), None)

    status = wait_for(lambda: view.status("j1"), "finished")
    assert [checkpoint.step for checkpoint in status.checkpoints] == [1, 2]
    assert status.checkpoints[0].metrics == {"loss": 1.0}
    # The worker recorded the parts that each checkpoint changed, so control reads them without the component.
    assert view.changed_parts("steps:base", status.checkpoints[1]) == {"step-2"}
    listed = job_json(queue.read(JOBS_QUEUE, "j1"))["checkpoints"][0]
    assert listed["changed_parts"] == ["step-1"]
    # The worker stored the files of each checkpoint, and the queue names the stored content and no directory.
    assert [(saved.content, saved.artifact_path) for saved in status.checkpoints] == [("j1:1", None), ("j1:2", None)]
    assert sorted(listed) == ["changed_parts", "content", "metrics", "path", "step"] and listed["content"] == "j1:1"
    assert (worker.backends("s").materialize_content("j1:2") / "weights.txt").read_text() == "step 2"
    # Control commits by that name, so it needs no directory of the training service.
    shutil.rmtree(component.jobs_root)
    state = JobLoopState("steps:base", (RunningJob("j1", "steps:base"),))
    publication = JobLoop(view, scenario="s").poll(state).result
    assert (publication.artifact.content, publication.artifact.local_path) == ("j1:1", None)
    assert publication.changed_kinds == {"step-1"}
    with pytest.raises(KeyError):
        view.status("unknown")


@pytest.mark.unit
def test_the_training_service_checks_a_job_before_it_enters_the_queue(worked, tmp_path):
    queue, _component, _worker = worked
    view = JobQueueComponent(StepFacts(), queue)
    # The first submit only asks: nothing is queued while the answer is not there.
    with pytest.raises(JobCheckPendingError):
        view.submit(job("bad", invalid=True))
    assert queue.read(JOBS_QUEUE, "bad") is None
    # The component's refusal comes back with its message, and the job never enters the queue.
    with pytest.raises(ValueError, match=r"^this component cannot train the job$"):
        submit(view, job("bad", invalid=True))
    assert queue.read(JOBS_QUEUE, "bad") is None
    # A submit after a refusal asks again: the job's one check item is queued once more.
    with pytest.raises(JobCheckPendingError):
        view.submit(job("bad", invalid=True))
    with pytest.raises(ValueError, match=r"^this component cannot train the job$"):
        submit(view, job("bad", invalid=True))
    assert [item.id for item in queue.items(CHECKS_QUEUE)] == ["job:bad"]

    submit(view, job("boom", fail=True))
    failed = wait_for(lambda: view.status("boom"), "failed")
    # A failed item has no result, so the metrics that the component reported cross the queue in its metadata.
    assert "boom" in failed.error and failed.metrics == {"reached": "start"}


@pytest.mark.unit
def test_a_check_that_could_not_run_is_no_refusal_and_the_next_submit_asks_again(worked):
    queue, component, _worker = worked
    view = JobQueueComponent(StepFacts(), queue)
    # The check fails for a reason that is not the job's: it uses up its attempts and ends.
    with pytest.raises(RuntimeError) as failed:
        submit(view, job("unlucky", unreachable=True))
    assert not isinstance(failed.value, ValueError)
    assert "could not check job unlucky, so nothing was decided about it" in str(failed.value)
    assert "the artifact store did not answer" in str(failed.value)
    assert queue.read(JOBS_QUEUE, "unlucky") is None
    attempts = len(component.checked)
    assert attempts > 1
    # The failure is read once. The same job, submitted again, gets a new check.
    with pytest.raises(JobCheckPendingError):
        view.submit(job("unlucky", unreachable=True))
    with pytest.raises(RuntimeError):
        submit(view, job("unlucky", unreachable=True))
    assert len(component.checked) > attempts


@pytest.mark.unit
def test_the_check_worker_drops_a_check_that_control_stopped_asking_for(tmp_path):
    queue = memory_coordination(lease_seconds=30).work_queue
    component = StepComponent(tmp_path / "jobs")
    view = JobQueueComponent(StepFacts(), queue)
    worker = TrainingCheckWorker(
        component, queue, MemoryDataStore(LocalTaskFileStore(tmp_path / "task-files")), None, worker_id="checks-1"
    )
    abandoned = job("abandoned")
    payload = {"check": "job", "job_id": "abandoned", "job": job_payload(abandoned)}
    old = time.time() - UNASKED_CHECK_SECONDS - 1
    queue.append(NewWork(CHECKS_QUEUE, "job:abandoned", "s", "training_check:steps", payload), now=old)
    assert worker.run_once()
    # Nobody waits for the answer, so the component was not asked and imported nothing.
    dropped = queue.read(CHECKS_QUEUE, "job:abandoned")
    assert (dropped.state, dropped.error, component.checked) == ("failed", UNASKED_CHECK_ERROR, [])
    # A trainer that comes back for the job asks again, and the check runs.
    with pytest.raises(JobCheckPendingError):
        view.submit(abandoned)
    assert worker.run_once()
    view.submit(abandoned)
    assert component.checked == ["abandoned"] and queue.read(JOBS_QUEUE, "abandoned") is not None
    component.close()


@pytest.mark.unit
def test_a_job_item_of_another_release_is_one_failed_job_and_the_next_job_runs(worked):
    queue, _component, _worker = worked
    view = JobQueueComponent(StepFacts(), queue)
    loop = JobLoop(view, scenario="s")
    # A worker of an older release reported a checkpoint by its directory, and still holds the job.
    queue.append(NewWork(JOBS_QUEUE, "old", "s", job_kind("old-release"), job_payload(job("old"))))
    queue.take(JOBS_QUEUE, "training-0", [job_kind("old-release")])
    entry = {"step": 1, "path": "/old/step-1", "artifact_path": "/old/step-1", "metrics": {}, "changed_parts": []}
    queue.append_progress(JOBS_QUEUE, "old", "training-0", entry, number=0)

    state = JobLoopState("steps:base", (RunningJob("old", "steps:base"),))
    ended = JobLoopState.from_dict(loop.poll(state).result.state)
    assert ended.jobs == () and ended.finished_jobs == 1
    assert ended.last_job.event == "failed"
    assert "another release of Reef may have written it" in ended.last_job.error
    assert "a checkpoint's content" in ended.last_job.error
    # The item is ended, so no worker runs it further, and a cancel and the job listing still read it.
    assert queue.read(JOBS_QUEUE, "old").state == "failed"
    assert view.status("old").state == "failed"
    view.cancel("old")
    assert [listed["id"] for listed in scenario_jobs(queue, "s")] == ["old"]

    # The trainer goes on: its next job is checked and runs.
    submit(view, job("next"))
    assert wait_for(lambda: view.status("next"), "finished").checkpoints


@pytest.mark.unit
def test_a_job_waits_unsubmitted_while_no_training_service_answers(tmp_path):
    queue = memory_coordination(lease_seconds=30).work_queue
    view = JobQueueComponent(StepFacts(), queue)
    for _ in range(3):
        with pytest.raises(JobCheckPendingError):
            view.submit(job("waiting"))
    # One check item waits, and no job.
    assert len(queue.items(CHECKS_QUEUE)) == 1 and not queue.items(JOBS_QUEUE)


@pytest.mark.unit
def test_control_names_the_snapshot_of_a_start_and_the_training_service_resolves_it(tmp_path):
    view = JobQueueComponent(StepFacts(), memory_coordination().work_queue)
    backends = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    backend = backends("s")
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "model.txt").write_text("w1")
    published = backend.register(Artifact.local(tree), metadata={})
    source = RootSource("default", inputs={"model": "stub"})
    root = ScenarioRoot(Artifact(published, backend), source)

    named = view.root_start(root)
    assert SnapshotStart.from_text(named) == SnapshotStart(published.revision, source)
    assert SnapshotStart.from_text("steps:base") is None
    imported = view.snapshot_start(root.artifact.materialize())
    assert SnapshotStart.from_text(imported) == SnapshotStart(published.revision)
    # The facts refuse at once what the component would refuse, and control imports nothing itself.
    refused = tmp_path / "refused"
    refused.mkdir()
    with pytest.raises(ValueError, match="cannot start from this snapshot"):
        view.snapshot_start(Artifact.local(refused))
    with pytest.raises(InternalError):
        view.import_start(tree)

    component = StepComponent(tmp_path / "jobs")
    first = TrainingJob("j1", "steps", named, job_data_ref(JobData("d1")), {}, scenario="s")
    assert resolved_job(component, first, backends, tmp_path / "staged").start == "steps:base"
    later = TrainingJob("j2", "steps", imported, job_data_ref(JobData("d2")), {}, scenario="s")
    start = resolved_job(component, later, backends, tmp_path / "staged").start
    assert (Path(start) / "model.txt").read_text() == "w1"
    # A component's own start stays as it is, and a worker without an artifact store refuses a named snapshot.
    own = TrainingJob("j3", "steps", "steps:base", job_data_ref(JobData("d3")), {}, scenario="s")
    assert resolved_job(component, own, None, None) is own
    with pytest.raises(ValueError, match="opens no artifact store"):
        resolved_job(component, first, None, None)
    component.close()


@pytest.mark.unit
def test_the_training_service_answers_the_remote_check_of_an_artifact(worked):
    queue, _component, _worker = worked
    view = JobQueueComponent(StepFacts(), queue, answer_seconds=WAIT_SECONDS)
    view.check_remote_artifact({"path": "known"})
    with pytest.raises(ValueError, match=r"^the service holds no lost$"):
        view.check_remote_artifact({"path": "lost"})


@pytest.mark.unit
def test_a_remote_check_without_a_training_service_ends_as_an_artifact_error():
    queue = memory_coordination(lease_seconds=30).work_queue
    view = JobQueueComponent(StepFacts(), queue, answer_seconds=0.1)
    with pytest.raises(ArtifactError, match="did not check the artifact"):
        view.check_remote_artifact({"path": "known"})
    # The check that nobody answered is cancelled, so a training service that starts later does not run it.
    assert [item.state for item in queue.items(CHECKS_QUEUE)] == ["failed"]


@pytest.mark.unit
def test_a_cancelled_job_fails_and_the_worker_owns_the_component(tmp_path):
    queue = memory_coordination(lease_seconds=30).work_queue
    component = StepComponent(tmp_path / "jobs")
    view = JobQueueComponent(StepFacts(), queue)
    queue.append(NewWork(JOBS_QUEUE, "waiting", "s", job_kind("steps"), job_payload(job("waiting"))))
    view.cancel("waiting")
    assert view.status("waiting").state == "failed" and view.status("waiting").error == "cancelled"

    worker = TrainingJobWorker(
        component,
        queue,
        MemoryDataStore(LocalTaskFileStore(tmp_path / "task-files")),
        registered_backends(tmp_path / "artifacts"),
        checks=ACCEPT_ALL,
        worker_id="training-1",
        poll_seconds=0.02,
    )
    assert worker.run_once() is False  # the cancelled job is final, so nothing is left to take
    worker.stop()
    assert component.closed_count == 1


@pytest.mark.unit
def test_a_worker_that_continues_a_job_reports_each_checkpoint_step_once(tmp_path):
    queue = memory_coordination(lease_seconds=30).work_queue
    queue.append(NewWork(JOBS_QUEUE, "j1", "s", job_kind("steps"), job_payload(job("j1")), serial_key="s"))
    # The first worker reported step 1, then stopped without a word; its lease expires.
    queue.take(JOBS_QUEUE, "training-0", [job_kind("steps")])
    queue.start(JOBS_QUEUE, "j1", "training-0")
    first = SavedCheckpoint(1, "/old/step-1", None, {"loss": 1.0}, content="j1:1")
    queue.append_progress(JOBS_QUEUE, "j1", "training-0", checkpoint_json(first), number=0)
    # It had stored step 2 as well, and stopped before it recorded that checkpoint.
    backends = registered_backends(tmp_path / "artifacts")
    unfinished = tmp_path / "unfinished"
    unfinished.mkdir()
    (unfinished / "weights.txt").write_text("half written")
    (unfinished / "leftover.txt").write_text("")
    backends("s").store_content("j1:2", unfinished)
    queue.requeue_expired(now=time.time() + 60)

    worker = TrainingJobWorker(
        StepComponent(tmp_path / "jobs"),
        queue,
        MemoryDataStore(LocalTaskFileStore(tmp_path / "task-files")),
        backends,
        checks=ACCEPT_ALL,
        worker_id="training-1",
        poll_seconds=0.02,
    )
    assert worker.run_once()
    item = queue.read(JOBS_QUEUE, "j1")
    assert item.state == "finished"
    assert [(entry["step"], entry["content"]) for entry in item.progress] == [(1, "j1:1"), (2, "j1:2")]
    # The step that was recorded is not stored again; the other is stored again under its one name.
    with pytest.raises(ArtifactError):
        backends("s").materialize_content("j1:1")
    stored = backends("s").materialize_content("j1:2")
    assert {path.name: path.read_text() for path in stored.iterdir()} == {"weights.txt": "step 2"}
    worker.stop()


class RefuseSecondStep(ComponentChecks):
    """Refuse a checkpoint whose weights say ``step 2``; it reads the checkpoint's files. ``seen`` lists the
    scenario and component of each check."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str | None]] = []

    def validate(self, scenario: str, component: str | None, directory: Path) -> None:
        self.seen.append((scenario, component))
        if (directory / "weights.txt").read_text() == "step 2":
            raise ValueError("the weights of step 2 are refused")


class HeldSteps(StepComponent):
    """Saves the two checkpoints of ``StepComponent`` and keeps the job running until ``release`` is set."""

    def __init__(self, jobs_root: Path) -> None:
        super().__init__(jobs_root)
        self.release = threading.Event()

    def run(self, job: TrainingJob, directory: Path) -> Mapping[str, Any]:
        result = super().run(job, directory)
        self.release.wait(WAIT_SECONDS)
        return result


@pytest.mark.unit
def test_the_worker_checks_each_checkpoint_before_it_stores_it_and_a_refusal_fails_the_job(tmp_path):
    """The training service runs the scenario's check on the checkpoint's own directory, so no other process
    fetches a checkpoint to check it. A refused checkpoint is not stored, and its job fails for good with the
    check's message; the job's earlier checkpoint stays recorded, and so do the job's record files."""
    queue = memory_coordination(lease_seconds=30).work_queue
    backends = registered_backends(tmp_path / "artifacts")
    queue.append(NewWork(JOBS_QUEUE, "j1", "s", job_kind("steps"), job_payload(replace(job("j1"), component="w"))))
    checks = RefuseSecondStep()
    # The job still runs when its second checkpoint is refused, so it has not stored its records yet.
    component = HeldSteps(tmp_path / "jobs")
    worker = TrainingJobWorker(
        component,
        queue,
        MemoryDataStore(LocalTaskFileStore(tmp_path / "task-files")),
        backends,
        checks=checks,
        worker_id="training-1",
        poll_seconds=0.02,
    )
    assert worker.run_once()
    item = queue.read(JOBS_QUEUE, "j1")
    assert item.state == "failed"
    assert item.error == "the scenario's check refused checkpoint 2 of job j1: the weights of step 2 are refused"
    assert [entry["content"] for entry in item.progress] == ["j1:1"]
    assert item.metadata[STORED_CONTENT_KEY] == ["j1:1"]
    assert checks.seen == [("s", "w"), ("s", "w")]
    with pytest.raises(ArtifactError):
        backends("s").materialize_content("j1:2")
    records = backends("s").materialize_content(job_records_content("j1"))
    assert (records / "weights.txt").read_text() == "step 1"
    component.release.set()
    worker.stop()


@pytest.mark.unit
def test_control_reads_a_jobs_progress_and_records_from_the_store_and_asks_for_an_archive(tmp_path):
    """Control reads no directory of the training service: the worker copies the running job's progress into the
    job's item and stores the ended job's record files, and the training service archives its own directories."""
    queue = memory_coordination(lease_seconds=30).work_queue
    component = StepComponent(tmp_path / "jobs")
    component.progress["j1"] = {"phase": "proposing"}
    data_store = MemoryDataStore(LocalTaskFileStore(tmp_path / "task-files"))
    backends = registered_backends(tmp_path / "artifacts")
    view = JobQueueComponent(StepFacts(), queue)
    seen: list[Mapping[str, Any] | None] = []
    status = component.status

    def status_and_look(job_id: str) -> JobStatus:
        # What control reads while the job runs, after the worker's first look at the job.
        seen.append(view.job_progress(job_id))
        return status(job_id)

    component.status = status_and_look
    queue.append(NewWork(JOBS_QUEUE, "j1", "s", job_kind("steps"), job_payload(job("j1"))))
    worker = TrainingJobWorker(
        component, queue, data_store, backends, checks=ACCEPT_ALL, worker_id="training-1", poll_seconds=0.02
    )
    assert view.job_progress("j1") is None
    assert worker.run_once()
    assert queue.read(JOBS_QUEUE, "j1").state == "finished"
    assert {"phase": "proposing"} in seen
    # The ended job reports no progress, and its record files are stored content.
    assert view.job_progress("j1") is None
    records = backends("s").materialize_content(job_records_content("j1"))
    assert (records / "weights.txt").read_text() == "step 1"

    # A deleted scenario: control only asks, however long the training service was away.
    view.archive_scenario("s")
    (asked,) = queue.items(CHECKS_QUEUE)
    assert (asked.scenario, asked.payload) == ("s", {"check": "archive"}) and component.archived == []
    checks = TrainingCheckWorker(component, queue, data_store, None, worker_id="checks-1")
    assert checks.run_once()
    assert component.archived == ["s"] and queue.read(CHECKS_QUEUE, asked.id).state == "finished"
    worker.stop()


@pytest.mark.unit
def test_the_sweep_discards_the_stored_checkpoints_of_an_ended_job_that_no_commit_took(worked):
    queue, _component, worker = worked
    backend = worker.backends("s")
    view = JobQueueComponent(StepFacts(), queue)
    for job_id in ("j1", "j2"):
        submit(view, job(job_id))
        wait_for(lambda job_id=job_id: view.status(job_id), "finished")
    ended_at = queue.read(JOBS_QUEUE, "j2").ended_at
    # A commit took and discarded one checkpoint. The trainer got a new start, so nobody commits the others.
    backend.discard_content("j1:1")
    assert discard_ended_content(queue, worker.backends, ended_before=0.0, give_up_before=0.0) == ()
    assert backend.materialize_content("j1:2").is_dir()
    assert discard_ended_content(queue, worker.backends, ended_before=ended_at + 1, give_up_before=0.0) == ()
    for name in ("j1:1", "j1:2", "j2:1", "j2:2"):
        with pytest.raises(ArtifactError):
            backend.materialize_content(name)


@pytest.mark.integration
@pytest.mark.parametrize("deleted", ["before the worker takes the job", "while the worker stores"])
def test_a_job_of_a_deleted_scenario_does_not_run_on_and_leaves_no_content(
    deleted: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The delete of a scenario cancels the jobs that it finds in the queue, and the cycle that added a job after
    that look cancels it when it returns. Here the cycle's process stopped in between, so the job stays queued
    after the delete. The worker finds no registration of the scenario, when it takes the job or after it stored
    for a delete that ended meanwhile: it cancels the job, which records nothing, and discards what it stored."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    remote = tmp_path / "artifacts.git"
    control = GitLFSRepositoryBackend.factory(remote, work_dir=tmp_path / "control", cache_dir=tmp_path / "c-cache")
    root = tmp_path / "root"
    root.mkdir()
    (root / "config.json").write_text("{}")
    control("s").register(Artifact.local(root), metadata={})

    class DeletedWhileStoring(GitLFSRepositoryBackend):
        """The delete ends while the worker stores the job's first content."""

        def store_content(self, name: str, directory: Path) -> None:
            super().store_content(name, directory)
            if control.has_registration("s"):
                control.archive_registration("s")

    backend_type = GitLFSRepositoryBackend if deleted.startswith("before") else DeletedWhileStoring
    backends = backend_type.factory(remote, work_dir=tmp_path / "training", cache_dir=tmp_path / "t-cache")
    queue = memory_coordination(lease_seconds=30).work_queue
    component = StepComponent(tmp_path / "jobs")
    data_store = MemoryDataStore(LocalTaskFileStore(tmp_path / "task-files"))
    queue.append(NewWork(JOBS_QUEUE, "j1", "s", job_kind("steps"), job_payload(job("j1"))))
    if deleted.startswith("before"):
        control.archive_registration("s")
    worker = TrainingJobWorker(
        component, queue, data_store, backends, checks=ACCEPT_ALL, worker_id="training-1", poll_seconds=0.02
    )
    assert worker.run_once()

    item = queue.read(JOBS_QUEUE, "j1")
    assert (item.state, item.error, item.progress) == ("failed", SCENARIO_DELETED_ERROR, ())
    assert (
        "/contents/"
        not in subprocess.run(["git", "--git-dir", str(remote), "show-ref"], capture_output=True, text=True).stdout
    )
    if deleted.startswith("before"):
        # The component never got the job.
        with pytest.raises(KeyError):
            component.status("j1")
    worker.stop()


class LosingQueue:
    """The work queue of a worker that loses its job between the store of a checkpoint and its record."""

    def __init__(self, queue) -> None:
        self.queue = queue

    def __getattr__(self, name: str):
        return getattr(self.queue, name)

    def append_progress(self, queue, item_id, worker_id, entry, *, number):
        self.queue.cancel(queue, item_id, "cancelled")
        raise WorkLeaseLostError(f"job {item_id} is no longer held by {worker_id}")


@pytest.mark.unit
def test_content_that_a_worker_stored_and_could_not_record_goes_with_the_jobs_item(tmp_path):
    """The item names the content before the worker stores it, so the sweep finds content that no progress entry
    names: of a job that was cancelled while the worker stored its checkpoint."""
    queue = memory_coordination(lease_seconds=30).work_queue
    backends = registered_backends(tmp_path / "artifacts")
    queue.append(NewWork(JOBS_QUEUE, "j1", "s", job_kind("steps"), job_payload(job("j1"))))
    worker = TrainingJobWorker(
        StepComponent(tmp_path / "jobs"),
        LosingQueue(queue),
        MemoryDataStore(LocalTaskFileStore(tmp_path / "task-files")),
        backends,
        checks=ACCEPT_ALL,
        worker_id="training-1",
        poll_seconds=0.02,
    )
    assert worker.run_once()
    item = queue.read(JOBS_QUEUE, "j1")
    assert (item.state, item.progress, item.metadata[STORED_CONTENT_KEY]) == ("failed", (), ["j1:1"])
    # The worker discarded what it could not record.
    with pytest.raises(ArtifactError):
        backends("s").materialize_content("j1:1")
    # A worker that stopped hard between the store and the record discards nothing: the sweep finds the name.
    files = tmp_path / "files"
    files.mkdir()
    backends("s").store_content("j1:1", files)
    assert discard_ended_content(queue, backends, ended_before=item.ended_at + 1, give_up_before=0.0) == ()
    with pytest.raises(ArtifactError):
        backends("s").materialize_content("j1:1")
    worker.stop()


class FailingBackends(RepositoryBackendFactory):
    """Opens the artifact stores of ``backends``, and raises for the scenario ``down``, as a plugin's backend can
    raise any error."""

    def __init__(self, backends: RepositoryBackendFactory) -> None:
        self.backends = backends
        self.down = True

    def __call__(self, scenario: str):
        if scenario == "down" and self.down:
            raise ConnectionError("the artifact store did not answer")
        return self.backends(scenario)


@pytest.mark.unit
def test_a_job_item_stays_until_its_content_is_discarded_and_does_not_stop_the_sweep_of_the_others(tmp_path):
    queue = memory_coordination(lease_seconds=30).work_queue
    stores = registered_backends(tmp_path / "artifacts")
    backends = FailingBackends(stores)
    files = tmp_path / "files"
    files.mkdir()
    for scenario, job_id in (("down", "a"), ("up", "b")):
        stores(scenario).store_content(f"{job_id}:1", files)
        queue.append(NewWork(JOBS_QUEUE, job_id, scenario, job_kind("steps"), job_payload(job(job_id))), now=0)
        queue.write_metadata(JOBS_QUEUE, job_id, STORED_CONTENT_KEY, [f"{job_id}:1"])
        queue.cancel(JOBS_QUEUE, job_id, "cancelled")
    queue.append(NewWork(CHECKS_QUEUE, "other", "up", "training_check:steps", {"check": "archive"}), now=0)
    queue.cancel(CHECKS_QUEUE, "other", "cancelled")
    ended_at = queue.read(JOBS_QUEUE, "a").ended_at
    sweeper = EndedWorkSweeper(queue, backends, retention_seconds=100.0)

    def sweep(now: float) -> int:
        kept = discard_ended_content(queue, backends, ended_before=now - 100, give_up_before=now - 200)
        return queue.delete_final(finished_before=now - 100, keep={JOBS_QUEUE: kept})

    # The store of one scenario is down: its item stays with the names, and every other ended item goes.
    assert sweep(ended_at + 101) == 2
    assert [item.id for item in queue.items(JOBS_QUEUE)] == ["a"] and not queue.items(CHECKS_QUEUE)
    with pytest.raises(ArtifactError):
        stores("up").materialize_content("b:1")
    # The next sweep after the store is back discards the content and deletes the item.
    backends.down = False
    assert sweep(ended_at + 102) == 1 and not queue.items(JOBS_QUEUE)
    with pytest.raises(ArtifactError):
        stores("down").materialize_content("a:1")

    # A store that never comes back: the item goes after one more retention time, so no item lives forever.
    backends.down = True
    queue.append(NewWork(JOBS_QUEUE, "c", "down", job_kind("steps"), job_payload(job("c"))), now=0)
    queue.write_metadata(JOBS_QUEUE, "c", STORED_CONTENT_KEY, ["c:1"])
    queue.cancel(JOBS_QUEUE, "c", "cancelled")
    ended_at = queue.read(JOBS_QUEUE, "c").ended_at
    assert sweep(ended_at + 101) == 0 and sweep(ended_at + 201) == 1
    assert sweeper.sweep() == 0


@pytest.mark.unit
def test_a_job_check_of_a_scenario_waits_for_the_archive_of_the_scenario_before_it(tmp_path):
    """An archive never moves the directory of a scenario that was created again under the name: the queue gives
    the checks and the archives of one scenario out one at a time, in order, to whichever worker asks."""
    queue = memory_coordination(lease_seconds=30).work_queue
    view = JobQueueComponent(StepFacts(), queue)
    view.archive_scenario("s")
    # The scenario exists again, and its first job asks for its check.
    with pytest.raises(JobCheckPendingError):
        view.submit(job("j1"))
    archive, check = queue.items(CHECKS_QUEUE)
    assert archive.serial_key == check.serial_key == "scenario:s"
    # Another training service instance asks for work while the first holds the archive: it gets no check.
    assert queue.take(CHECKS_QUEUE, "instance-1", ["training_check:steps"]).id == archive.id
    assert queue.take(CHECKS_QUEUE, "instance-2", ["training_check:steps"]) is None
    queue.start(CHECKS_QUEUE, archive.id, "instance-1")
    queue.finish(CHECKS_QUEUE, archive.id, "instance-1", {"refusal": None})
    assert queue.take(CHECKS_QUEUE, "instance-2", ["training_check:steps"]).id == check.id


@pytest.mark.unit
def test_the_sweep_opens_no_artifact_store_for_a_job_that_stored_nothing(tmp_path):
    class NoStore(RepositoryBackendFactory):
        def __call__(self, scenario: str):
            raise AssertionError("a job with no stored content needs no artifact store")

    queue = memory_coordination(lease_seconds=30).work_queue
    queue.append(NewWork(JOBS_QUEUE, "j1", "s", job_kind("steps"), job_payload(job("j1"))), now=0)
    queue.cancel(JOBS_QUEUE, "j1", "cancelled")
    ended_at = queue.read(JOBS_QUEUE, "j1").ended_at
    assert discard_ended_content(queue, NoStore(), ended_before=ended_at + 1, give_up_before=0.0) == ()


@pytest.mark.unit
def test_the_job_listing_shows_only_checkpoints_of_the_current_form(tmp_path):
    queue = memory_coordination(lease_seconds=30).work_queue
    queue.append(NewWork(JOBS_QUEUE, "old", "s", job_kind("steps"), job_payload(job("old"))))
    queue.take(JOBS_QUEUE, "training-0", [job_kind("steps")])
    old = {"step": 1, "path": "/old/step-1", "artifact_path": "/old/step-1", "metrics": {}, "changed_parts": []}
    queue.append_progress(JOBS_QUEUE, "old", "training-0", old, number=0)
    current = checkpoint_json(SavedCheckpoint(2, "/old/step-2", None, content="old:2"))
    queue.append_progress(JOBS_QUEUE, "old", "training-0", current, number=1)
    assert job_json(queue.read(JOBS_QUEUE, "old"))["checkpoints"] == [current]


@pytest.mark.unit
def test_a_queue_component_serves_the_parameters_schema_of_its_facts(tmp_path):
    class Declaring(StepFacts):
        def parameters_schema(self, kind):
            return {"title": kind}

    queue = memory_coordination().work_queue
    assert JobQueueComponent(Declaring(), queue).parameters_schema("task") == {"title": "task"}
    assert JobQueueComponent(StepFacts(), queue).parameters_schema("task") is None


@pytest.mark.unit
def test_a_listed_job_carries_its_parameters_request_data_and_times(tmp_path):
    queue = memory_coordination(lease_seconds=300).work_queue
    request = TrainingRequest("prefer tests", "session-1", "artifact-1", id="r1", parameters={"learning_rate": 1e-5})
    sample = policy_trajectory("s-j1", (1, 2, 3), (1, 1), (-0.1, -0.2), 1.0, artifact_id="v1")
    data = JobData("s:request:r1", (sample, sample), request=request)
    parameters = {"objective": "sft", "learning_rate": 1e-5}
    instructed = TrainingJob("j1", "steps", "steps:base", job_data_ref(data), parameters, scenario="s")
    summary = {"id": "r1", "text": "prefer tests", "parameters": {"learning_rate": 1e-5}}
    queue.append(
        NewWork(JOBS_QUEUE, "j1", "s", job_kind("steps"), job_payload(instructed, summary), serial_key="s"), now=0
    )

    queued = job_json(queue.read(JOBS_QUEUE, "j1"))
    assert queued["parameters"] == parameters
    assert (queued["sample_count"], queued["sample_kind"]) == (2, "trajectory")
    assert queued["request"] == {"id": "r1", "text": "prefer tests", "parameters": {"learning_rate": 1e-5}}
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

    # A job of the auto mode has no request; a task job names its kind, and an empty job has none.
    automatic = TrainingJob(
        "j2", "steps", "steps:base", job_data_ref(JobData("s:data:1", (TaskSample(tmp_path),))), {}, scenario="s"
    )
    empty = TrainingJob("j3", "steps", "steps:base", job_data_ref(JobData("s:data:2")), {}, scenario="s")
    for value in (automatic, empty):
        queue.append(NewWork(JOBS_QUEUE, value.job_id, "s", job_kind("steps"), job_payload(value), serial_key="s"))
    listed = {listed_job["id"]: listed_job for listed_job in scenario_jobs(queue, "s")}
    assert (listed["j2"]["request"], listed["j2"]["sample_kind"], listed["j2"]["parameters"]) == (None, "task", {})
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
    ref = SnapshotRef(revision="r1", parent_revision=None)
    context = TrainingExperimentContext(
        scenario="s", recipe="recipe", commit_number=1, source_artifacts=ArtifactSelection({"weights": ref.revision})
    )
    return TrainingExperimentEvent(context, context.source_artifacts, {}, "committed", training_job_id=job_id)


@pytest.mark.unit
def test_control_writes_the_run_that_records_a_job_into_the_jobs_item():
    queue = memory_coordination().work_queue
    queue.append(NewWork(JOBS_QUEUE, "j1", "s", job_kind("steps"), job_payload(job("j1")), serial_key="s"))
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
    sequence = queue.read(JOBS_QUEUE, "j1").changed_sequence
    recorder.record(job_event("j1"))
    assert queue.read(JOBS_QUEUE, "j1").changed_sequence == sequence

    with pytest.raises(TypeError):
        JobExperimentRecorder(recorder, queue)


@pytest.mark.unit
def test_a_job_has_no_experiment_when_the_tracker_has_no_run():
    queue = memory_coordination().work_queue
    queue.append(NewWork(JOBS_QUEUE, "j1", "s", job_kind("steps"), job_payload(job("j1")), serial_key="s"))
    JobExperimentRecorder(RunTracker(open_run=False), queue).record(job_event("j1"))
    JobExperimentRecorder(ExperimentTracker(), queue).record(job_event("j1"))
    assert scenario_jobs(queue, "s")[0]["experiment"] is None


@pytest.mark.unit
def test_the_job_of_a_request_that_was_cancelled_meanwhile_is_cancelled_when_it_is_queued(worked):
    queue, _component, worker = worked
    # The jobs stay queued: only the checks are answered.
    worker.stop()
    requests = ScenarioTrainingRequests(queue, "s")
    view = JobQueueComponent(StepFacts(), queue)
    sample = policy_trajectory("s-j", (1, 2, 3), (1, 1), (-0.1, -0.2), 1.0, artifact_id="v1")

    def request_job(job_id: str, request_id: str) -> TrainingJob:
        request = TrainingRequest("prefer tests", "session-1", "artifact-1", id=request_id)
        requests.file(request_id, request.to_dict())
        data = JobData(f"s:request:{request_id}", (sample,), request=request)
        return TrainingJob(job_id, "steps", "steps:base", job_data_ref(data), {}, scenario="s", request_id=request_id)

    # The client cancelled the request while control built the data of its job.
    late = request_job("late", "r1")
    assert requests.cancel("r1") is not None and requests.failures() == {"r1": REQUEST_CANCELLED_ERROR}
    submit(view, late)
    assert (view.status("late").state, view.status("late").error) == ("failed", JOB_CANCELLED_ERROR)

    # A cancel of the jobs of a request leaves the jobs of other requests and the jobs without a request.
    submit(view, request_job("kept", "r2"))
    submit(view, job("automatic"))
    assert cancel_request_jobs(queue, "s", "missing") is None
    assert cancel_request_jobs(queue, "s", "r1")["id"] == "late"
    assert [view.status(job_id).state for job_id in ("kept", "automatic")] == ["running", "running"]
    assert cancel_request_jobs(queue, "s", "r2")["error"] == JOB_CANCELLED_ERROR
    assert requests.cancel("missing") is None
