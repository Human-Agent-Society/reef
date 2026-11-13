"""Training jobs through the jobs queue: control adds them, a training worker runs them on the component."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from tests.support.job_data import job_data_ref
from tests.support.training_deployment import LocalFacts
from tests.support.trajectories import policy_trajectory

from reef.artifact import Artifact, ArtifactError, InMemoryRepositoryBackend
from reef.core.artifact_selection import ArtifactSelection
from reef.core.errors import InternalError
from reef.core.job_data import JobData, TaskSample
from reef.core.snapshot import RootSource, SnapshotRef
from reef.core.training_request import TrainingRequest
from reef.core.weight_layout import ArtifactFormat
from reef.observability.base import (
    ExperimentRun,
    ExperimentTracker,
    TrainingExperimentContext,
    TrainingExperimentEvent,
)
from reef.storage.data import MemoryDataStore
from reef.storage.sql_coordination import memory_coordination
from reef.storage.tasks import LocalTaskFileStore
from reef.storage.work_queue import NewWork
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
    UNASKED_CHECK_ERROR,
    UNASKED_CHECK_SECONDS,
    JobExperimentRecorder,
    JobQueueComponent,
    TrainingCheckWorker,
    TrainingJobWorker,
    cancel_request_jobs,
    checkpoint_json,
    job_from_item,
    job_json,
    job_kind,
    job_payload,
    resolved_job,
    scenario_jobs,
)
from reef.train.jobs import JobLoop, JobLoopState, RunningJob
from reef.train.local_jobs import LocalJobComponent
from reef.train.training_requests import REQUEST_CANCELLED_ERROR, ScenarioTrainingRequests

WAIT_SECONDS = 10.0


class StepComponent(LocalJobComponent):
    """Saves two checkpoints per job; a job with ``{"fail": true}`` raises and reports what it reached, and
    ``{"invalid": true}`` is refused."""

    def __init__(self, jobs_root: Path) -> None:
        super().__init__(jobs_root)
        self.closed_count = 0
        self.checked: list[str] = []
        self.start_jobs()

    @property
    def name(self) -> str:
        return "steps"

    @property
    def artifact_format(self) -> ArtifactFormat:
        return ArtifactFormat("full")

    def root_start(self, root: ScenarioRoot) -> str:
        return "steps:base"

    def import_start(self, release: Path) -> str:
        return str(release)

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
            path = str(directory / f"step-{step}")
            self.save_checkpoint(directory, SavedCheckpoint(step, path, path, {"loss": 1.0 / step}))
        return {"steps": 2}

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


@pytest.fixture
def worked(tmp_path):
    queue = memory_coordination(lease_seconds=30).work_queue
    component = StepComponent(tmp_path / "jobs")
    data_store = MemoryDataStore(LocalTaskFileStore(tmp_path / "task-files"))
    worker = TrainingJobWorker(component, queue, data_store, None, worker_id="training-1", poll_seconds=0.02)
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
    queue, _component, _worker = worked
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
    assert job_json(item := queue.read(JOBS_QUEUE, "j1"))["checkpoints"][0]["changed_parts"] == ["step-1"]
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
    # A worker of an older release reported a checkpoint without its changed parts, and still holds the job.
    queue.append(NewWork(JOBS_QUEUE, "old", "s", job_kind("old-release"), job_payload(job("old"))))
    queue.take(JOBS_QUEUE, "training-0", [job_kind("old-release")])
    entry = {"step": 1, "path": "/old/step-1", "artifact_path": "/old/step-1", "metrics": {}}
    queue.append_progress(JOBS_QUEUE, "old", "training-0", entry, number=0)

    state = JobLoopState("steps:base", (RunningJob("old", "steps:base"),))
    ended = JobLoopState.from_dict(loop.poll(state).result.state)
    assert ended.jobs == () and ended.finished_jobs == 1
    assert ended.last_job.event == "failed"
    assert "another release of Reef may have written it" in ended.last_job.error
    assert "changed_parts" in ended.last_job.error
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
        None,
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
    first = SavedCheckpoint(1, "/old/step-1", "/old/step-1", {"loss": 1.0})
    queue.append_progress(JOBS_QUEUE, "j1", "training-0", checkpoint_json(first), number=0)
    queue.requeue_expired(now=time.time() + 60)

    worker = TrainingJobWorker(
        StepComponent(tmp_path / "jobs"),
        queue,
        MemoryDataStore(LocalTaskFileStore(tmp_path / "task-files")),
        None,
        worker_id="training-1",
        poll_seconds=0.02,
    )
    assert worker.run_once()
    item = queue.read(JOBS_QUEUE, "j1")
    assert item.state == "finished"
    assert [entry["step"] for entry in item.progress] == [1, 2]
    worker.stop()


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
    request = TrainingRequest("prefer tests", "session-1", "release-1", id="r1", parameters={"learning_rate": 1e-5})
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
        request = TrainingRequest("prefer tests", "session-1", "release-1", id=request_id)
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
