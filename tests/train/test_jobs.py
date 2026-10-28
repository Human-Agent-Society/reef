"""The job loop: plan, submission, checkpoint publications and job ends, against a fake training component."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from tests.support._threshold_processor import ThresholdJobData, threshold_pipeline
from tests.support._trajectories import policy_trajectory

from reef.artifact import Artifact, BaseArtifactSource
from reef.core import AgentRecord, RequestType
from reef.core.errors import CommitPending, TrainingStartRefused
from reef.core.job_data import JobOutcome
from reef.pipeline_context import PipelineContext
from reef.runtime.jobs import JobStatus, ReleaseFormat, SavedCheckpoint, TrainingComponent, TrainingJob
from reef.storage.sqlite import SQLiteRecordStore
from reef.train.jobs import FixedJobPlan, JobLoop, JobLoopState, JobPlan, JobPlanContext, JobRequest, RunningJob
from reef.train.pipeline import TrainingPipeline
from reef.train.trainer import Trainer
from reef.train.types import JobData


class FakeComponent(TrainingComponent):
    """Records each job; a test sets what ``status`` reports."""

    def __init__(self) -> None:
        self.jobs: list[TrainingJob] = []
        self.statuses: dict[str, JobStatus] = {}
        self.cancelled: list[str] = []
        self.imported: list[Path] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def release_format(self) -> ReleaseFormat:
        return ReleaseFormat("full")

    def base_artifact_source(self) -> BaseArtifactSource:
        return BaseArtifactSource()

    @property
    def base_start(self) -> str:
        return "fake:base"

    def import_start(self, release: Path) -> str:
        self.imported.append(release)
        return f"fake:{release.name}"

    def submit(self, job: TrainingJob) -> None:
        self.jobs.append(job)
        self.statuses.setdefault(job.job_id, JobStatus(job.job_id, "running"))

    def status(self, job_id: str) -> JobStatus:
        return self.statuses[job_id]

    def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)

    def fresh_start(self, start: str) -> str:
        return f"fresh:{start}"


class SupervisedThenPolicy(JobPlan):
    """The SFT objective for the first ``supervised_jobs`` jobs, then a policy objective."""

    def __init__(self, supervised_jobs: int) -> None:
        self.supervised_jobs = supervised_jobs

    def next_job(self, context: JobPlanContext) -> JobRequest:
        objective = "sft" if context.finished_jobs < self.supervised_jobs else "policy"
        # The first policy job restarts the optimizer, the schedule and the objective state.
        return JobRequest({"method": {"objective": objective}}, restart=context.finished_jobs == self.supervised_jobs)


def batch(name: str = "b") -> JobData:
    return JobData(name, (policy_trajectory(f"s-{name}", (1, 2, 3), (1, 1), (-0.1, -0.2), 1.0, "v1"),))


FINISHED = JobOutcome("j0", "fake", "finished")


@pytest.fixture
def component():
    return FakeComponent()


def loop_with(component, *, plan=None, scenario="math"):
    return JobLoop(component, plan or FixedJobPlan(JobRequest({"method": "sft"})), scenario=scenario)


@pytest.mark.unit
def test_a_job_request_keeps_its_own_copy_of_json_parameters():
    parameters = {"method": {"objective": "sft"}}
    request = JobRequest(parameters)
    parameters["method"] = "changed"
    assert request.parameters == {"method": {"objective": "sft"}}
    with pytest.raises(TypeError, match="mapping"):
        JobRequest(["method"])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        JobRequest({"start": Path("/weights")})
    with pytest.raises(ValueError):
        JobRequest({"learning_rate": math.nan})
    with pytest.raises(TypeError, match="restart must be a bool"):
        JobRequest({}, restart=1)  # type: ignore[arg-type]


@pytest.mark.unit
def test_last_job_is_set_exactly_when_a_job_has_ended():
    assert JobPlanContext(batch(), 1, FINISHED).last_job == FINISHED
    assert JobLoopState("/s", None, 1, FINISHED).last_job == FINISHED
    for finished_jobs, last_job in ((1, None), (0, FINISHED)):
        with pytest.raises(ValueError, match="exactly when no job has ended"):
            JobPlanContext(batch(), finished_jobs, last_job)
        with pytest.raises(ValueError, match="exactly when no job has ended"):
            JobLoopState("fake:base", finished_jobs=finished_jobs, last_job=last_job)
    with pytest.raises(ValueError, match="finished_jobs"):
        JobPlanContext(batch(), True, FINISHED)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="JobData"):
        JobPlanContext("b", 0, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="JobOutcome"):
        JobLoopState("/s", finished_jobs=1, last_job=FINISHED.to_dict())  # type: ignore[arg-type]


@pytest.mark.unit
def test_a_job_outcome_has_an_error_exactly_when_the_job_failed():
    failed = JobOutcome("j", "fake", "failed", error="boom")
    assert JobOutcome.from_dict(failed.to_dict()) == failed
    assert FINISHED.to_dict() == {"job_id": "j0", "backend": "fake", "event": "finished", "metrics": {}}
    with pytest.raises(ValueError, match="exactly when the job failed"):
        JobOutcome("j", "fake", "failed")
    with pytest.raises(ValueError, match="exactly when the job failed"):
        JobOutcome("j", "fake", "finished", error="boom")
    with pytest.raises(ValueError, match="finished or failed"):
        JobOutcome("j", "fake", "running")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        JobOutcome("j", "fake", "finished", {"loss": math.nan})
    with pytest.raises(ValueError, match="keys"):
        JobOutcome.from_dict({**FINISHED.to_dict(), "extra": 1})


@pytest.mark.unit
def test_a_ready_batch_is_one_submitted_job_whose_commit_consumes_it(component):
    loop = loop_with(component)
    submitted = loop.submit(JobLoopState("fake:base"), batch())
    result = submitted.result
    [job] = component.jobs
    assert (job.backend, job.start, job.parameters) == ("fake", "fake:base", {"method": "sft"})
    assert submitted.ended is None
    assert result.artifact is None and result.training_job_id == job.job_id
    assert result.metrics["training_job"] == {"job_id": job.job_id, "backend": "fake", "event": "submitted"}
    assert JobLoopState.from_dict(result.state).job == RunningJob(job.job_id)
    # The same rows are the same job, whatever the batch's number; another scenario's are another job.
    assert (
        loop.submit(JobLoopState("fake:base"), JobData("renumbered", batch().samples)).result.training_job_id
        == job.job_id
    )
    other_scenario = loop_with(component, scenario="code").submit(JobLoopState("fake:base"), batch())
    assert other_scenario.result.training_job_id != job.job_id
    # The next job of the same trainer is another job, even with the same data from the same start.
    later = loop.submit(JobLoopState("fake:base", finished_jobs=1, last_job=FINISHED), batch())
    assert later.result.training_job_id != job.job_id


@pytest.mark.unit
def test_one_job_runs_at_a_time_and_only_a_running_job_is_polled_or_ended(component):
    loop = loop_with(component)
    running = JobLoopState("fake:base", job=RunningJob("job-1"))
    with pytest.raises(ValueError, match="one job at a time"):
        loop.submit(running, batch())
    assert component.jobs == []
    with pytest.raises(ValueError, match="only a running job"):
        loop.poll(JobLoopState("fake:base"))
    with pytest.raises(ValueError, match="has not ended"):
        loop.end(running, JobStatus("job-1", "running"))
    with pytest.raises(ValueError, match="only a running job ends"):
        loop.end(JobLoopState("fake:base"), JobStatus("job-1", "finished"))


@pytest.mark.unit
def test_a_plan_that_returns_no_job_request_is_a_type_error(component):
    class NoJob(JobPlan):
        def next_job(self, context: JobPlanContext) -> JobRequest:
            return None  # type: ignore[return-value]

    with pytest.raises(TypeError, match=r"NoJob\.next_job must return a JobRequest"):
        loop_with(component, plan=NoJob()).submit(JobLoopState("fake:base"), batch())
    assert component.jobs == []


@pytest.mark.unit
def test_a_phased_plan_sends_each_job_its_method_on_the_one_component(component):
    loop = loop_with(component, plan=SupervisedThenPolicy(supervised_jobs=2))
    state = JobLoopState("fake:base")
    for index in range(3):
        state = JobLoopState.from_dict(loop.submit(state, batch(str(index))).result.state)
        component.statuses[state.job.job_id] = JobStatus(state.job.job_id, "finished")
        state = JobLoopState.from_dict(loop.poll(state).result.state)
    assert [job.parameters["method"]["objective"] for job in component.jobs] == ["sft", "sft", "policy"]
    assert {job.backend for job in component.jobs} == {"fake"}
    assert state.finished_jobs == 3 and state.last_job.job_id == component.jobs[-1].job_id
    # The jobs saved no checkpoint, so only the restarted job starts elsewhere: from the component's fresh start.
    assert [job.start for job in component.jobs] == ["fake:base", "fake:base", "fresh:fake:base"]
    assert state.start == "fresh:fake:base"


@pytest.mark.unit
def test_a_restart_submits_from_the_fresh_start_and_a_component_without_one_refuses(component):
    restart = FixedJobPlan(JobRequest({"method": "sft"}, restart=True))
    result = loop_with(component, plan=restart).submit(JobLoopState("/c"), batch()).result
    [job] = component.jobs
    assert job.start == "fresh:/c" and result.metrics["training_job"]["restart"] is True
    # The job ID names the fresh start, so a restarted job differs from the same job without a restart.
    assert loop_with(component).submit(JobLoopState("/c"), batch()).result.training_job_id != job.job_id

    class NoRestart(FakeComponent):
        fresh_start = TrainingComponent.fresh_start

    with pytest.raises(ValueError, match="the fake training component cannot restart training from '/c'"):
        loop_with(NoRestart(), plan=restart).submit(JobLoopState("/c"), batch())


@pytest.mark.unit
def test_each_saved_checkpoint_is_published_and_the_next_job_continues_from_it(component, tmp_path):
    loop = loop_with(component)
    state = JobLoopState.from_dict(loop.submit(JobLoopState("fake:base"), batch()).result.state)
    job_id = state.job.job_id
    assert loop.poll(state) is None
    first = SavedCheckpoint(1, str(tmp_path / "1"), str(tmp_path / "1-release"), {"loss": 0.5})
    component.statuses[job_id] = JobStatus(job_id, "running", (first,))
    publication = loop.poll(state)
    assert publication.ended is None
    published = publication.result
    assert published.artifact is not None and published.artifact.local_path == tmp_path / "1-release"
    # The commit carries the training metrics and the checkpoint's place in its job, and no selection.
    assert published.metrics == {
        "loss": 0.5,
        "training_job": {"job_id": job_id, "backend": "fake", "event": "checkpoint", "step": 1},
    }
    state = JobLoopState.from_dict(published.state)
    assert state.start == first.path and state.job.checkpoints_handled == 1
    # Nothing new: the job runs on.
    assert loop.poll(state) is None
    component.statuses[job_id] = JobStatus(job_id, "finished", (first,), metrics={"optimizer_steps": 2})
    end = loop.poll(state)
    ended = end.result
    state = JobLoopState.from_dict(ended.state)
    assert state.job is None and state.finished_jobs == 1 and state.start == first.path
    assert end.ended == state.last_job == JobOutcome(job_id, "fake", "finished", {"optimizer_steps": 2})
    assert ended.metrics["training_job"]["event"] == "finished" and ended.metrics["optimizer_steps"] == 2
    loop.submit(state, batch("next"))
    assert component.jobs[-1].start == first.path


@pytest.mark.unit
def test_a_failed_or_unknown_job_ends_and_keeps_the_start(component):
    loop = loop_with(component)
    state = JobLoopState("/start", RunningJob("lost"))
    ended = JobLoopState.from_dict(loop.poll(state).result.state)
    assert ended.job is None and ended.start == "/start"
    assert ended.last_job.event == "failed" and "no such job" in ended.last_job.error


@pytest.mark.unit
def test_the_committed_state_round_trips_and_checks_its_values():
    state = JobLoopState("/s", RunningJob("j", 2), 3, FINISHED)
    assert state.to_dict()["job"] == {"job_id": "j", "checkpoints_handled": 2}
    assert JobLoopState.from_dict(state.to_dict()) == state
    # A state that cannot be read is refused, not taken for an untrained scenario.
    for unreadable in ({}, {"start": "/s"}):
        with pytest.raises(ValueError, match="keys"):
            JobLoopState.from_dict(unreadable)
    with pytest.raises(ValueError, match="job_id and checkpoints_handled"):
        JobLoopState.from_dict({**state.to_dict(), "job": {"job_id": "j"}})
    for start in ("", None):
        with pytest.raises(ValueError, match="start"):
            JobLoopState(start=start)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="RunningJob"):
        JobLoopState("/s", job={"job_id": "j"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="checkpoints_handled"):
        RunningJob("j", -1)
    with pytest.raises(ValueError, match="job_id"):
        RunningJob("")
    with pytest.raises(ValueError, match="keys"):
        JobLoopState.from_dict({**state.to_dict(), "evaluations": []})


@pytest.mark.unit
def test_a_checkpoint_publication_refuses_a_training_metric_that_it_would_overwrite(component, tmp_path):
    loop = loop_with(component)
    state = JobLoopState.from_dict(loop.submit(JobLoopState("fake:base"), batch()).result.state)
    reserved = SavedCheckpoint(1, str(tmp_path / "1"), str(tmp_path / "1"), {"training_job": 1})
    with pytest.raises(ValueError, match="training metrics name 'training_job'"):
        loop.publish(state, reserved)
    with pytest.raises(ValueError, match="only a running job's checkpoint"):
        loop.publish(JobLoopState("fake:base"), reserved)


def scored_records() -> SQLiteRecordStore:
    """A record store with one inference and its report: one ready batch for the threshold pipeline."""
    records = SQLiteRecordStore()
    records.append(
        AgentRecord.create(
            scenario="math",
            request_type=RequestType.INFERENCE,
            payload={"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]},
            agent_record_id="i1",
        )
    )
    records.append(
        AgentRecord.create(
            scenario="math",
            request_type=RequestType.REPORT,
            payload={"score": 1.0, "references": ["i1"]},
            agent_record_id="r1",
            references=("i1",),
        )
    )
    return records


def job_trainer(state: JobLoopState, component: TrainingComponent) -> Trainer:
    return Trainer.build(
        "math",
        scored_records(),
        pipeline_factory=threshold_pipeline,
        job_loop=loop_with(component),
        algorithm_state=state.to_dict(),
        default_job_samples=1,
    )


@pytest.mark.unit
def test_start_from_imports_the_release_cancels_the_running_job_and_restarts_from_it(component):
    trainer = job_trainer(JobLoopState("/start", RunningJob("job-1"), 1, FINISHED), component)
    trainer.start_from(Artifact.local(Path("/releases/old")), None)
    assert component.imported == [Path("/releases/old")]
    assert component.cancelled == ["job-1"]
    # The jobs that ended stay counted.
    assert JobLoopState.from_dict(trainer.state) == JobLoopState("fake:old", None, 1, FINISHED)


@pytest.mark.unit
def test_start_from_is_refused_without_a_change_when_the_trainer_cannot_start(component, monkeypatch):
    trainer = job_trainer(JobLoopState("/start", RunningJob("job-1")), component)

    def refuse(release):
        raise ValueError("these weights cannot train")

    monkeypatch.setattr(component, "import_start", refuse)
    with pytest.raises(TrainingStartRefused, match="these weights cannot train"):
        trainer.start_from(Artifact.local(Path("/releases/old")), None)
    assert component.cancelled == []
    assert JobLoopState.from_dict(trainer.state) == JobLoopState("/start", RunningJob("job-1"))
    # A trainer that trains nothing has no start to set.
    observer = Trainer.build("math", SQLiteRecordStore(), pipeline_factory=threshold_pipeline)
    with pytest.raises(TrainingStartRefused, match="trains nothing"):
        observer.start_from(Artifact.local(Path("/releases/old")), None)


@pytest.mark.unit
def test_start_from_is_refused_while_a_commit_is_pending(component):
    pending = job_trainer(JobLoopState("fake:base", job=RunningJob("job-2")), component)
    component.statuses["job-2"] = JobStatus("job-2", "finished")
    assert not pending.has_pending_commit
    assert pending.run_once() is not None
    assert pending.has_pending_commit
    with pytest.raises(CommitPending, match="waits for its commit"):
        pending.start_from(Artifact.local(Path("/releases/old")), None)
    assert component.imported == []


@pytest.mark.unit
def test_a_poll_result_is_dropped_when_a_new_start_was_set_while_the_poll_ran(tmp_path):
    saved = SavedCheckpoint(0, str(tmp_path / "0"), str(tmp_path / "0"))

    class StartDuringStatus(FakeComponent):
        trainer: Trainer

        def status(self, job_id: str) -> JobStatus:
            # A new training start lands while the trainer waits on the component.
            self.trainer.start_from(Artifact.local(Path("/releases/old")), None)
            return JobStatus(job_id, "running", (saved,))

    component = StartDuringStatus()
    trainer = job_trainer(JobLoopState("/start", RunningJob("job-1")), component)
    component.trainer = trainer
    assert trainer.run_once() is None
    assert not trainer.has_pending_commit and component.cancelled == ["job-1"]
    assert JobLoopState.from_dict(trainer.state) == JobLoopState("fake:old")


@pytest.mark.unit
def test_a_restarted_trainer_polls_its_running_job_before_it_reserves_a_batch(component, tmp_path):
    trainer = job_trainer(JobLoopState("fake:base", job=RunningJob("job-1")), component)
    component.statuses["job-1"] = JobStatus("job-1", "running")
    assert trainer.job_running
    assert trainer.run_once() is None
    assert trainer.pending_job_data is None and component.jobs == []
    saved = SavedCheckpoint(0, str(tmp_path / "0"), str(tmp_path / "0"))
    component.statuses["job-1"] = JobStatus("job-1", "running", (saved,))
    published = trainer.run_once()
    assert published is not None and published.metrics["training_job"]["event"] == "checkpoint"
    assert trainer.pending_job_data is None
    trainer.commit(trainer.prepare_commit(published))
    assert JobLoopState.from_dict(trainer.state).job == RunningJob("job-1", 1)


@pytest.mark.unit
def test_a_running_job_the_component_does_not_know_ends_failed_and_keeps_the_start(component):
    trainer = job_trainer(JobLoopState("/start", RunningJob("lost")), component)
    ended = trainer.run_once()
    assert ended is not None and ended.metrics["training_job"]["event"] == "failed"
    trainer.commit(trainer.prepare_commit(ended))
    state = JobLoopState.from_dict(trainer.state)
    assert (state.job, state.start, state.finished_jobs) == (None, "/start", 1)
    # With the job ended, the ready batch becomes the next job, from the kept start.
    assert trainer.run_once() is not None
    [job] = component.jobs
    assert job.start == "/start"


class EndedJobs(ThresholdJobData):
    """Threshold job data that keeps each job outcome the pipeline gives it, in order."""

    def __init__(self, context: PipelineContext) -> None:
        super().__init__(context, None, None)
        self.outcomes: list[JobOutcome] = []

    def job_ended(self, outcome: JobOutcome) -> None:
        self.outcomes.append(outcome)


@pytest.mark.unit
def test_the_pipeline_learns_a_jobs_outcome_exactly_when_the_commit_of_its_end_settles(component, tmp_path):
    stages: list[EndedJobs] = []

    def pipeline(context: PipelineContext) -> TrainingPipeline:
        stages.append(EndedJobs(context))
        return threshold_pipeline(context, sample2job=stages[-1])

    trainer = Trainer.build(
        "math", scored_records(), pipeline_factory=pipeline, job_loop=loop_with(component), default_job_samples=1
    )
    [ended] = stages
    submitted = trainer.run_once()
    trainer.commit(trainer.prepare_commit(submitted))
    [job] = component.jobs
    saved = SavedCheckpoint(0, str(tmp_path / "0"), str(tmp_path / "0"))
    component.statuses[job.job_id] = JobStatus(job.job_id, "running", (saved,))
    published = trainer.run_once()
    trainer.commit(trainer.prepare_commit(published))
    assert ended.outcomes == [], "neither the submission nor a checkpoint publication ends the job"

    metrics = {"rollout_rewards": {"task-a": [1.0, 0.0]}}
    component.statuses[job.job_id] = JobStatus(job.job_id, "finished", (saved,), metrics)
    finished = trainer.run_once()
    prepared = trainer.prepare_commit(finished)
    assert ended.outcomes == [], "the end is not durable before its commit settles"
    trainer.commit(prepared)
    assert ended.outcomes == [JobOutcome(job.job_id, "fake", "finished", metrics)]
    trainer.commit(prepared)
    assert len(ended.outcomes) == 1, "a settled commit is not applied again"

    # A restart can fall between the end's commit and the stage taking it: the next trainer gives it again.
    Trainer.build(
        "math",
        scored_records(),
        pipeline_factory=pipeline,
        job_loop=loop_with(component),
        algorithm_state=trainer.algorithm_state_dict(),
        default_job_samples=1,
    )
    assert stages[-1].outcomes == ended.outcomes
