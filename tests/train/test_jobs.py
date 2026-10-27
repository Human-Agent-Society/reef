"""The job loop: plan, submission, checkpoint decisions and job ends, against a fake training component."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from tests.support._threshold_processor import ThresholdJobData, threshold_pipeline
from tests.support._trajectories import policy_trajectory

from reef.core import AgentRecord, RequestType
from reef.core.evaluation import CheckpointCandidate, EvaluationResult, SelectionDecision
from reef.core.job_data import JobOutcome, TaskSample
from reef.evaluation.evaluators import AlwaysSelectCheckpointPlugin
from reef.pipeline_context import PipelineContext
from reef.runtime.jobs import JobStatus, ReleaseFormat, SavedCheckpoint, TrainingComponent, TrainingJob
from reef.storage.sqlite import SQLiteRecordStore
from reef.train.jobs import (
    ArtifactEvaluation,
    EvaluationJob,
    FixedJobPlan,
    JobLoop,
    JobLoopState,
    JobPlan,
    JobPlanContext,
    JobRequest,
    PendingEvaluation,
    RunningJob,
)
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

    def import_start(self, release: Path) -> str | None:
        self.imported.append(release)
        return f"fake:{release.name}"

    def submit(self, job: TrainingJob) -> None:
        self.jobs.append(job)
        self.statuses.setdefault(job.job_id, JobStatus(job.job_id, "running"))

    def status(self, job_id: str) -> JobStatus:
        return self.statuses[job_id]

    def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)

    def fresh_start(self, start: str | None) -> str | None:
        return f"fresh:{start}"


class RejectAll(AlwaysSelectCheckpointPlugin):
    def decide(self, candidate, evaluation: EvaluationResult) -> SelectionDecision:
        return SelectionDecision(selected=False, policy="test", reason="rejected", evaluation=evaluation)


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


def loop_with(component, *, plan=None, evaluator=None, scenario="math"):
    return JobLoop(
        component,
        plan or FixedJobPlan(JobRequest({"method": "sft"})),
        evaluator or AlwaysSelectCheckpointPlugin(),
        scenario=scenario,
    )


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
            JobLoopState(finished_jobs=finished_jobs, last_job=last_job)
    with pytest.raises(ValueError, match="finished_jobs"):
        JobPlanContext(batch(), True, FINISHED)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="JobData"):
        JobPlanContext("b", 0, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="JobOutcome"):
        JobLoopState(finished_jobs=1, last_job=FINISHED.to_dict())  # type: ignore[arg-type]


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
def test_a_checkpoint_candidate_id_names_its_job_and_step():
    candidate = CheckpointCandidate(candidate_id="j:2", training_job_id="j", step=2, release_path="/c")
    assert candidate.training_metrics == {}
    with pytest.raises(ValueError, match="'<training_job_id>:<step>'"):
        CheckpointCandidate(candidate_id="j:1", training_job_id="j", step=2, release_path="/c")
    with pytest.raises(ValueError, match="release_path"):
        CheckpointCandidate(candidate_id="j:2", training_job_id="j", step=2, release_path="")
    with pytest.raises(ValueError, match="step"):
        CheckpointCandidate(candidate_id="j:-1", training_job_id="j", step=-1, release_path="/c")


@pytest.mark.unit
def test_a_ready_batch_is_one_submitted_job_whose_commit_consumes_it(component):
    loop = loop_with(component)
    submitted = loop.submit(JobLoopState(), batch())
    result = submitted.result
    [job] = component.jobs
    assert (job.backend, job.start, job.parameters) == ("fake", None, {"method": "sft"})
    assert submitted.ended is None
    assert result.artifact is None and result.training_job_id == job.job_id
    assert result.metrics["training_job"] == {"job_id": job.job_id, "backend": "fake", "event": "submitted"}
    assert JobLoopState.from_dict(result.state).job == RunningJob(job.job_id)
    # The same rows are the same job, whatever the batch's number; another scenario's are another job.
    assert loop.submit(JobLoopState(), JobData("renumbered", batch().samples)).result.training_job_id == job.job_id
    other_scenario = loop_with(component, scenario="code").submit(JobLoopState(), batch())
    assert other_scenario.result.training_job_id != job.job_id
    # The next job of the same trainer is another job, even with the same data from the same start.
    later = loop.submit(JobLoopState(finished_jobs=1, last_job=FINISHED), batch())
    assert later.result.training_job_id != job.job_id


@pytest.mark.unit
def test_one_job_runs_at_a_time_and_only_a_running_job_is_polled_or_ended(component):
    loop = loop_with(component)
    running = JobLoopState(job=RunningJob("job-1"))
    with pytest.raises(ValueError, match="one job at a time"):
        loop.submit(running, batch())
    assert component.jobs == []
    with pytest.raises(ValueError, match="only a running job"):
        loop.poll(JobLoopState())
    with pytest.raises(ValueError, match="has not ended"):
        loop.end(running, JobStatus("job-1", "running"))
    with pytest.raises(ValueError, match="only a running job ends"):
        loop.end(JobLoopState(), JobStatus("job-1", "finished"))


@pytest.mark.unit
def test_a_plan_that_returns_no_job_request_is_a_type_error(component):
    class NoJob(JobPlan):
        def next_job(self, context: JobPlanContext) -> JobRequest:
            return None  # type: ignore[return-value]

    with pytest.raises(TypeError, match=r"NoJob\.next_job must return a JobRequest"):
        loop_with(component, plan=NoJob()).submit(JobLoopState(), batch())
    assert component.jobs == []


@pytest.mark.unit
def test_a_phased_plan_sends_each_job_its_method_on_the_one_component(component):
    loop = loop_with(component, plan=SupervisedThenPolicy(supervised_jobs=2))
    state = JobLoopState()
    for index in range(3):
        state = JobLoopState.from_dict(loop.submit(state, batch(str(index))).result.state)
        component.statuses[state.job.job_id] = JobStatus(state.job.job_id, "finished")
        state = JobLoopState.from_dict(loop.poll(state).result.state)
    assert [job.parameters["method"]["objective"] for job in component.jobs] == ["sft", "sft", "policy"]
    assert {job.backend for job in component.jobs} == {"fake"}
    assert state.finished_jobs == 3 and state.last_job.job_id == component.jobs[-1].job_id
    # The jobs saved no checkpoint, so only the restarted job starts elsewhere: from the component's fresh start.
    assert [job.start for job in component.jobs] == [None, None, "fresh:None"]
    assert state.start == "fresh:None"


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
def test_each_saved_checkpoint_is_a_candidate_and_the_next_job_continues_from_it(component, tmp_path):
    loop = loop_with(component)
    state = JobLoopState.from_dict(loop.submit(JobLoopState(), batch()).result.state)
    job_id = state.job.job_id
    assert loop.poll(state) is None
    first = SavedCheckpoint(1, str(tmp_path / "1"), str(tmp_path / "1-release"), {"loss": 0.5})
    component.statuses[job_id] = JobStatus(job_id, "running", (first,))
    decision = loop.poll(state)
    assert decision.ended is None
    decided = decision.result
    assert decided.artifact is not None and decided.artifact.local_path == tmp_path / "1-release"
    assert decided.metrics["selected"] is True and decided.metrics["loss"] == 0.5
    assert decided.metrics["selection"]["candidate_id"] == f"{job_id}:1"
    state = JobLoopState.from_dict(decided.state)
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
def test_a_rejected_checkpoint_is_held_for_review_and_training_still_continues_from_it(component, tmp_path):
    loop = loop_with(component, evaluator=RejectAll())
    state = JobLoopState.from_dict(loop.submit(JobLoopState(), batch()).result.state)
    saved = SavedCheckpoint(1, str(tmp_path / "1"), str(tmp_path / "1"))
    component.statuses[state.job.job_id] = JobStatus(state.job.job_id, "running", (saved,))
    decided = loop.poll(state).result
    assert decided.pending and decided.metrics["selected"] is False
    assert decided.artifact is not None and decided.artifact.local_path == tmp_path / "1"
    assert JobLoopState.from_dict(decided.state).start == saved.path


@pytest.mark.unit
def test_a_failed_or_unknown_job_ends_and_keeps_the_start(component):
    loop = loop_with(component)
    state = JobLoopState("/start", RunningJob("lost"))
    ended = JobLoopState.from_dict(loop.poll(state).result.state)
    assert ended.job is None and ended.start == "/start"
    assert ended.last_job.event == "failed" and "no such job" in ended.last_job.error


@pytest.mark.unit
def test_a_rollback_cancels_the_running_job_and_restarts_from_the_imported_start(component):
    loop = loop_with(component)
    state = JobLoopState("/start", RunningJob("job-1"), 1, FINISHED)
    restored = loop.restored(state, "fake:old")
    assert component.cancelled == ["job-1"]
    assert restored == JobLoopState("fake:old", None, 1, FINISHED)
    assert loop.restored(JobLoopState(), None) == JobLoopState()


@pytest.mark.unit
def test_the_committed_state_round_trips_and_checks_its_values():
    state = JobLoopState("/s", RunningJob("j", 2), 3, FINISHED)
    assert state.to_dict()["job"] == {"job_id": "j", "checkpoints_handled": 2}
    assert JobLoopState.from_dict(state.to_dict()) == state
    assert JobLoopState.from_dict({}) == JobLoopState()
    # A state that cannot be read is refused, not taken for an untrained scenario.
    with pytest.raises(ValueError, match="keys"):
        JobLoopState.from_dict({"start": "/s"})
    with pytest.raises(ValueError, match="job_id and checkpoints_handled"):
        JobLoopState.from_dict({**state.to_dict(), "job": {"job_id": "j"}})
    with pytest.raises(ValueError, match="start"):
        JobLoopState(start="")
    with pytest.raises(TypeError, match="RunningJob"):
        JobLoopState(job={"job_id": "j"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="checkpoints_handled"):
        RunningJob("j", -1)
    with pytest.raises(ValueError, match="job_id"):
        RunningJob("")
    evaluating = JobLoopState("/s", None, 3, FINISHED, (PendingEvaluation("e1", "j:1"),))
    assert JobLoopState.from_dict(evaluating.to_dict()) == evaluating
    with pytest.raises(ValueError, match="distinct job IDs"):
        JobLoopState(evaluations=(PendingEvaluation("e1", "j:1"), PendingEvaluation("e1", "j:2")))
    with pytest.raises(ValueError, match="job_id and candidate_id"):
        JobLoopState.from_dict({**evaluating.to_dict(), "evaluations": [{"job_id": "e1"}]})


@pytest.mark.unit
def test_the_default_evaluation_serves_every_checkpoint_with_its_training_metrics():
    plugin = AlwaysSelectCheckpointPlugin()
    candidate = CheckpointCandidate(
        candidate_id="j:1", training_job_id="j", step=1, release_path="/c", training_metrics={"loss": 1.0}
    )
    decision = plugin.evaluate_and_decide(candidate)
    # The commit already carries the training metrics, so the evaluation adds none.
    assert decision.evaluation.metrics == {}
    assert decision.selected


@pytest.mark.unit
def test_a_checkpoint_decision_refuses_metrics_that_would_overwrite_each_other(component, tmp_path):
    class ScoredEvaluation(AlwaysSelectCheckpointPlugin):
        def evaluate(self, candidate) -> EvaluationResult:
            return EvaluationResult("scored", "1", {"loss": 0.1})

    loop = loop_with(component, evaluator=ScoredEvaluation())
    state = JobLoopState.from_dict(loop.submit(JobLoopState(), batch()).result.state)
    saved = SavedCheckpoint(1, str(tmp_path / "1"), str(tmp_path / "1"), {"loss": 0.5})
    with pytest.raises(ValueError, match=r"share the keys \['loss'\]"):
        loop.decide(state, saved)
    reserved = SavedCheckpoint(1, str(tmp_path / "1"), str(tmp_path / "1"), {"selected": 1})
    with pytest.raises(ValueError, match=r"share the keys \['selected'\]"):
        loop_with(component).decide(state, reserved)


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
def test_a_rollback_release_is_imported_into_the_trainer_component(component):
    trainer = job_trainer(JobLoopState("/start"), component)
    assert trainer.import_rollback_release(Path("/releases/old")) == "fake:old"
    assert component.imported == [Path("/releases/old")]
    observer = Trainer.build("math", SQLiteRecordStore(), pipeline_factory=threshold_pipeline)
    with pytest.raises(RuntimeError, match="only a weight trainer"):
        observer.import_rollback_release(Path("/releases/old"))


@pytest.mark.unit
def test_a_rollback_start_cancels_the_running_job_and_is_refused_while_a_commit_is_pending(component):
    trainer = job_trainer(JobLoopState("/start", RunningJob("job-1")), component)
    trainer.apply_rollback_start("fake:old")
    assert component.cancelled == ["job-1"]
    assert JobLoopState.from_dict(trainer.state) == JobLoopState("fake:old")

    pending = job_trainer(JobLoopState(job=RunningJob("job-2")), component)
    component.statuses["job-2"] = JobStatus("job-2", "finished")
    assert not pending.has_pending_commit
    assert pending.run_once() is not None
    assert pending.has_pending_commit
    with pytest.raises(RuntimeError, match="commit is pending"):
        pending.apply_rollback_start("fake:old")


@pytest.mark.unit
def test_a_poll_result_is_dropped_when_a_rollback_ran_while_the_poll_ran(tmp_path):
    saved = SavedCheckpoint(0, str(tmp_path / "0"), str(tmp_path / "0"))

    class RollbackDuringStatus(FakeComponent):
        trainer: Trainer

        def status(self, job_id: str) -> JobStatus:
            # A rollback commits while the trainer waits on the component.
            self.trainer.apply_rollback_start("fake:old")
            return JobStatus(job_id, "running", (saved,))

    component = RollbackDuringStatus()
    trainer = job_trainer(JobLoopState("/start", RunningJob("job-1")), component)
    component.trainer = trainer
    assert trainer.run_once() is None
    assert not trainer.has_pending_commit and component.cancelled == ["job-1"]
    assert JobLoopState.from_dict(trainer.state) == JobLoopState("fake:old")


@pytest.mark.unit
def test_a_restarted_trainer_polls_its_running_job_before_it_reserves_a_batch(component, tmp_path):
    trainer = job_trainer(JobLoopState(job=RunningJob("job-1")), component)
    component.statuses["job-1"] = JobStatus("job-1", "running")
    assert trainer.job_running
    assert trainer.run_once() is None
    assert trainer.pending_job_data is None and component.jobs == []
    saved = SavedCheckpoint(0, str(tmp_path / "0"), str(tmp_path / "0"))
    component.statuses["job-1"] = JobStatus("job-1", "running", (saved,))
    decided = trainer.run_once()
    assert decided is not None and decided.metrics["training_job"]["event"] == "checkpoint"
    assert trainer.pending_job_data is None
    trainer.commit(trainer.prepare_commit(decided))
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
    decided = trainer.run_once()
    trainer.commit(trainer.prepare_commit(decided))
    assert ended.outcomes == [], "neither the submission nor a checkpoint decision ends the job"

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


class EvalTasks(ArtifactEvaluation):
    """Evaluates each artifact on one task, or on none when ``empty``."""

    def __init__(self, tmp_path: Path, *, empty: bool = False) -> None:
        self.task = TaskSample(tmp_path, metadata={"task": {"name": "held-out", "split": "eval"}})
        self.empty = empty

    def evaluation_job(self, candidate):
        if self.empty:
            return None
        return EvaluationJob(
            JobData(f"{candidate.candidate_id}:evaluation", (self.task,)), {"evaluation_of": candidate.candidate_id}
        )


@pytest.mark.unit
def test_every_artifact_is_evaluated_from_its_own_checkpoint_and_its_result_is_committed(component, tmp_path):
    loop = JobLoop(
        component,
        FixedJobPlan(JobRequest({"method": "sft"})),
        AlwaysSelectCheckpointPlugin(),
        scenario="math",
        artifact_evaluation=EvalTasks(tmp_path),
    )
    state = JobLoopState.from_dict(loop.submit(JobLoopState(), batch()).result.state)
    training = state.job.job_id
    first, second = (SavedCheckpoint(step, str(tmp_path / str(step)), str(tmp_path / str(step))) for step in (1, 2))
    component.statuses[training] = JobStatus(training, "running", (first, second))

    decided = loop.poll(state).result
    state = JobLoopState.from_dict(decided.state)
    (evaluation,) = state.evaluations
    submitted = component.jobs[-1]
    assert (evaluation.candidate_id, submitted.job_id) == (f"{training}:1", evaluation.job_id)
    assert submitted.start == first.path and submitted.parameters == {"evaluation_of": f"{training}:1"}
    assert decided.metrics["artifact_evaluation"] == {"job_id": evaluation.job_id, "event": "submitted"}
    state = JobLoopState.from_dict(loop.poll(state).result.state)
    assert [pending.candidate_id for pending in state.evaluations] == [f"{training}:1", f"{training}:2"]
    assert component.jobs[-1].start == second.path, "each artifact is scored from its own checkpoint"

    # The training job ends while both evaluations run; the trainer polls them alone afterwards.
    component.statuses[training] = JobStatus(training, "finished", (first, second))
    state = JobLoopState.from_dict(loop.poll(state).result.state)
    assert state.job is None and len(state.evaluations) == 2
    assert loop.poll(state) is None
    later = state.evaluations[1].job_id
    component.statuses[later] = JobStatus(later, "finished", metrics={"evaluation_reward_mean": 0.75})
    recorded = loop.poll(state).result
    assert recorded.artifact is None and recorded.metrics["artifact_evaluation"] == {
        "candidate_id": f"{training}:2",
        "job_id": later,
        "event": "finished",
        "metrics": {"evaluation_reward_mean": 0.75},
    }
    state = JobLoopState.from_dict(recorded.state)
    earlier = state.evaluations[0].job_id
    component.statuses[earlier] = JobStatus(earlier, "failed", error="the sandbox did not start")
    failed = loop.poll(state).result
    assert failed.metrics["artifact_evaluation"]["error"] == "the sandbox did not start"
    assert JobLoopState.from_dict(failed.state).evaluations == ()
    with pytest.raises(ValueError, match="polls only a running job or an artifact evaluation"):
        loop.poll(JobLoopState.from_dict(failed.state))


@pytest.mark.unit
def test_an_artifact_with_nothing_to_evaluate_on_submits_no_evaluation(component, tmp_path):
    loop = JobLoop(
        component,
        FixedJobPlan(JobRequest({"method": "sft"})),
        AlwaysSelectCheckpointPlugin(),
        scenario="math",
        artifact_evaluation=EvalTasks(tmp_path, empty=True),
    )
    state = JobLoopState.from_dict(loop.submit(JobLoopState(), batch()).result.state)
    saved = SavedCheckpoint(1, str(tmp_path / "1"), str(tmp_path / "1"))
    component.statuses[state.job.job_id] = JobStatus(state.job.job_id, "running", (saved,))
    decided = loop.poll(state).result
    assert JobLoopState.from_dict(decided.state).evaluations == () and "artifact_evaluation" not in decided.metrics
    assert len(component.jobs) == 1


@pytest.mark.unit
def test_a_trainer_with_no_job_polls_its_artifact_evaluations_until_they_end(component):
    state = JobLoopState("/start", None, 1, FINISHED, (PendingEvaluation("e1", "j0:1"),))
    trainer = Trainer.build(
        "math",
        SQLiteRecordStore(),
        pipeline_factory=threshold_pipeline,
        job_loop=loop_with(component),
        algorithm_state=state.to_dict(),
        default_job_samples=1,
    )
    component.statuses["e1"] = JobStatus("e1", "running")
    assert trainer.needs_polling() and trainer.run_once() is None
    component.statuses["e1"] = JobStatus("e1", "finished", metrics={"evaluation_reward_mean": 1.0})
    recorded = trainer.run_once()
    assert recorded is not None and recorded.metrics["artifact_evaluation"]["candidate_id"] == "j0:1"
    trainer.commit(trainer.prepare_commit(recorded))
    assert JobLoopState.from_dict(trainer.state).evaluations == () and not trainer.needs_polling()
