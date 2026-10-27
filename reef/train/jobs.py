"""The control plane's side of training jobs: the recipe's job plan and one weight trainer's job loop.

A weight trainer does not drive training steps. When a job's data is ready, the
recipe's ``JobPlan`` gives the job's parameters, and the ``JobLoop`` submits
the job to the trainer's ``TrainingComponent``. The trainer commits the
submission, which consumes the records of the job's data. While the job runs, each
checkpoint it saves becomes a ``CheckpointCandidate``: candidate evaluation
decides whether Reef serves it, and the trainer commits the decision. Every
checkpoint becomes a release (an artifact): a selected one serves, and any
other is held for review, for a person to promote. With an
``ArtifactEvaluation``, the decision also submits the artifact's evaluation
job, which starts from the artifact's own checkpoint and trains nothing. The
component runs it after the jobs before it, and the trainer commits its
result when it ends. When the
job ends, the trainer commits its outcome. Training is linear: the next job
starts from the job's last saved checkpoint, whatever evaluation decided.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.evaluation import CandidateEvaluationPlugin, CheckpointCandidate
from reef.core.job_data import JobData, JobOutcome
from reef.runtime.jobs import JobStatus, SavedCheckpoint, TrainingComponent, TrainingJob, job_id_for
from reef.train.types import TrainStepResult

#: The keys of a checkpoint decision's commit metrics that the job loop writes itself.
DECISION_KEYS = frozenset({"selected", "selection", "training_job", "artifact_evaluation"})


def require_count(value: object, name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is a non-negative int (not a bool)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")


@dataclass(frozen=True)
class JobRequest:
    """The parameters of the job a plan asks for, and whether the job restarts training.

    A value holds ``parameters``: a mapping whose values encode as JSON. The
    value keeps its own copy, because the parameters are part of the job's
    identity. ``restart`` is a bool: when it is True, the job starts from the
    weights of the trainer's start with no training state
    (``TrainingComponent.fresh_start``), for example at the switch from SFT to
    RL. A ``JobPlan`` makes the value, and ``JobLoop.submit`` reads it. Reef
    reads none of the parameters; the component's own defaults fill the ones
    a request leaves out.
    """

    parameters: Mapping[str, Any] = field(default_factory=dict)
    restart: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, Mapping):
            raise TypeError("JobRequest.parameters must be a mapping")
        if not isinstance(self.restart, bool):
            raise TypeError("JobRequest.restart must be a bool")
        parameters = dict(self.parameters)
        json.dumps(parameters, allow_nan=False)
        object.__setattr__(self, "parameters", parameters)


@dataclass(frozen=True)
class JobPlanContext:
    """What a plan reads when it chooses the next job.

    ``data`` is the data the job trains. ``finished_jobs`` is the number of
    this trainer's jobs that ended, finished or failed, a non-negative int.
    ``last_job`` is the outcome of the last of them, and it is ``None``
    exactly when ``finished_jobs`` is 0. ``JobLoop.submit`` makes the value
    and ``JobPlan.next_job`` reads it.
    """

    data: JobData
    finished_jobs: int
    last_job: JobOutcome | None

    def __post_init__(self) -> None:
        if not isinstance(self.data, JobData):
            raise TypeError("JobPlanContext.data must be a JobData")
        require_count(self.finished_jobs, "JobPlanContext.finished_jobs")
        if self.last_job is not None and not isinstance(self.last_job, JobOutcome):
            raise TypeError("JobPlanContext.last_job must be a JobOutcome or None")
        if (self.last_job is None) != (self.finished_jobs == 0):
            raise ValueError("JobPlanContext.last_job is None exactly when no job has ended")


class JobPlan(ABC):
    """The recipe's choice of each training job's parameters.

    A recipe that trains in phases, such as SFT for its first jobs and then a
    policy objective, returns a ``method`` parameter per phase. The same
    training component switches the objective from one job to the next.
    """

    @abstractmethod
    def next_job(self, context: JobPlanContext) -> JobRequest:
        """The request for the job that trains ``context.data``.

        The result must depend only on ``context``: after the service restarts, the
        trainer asks again with the same context and must get the same
        request, because the request is part of the job's identity.
        """

    def experiment_config(self) -> Mapping[str, Any]:
        """A non-secret description of the plan, which experiment runs record."""
        return {"job_plan": f"{type(self).__module__}.{type(self).__qualname__}"}


class FixedJobPlan(JobPlan):
    """A plan that gives every job the same ``request``: one job, with the same parameters, whenever job data is ready."""

    def __init__(self, request: JobRequest) -> None:
        if not isinstance(request, JobRequest):
            raise TypeError(f"FixedJobPlan requires a JobRequest, got {type(request).__name__}")
        self.request = request

    def next_job(self, context: JobPlanContext) -> JobRequest:
        return self.request

    def experiment_config(self) -> Mapping[str, Any]:
        return {"job_plan": "fixed", "parameters": dict(self.request.parameters)}


@dataclass(frozen=True)
class EvaluationJob:
    """The data and the parameters of one artifact's evaluation job.

    Values: ``data`` is a ``JobData`` of one or more items; ``parameters`` is a mapping whose values encode as
    JSON, which the value copies, and which must ask the component to score its start without training, such as
    the Tinker component's ``evaluation_of``. Purpose: ``ArtifactEvaluation.evaluation_job`` makes it, and
    ``JobLoop.decide`` submits it from the artifact's checkpoint.
    """

    data: JobData
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.data, JobData) or not self.data.items:
            raise ValueError("EvaluationJob.data must be a JobData with one or more items")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("EvaluationJob.parameters must be a mapping")
        parameters = dict(self.parameters)
        json.dumps(parameters, allow_nan=False)
        object.__setattr__(self, "parameters", parameters)


class ArtifactEvaluation(ABC):
    """The recipe's evaluation of every artifact that its weight trainer publishes."""

    @abstractmethod
    def evaluation_job(self, candidate: CheckpointCandidate) -> EvaluationJob | None:
        """The evaluation job of ``candidate``'s artifact, or ``None`` when there is nothing to evaluate it on.

        It runs on the trainer thread, without network or model calls.
        """


@dataclass(frozen=True)
class PendingEvaluation:
    """An artifact's evaluation job that the trainer submitted and has not seen end.

    ``job_id`` is the evaluation job's non-empty ID, and ``candidate_id`` the non-empty ``candidate_id`` of the
    checkpoint it scores. ``JobLoop.decide`` makes the value; ``JobLoopState`` holds it until ``JobLoop.poll``
    commits the evaluation's result.
    """

    job_id: str
    candidate_id: str

    def __post_init__(self) -> None:
        for name, value in (("job_id", self.job_id), ("candidate_id", self.candidate_id)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"PendingEvaluation.{name} must be a non-empty string")


@dataclass(frozen=True)
class RunningJob:
    """The job a trainer submitted and has not seen end.

    ``job_id`` is the job's non-empty ID. ``checkpoints_handled`` is the number
    of its saved checkpoints that the trainer committed a decision for, a
    non-negative int. ``JobLoop`` makes the value; ``JobLoopState`` holds it.
    """

    job_id: str
    checkpoints_handled: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id:
            raise ValueError("RunningJob.job_id must be a non-empty string")
        require_count(self.checkpoints_handled, "RunningJob.checkpoints_handled")


@dataclass(frozen=True)
class JobLoopState:
    """The state a weight trainer commits: where the next job starts, the running job, the jobs that ended, and
    the artifact evaluations that run.

    ``start`` is the start of the next job as the training component reads
    it (a non-empty string), or ``None`` for the base model. ``job`` is the
    running job, or ``None``. ``finished_jobs`` is the number of jobs that
    ended, finished or failed, a non-negative int. ``last_job`` is the outcome
    of the last of them, and it is ``None`` exactly when ``finished_jobs`` is
    0. ``evaluations`` is a tuple of ``PendingEvaluation`` values with distinct
    job IDs, in submission order. The commit history stores the value as
    ``to_dict()``, and ``from_dict`` reads it back.
    """

    start: str | None = None
    job: RunningJob | None = None
    finished_jobs: int = 0
    last_job: JobOutcome | None = None
    evaluations: tuple[PendingEvaluation, ...] = ()

    def __post_init__(self) -> None:
        if self.start is not None and (not isinstance(self.start, str) or not self.start):
            raise ValueError("JobLoopState.start must be a non-empty string or None")
        if self.job is not None and not isinstance(self.job, RunningJob):
            raise TypeError("JobLoopState.job must be a RunningJob or None")
        require_count(self.finished_jobs, "JobLoopState.finished_jobs")
        if self.last_job is not None and not isinstance(self.last_job, JobOutcome):
            raise TypeError("JobLoopState.last_job must be a JobOutcome or None")
        if (self.last_job is None) != (self.finished_jobs == 0):
            raise ValueError("JobLoopState.last_job is None exactly when no job has ended")
        if not isinstance(self.evaluations, tuple) or not all(
            isinstance(evaluation, PendingEvaluation) for evaluation in self.evaluations
        ):
            raise TypeError("JobLoopState.evaluations must be a tuple of PendingEvaluation values")
        if len({evaluation.job_id for evaluation in self.evaluations}) != len(self.evaluations):
            raise ValueError("JobLoopState.evaluations must have distinct job IDs")

    def to_dict(self) -> dict[str, Any]:
        """The value as a JSON object; ``from_dict`` of the result is an equal value."""
        return {
            "start": self.start,
            "job": (
                None
                if self.job is None
                else {"job_id": self.job.job_id, "checkpoints_handled": self.job.checkpoints_handled}
            ),
            "finished_jobs": self.finished_jobs,
            "last_job": None if self.last_job is None else self.last_job.to_dict(),
            "evaluations": [
                {"job_id": evaluation.job_id, "candidate_id": evaluation.candidate_id}
                for evaluation in self.evaluations
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JobLoopState:
        """Read ``to_dict``'s object; an empty mapping is the state of a trainer that has committed nothing.

        ``ValueError`` for any other key set or a malformed value: a state that cannot be read must not be
        taken for an untrained scenario.
        """
        if not value:
            return cls()
        keys = {"start", "job", "finished_jobs", "last_job", "evaluations"}
        if set(value) != keys:
            raise ValueError(f"a job loop state has the keys {sorted(keys)}, not {sorted(value)}")
        job = value["job"]
        if job is not None and (not isinstance(job, Mapping) or set(job) != {"job_id", "checkpoints_handled"}):
            raise ValueError("a job loop state's job has the keys job_id and checkpoints_handled")
        evaluations = value["evaluations"]
        if not isinstance(evaluations, list) or not all(
            isinstance(evaluation, Mapping) and set(evaluation) == {"job_id", "candidate_id"}
            for evaluation in evaluations
        ):
            raise ValueError("a job loop state's evaluations are objects with the keys job_id and candidate_id")
        return cls(
            start=value["start"],
            job=None if job is None else RunningJob(job["job_id"], job["checkpoints_handled"]),
            finished_jobs=value["finished_jobs"],
            last_job=None if value["last_job"] is None else JobOutcome.from_dict(value["last_job"]),
            evaluations=tuple(
                PendingEvaluation(evaluation["job_id"], evaluation["candidate_id"]) for evaluation in evaluations
            ),
        )


@dataclass(frozen=True)
class JobCommit:
    """One commit that a job loop gives its trainer: the result to commit, and the job end it records.

    Values: ``result`` is the ``TrainStepResult`` whose ``state`` is the next ``JobLoopState`` as a dict.
    ``ended`` is the outcome of the job whose end the commit records, which is that state's ``last_job``, or
    None for a commit that submits a job, decides one of its checkpoints, or records an artifact's evaluation. Purpose: ``JobLoop.submit``,
    ``poll``, ``decide`` and ``end`` make it; the trainer commits ``result`` and, once the commit is durable,
    gives ``ended`` to its training pipeline.
    """

    result: TrainStepResult
    ended: JobOutcome | None = None

    def __post_init__(self) -> None:
        if self.ended is not None and not isinstance(self.ended, JobOutcome):
            raise TypeError("JobCommit.ended must be a JobOutcome or None")


class JobLoop:
    """Submit a weight trainer's jobs to its training component, and turn their checkpoints into commits.

    ``component`` runs the jobs; ``plan`` gives their parameters;
    ``evaluator`` decides which checkpoints Reef serves; ``artifact_evaluation``
    gives each artifact's evaluation job, or is ``None`` for no evaluation job;
    ``scenario`` is the non-empty name of the trainer's scenario, which is part
    of each job's identity, so two scenarios never share a job. ``submit`` and
    ``poll`` return the ``JobCommit`` that the trainer commits next.
    """

    def __init__(
        self,
        component: TrainingComponent,
        plan: JobPlan,
        evaluator: CandidateEvaluationPlugin,
        *,
        scenario: str,
        artifact_evaluation: ArtifactEvaluation | None = None,
    ) -> None:
        if not isinstance(component, TrainingComponent):
            raise TypeError(f"component must be a TrainingComponent, got {type(component).__name__}")
        if not isinstance(plan, JobPlan):
            raise TypeError(f"plan must be a JobPlan, got {type(plan).__name__}")
        if not isinstance(evaluator, CandidateEvaluationPlugin):
            raise TypeError(f"evaluator must be a CandidateEvaluationPlugin, got {type(evaluator).__name__}")
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("scenario must be a non-empty string")
        if artifact_evaluation is not None and not isinstance(artifact_evaluation, ArtifactEvaluation):
            raise TypeError(
                f"artifact_evaluation must be an ArtifactEvaluation, got {type(artifact_evaluation).__name__}"
            )
        self.artifact_evaluation = artifact_evaluation
        self.component = component
        self.plan = plan
        self.evaluator = evaluator
        self.scenario = scenario

    def job_id(self, state: JobLoopState, start: str | None, data: JobData, parameters: Mapping[str, Any]) -> str:
        """The ID of the job that ``state``'s trainer submits next: a hash of the scenario, the job's ordinal
        (``state.finished_jobs``), the component, the start, the data's rows and request, and the parameters.

        Two jobs of one trainer never share an ID, even with the same data from the same start, since their
        ordinals differ; a resubmission after the service restarts names the same job, so the component does
        not run it twice.
        """
        encoded = data.to_dict()
        # The data's number is process-local, so the rows name the data.
        identity = {
            "scenario": self.scenario,
            "job": state.finished_jobs,
            "items": encoded["items"],
            "request": encoded["request"],
        }
        return job_id_for(self.component.name, start, identity, parameters)

    def submit(self, state: JobLoopState, data: JobData) -> JobCommit:
        """Submit the plan's job for ``data``; the result records the running job and its start.

        The job starts from ``state.start``, or from a fresh start of those
        weights when the request restarts training, under ``job_id``. Raise
        ``ValueError`` when ``state`` has a running job, or when the component
        cannot restart training from its start.
        """
        if state.job is not None:
            raise ValueError("a weight trainer runs one job at a time")
        request = self.plan.next_job(JobPlanContext(data, state.finished_jobs, state.last_job))
        if not isinstance(request, JobRequest):
            raise TypeError(f"{type(self.plan).__name__}.next_job must return a JobRequest")
        backend = self.component.name
        start = self.component.fresh_start(state.start) if request.restart else state.start
        job_id = self.job_id(state, start, data, request.parameters)
        self.component.submit(TrainingJob(job_id, backend, start, data, request.parameters))
        # The fresh start is the trainer's start from now on: a job that saves nothing leaves it in place.
        submitted = JobLoopState(start, RunningJob(job_id), state.finished_jobs, state.last_job, state.evaluations)
        event: dict[str, Any] = {"job_id": job_id, "backend": backend, "event": "submitted"}
        if request.restart:
            event["restart"] = True
        return JobCommit(
            TrainStepResult(state=submitted.to_dict(), metrics={"training_job": event}, training_job_id=job_id)
        )

    def job_status(self, job_id: str) -> JobStatus:
        """The component's status of ``job_id``; a job the component does not know has failed."""
        try:
            return self.component.status(job_id)
        except KeyError:
            return JobStatus(job_id, "failed", error=f"training component {self.component.name!r} has no such job")

    def poll(self, state: JobLoopState) -> JobCommit | None:
        """The next commit about the running job or the artifact evaluations of ``state``, or ``None`` while
        they run with nothing new.

        The result is the decision on the running job's next undecided
        checkpoint (``decide``) while one exists, else the job's end once it
        ended (``end``), else the result of the first evaluation that ended
        (``evaluated``). A job the component does not know ends as failed.
        Raise ``ValueError`` when ``state`` has no running job and no
        evaluation.
        """
        job = state.job
        if job is None and not state.evaluations:
            raise ValueError("a weight trainer polls only a running job or an artifact evaluation")
        if job is not None:
            status = self.job_status(job.job_id)
            if len(status.checkpoints) > job.checkpoints_handled:
                return self.decide(state, status.checkpoints[job.checkpoints_handled])
            if status.state != "running":
                return self.end(state, status)
        for evaluation in state.evaluations:
            evaluated = self.job_status(evaluation.job_id)
            if evaluated.state != "running":
                return self.evaluated(state, evaluation, evaluated)
        return None

    def decide(self, state: JobLoopState, checkpoint: SavedCheckpoint) -> JobCommit:
        """The commit that decides whether Reef serves ``checkpoint``, the running job's next undecided one.

        The next job starts from ``checkpoint.path`` whether or not the
        evaluation selects it. The commit publishes the checkpoint's
        ``release_path``: a selected checkpoint serves, and any other is a
        pending release, held for a person to promote. The commit's metrics are the checkpoint's training
        metrics with the evaluation's and the decision's own metrics, and
        ``selected``, ``selection`` (the whole decision) and ``training_job``.
        Raise ``ValueError`` when ``state`` has no running job, or when two of
        these share a key: a metric is never overwritten.
        """
        job = state.job
        if job is None:
            raise ValueError("only a running job's checkpoint is decided")
        candidate = CheckpointCandidate(
            candidate_id=f"{job.job_id}:{checkpoint.step}",
            training_job_id=job.job_id,
            step=checkpoint.step,
            release_path=checkpoint.release_path,
            training_metrics=dict(checkpoint.metrics),
        )
        decision = self.evaluator.evaluate_and_decide(candidate)
        evaluations = state.evaluations
        evaluation_event: dict[str, Any] | None = None
        request = None if self.artifact_evaluation is None else self.artifact_evaluation.evaluation_job(candidate)
        if request is not None:
            identity = {"scenario": self.scenario, "evaluates": candidate.candidate_id, **request.data.to_dict()}
            evaluation_id = job_id_for(self.component.name, checkpoint.path, identity, request.parameters)
            # Submitted before the decision commits: a retry after a restart names the same job, which runs once.
            self.component.submit(
                TrainingJob(evaluation_id, self.component.name, checkpoint.path, request.data, request.parameters)
            )
            evaluations = (*evaluations, PendingEvaluation(evaluation_id, candidate.candidate_id))
            evaluation_event = {"job_id": evaluation_id, "event": "submitted"}
        handled = JobLoopState(
            checkpoint.path,
            RunningJob(job.job_id, job.checkpoints_handled + 1),
            state.finished_jobs,
            state.last_job,
            evaluations,
        )
        parts = (dict(checkpoint.metrics), dict(decision.evaluation.metrics), dict(decision.metrics))
        keys = [key for part in parts for key in part]
        shared = {key for key in keys if keys.count(key) > 1} | (set(keys) & DECISION_KEYS)
        if shared:
            raise ValueError(
                f"checkpoint {candidate.candidate_id}'s training, evaluation and decision metrics share the keys "
                f"{sorted(shared)}"
            )
        metrics = {
            **parts[0],
            **parts[1],
            **parts[2],
            "selected": decision.selected,
            "selection": {"candidate_id": candidate.candidate_id, **decision.to_dict()},
            "training_job": {
                "job_id": job.job_id,
                "backend": self.component.name,
                "event": "checkpoint",
                "step": checkpoint.step,
            },
        }
        if evaluation_event is not None:
            metrics["artifact_evaluation"] = evaluation_event
        return JobCommit(
            TrainStepResult(
                state=handled.to_dict(),
                metrics=metrics,
                artifact=Artifact.local(Path(checkpoint.release_path)),
                training_job_id=job.job_id,
                pending=not decision.selected,
            )
        )

    def end(self, state: JobLoopState, status: JobStatus) -> JobCommit:
        """The commit that records the end of the running job, whose ended ``status`` the component reported.

        The job's outcome becomes ``last_job`` and ``finished_jobs`` grows by
        one; the start stays the last saved checkpoint. Raise ``ValueError``
        when ``state`` has no running job or ``status`` is still running.
        """
        job = state.job
        if job is None:
            raise ValueError("only a running job ends")
        if status.state == "running":
            raise ValueError("a running job has not ended")
        outcome = JobOutcome(job.job_id, self.component.name, status.state, status.metrics, status.error)
        ended = JobLoopState(state.start, None, state.finished_jobs + 1, outcome, state.evaluations)
        event = {key: value for key, value in outcome.to_dict().items() if key != "metrics"}
        return JobCommit(
            TrainStepResult(
                state=ended.to_dict(),
                metrics={**status.metrics, "training_job": event},
                training_job_id=job.job_id,
            ),
            ended=outcome,
        )

    def evaluated(self, state: JobLoopState, evaluation: PendingEvaluation, status: JobStatus) -> JobCommit:
        """The commit that records the end of ``evaluation``, whose ended ``status`` the component reported.

        The evaluation leaves ``state.evaluations``. The commit's metrics are
        ``artifact_evaluation``: the evaluated ``candidate_id``, the job's ID,
        its ``event`` (``finished`` or ``failed``), its ``metrics`` and, when it
        failed, its ``error``. Raise ``ValueError`` when ``status`` is still
        running or ``evaluation`` is not in ``state``.
        """
        if status.state == "running":
            raise ValueError("a running evaluation has not ended")
        if evaluation not in state.evaluations:
            raise ValueError(f"evaluation {evaluation.job_id!r} is not pending")
        remaining = tuple(pending for pending in state.evaluations if pending != evaluation)
        recorded = JobLoopState(state.start, state.job, state.finished_jobs, state.last_job, remaining)
        event: dict[str, Any] = {
            "candidate_id": evaluation.candidate_id,
            "job_id": evaluation.job_id,
            "event": status.state,
            "metrics": dict(status.metrics),
        }
        if status.error is not None:
            event["error"] = status.error
        return JobCommit(
            TrainStepResult(
                state=recorded.to_dict(), metrics={"artifact_evaluation": event}, training_job_id=evaluation.job_id
            )
        )

    def import_start(self, release: Path) -> str | None:
        """The start that the component makes of a rolled-back ``release`` (``TrainingComponent.import_start``)."""
        return self.component.import_start(release)

    def cancel(self, state: JobLoopState) -> None:
        """Cancel the running job of ``state`` at the component; nothing when no job runs."""
        if state.job is not None:
            self.component.cancel(state.job.job_id)

    def restored(self, state: JobLoopState, start: str | None) -> JobLoopState:
        """The state after a rollback whose release the component imported as ``start``.

        The running job is cancelled, and the next job starts from ``start``
        (``None``: the base model). The jobs that ended stay counted.
        """
        self.cancel(state)
        return JobLoopState(start, None, state.finished_jobs, state.last_job, state.evaluations)
