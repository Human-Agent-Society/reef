"""The control plane's side of training jobs: the recipe's job plan and one weight trainer's job loop.

A weight trainer does not drive training steps. When a job's data is ready, the
recipe's ``JobPlan`` gives the job's parameters, and the ``JobLoop`` submits
the job to the trainer's ``TrainingComponent``. The trainer commits the
submission, which consumes the records of the job's data. While the job runs, each
checkpoint it saves becomes a ``CheckpointCandidate``: candidate evaluation
decides whether Reef serves it, and the trainer commits the decision. When the
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
from reef.core.evaluation import CandidateEvaluationPlugin, EvaluationResult, UpdateCandidate
from reef.core.job_data import JobData
from reef.evaluation.evaluators import AlwaysSelectMixin
from reef.runtime.jobs import JobStatus, SavedCheckpoint, TrainingComponent, TrainingJob, job_id_for
from reef.train.types import TrainStepResult


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
    ``last_job`` is the outcome of the last of them, as ``JobLoop.end``
    commits it (``{"job_id", "backend", "event", "metrics"}``, with ``"error"``
    when ``event`` is ``"failed"``), and it is ``None`` exactly when
    ``finished_jobs`` is 0. ``JobLoop.submit`` makes the value and
    ``JobPlan.next_job`` reads it.
    """

    data: JobData
    finished_jobs: int
    last_job: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        if not isinstance(self.data, JobData):
            raise TypeError("JobPlanContext.data must be a JobData")
        require_count(self.finished_jobs, "JobPlanContext.finished_jobs")
        if self.last_job is not None and not isinstance(self.last_job, Mapping):
            raise TypeError("JobPlanContext.last_job must be a mapping or None")
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
    """The state a weight trainer commits: where the next job starts, the running job and the jobs that ended.

    ``start`` is the start of the next job as the training component reads
    it (a non-empty string), or ``None`` for the base model. ``job`` is the
    running job, or ``None``. ``finished_jobs`` is the number of jobs that
    ended, finished or failed, a non-negative int. ``last_job`` is the outcome
    of the last of them, a JSON object, and it is ``None`` exactly when
    ``finished_jobs`` is 0. The commit history stores the value as
    ``to_dict()``, and ``from_dict`` reads it back.
    """

    start: str | None = None
    job: RunningJob | None = None
    finished_jobs: int = 0
    last_job: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.start is not None and (not isinstance(self.start, str) or not self.start):
            raise ValueError("JobLoopState.start must be a non-empty string or None")
        if self.job is not None and not isinstance(self.job, RunningJob):
            raise TypeError("JobLoopState.job must be a RunningJob or None")
        require_count(self.finished_jobs, "JobLoopState.finished_jobs")
        if self.last_job is not None and not isinstance(self.last_job, Mapping):
            raise TypeError("JobLoopState.last_job must be a mapping or None")
        if (self.last_job is None) != (self.finished_jobs == 0):
            raise ValueError("JobLoopState.last_job is None exactly when no job has ended")

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
            "last_job": None if self.last_job is None else dict(self.last_job),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JobLoopState:
        """Read a committed state. An empty mapping is the state of a trainer that has committed nothing."""
        job = value.get("job")
        return cls(
            start=value.get("start"),
            job=None if job is None else RunningJob(job["job_id"], job["checkpoints_handled"]),
            finished_jobs=value.get("finished_jobs", 0),
            last_job=value.get("last_job"),
        )


@dataclass(frozen=True, kw_only=True)
class CheckpointCandidate(UpdateCandidate):
    """A checkpoint that a job saved and that Reef has not decided to serve.

    ``training_job_id`` (non-empty) and ``step`` (a non-negative int) name the
    checkpoint within its job, and ``candidate_id`` is
    ``"<training_job_id>:<step>"``. ``checkpoint_path`` is the non-empty
    directory that Reef publishes when it serves the checkpoint
    (``SavedCheckpoint.release_path``). ``training_metrics`` are the training
    component's metrics for it. ``JobLoop.decide`` makes the value, and the
    recipe's candidate evaluation reads it.
    """

    training_job_id: str
    step: int
    checkpoint_path: str
    training_metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.training_job_id, str) or not self.training_job_id:
            raise ValueError("CheckpointCandidate.training_job_id must be a non-empty string")
        require_count(self.step, "CheckpointCandidate.step")
        if not isinstance(self.checkpoint_path, str) or not self.checkpoint_path:
            raise ValueError("CheckpointCandidate.checkpoint_path must be a non-empty string")
        if self.candidate_id != f"{self.training_job_id}:{self.step}":
            raise ValueError("CheckpointCandidate.candidate_id must be '<training_job_id>:<step>'")
        if not isinstance(self.training_metrics, Mapping):
            raise TypeError("CheckpointCandidate.training_metrics must be a mapping")


class TrainingMetricsSelectPlugin(AlwaysSelectMixin):
    """The default candidate evaluation of a weight trainer: record the training metrics and serve every checkpoint.

    The evaluation's metrics are the candidate's ``training_metrics`` for a
    ``CheckpointCandidate``, and empty for any other candidate.
    """

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        metrics = candidate.training_metrics if isinstance(candidate, CheckpointCandidate) else {}
        return EvaluationResult(evaluator="training_job", evaluator_version="1", metrics=dict(metrics))


class JobLoop:
    """Submit a weight trainer's jobs to its training component, and turn their checkpoints into commits.

    ``component`` runs the jobs; ``plan`` gives their parameters;
    ``evaluator`` decides which checkpoints Reef serves; ``scenario`` is the
    non-empty name of the trainer's scenario, which is part of each job's
    identity, so two scenarios never share a job. ``submit`` and ``poll``
    return the ``TrainStepResult`` that the trainer commits next; its
    ``state`` is the next ``JobLoopState`` as a dict.
    """

    def __init__(
        self,
        component: TrainingComponent,
        plan: JobPlan,
        evaluator: CandidateEvaluationPlugin,
        *,
        scenario: str,
    ) -> None:
        if not isinstance(component, TrainingComponent):
            raise TypeError(f"component must be a TrainingComponent, got {type(component).__name__}")
        if not isinstance(plan, JobPlan):
            raise TypeError(f"plan must be a JobPlan, got {type(plan).__name__}")
        if not isinstance(evaluator, CandidateEvaluationPlugin):
            raise TypeError(f"evaluator must be a CandidateEvaluationPlugin, got {type(evaluator).__name__}")
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("scenario must be a non-empty string")
        self.component = component
        self.plan = plan
        self.evaluator = evaluator
        self.scenario = scenario

    def submit(self, state: JobLoopState, data: JobData) -> TrainStepResult:
        """Submit the plan's job for ``data``; the result records the running job and its start.

        The job starts from ``state.start``, or from a fresh start of those
        weights when the request restarts training. Raise ``ValueError`` when
        ``state`` has a running job, or when the component cannot restart
        training from its start. The job ID derives from the scenario, the
        component, the start, the data's records and the parameters. A
        resubmission after the service restarts therefore names the same job,
        and the component does not run it twice.
        """
        if state.job is not None:
            raise ValueError("a weight trainer runs one job at a time")
        request = self.plan.next_job(JobPlanContext(data, state.finished_jobs, state.last_job))
        if not isinstance(request, JobRequest):
            raise TypeError(f"{type(self.plan).__name__}.next_job must return a JobRequest")
        backend = self.component.name
        start = self.component.fresh_start(state.start) if request.restart else state.start
        encoded = data.to_dict()
        # The scenario and the rows name the data: the data's number is process-local.
        identity = {"scenario": self.scenario, "items": encoded["items"], "request": encoded["request"]}
        job_id = job_id_for(backend, start, identity, request.parameters)
        self.component.submit(TrainingJob(job_id, backend, start, data, request.parameters))
        # The fresh start is the trainer's start from now on: a job that saves nothing leaves it in place.
        submitted = JobLoopState(start, RunningJob(job_id), state.finished_jobs, state.last_job)
        event: dict[str, Any] = {"job_id": job_id, "backend": backend, "event": "submitted"}
        if request.restart:
            event["restart"] = True
        return TrainStepResult(state=submitted.to_dict(), metrics={"training_job": event}, training_job_id=job_id)

    def poll(self, state: JobLoopState) -> TrainStepResult | None:
        """The next commit about the running job of ``state``, or ``None`` while it runs with nothing new.

        The result is the decision on the job's next undecided checkpoint
        (``decide``) while one exists, else the job's end once it ended
        (``end``). A job the component does not know ends as failed. Raise
        ``ValueError`` when ``state`` has no running job.
        """
        job = state.job
        if job is None:
            raise ValueError("a weight trainer polls only a running job")
        try:
            status = self.component.status(job.job_id)
        except KeyError:
            status = JobStatus(
                job.job_id, "failed", error=f"training component {self.component.name!r} has no such job"
            )
        if len(status.checkpoints) > job.checkpoints_handled:
            return self.decide(state, status.checkpoints[job.checkpoints_handled])
        if status.state == "running":
            return None
        return self.end(state, status)

    def decide(self, state: JobLoopState, checkpoint: SavedCheckpoint) -> TrainStepResult:
        """The commit that decides whether Reef serves ``checkpoint``, the running job's next undecided one.

        The next job starts from ``checkpoint.path`` whether or not the
        evaluation selects it. A selected checkpoint publishes its
        ``release_path``. Raise ``ValueError`` when ``state`` has no running job.
        """
        job = state.job
        if job is None:
            raise ValueError("only a running job's checkpoint is decided")
        candidate = CheckpointCandidate(
            candidate_id=f"{job.job_id}:{checkpoint.step}",
            training_job_id=job.job_id,
            step=checkpoint.step,
            checkpoint_path=checkpoint.release_path,
            training_metrics=dict(checkpoint.metrics),
        )
        decision = self.evaluator.evaluate_and_decide(candidate)
        handled = JobLoopState(
            checkpoint.path,
            RunningJob(job.job_id, job.checkpoints_handled + 1),
            state.finished_jobs,
            state.last_job,
        )
        metrics = {
            **checkpoint.metrics,
            **decision.evaluation.metrics,
            **decision.metrics,
            "selected": decision.selected,
            "selection": {"candidate_id": candidate.candidate_id, **decision.to_dict()},
            "training_job": {
                "job_id": job.job_id,
                "backend": self.component.name,
                "event": "checkpoint",
                "step": checkpoint.step,
            },
        }
        return TrainStepResult(
            state=handled.to_dict(),
            metrics=metrics,
            artifact=Artifact.local(Path(checkpoint.release_path)) if decision.selected else None,
            training_job_id=job.job_id,
        )

    def end(self, state: JobLoopState, status: JobStatus) -> TrainStepResult:
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
        outcome: dict[str, Any] = {"job_id": job.job_id, "backend": self.component.name, "event": status.state}
        if status.error is not None:
            outcome["error"] = status.error
        ended = JobLoopState(state.start, None, state.finished_jobs + 1, {**outcome, "metrics": dict(status.metrics)})
        return TrainStepResult(
            state=ended.to_dict(),
            metrics={**status.metrics, "training_job": outcome},
            training_job_id=job.job_id,
        )

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
        return JobLoopState(start, None, state.finished_jobs, state.last_job)
