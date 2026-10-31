"""The control plane's side of training jobs: one weight trainer's job loop.

A weight trainer does not drive training steps. When a job's data is ready, the
pipeline's sample-to-job stage gives the job's data and parameters
(``Sample2Job.make_job_data`` and ``Sample2Job.job_parameters``), and the
``JobLoop`` submits the job to the trainer's ``TrainingComponent``. The trainer commits the
submission, which consumes the records of the job's data. While the job runs, the
trainer commits each checkpoint that it saves as an artifact. Training never
activates an artifact: the scenario's activation rule decides whether it
serves. The training component only trains; it evaluates no artifact. When the
job ends, the trainer commits its outcome. Training is linear: the next job
starts from the job's last saved checkpoint, whether or not it serves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.job_data import JobData, JobOutcome
from reef.runtime.jobs import JobStatus, SavedCheckpoint, TrainingComponent, TrainingJob, job_id_for
from reef.train.types import TrainStepResult

#: The key of a checkpoint commit's metrics that the job loop writes itself.
TRAINING_JOB_KEY = "training_job"


def require_count(value: object, name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is a non-negative int (not a bool)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")


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
    """The state a weight trainer commits: where the next job starts, the running job, and the jobs that ended.

    ``start`` is the start of the next job as the training component reads
    it, a non-empty string; a new trainer starts at ``TrainingComponent.base_start``. ``job`` is the
    running job, or ``None``. ``finished_jobs`` is the number of jobs that
    ended, finished or failed, a non-negative int. ``last_job`` is the outcome
    of the last of them, and it is ``None`` exactly when ``finished_jobs`` is
    0. The commit history stores the value as ``to_dict()``, and ``from_dict``
    reads it back.
    """

    start: str
    job: RunningJob | None = None
    finished_jobs: int = 0
    last_job: JobOutcome | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.start, str) or not self.start:
            raise ValueError("JobLoopState.start must be a non-empty string")
        if self.job is not None and not isinstance(self.job, RunningJob):
            raise TypeError("JobLoopState.job must be a RunningJob or None")
        require_count(self.finished_jobs, "JobLoopState.finished_jobs")
        if self.last_job is not None and not isinstance(self.last_job, JobOutcome):
            raise TypeError("JobLoopState.last_job must be a JobOutcome or None")
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
            "last_job": None if self.last_job is None else self.last_job.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JobLoopState:
        """Read ``to_dict``'s object; ``ValueError`` for any other key set or a malformed value."""
        keys = {"start", "job", "finished_jobs", "last_job"}
        if set(value) != keys:
            raise ValueError(f"a job loop state has the keys {sorted(keys)}, not {sorted(value)}")
        job = value["job"]
        if job is not None and (not isinstance(job, Mapping) or set(job) != {"job_id", "checkpoints_handled"}):
            raise ValueError("a job loop state's job has the keys job_id and checkpoints_handled")
        return cls(
            start=value["start"],
            job=None if job is None else RunningJob(job["job_id"], job["checkpoints_handled"]),
            finished_jobs=value["finished_jobs"],
            last_job=None if value["last_job"] is None else JobOutcome.from_dict(value["last_job"]),
        )


@dataclass(frozen=True)
class JobCommit:
    """One commit that a job loop gives its trainer: the result to commit, and the job end it records.

    Values: ``result`` is the ``TrainStepResult`` whose ``state`` is the next ``JobLoopState`` as a dict.
    ``ended`` is the outcome of the job whose end the commit records, which is that state's ``last_job``, or
    None for a commit that submits a job or publishes one of its checkpoints. Purpose: ``JobLoop.submit``,
    ``poll``, ``publish`` and ``end`` make it; the trainer commits ``result`` and, once the commit is durable,
    gives ``ended`` to its training pipeline.
    """

    result: TrainStepResult
    ended: JobOutcome | None = None

    def __post_init__(self) -> None:
        if self.ended is not None and not isinstance(self.ended, JobOutcome):
            raise TypeError("JobCommit.ended must be a JobOutcome or None")


class SubmitParameters(ABC):
    """The parameters of a scenario's job that only the scenario knows when the job is submitted.

    Values: ``parameters(data, state)`` returns the parameters of the job that trains ``data`` from the trainer's
    committed ``state``, a mapping that encodes as JSON, read from the scenario's state at that moment, such as the
    scenario's own model, the tasks of its eval split, or the last job that ended (``state.last_job``). Purpose: a
    recipe whose jobs depend on such state gives one to its ``JobLoop``; ``JobLoop.submit`` adds the parameters to
    the ones that the pipeline's sample-to-job stage gives.
    """

    @abstractmethod
    def parameters(self, data: JobData, state: JobLoopState) -> Mapping[str, Any]: ...


class JobLoop:
    """Submit a weight trainer's jobs to its training component, and turn their checkpoints into commits.

    ``component`` runs the jobs; ``scenario`` is the non-empty name of the
    trainer's scenario, which is part of each job's identity, so two scenarios
    never share a job. ``submitted`` adds the parameters that only the
    scenario knows at submit time, or is None. ``submit`` and
    ``poll`` return the ``JobCommit`` that the trainer commits next.
    """

    def __init__(
        self, component: TrainingComponent, *, scenario: str, submitted: SubmitParameters | None = None
    ) -> None:
        if not isinstance(component, TrainingComponent):
            raise TypeError(f"component must be a TrainingComponent, got {type(component).__name__}")
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("scenario must be a non-empty string")
        if submitted is not None and not isinstance(submitted, SubmitParameters):
            raise TypeError("submitted must be a SubmitParameters or None")
        self.component = component
        self.scenario = scenario
        self.submitted = submitted

    def job_id(self, state: JobLoopState, start: str, data: JobData, parameters: Mapping[str, Any]) -> str:
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
            "samples": encoded["samples"],
            "request": encoded["request"],
        }
        return job_id_for(self.component.name, start, identity, parameters)

    def submit(self, state: JobLoopState, data: JobData, parameters: Mapping[str, Any]) -> JobCommit:
        """Submit the job that trains ``data`` with ``parameters``; the result records the running job.

        The job starts from ``state.start``, under ``job_id``. ``parameters``
        is the mapping that the pipeline's sample-to-job stage gives for
        ``data``, which encodes as JSON; ``submitted`` adds its own. Raise
        ``ValueError`` when ``state`` has a running job, or when ``submitted``
        names a parameter that the stage gives too.
        """
        if state.job is not None:
            raise ValueError("a weight trainer runs one job at a time")
        if self.submitted is not None:
            added = dict(self.submitted.parameters(data, state))
            shared = sorted(set(added) & set(parameters))
            if shared:
                raise ValueError(f"the stage's job parameters already name {shared}")
            parameters = {**parameters, **added}
        backend = self.component.name
        job_id = self.job_id(state, state.start, data, parameters)
        self.component.submit(TrainingJob(job_id, backend, state.start, data, parameters, scenario=self.scenario))
        submitted = JobLoopState(state.start, RunningJob(job_id), state.finished_jobs, state.last_job)
        event: dict[str, Any] = {"job_id": job_id, "backend": backend, "event": "submitted"}
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
        """The next commit about the running job of ``state``, or ``None`` while it runs with nothing new.

        The result is the publication of the job's next unpublished checkpoint
        (``publish``) while one exists, else the job's end once it ended
        (``end``). A job the component does not know ends as failed. Raise
        ``ValueError`` when ``state`` has no running job.
        """
        job = state.job
        if job is None:
            raise ValueError("a weight trainer polls only a running job")
        status = self.job_status(job.job_id)
        if len(status.checkpoints) > job.checkpoints_handled:
            return self.publish(state, status.checkpoints[job.checkpoints_handled])
        if status.state != "running":
            return self.end(state, status)
        return None

    def publish(self, state: JobLoopState, checkpoint: SavedCheckpoint) -> JobCommit:
        """The commit that publishes ``checkpoint``, the running job's next unpublished one, as an artifact.

        The next job starts from ``checkpoint.path``. The commit publishes the checkpoint's ``release_path``, and
        the scenario's activation rule decides whether it serves. The commit's metrics are the checkpoint's
        training metrics and ``training_job``. Raise ``ValueError`` when ``state`` has no running job, or when a
        training metric is named ``training_job``: a metric is never overwritten.
        """
        job = state.job
        if job is None:
            raise ValueError("only a running job's checkpoint is published")
        if TRAINING_JOB_KEY in checkpoint.metrics:
            raise ValueError(f"checkpoint {job.job_id}:{checkpoint.step}'s training metrics name {TRAINING_JOB_KEY!r}")
        handled = JobLoopState(
            checkpoint.path, RunningJob(job.job_id, job.checkpoints_handled + 1), state.finished_jobs, state.last_job
        )
        metrics = {
            **dict(checkpoint.metrics),
            TRAINING_JOB_KEY: {
                "job_id": job.job_id,
                "backend": self.component.name,
                "event": "checkpoint",
                "step": checkpoint.step,
            },
        }
        return JobCommit(
            TrainStepResult(
                state=handled.to_dict(),
                metrics=metrics,
                artifact=Artifact.local(Path(checkpoint.release_path)),
                training_job_id=job.job_id,
                changed_kinds=checkpoint.changed_kinds,
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
        ended = JobLoopState(state.start, None, state.finished_jobs + 1, outcome)
        event = {key: value for key, value in outcome.to_dict().items() if key != "metrics"}
        return JobCommit(
            TrainStepResult(
                state=ended.to_dict(),
                metrics={**status.metrics, "training_job": event},
                training_job_id=job.job_id,
            ),
            ended=outcome,
        )

    def cancel(self, state: JobLoopState) -> None:
        """Cancel the running job of ``state`` at the component; nothing when no job runs."""
        if state.job is not None:
            self.component.cancel(state.job.job_id)

    def started_from(self, state: JobLoopState, start: str) -> JobLoopState:
        """The state after the training start is set to ``start``, which the component imported of a release.

        The running job is cancelled, and the next job starts from ``start``.
        The jobs that ended stay counted.
        """
        self.cancel(state)
        return JobLoopState(start, None, state.finished_jobs, state.last_job)
