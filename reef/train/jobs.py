"""The control plane's side of training jobs: one trainer's job loop.

A trainer does not drive training steps. When a job's data is ready, the
pipeline's sample-to-job stage gives the job's data and parameters
(``Sample2Job.make_job_data`` and ``Sample2Job.job_parameters``), and the
``JobLoop`` submits the job to the trainer's ``TrainingComponent``. The trainer commits the
submission, which consumes the records of the job's data. While the job runs, the
trainer commits each checkpoint that it saves as an artifact. Training never
activates an artifact: the scenario's activation rule decides whether it
serves. The training component only trains; it evaluates no artifact. When the
job ends, the trainer commits its outcome.

A job continues the trainer's training start unless its request names another start (``train_from``, RFC #1324).
A job that continues the training start begins there and moves it to each checkpoint that it saves, so the next such
job continues where it ended; the trainer runs one such job at a time. A job with its own start never moves the
training start, and it runs beside the other jobs when the component allows it
(``TrainingComponent.concurrent_jobs``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.job_data import JobData, JobOutcome
from reef.storage.data import DataStore, MemoryDataStore
from reef.storage.tasks import LocalTaskFileStore
from reef.train.component import JobStatus, SavedCheckpoint, TrainingComponent, TrainingJob, job_id_for
from reef.train.types import TrainingCommit

#: The key of a checkpoint commit's metrics that the job loop writes itself.
TRAINING_JOB_KEY = "training_job"


def require_count(value: object, name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is a non-negative int (not a bool)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")


@dataclass(frozen=True)
class RunningJob:
    """A job that a trainer submitted and has not seen end.

    Values: ``job_id`` is the job's non-empty ID. ``start`` is the non-empty start that the job trains from, as the
    training component reads it. ``checkpoints_handled`` is the number of its saved checkpoints that the trainer
    committed, a non-negative int. ``train_from`` is the non-empty ID of the artifact that the job's request named as
    its start, or None for a job that continues the training start.
    Purpose: ``JobLoop`` makes the value; ``JobLoopState`` holds it.
    """

    job_id: str
    start: str
    checkpoints_handled: int = 0
    train_from: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id:
            raise ValueError("RunningJob.job_id must be a non-empty string")
        if not isinstance(self.start, str) or not self.start:
            raise ValueError("RunningJob.start must be a non-empty string")
        require_count(self.checkpoints_handled, "RunningJob.checkpoints_handled")
        if self.train_from is not None and (not isinstance(self.train_from, str) or not self.train_from):
            raise ValueError("RunningJob.train_from must be a non-empty string or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "start": self.start,
            "checkpoints_handled": self.checkpoints_handled,
            "train_from": self.train_from,
        }


@dataclass(frozen=True)
class JobLoopState:
    """The state a trainer commits: its training start, the running jobs, and the jobs that ended.

    Values: ``start`` is the training start as the training component reads it, a non-empty string: the start of a job
    that names none. A new trainer starts at ``TrainingComponent.root_start``. ``jobs`` are the running jobs in the
    order of their submission, with distinct IDs; at most one of them continues the training start. ``finished_jobs`` is the number of jobs that ended, finished or
    failed, a non-negative int. ``last_job`` is the outcome of the last of them, and it is ``None`` exactly when
    ``finished_jobs`` is 0.
    Purpose: ``JobLoop`` makes each next value, and the commit history stores it as ``to_dict()``; ``from_dict`` reads
    it back.
    """

    start: str
    jobs: tuple[RunningJob, ...] = ()
    finished_jobs: int = 0
    last_job: JobOutcome | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.start, str) or not self.start:
            raise ValueError("JobLoopState.start must be a non-empty string")
        if not isinstance(self.jobs, tuple) or not all(isinstance(job, RunningJob) for job in self.jobs):
            raise TypeError("JobLoopState.jobs must be a tuple of RunningJob")
        if len({job.job_id for job in self.jobs}) != len(self.jobs):
            raise ValueError("JobLoopState.jobs must have distinct job IDs")
        if sum(job.train_from is None for job in self.jobs) > 1:
            raise ValueError("at most one job of JobLoopState.jobs continues the training start")
        require_count(self.finished_jobs, "JobLoopState.finished_jobs")
        if self.last_job is not None and not isinstance(self.last_job, JobOutcome):
            raise TypeError("JobLoopState.last_job must be a JobOutcome or None")
        if (self.last_job is None) != (self.finished_jobs == 0):
            raise ValueError("JobLoopState.last_job is None exactly when no job has ended")

    @property
    def continuing_job(self) -> RunningJob | None:
        """The running job that continues the training start, or None."""
        return next((job for job in self.jobs if job.train_from is None), None)

    def to_dict(self) -> dict[str, Any]:
        """The value as a JSON object; ``from_dict`` of the result is an equal value."""
        return {
            "start": self.start,
            "jobs": [job.to_dict() for job in self.jobs],
            "finished_jobs": self.finished_jobs,
            "last_job": None if self.last_job is None else self.last_job.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JobLoopState:
        """Read ``to_dict``'s object; ``ValueError`` for any other key set or a malformed value.

        A record from before RFC #1324 holds ``job``, one running job with no start of its own, or None. It reads as
        that job, which continues ``start``.
        """
        keys = {"start", "finished_jobs", "last_job"}
        if set(value) == keys | {"job"}:
            job = value["job"]
            if job is not None and (not isinstance(job, Mapping) or set(job) != {"job_id", "checkpoints_handled"}):
                raise ValueError("a job loop state's job has the keys job_id and checkpoints_handled")
            jobs: tuple[RunningJob, ...] = (
                () if job is None else (RunningJob(job["job_id"], value["start"], job["checkpoints_handled"]),)
            )
        elif set(value) == keys | {"jobs"}:
            job_keys = {"job_id", "start", "checkpoints_handled", "train_from"}
            if not isinstance(value["jobs"], list) or not all(
                isinstance(job, Mapping) and set(job) == job_keys for job in value["jobs"]
            ):
                raise ValueError(f"a job loop state's jobs each have the keys {sorted(job_keys)}")
            jobs = tuple(RunningJob(**job) for job in value["jobs"])
        else:
            raise ValueError(f"a job loop state has the keys {sorted(keys | {'jobs'})}, not {sorted(value)}")
        return cls(
            start=value["start"],
            jobs=jobs,
            finished_jobs=value["finished_jobs"],
            last_job=None if value["last_job"] is None else JobOutcome.from_dict(value["last_job"]),
        )


@dataclass(frozen=True)
class JobCommit:
    """One commit that a job loop gives its trainer.

    Values: ``result`` is the ``TrainingCommit`` whose ``state`` is the next ``JobLoopState`` as a dict.
    Purpose: ``JobLoop.submit``, ``poll``, ``publish`` and ``end`` make it; the trainer commits ``result``.
    """

    result: TrainingCommit


class JobLoop:
    """Submit a trainer's jobs to its training component, and turn their checkpoints into commits. Weight training
    and harness evolution share it: each job carries the same fields (``TrainingJob``).

    ``component`` runs the jobs; ``scenario`` is the non-empty name of the
    trainer's scenario, which is part of each job's identity, so two scenarios
    never share a job. ``scenario_component`` is the scenario component that the jobs train, or None.
    ``data_store`` keeps the job data directory of each job that ``submit`` submits; None keeps
    them in a new temporary directory, for a process that keeps no state across restarts. ``submit`` and
    ``poll`` return the ``JobCommit`` that the trainer commits next. ``waiting_job`` is the job whose data
    directory ``submit`` wrote and that the component has not taken yet (``JobCheckPendingError``), or None.
    """

    def __init__(
        self,
        component: TrainingComponent,
        *,
        scenario: str,
        scenario_component: str | None = None,
        data_store: DataStore | None = None,
    ) -> None:
        if not isinstance(component, TrainingComponent):
            raise TypeError(f"component must be a TrainingComponent, got {type(component).__name__}")
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("scenario must be a non-empty string")
        self.component = component
        self.scenario = scenario
        self.scenario_component = scenario_component
        self.data_store = MemoryDataStore(LocalTaskFileStore()) if data_store is None else data_store
        self.waiting_job: TrainingJob | None = None

    def job_id(self, state: JobLoopState, start: str, data: JobData, parameters: Mapping[str, Any]) -> str:
        """The ID of the job that ``state``'s trainer submits next: a hash of the scenario, the job's ordinal
        (``state.finished_jobs``), the component, the start, the data's rows and request, and the parameters.

        Two jobs of one trainer never share an ID, even with the same data from the same start: their ordinals
        differ, or they run at the same time for two requests. A resubmission after the service restarts names the
        same job, so the component does not run it twice.
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

    def accepts_job(self, state: JobLoopState, *, continues: bool) -> bool:
        """Whether ``submit`` takes one more job now; ``continues`` is whether the job continues the training start.
        Such jobs run one at a time, since each begins where the one before ended. So does every job of a component
        without ``TrainingComponent.concurrent_jobs``."""
        if not self.component.concurrent_jobs:
            return not state.jobs
        return not continues or state.continuing_job is None

    def submit(
        self,
        state: JobLoopState,
        data: JobData,
        parameters: Mapping[str, Any],
        *,
        start: str | None = None,
        train_from: str | None = None,
        train_from_artifact_id: str | None = None,
    ) -> JobCommit:
        """Submit the job that trains ``data`` with ``parameters``; the result records the running job.

        A job whose request names a start gives the artifact's ID as ``train_from`` and what the component imported
        of it as ``start``. Any other job gives neither, and continues ``state.start``. The job runs under
        ``job_id``. ``parameters``
        is the mapping that the pipeline's sample-to-job stage gives for
        ``data``, which encodes as JSON. The loop writes ``data`` as the job's data directory first. The directory
        holds no request: the job keeps only its ID (``TrainingJob.request_id``). ``train_from_artifact_id`` is the
        artifact of the training start, for the job listing of a job that continues it, or None. Raise ``ValueError``
        when ``accepts_job`` is false, and when the component refuses the job. Raise ``JobCheckPendingError`` while
        the component waits for the check of the job: a later call with the same rows, start and parameters names
        the same job and submits it, and its data directory is written once.
        """
        if (start is None) != (train_from is None):
            raise ValueError("a job names its start and the start's artifact together, or neither")
        if not self.accepts_job(state, continues=train_from is None):
            raise ValueError("the trainer runs one such job at a time")
        backend = self.component.name
        start = state.start if start is None else start
        job_id = self.job_id(state, start, data, parameters)
        job = self.waiting_job
        if job is None or job.job_id != job_id:
            job = TrainingJob(
                job_id,
                backend,
                start,
                self.data_store.write_job_directory(data),
                parameters,
                scenario=self.scenario,
                request_id=None if data.request is None else data.request.id,
                component=self.scenario_component,
                previous_job_id=None if state.last_job is None else state.last_job.job_id,
                train_from_artifact_id=train_from if train_from is not None else train_from_artifact_id,
            )
        self.waiting_job = job
        self.component.submit(job)
        self.waiting_job = None
        submitted = replace(state, jobs=(*state.jobs, RunningJob(job_id, start, train_from=train_from)))
        event: dict[str, Any] = {"job_id": job_id, "backend": backend, "event": "submitted"}
        return JobCommit(
            TrainingCommit(state=submitted.to_dict(), metrics={"training_job": event}, training_job_id=job_id)
        )

    def job_status(self, job_id: str) -> JobStatus:
        """The component's status of ``job_id``; a job the component does not know has failed."""
        try:
            return self.component.status(job_id)
        except KeyError:
            return JobStatus(job_id, "failed", error=f"training component {self.component.name!r} has no such job")

    def poll(self, state: JobLoopState) -> JobCommit | None:
        """The next commit about a running job of ``state``, or ``None`` while every job runs with nothing new.

        For the first job with news, in the order of submission, the result is the publication of its next
        unpublished checkpoint (``publish``) while one exists, else its end once it ended (``end``). A job the
        component does not know ends as failed.
        """
        for job in state.jobs:
            status = self.job_status(job.job_id)
            if len(status.checkpoints) > job.checkpoints_handled:
                return self.publish(state, job, status.checkpoints[job.checkpoints_handled])
            if status.state != "running":
                return self.end(state, job, status)
        return None

    def publish(self, state: JobLoopState, job: RunningJob, checkpoint: SavedCheckpoint) -> JobCommit:
        """The commit that publishes ``checkpoint``, the next unpublished one of ``job``, as an artifact.

        ``job`` is a running job of ``state``. The commit publishes the checkpoint's ``artifact_path``, and
        the scenario's activation rule decides whether it serves. A job that continues the training start moves it to
        ``checkpoint.path``; the commit of any other job names the job's start (``TrainingCommit.train_from``). The
        commit's metrics are the checkpoint's training metrics and ``training_job``. Raise ``ValueError`` when a
        training metric is named ``training_job``: a metric is never overwritten.
        """
        if TRAINING_JOB_KEY in checkpoint.metrics:
            raise ValueError(f"checkpoint {job.job_id}:{checkpoint.step}'s training metrics name {TRAINING_JOB_KEY!r}")
        handled = replace(job, checkpoints_handled=job.checkpoints_handled + 1)
        published = replace(
            state,
            start=checkpoint.path if job.train_from is None else state.start,
            jobs=tuple(handled if running is job else running for running in state.jobs),
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
            TrainingCommit(
                state=published.to_dict(),
                metrics=metrics,
                artifact=Artifact.local(Path(checkpoint.artifact_path)),
                training_job_id=job.job_id,
                train_from=job.train_from,
                changed_kinds=self.component.changed_parts(job.start, checkpoint),
            )
        )

    def end(self, state: JobLoopState, job: RunningJob, status: JobStatus) -> JobCommit:
        """The commit that records the end of ``job``, a running job of ``state`` whose ended ``status`` the
        component reported.

        The job leaves ``jobs``, its outcome becomes ``last_job``, and ``finished_jobs`` grows by
        one. Raise ``ValueError`` when ``status`` is still running.
        """
        if status.state == "running":
            raise ValueError("a running job has not ended")
        outcome = JobOutcome(job.job_id, self.component.name, status.state, status.metrics, status.error)
        ended = JobLoopState(
            state.start,
            tuple(running for running in state.jobs if running is not job),
            state.finished_jobs + 1,
            outcome,
        )
        event = {key: value for key, value in outcome.to_dict().items() if key != "metrics"}
        return JobCommit(
            TrainingCommit(
                state=ended.to_dict(),
                metrics={**status.metrics, "training_job": event},
                training_job_id=job.job_id,
            )
        )

    def cancel(self, state: JobLoopState) -> None:
        """Cancel every running job of ``state`` at the component."""
        for job in state.jobs:
            self.component.cancel(job.job_id)

    def started_from(self, state: JobLoopState, start: str) -> JobLoopState:
        """The state after the training start is set to ``start``, which the component imported of a snapshot.

        The running job that continues the start before is cancelled; the next such job starts from ``start``. The
        jobs with their own start keep running, and the jobs that ended stay counted.
        """
        continuing = state.continuing_job
        if continuing is not None:
            self.component.cancel(continuing.job_id)
        return replace(state, start=start, jobs=tuple(job for job in state.jobs if job is not continuing))
