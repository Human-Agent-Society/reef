"""Training jobs through the jobs queue: control adds jobs, the training service runs them.

Control does not call a training component directly. ``JobQueueComponent`` is the component that a trainer's
``JobLoop`` sees: it adds each ``TrainingJob`` to the ``jobs`` queue and reads the job's ``JobStatus`` back from
the queue item. ``TrainingJobWorker`` runs in the training service: it takes the jobs of its component, has the
component run each one, copies each saved checkpoint into the item's progress, and ends the item with the job.

A jobs-queue item:

- ``id`` is the job ID, which ``JobLoop.job_id`` makes from the job's inputs;
- ``kind`` is ``training_job:<component name>``, so a worker takes only the jobs of its component;
- ``scenario`` and ``serial_key`` are the job's scenario, so the jobs of one scenario run one at a time, in the order control added them;
- ``payload`` is the job (``job_payload``);
- each ``progress`` entry is one ``SavedCheckpoint``, in step order;
- ``result`` is ``{"state": "finished", "metrics": {...}}``;
- ``metadata["experiment"]``, when present, is the ``ExperimentRun`` that records the job, as ``to_dict`` gives
  it. Control writes it (``JobExperimentRecorder``) when its experiment tracker records a commit of the job, so the
  job listing of any service reads it from the item.

A checkpoint's ``path`` and ``release_path`` are paths of the training service. In the one-process layout they
are local; a split deployment gives control and the training service one volume for them.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from reef.artifact.files import ArtifactFiles
from reef.artifact.repository import BaseArtifactSource
from reef.core.artifact_ref import ArtifactRef
from reef.core.job_data import JobData
from reef.execution.workers import ItemFailed, QueueWorker, WorkerStopped
from reef.observability.base import (
    ExperimentLogger,
    ExperimentRun,
    ExperimentTracker,
    TrainFromExperimentEvent,
    TrainingExperimentContext,
    TrainingExperimentEvent,
)
from reef.runtime.jobs import JobStatus, ReleaseFormat, SavedCheckpoint, TrainingComponent, TrainingJob
from reef.storage.work_queue import JsonValue, NewWork, WorkItem, WorkLeaseLost, WorkQueue

logger = logging.getLogger(__name__)

#: The queue of training jobs.
JOBS_QUEUE = "jobs"
#: The key of a job item's ``metadata`` that holds the experiment run that records the job.
EXPERIMENT_METADATA_KEY = "experiment"


def job_kind(component_name: str) -> str:
    """The work-item kind of the jobs of the component named ``component_name``."""
    return f"training_job:{component_name}"


def job_payload(job: TrainingJob) -> dict[str, JsonValue]:
    """The jobs-queue payload of ``job``: its backend, start, data and parameters, as JSON."""
    return cast(
        dict[str, JsonValue],
        {
            "backend": job.backend,
            "start": job.start,
            "data": job.data.to_dict(),
            "parameters": dict(job.parameters),
        },
    )


def json_object(value: JsonValue, name: str) -> Mapping[str, JsonValue]:
    """``value`` as a JSON object; ``ValueError`` when it is another JSON value."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} is a JSON object")
    return value


def json_text(value: JsonValue, name: str) -> str:
    """``value`` as a non-empty string; ``ValueError`` when it is another JSON value."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is a non-empty string")
    return value


def job_from_item(item: WorkItem) -> TrainingJob:
    """The ``TrainingJob`` that ``job_payload`` wrote into ``item``; ``ValueError`` for another payload."""
    payload = item.payload
    return TrainingJob(
        item.id,
        json_text(payload.get("backend"), "a job's backend"),
        json_text(payload.get("start"), "a job's start"),
        JobData.from_dict(json_object(payload.get("data"), "a job's data")),
        json_object(payload.get("parameters"), "a job's parameters"),
        scenario=item.scenario,
    )


def json_texts(value: JsonValue, name: str) -> list[str]:
    """``value`` as a list of non-empty strings; ``ValueError`` when it is another JSON value."""
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{name} is a list of non-empty strings")
    return list(value)


def checkpoint_json(checkpoint: SavedCheckpoint) -> dict[str, JsonValue]:
    return {
        "step": checkpoint.step,
        "path": checkpoint.path,
        "release_path": checkpoint.release_path,
        "metrics": cast(JsonValue, dict(checkpoint.metrics)),
        "changed_kinds": cast(JsonValue, sorted(checkpoint.changed_kinds)),
    }


def checkpoint_from_json(entry: Mapping[str, JsonValue]) -> SavedCheckpoint:
    """The ``SavedCheckpoint`` that ``checkpoint_json`` wrote; ``ValueError`` for another object."""
    step = entry.get("step")
    if isinstance(step, bool) or not isinstance(step, int):
        raise ValueError("a checkpoint's step is an integer")
    return SavedCheckpoint(
        step,
        json_text(entry.get("path"), "a checkpoint's path"),
        json_text(entry.get("release_path"), "a checkpoint's release_path"),
        json_object(entry.get("metrics"), "a checkpoint's metrics"),
        frozenset(json_texts(entry.get("changed_kinds"), "a checkpoint's changed_kinds")),
    )


def status_from_item(item: WorkItem) -> JobStatus:
    """The ``JobStatus`` of the job in ``item``: running until the item is final, then finished or failed."""
    checkpoints = tuple(checkpoint_from_json(entry) for entry in item.progress)
    if item.state == "finished" and item.result is not None:
        return JobStatus(
            item.id, "finished", checkpoints, metrics=json_object(item.result.get("metrics", {}), "metrics")
        )
    if item.state == "failed":
        # A failed item always carries its error (``WorkItem``).
        return JobStatus(item.id, "failed", checkpoints, error=item.error)
    return JobStatus(item.id, "running", checkpoints)


#: The error of a job that a client cancelled.
JOB_CANCELLED_ERROR = "cancelled by request"


def job_json(item: WorkItem) -> dict[str, JsonValue]:
    """The jobs-queue ``item`` as the JSON object that the job listing gives; ``ValueError`` for an item whose
    payload ``job_payload`` did not write.

    The object has these keys:

    - ``id``, ``state``, ``attempt_count`` and ``error``: the item's fields (``WorkItem``).
    - ``checkpoints``: the saved checkpoints that the job reported, in step order, each as ``checkpoint_json``.
    - ``result``: ``{"state": "finished", "metrics": {...}}`` once the job finished, else None.
    - ``parameters``: the job's parameters, a JSON object: the recipe's parameters for the kind of the job's
      samples, with the instruction's parameters over them. They hold the training ``method`` and the
      ``scheduling`` when the recipe or the instruction gives them.
    - ``sample_count``: the number of samples in the job's data, an integer from 0.
    - ``sample_kind``: ``"trajectory"`` or ``"task"``, the kind of the job's samples, which one job never mixes;
      None for data with no sample.
    - ``instruction``: ``{id, text, parameters}`` of the training instruction (``POST /reef/train``) that the job
      trains, where ``parameters`` is the JSON object that the instruction gave, ``{}`` when it gave none; None for
      a job that the training mode ``auto`` made without an instruction.
    - ``created_at``, ``started_at`` and ``ended_at``: ISO 8601 UTC times of the item, when control added the job,
      when a worker first took it, and when it finished or failed. ``started_at`` is None until a worker takes the
      job, and ``ended_at`` until the job ends.
    - ``experiment``: ``{tracker, project, entity, run_id, url}`` of the experiment run that records the job
      (``ExperimentRun.to_dict``), once control's tracker recorded a commit of the job; else None, and always None
      without an experiment tracker.
    """
    payload = item.payload
    data = json_object(payload.get("data"), "a job's data")
    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError("a job's samples are a JSON array")
    kinds = {json_text(json_object(sample, "a job's sample").get("kind"), "a sample's kind") for sample in samples}
    if len(kinds) > 1:
        raise ValueError(f"job {item.id} mixes the sample kinds {sorted(kinds)}")
    request_value = data.get("request")
    if request_value is None:
        instruction: JsonValue = None
    else:
        request = json_object(request_value, "a job's instruction")
        instruction = {
            "id": json_text(request.get("id"), "an instruction's id"),
            "text": json_text(request.get("text"), "an instruction's text"),
            "parameters": dict(json_object(request.get("parameters", {}), "an instruction's parameters")),
        }
    experiment = item.metadata.get(EXPERIMENT_METADATA_KEY)
    return {
        "id": item.id,
        "state": item.state,
        "attempt_count": item.attempt_count,
        "checkpoints": [dict(progress) for progress in item.progress],
        "result": None if item.result is None else dict(item.result),
        "error": item.error,
        "parameters": dict(json_object(payload.get("parameters"), "a job's parameters")),
        "sample_count": len(samples),
        "sample_kind": next(iter(kinds), None),
        "instruction": instruction,
        "created_at": utc_time(item.created_at),
        "started_at": None if item.started_at is None else utc_time(item.started_at),
        "ended_at": None if item.ended_at is None else utc_time(item.ended_at),
        "experiment": (
            None
            if experiment is None
            else cast(JsonValue, ExperimentRun.from_dict(json_object(experiment, "a job's experiment")).to_dict())
        ),
    }


def utc_time(seconds: float) -> str:
    """The time ``seconds`` after the epoch as an ISO 8601 string in UTC, such as ``2026-10-31T08:15:00.250000+00:00``."""
    return datetime.fromtimestamp(seconds, UTC).isoformat()


def scenario_jobs(queue: WorkQueue, scenario: str) -> list[dict[str, JsonValue]]:
    """The training jobs of ``scenario`` in ``queue``, oldest first, each as ``job_json`` gives it."""
    return [job_json(item) for item in queue.items(JOBS_QUEUE, scenario=scenario)]


def cancel_scenario_job(queue: WorkQueue, scenario: str, job_id: str) -> dict[str, JsonValue] | None:
    """Cancel the training job ``job_id`` of ``scenario``; return the job, as ``job_json`` gives it, after the cancel.

    A job that has not ended fails with ``JOB_CANCELLED_ERROR``: a queued job at once, and a running job's training
    worker cancels the run at its next poll. The trainer reads the failed job and records its end. A job that ended
    stays as it is. Return None when ``scenario`` has no job ``job_id``.
    """
    item = queue.get(JOBS_QUEUE, job_id)
    if item is None or item.scenario != scenario:
        return None
    cancelled = queue.cancel(JOBS_QUEUE, job_id, JOB_CANCELLED_ERROR)
    return job_json(item if cancelled is None else cancelled)


class JobQueueComponent(TrainingComponent):
    """The training component that control uses: it adds jobs to the jobs queue instead of running them.

    Values: ``component`` is the training component whose jobs the queue carries; its name, release format, base
    artifact and starts describe the jobs. ``queue`` is the work queue. ``owns_component`` is whether ``close``
    closes ``component``; a training worker owns it otherwise. A job goes to the queue under its scenario, which is
    also its serial key, so the jobs of one scenario run one at a time, in the order of their submission.

    Purpose: control's trainers submit jobs and read their status through it (``JobLoop``); the training service
    runs the jobs with ``TrainingJobWorker``.
    """

    def __init__(
        self,
        component: TrainingComponent,
        queue: WorkQueue,
        *,
        owns_component: bool = False,
    ) -> None:
        if isinstance(component, JobQueueComponent):
            raise TypeError("a jobs-queue component wraps the component that runs the jobs")
        self.component = component
        self.queue = queue
        #: Whether ``close`` closes ``component``: true in a process that runs no training worker for it.
        self.owns_component = owns_component

    @property
    def name(self) -> str:
        return self.component.name

    @property
    def release_format(self) -> ReleaseFormat | None:
        return self.component.release_format

    def base_artifact_source(self) -> BaseArtifactSource:
        return self.component.base_artifact_source()

    @property
    def base_start(self) -> str:
        return self.component.base_start

    @property
    def shares_scenarios(self) -> bool:
        return self.component.shares_scenarios

    @property
    def starts_from_releases(self) -> bool:
        return self.component.starts_from_releases

    def check_job(self, job: TrainingJob) -> None:
        self.component.check_job(job)

    def check_task_jobs(self, parameters: Mapping[str, Any]) -> None:
        self.component.check_task_jobs(parameters)

    def check_release(self, files: ArtifactFiles) -> None:
        self.component.check_release(files)

    def import_start(self, release: Path) -> str:
        return self.component.import_start(release)

    def submit(self, job: TrainingJob) -> None:
        if job.backend != self.name:
            raise ValueError(f"the job is for {job.backend!r}, not the {self.name!r} training component")
        if self.queue.get(JOBS_QUEUE, job.job_id) is not None:
            return
        self.component.check_job(job)
        self.queue.add(
            NewWork(
                JOBS_QUEUE, job.job_id, job.scenario, job_kind(self.name), job_payload(job), serial_key=job.scenario
            )
        )

    def status(self, job_id: str) -> JobStatus:
        item = self.queue.get(JOBS_QUEUE, job_id)
        if item is None:
            raise KeyError(job_id)
        return status_from_item(item)

    def cancel(self, job_id: str) -> None:
        self.queue.cancel(JOBS_QUEUE, job_id, "cancelled")

    def close(self) -> None:
        if self.owns_component:
            self.component.close()


class JobExperimentRecorder(ExperimentTracker):
    """Control's experiment tracker: ``tracker``, which also writes into each training job's item the run that
    records the job.

    Values: ``tracker`` is the experiment tracker that records the scenarios' training, and not a
    ``JobExperimentRecorder``; ``queue`` holds the jobs queue. Every method but ``record`` is ``tracker``'s.

    ``record(event)`` gives ``event`` to ``tracker``. Then, for an event of a training job, it sets
    ``metadata["experiment"]`` of the job's item to ``tracker.experiment_run(event.context).to_dict()``
    (``WorkQueue.set_metadata``). Nothing is written when the tracker has no run for the event, or the queue
    has no such item, such as for the step of a candidate backend. Each commit of a job (its submission, its
    checkpoints and its end) records the same run, so a write after the first changes nothing.

    Purpose: the dispatcher of control wraps its tracker in it when the recipe trains with a training component.
    The tracker reports from control's report thread (``ExperimentReportQueue``), after each commit lands. The
    run is open in this process, so its URL is the one that the run gives; the job listing of any service reads
    it from the item (``job_json``).
    """

    def __init__(self, tracker: ExperimentTracker, queue: WorkQueue) -> None:
        if isinstance(tracker, JobExperimentRecorder):
            raise TypeError("a job experiment recorder wraps the tracker that records the training")
        self.tracker = tracker
        self.queue = queue

    @property
    def operational_metrics_enabled(self) -> bool:
        return self.tracker.operational_metrics_enabled

    def bind_scenario(
        self, *, scenario: str, recipe: str, source_artifact_ref: ArtifactRef, run_segment: int
    ) -> ExperimentLogger:
        return self.tracker.bind_scenario(
            scenario=scenario, recipe=recipe, source_artifact_ref=source_artifact_ref, run_segment=run_segment
        )

    def correlation_metrics(self, context: TrainingExperimentContext) -> Mapping[str, Any]:
        return self.tracker.correlation_metrics(context)

    def experiment_run(self, context: TrainingExperimentContext) -> ExperimentRun | None:
        return self.tracker.experiment_run(context)

    def record(self, event: TrainingExperimentEvent) -> None:
        self.tracker.record(event)
        if event.training_job_id is None:
            return
        run = self.tracker.experiment_run(event.context)
        if run is not None:
            self.queue.set_metadata(
                JOBS_QUEUE, event.training_job_id, EXPERIMENT_METADATA_KEY, cast(JsonValue, run.to_dict())
            )

    def record_train_from(self, event: TrainFromExperimentEvent) -> None:
        self.tracker.record_train_from(event)

    def close(self) -> None:
        self.tracker.close()


class TrainingJobWorker(QueueWorker):
    """One worker of the training service: it takes the jobs of one component from the jobs queue and runs them.

    Values: as ``QueueWorker``, on the jobs of ``component``, which runs them; ``poll_seconds`` is also the wait
    between two looks at a running job.

    Purpose: the training service starts one per training component (``start``); ``stop`` stops taking jobs and
    closes the component, which the worker owns. Each job runs to its end on this worker unless the worker loses
    the lease; then the component cancels its local run, and the worker that takes the job next continues it from
    the job's checkpoints. A job that the component reports failed fails for good.
    """

    def __init__(
        self, component: TrainingComponent, queue: WorkQueue, *, worker_id: str, poll_seconds: float = 0.5
    ) -> None:
        if isinstance(component, JobQueueComponent):
            raise TypeError("a training worker runs jobs on the component itself, not on a jobs-queue component")
        super().__init__(queue, JOBS_QUEUE, [job_kind(component.name)], worker_id=worker_id, poll_seconds=poll_seconds)
        self.component = component

    def start(self) -> None:
        """Start the component's jobs in this process (``TrainingComponent.start_jobs``), then the worker."""
        if self.thread is None:
            self.component.start_jobs()
        super().start()

    def stop(self) -> None:
        """Stop taking jobs, give a job in progress back to the queue, wait for the thread, and close the
        component. Another worker continues the job."""
        super().stop()
        self.component.close()

    def run_item(self, item: WorkItem, lease_lost: threading.Event) -> Mapping[str, JsonValue]:
        """Run the job of ``item`` on the component and mirror its checkpoints, by step, into the item's progress.

        At each poll the worker reads the item: a job that control cancelled, or that another worker took, ends
        here, and the component cancels its local run. A worker that continues a job reports only the checkpoints
        whose step the item does not hold yet. When the worker stops, the job goes back to the queue, so another
        worker continues it at once: the component keeps running it, and that worker's ``submit`` continues from
        the job's latest checkpoint.
        """
        job = job_from_item(item)
        self.component.submit(job)
        reported = {checkpoint_from_json(entry).step for entry in item.progress}
        progress_count = len(item.progress)
        try:
            while True:
                status = self.component.status(job.job_id)
                for checkpoint in status.checkpoints:
                    if checkpoint.step in reported:
                        continue
                    self.queue.add_progress(
                        JOBS_QUEUE, item.id, self.worker_id, checkpoint_json(checkpoint), number=progress_count
                    )
                    reported.add(checkpoint.step)
                    progress_count += 1
                if status.state == "finished":
                    return {"state": "finished", "metrics": cast(JsonValue, dict(status.metrics))}
                if status.state == "failed":
                    raise ItemFailed(status.error or "the job failed")
                if self.stopped.wait(self.poll_seconds):
                    raise WorkerStopped
                # Control can cancel the job (a deleted scenario, a new training start): stop at the next poll.
                held = self.queue.get(JOBS_QUEUE, item.id)
                if lease_lost.is_set() or held is None or held.worker_id != self.worker_id or held.final:
                    raise WorkLeaseLost(f"job {item.id} is no longer held by {self.worker_id}")
        except WorkLeaseLost:
            logger.warning("training worker %s lost job %s; cancelling its local run", self.worker_id, item.id)
            self.component.cancel(item.id)
            raise


__all__ = [
    "EXPERIMENT_METADATA_KEY",
    "JOBS_QUEUE",
    "JOB_CANCELLED_ERROR",
    "JobExperimentRecorder",
    "JobQueueComponent",
    "TrainingJobWorker",
    "cancel_scenario_job",
    "job_from_item",
    "job_json",
    "job_kind",
    "job_payload",
    "scenario_jobs",
    "status_from_item",
]
