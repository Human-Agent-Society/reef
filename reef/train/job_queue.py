"""Training jobs through the jobs queue: control adds jobs, the training service runs them.

Control does not call a training component directly. ``JobQueueComponent`` is the component that a trainer's
``JobLoop`` sees: it adds each ``TrainingJob`` to the ``jobs`` queue and reads the job's ``JobStatus`` back from
the queue item. ``TrainingJobWorker`` runs in the training service: it takes the jobs of its component, has the
component run each one, copies each saved checkpoint into the item's progress, and ends the item with the job.

A jobs-queue item:

- ``id`` is the job ID, which ``JobLoop.job_id`` makes from the job's inputs;
- ``kind`` is ``training_job:<component name>``, so a worker takes only the jobs of its component;
- ``serial_key`` is the scenario, so the jobs of one scenario run one at a time, in the order control added them;
- ``payload`` is the job (``job_payload``);
- each ``progress`` entry is one ``SavedCheckpoint``, in step order;
- ``result`` is ``{"state": "finished", "metrics": {...}}``.

A checkpoint's ``path`` and ``release_path`` are paths of the training service. In the one-process layout they
are local; a split deployment gives control and the training service one volume for them.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

from reef.artifact.files import ArtifactFiles
from reef.artifact.repository import BaseArtifactSource
from reef.core.job_data import JobData
from reef.execution.workers import ServiceWorker
from reef.runtime.jobs import JobStatus, ReleaseFormat, SavedCheckpoint, TrainingComponent, TrainingJob
from reef.storage.work_queue import JsonValue, NewWork, WorkItem, WorkLeaseLost, WorkQueue

logger = logging.getLogger(__name__)

#: The queue of training jobs.
JOBS_QUEUE = "jobs"


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
    )


def checkpoint_json(checkpoint: SavedCheckpoint) -> dict[str, JsonValue]:
    return {
        "step": checkpoint.step,
        "path": checkpoint.path,
        "release_path": checkpoint.release_path,
        "metrics": cast(JsonValue, dict(checkpoint.metrics)),
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


class JobQueueComponent(TrainingComponent):
    """The training component that control uses: it adds jobs to the jobs queue instead of running them.

    Values: ``component`` is the training component whose jobs the queue carries; its name, release format, base
    artifact and starts describe the jobs. ``queue`` is the work queue. ``scenario`` is None for the component
    itself, and the scenario name for the view that ``for_scenario`` returns; only a view submits jobs.
    ``owns_component`` is whether ``close`` closes ``component``; a training worker owns it otherwise.

    Purpose: control's trainers submit jobs and read their status through it (``JobLoop``); the training service
    runs the jobs with ``TrainingJobWorker``.
    """

    def __init__(
        self,
        component: TrainingComponent,
        queue: WorkQueue,
        *,
        scenario: str | None = None,
        owns_component: bool = False,
    ) -> None:
        if isinstance(component, JobQueueComponent):
            raise TypeError("a jobs-queue component wraps the component that runs the jobs")
        self.component = component
        self.queue = queue
        self.scenario = scenario
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

    def for_scenario(self, scenario: str) -> TrainingComponent:
        return JobQueueComponent(self.component, self.queue, scenario=scenario)

    def submit(self, job: TrainingJob) -> None:
        if self.scenario is None:
            raise RuntimeError("submit jobs through the view that for_scenario returns")
        if job.backend != self.name:
            raise ValueError(f"the job is for {job.backend!r}, not the {self.name!r} training component")
        if self.queue.get(JOBS_QUEUE, job.job_id) is not None:
            return
        self.component.check_job(job)
        self.queue.add(
            NewWork(
                JOBS_QUEUE, job.job_id, self.scenario, job_kind(self.name), job_payload(job), serial_key=self.scenario
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
        if self.owns_component and self.scenario is None:
            self.component.close()


class TrainingJobWorker(ServiceWorker):
    """One worker of the training service: it takes the jobs of one component from the jobs queue and runs them.

    Values: ``component`` runs the jobs; ``queue`` is the work queue; ``worker_id`` names this worker in leases;
    ``poll_seconds`` is the wait between two looks at the queue or at a running job.

    Purpose: the training service starts one per training component (``start``); ``stop`` stops taking jobs and
    closes the component, which the worker owns. Each job runs to its end on this worker unless the worker loses the lease; then the
    component cancels its local run, and the worker that takes the job next continues it from the job's
    checkpoints.
    """

    def __init__(
        self, component: TrainingComponent, queue: WorkQueue, *, worker_id: str, poll_seconds: float = 0.5
    ) -> None:
        if isinstance(component, JobQueueComponent):
            raise TypeError("a training worker runs jobs on the component itself, not on a jobs-queue component")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds is positive")
        self.component = component
        self.queue = queue
        self.worker_id = worker_id
        self.poll_seconds = poll_seconds
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        """Run the worker on a daemon thread."""
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self.run, name=f"reef-training-{self.component.name}", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop taking jobs, give a job in progress back to the queue, wait for the thread, and close the
        component. Another worker continues the job."""
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        self.component.close()

    def run(self) -> None:
        while not self.stopped.is_set():
            try:
                took = self.run_once()
            except Exception:
                # The store can fail for a moment; the worker tries again after its poll time.
                logger.exception("training worker %s could not read the jobs queue", self.worker_id)
                took = False
            if not took:
                self.stopped.wait(self.poll_seconds)

    def run_once(self) -> bool:
        """Take one job and run it to its end; return whether there was a job."""
        self.queue.requeue_expired()
        item = self.queue.take(JOBS_QUEUE, self.worker_id, [job_kind(self.component.name)])
        if item is None:
            return False
        try:
            self.run_job(item)
        except WorkLeaseLost:
            logger.warning("training worker %s lost job %s; cancelling its local run", self.worker_id, item.id)
            self.component.cancel(item.id)
        except Exception as exc:
            logger.exception("training worker %s could not run job %s", self.worker_id, item.id)
            # A raised error can pass, such as a dropped connection to the backend: the job runs again under its
            # ID until it uses up its attempts. A job that the component reports failed fails at once (``run_job``).
            with suppress(WorkLeaseLost):
                self.queue.fail(JOBS_QUEUE, item.id, self.worker_id, f"{type(exc).__name__}: {exc}", retryable=True)
        return True

    def run_job(self, item: WorkItem) -> None:
        """Run the job of ``item`` on the component and mirror its checkpoints, by step, into the item's progress.

        The lease is extended a third of a lease after each extension, measured on a monotonic clock, so a slow
        ``submit`` or ``status`` call counts too. At each poll the worker reads the item, and a job that control
        cancelled, or that another worker took, ends here with ``WorkLeaseLost``, which cancels the local run. A worker that continues a job reports only the checkpoints whose
        step the item does not hold yet.
        """
        job = job_from_item(item)
        extend_every = self.queue.lease_seconds / 3
        self.component.submit(job)
        self.queue.start(JOBS_QUEUE, item.id, self.worker_id)
        extended_at = time.monotonic()
        reported = {checkpoint_from_json(entry).step for entry in item.progress}
        progress_count = len(item.progress)
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
                self.queue.finish(
                    JOBS_QUEUE,
                    item.id,
                    self.worker_id,
                    {"state": "finished", "metrics": cast(JsonValue, dict(status.metrics))},
                )
                return
            if status.state == "failed":
                self.queue.fail(JOBS_QUEUE, item.id, self.worker_id, status.error or "the job failed", retryable=False)
                return
            if time.monotonic() - extended_at >= extend_every:
                self.queue.extend_lease(JOBS_QUEUE, item.id, self.worker_id)
                extended_at = time.monotonic()
            if self.stopped.wait(self.poll_seconds):
                # The job goes back to the queue, so another worker continues it at once: the component keeps
                # running it, and that worker's ``submit`` continues from the job's latest checkpoint.
                self.queue.fail(JOBS_QUEUE, item.id, self.worker_id, "the training worker stopped", retryable=True)
                return
            # Control can cancel the job (a deleted scenario, a new training start): stop at the next poll.
            held = self.queue.get(JOBS_QUEUE, item.id)
            if held is None or held.worker_id != self.worker_id or held.final:
                raise WorkLeaseLost(f"job {item.id} is no longer held by {self.worker_id}")


__all__ = [
    "JOBS_QUEUE",
    "JobQueueComponent",
    "TrainingJobWorker",
    "job_from_item",
    "job_kind",
    "job_payload",
    "status_from_item",
]
