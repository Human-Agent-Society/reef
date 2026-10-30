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
from contextlib import suppress
from collections.abc import Mapping
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


def job_from_item(item: WorkItem) -> TrainingJob:
    """The ``TrainingJob`` that ``job_payload`` wrote into ``item``."""
    payload = cast(Mapping[str, Any], item.payload)
    return TrainingJob(
        item.id,
        payload["backend"],
        payload["start"],
        JobData.from_dict(payload["data"]),
        payload["parameters"],
    )


def checkpoint_json(checkpoint: SavedCheckpoint) -> dict[str, JsonValue]:
    return cast(
        dict[str, JsonValue],
        {
            "step": checkpoint.step,
            "path": checkpoint.path,
            "release_path": checkpoint.release_path,
            "metrics": dict(checkpoint.metrics),
        },
    )


def checkpoint_from_json(value: Mapping[str, Any]) -> SavedCheckpoint:
    return SavedCheckpoint(value["step"], value["path"], value["release_path"], value["metrics"])


def status_from_item(item: WorkItem) -> JobStatus:
    """The ``JobStatus`` of the job in ``item``: running until the item is final, then finished or failed."""
    checkpoints = tuple(checkpoint_from_json(cast(Mapping[str, Any], entry)) for entry in item.progress)
    if item.state == "finished":
        result = cast(Mapping[str, Any], item.result)
        return JobStatus(item.id, "finished", checkpoints, metrics=result.get("metrics", {}))
    if item.state == "failed":
        return JobStatus(item.id, "failed", checkpoints, error=item.error or "the job failed")
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
        """Stop taking jobs, wait for the thread, and close the component. A job in progress keeps its lease until
        it expires; then another worker continues it."""
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        self.component.close()

    def run(self) -> None:
        while not self.stopped.is_set():
            if not self.run_once():
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
            with suppress(WorkLeaseLost):
                self.queue.fail(JOBS_QUEUE, item.id, self.worker_id, f"{type(exc).__name__}: {exc}", retryable=False)
        return True

    def run_job(self, item: WorkItem) -> None:
        job = job_from_item(item)
        self.component.submit(job)
        self.queue.start(JOBS_QUEUE, item.id, self.worker_id)
        reported = len(item.progress)
        lease_at = 0.0
        while True:
            status = self.component.status(job.job_id)
            for number in range(reported, len(status.checkpoints)):
                self.queue.add_progress(
                    JOBS_QUEUE, item.id, self.worker_id, checkpoint_json(status.checkpoints[number]), number=number
                )
            reported = len(status.checkpoints)
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
            lease_at += self.poll_seconds
            if lease_at >= self.queue.lease_seconds / 3:
                self.queue.extend_lease(JOBS_QUEUE, item.id, self.worker_id)
                lease_at = 0.0
            if self.stopped.wait(self.poll_seconds):
                return


__all__ = [
    "JOBS_QUEUE",
    "JobQueueComponent",
    "TrainingJobWorker",
    "job_from_item",
    "job_kind",
    "job_payload",
    "status_from_item",
]
