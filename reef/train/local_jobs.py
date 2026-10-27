"""Training components that run their jobs in a thread of this process and keep each job in a directory.

``LocalJobComponent`` is the base of the Tinker and Slime components. It owns
the job bookkeeping that the ``TrainingComponent`` contract asks for: a job is
durable once ``submit`` returns, jobs run one at a time in submission order,
``status`` reads a job from its files, and a restarted component resumes each
job that has not ended. A subclass trains one job in :meth:`run`.
"""

from __future__ import annotations

import logging
import threading
from abc import abstractmethod
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.core.job_data import JobData
from reef.runtime.durable_files import read_json, write_json
from reef.runtime.jobs import JobStatus, SavedCheckpoint, TrainingComponent, TrainingJob

logger = logging.getLogger(__name__)

#: The submitted job, which a restarted component resumes from.
JOB_FILE = "job.json"
#: The job's end: its state, metrics and error. A job without it has not ended.
RESULT_FILE = "result.json"
#: The checkpoints a job saved so far, in step order.
CHECKPOINTS_FILE = "checkpoints.json"


class LocalJobComponent(TrainingComponent):
    """A training component that runs each job in its own thread, one job at a time.

    ``jobs_root`` holds one directory per job, named by the job ID, with
    ``job.json`` (the submitted job), ``checkpoints.json`` (the checkpoints
    saved so far, each ``{"step", "path", "release_path", "metrics"}``) and,
    once the job ended, ``result.json`` (``{"state", "metrics", "error"?}``).
    A job's metrics and its checkpoints' metrics must encode as JSON. The job
    thread writes a job's last checkpoint before its result, so a job with a
    result has its complete checkpoint list. The subclass calls
    :meth:`start_jobs` once it can train; until then no job runs.
    """

    def __init__(self, jobs_root: Path) -> None:
        self.jobs_root = jobs_root
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.jobs_lock = threading.Condition()
        self.queue: deque[str] = deque()
        self.closed = False
        self.job_thread: threading.Thread | None = None

    def start_jobs(self) -> None:
        """Run the jobs that have not ended, in submission order: those a previous process left and those submitted."""
        with self.jobs_lock:
            unfinished = [
                path
                for path in self.jobs_root.iterdir()
                if (path / JOB_FILE).is_file() and not (path / RESULT_FILE).exists()
            ]
            # Every queued job has a job file and no result, so the directory listing is the whole queue.
            # The job files' modification times keep the submission order.
            self.queue = deque(
                path.name for path in sorted(unfinished, key=lambda path: (path / JOB_FILE).stat().st_mtime)
            )
        self.job_thread = threading.Thread(target=self.run_jobs, name=f"reef-{self.name}-jobs", daemon=True)
        self.job_thread.start()

    def job_directory(self, job_id: str) -> Path:
        return self.jobs_root / job_id

    # -- What a subclass provides

    @abstractmethod
    def check_job(self, job: TrainingJob) -> None:
        """Raise ``ValueError`` for a job this component cannot train; ``submit`` then records nothing."""

    @abstractmethod
    def run(self, job: TrainingJob, directory: Path) -> Mapping[str, Any]:
        """Train ``job`` to its end and return its metrics; record each checkpoint with :meth:`save_checkpoint`.

        A resumed job finds the checkpoints it saved before in :meth:`saved`
        and continues after them. Raising ends the job as failed.
        """

    def close_resources(self) -> None:
        """Release what the subclass owns, after the job thread stopped."""
        return

    # -- Helpers for run

    def saved(self, directory: Path) -> tuple[SavedCheckpoint, ...]:
        """The checkpoints the job in ``directory`` saved so far."""
        value = read_json(directory / CHECKPOINTS_FILE)
        if value is None:
            return ()
        return tuple(
            SavedCheckpoint(entry["step"], entry["path"], entry["release_path"], entry["metrics"])
            for entry in value["checkpoints"]
        )

    def save_checkpoint(self, directory: Path, checkpoint: SavedCheckpoint) -> None:
        """Record a complete checkpoint: once this returns, ``status`` reports it.

        A job that has ended, which only a cancel does while the job runs, records nothing: its checkpoint list
        was complete when its result was written.
        """
        with self.jobs_lock:
            if (directory / RESULT_FILE).exists():
                logger.info("%s dropped checkpoint %d of an ended job", self.name, checkpoint.step)
                return
            entries = [
                {
                    "step": saved.step,
                    "path": saved.path,
                    "release_path": saved.release_path,
                    "metrics": dict(saved.metrics),
                }
                for saved in (*self.saved(directory), checkpoint)
            ]
            write_json(directory / CHECKPOINTS_FILE, {"checkpoints": entries})

    def stopping(self, directory: Path) -> bool:
        """Whether the job in ``directory`` should stop at its next save: it was cancelled or the component closes."""
        with self.jobs_lock:
            return self.closed or (directory / RESULT_FILE).exists()

    # -- TrainingComponent

    def submit(self, job: TrainingJob) -> None:
        directory = self.job_directory(job.job_id)
        with self.jobs_lock:
            if self.closed:
                raise RuntimeError(f"the {self.name} training component is closed")
            if (directory / JOB_FILE).exists():
                return
            self.check_job(job)
            write_json(
                directory / JOB_FILE,
                {"start": job.start, "data": job.data.to_dict(), "parameters": dict(job.parameters)},
            )
            self.queue.append(job.job_id)
            self.jobs_lock.notify_all()

    def status(self, job_id: str) -> JobStatus:
        directory = self.job_directory(job_id)
        if not (directory / JOB_FILE).exists():
            raise KeyError(job_id)
        # The result first: the job thread writes it after the last checkpoint, so the list read next is complete.
        result = read_json(directory / RESULT_FILE)
        checkpoints = self.saved(directory)
        if result is None:
            return JobStatus(job_id, "running", checkpoints)
        return JobStatus(job_id, result["state"], checkpoints, metrics=result["metrics"], error=result.get("error"))

    def cancel(self, job_id: str) -> None:
        """A queued job never starts; a running job ends at its next save; an unknown or ended job stays as it is."""
        directory = self.job_directory(job_id)
        with self.jobs_lock:
            if not (directory / JOB_FILE).exists() or (directory / RESULT_FILE).exists():
                return
            write_json(directory / RESULT_FILE, {"state": "failed", "metrics": {}, "error": "cancelled"})
            if job_id in self.queue and self.queue[0] != job_id:
                self.queue.remove(job_id)

    def close(self) -> None:
        with self.jobs_lock:
            if self.closed:
                return
            self.closed = True
            self.jobs_lock.notify_all()
        if self.job_thread is not None:
            self.job_thread.join()
        self.close_resources()

    # -- The job thread

    def run_jobs(self) -> None:
        """The job thread: run the queued jobs one at a time until the component closes."""
        while True:
            with self.jobs_lock:
                while not self.queue and not self.closed:
                    self.jobs_lock.wait()
                if self.closed:
                    return
                job_id = self.queue[0]
                # A job cancelled while it waited at the head of the queue never starts.
                cancelled = (self.job_directory(job_id) / RESULT_FILE).exists()
            directory = self.job_directory(job_id)
            result: dict[str, Any] | None = None
            if not cancelled:
                try:
                    submitted = read_json(directory / JOB_FILE)
                    if submitted is None:
                        raise RuntimeError(f"training job {job_id} has no job file")
                    job = TrainingJob(
                        job_id,
                        self.name,
                        submitted["start"],
                        JobData.from_dict(submitted["data"]),
                        submitted["parameters"],
                    )
                    result = {"state": "finished", "metrics": dict(self.run(job, directory))}
                except Exception as exc:
                    logger.exception("%s training job %s failed", self.name, job_id)
                    result = {"state": "failed", "metrics": {}, "error": f"{type(exc).__name__}: {exc}"}
            with self.jobs_lock:
                if self.closed:
                    # The job did not end: a restart resumes it.
                    return
                if result is not None and not (directory / RESULT_FILE).exists():
                    write_json(directory / RESULT_FILE, result)
                if self.queue and self.queue[0] == job_id:
                    self.queue.popleft()
