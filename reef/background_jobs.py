"""Asynchronous jobs off the trainer thread: a job runner and the worker that runs it.

Pipeline stages use the worker for slow work that must not block the trainer,
such as judge calls or task-generation rounds. The module uses only the standard
library, so every package can depend on it.
"""

from __future__ import annotations

import asyncio
import logging
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Lock, Thread

logger = logging.getLogger(__name__)

#: How long ``close()`` lets in-flight jobs finish before cancelling.
CLOSE_GRACE_S = 5.0


@dataclass(frozen=True)
class SupportsReceipt:
    """The receipt of a job or of its result. The worker reads only this field.

    The caller defines the job and result types. Both types, and
    :class:`Failed`, use this class as their base.
    """

    receipt: str


@dataclass(frozen=True)
class Failed(SupportsReceipt):
    """The worker's result for a job that never finished."""


class BackgroundJobRunner(ABC):
    """Run one job and return its result. The worker calls ``run`` on its own thread."""

    @abstractmethod
    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        """Return the result for ``job``. An exception resolves the job as :class:`Failed`."""


class BackgroundJobWorker:
    """Run jobs off the trainer thread: jobs go in one queue, results come out of another.

    The worker owns only execution: one daemon thread, an event loop, and the
    two queues. The ``BackgroundJobRunner`` that the caller gives at construction
    decides what a job does.

    A fault never makes the worker hang. A failed or cancelled job resolves as
    :class:`Failed`. If the event loop stops, the worker marks itself broken
    and resolves each queued job as :class:`Failed`. It then rejects new jobs,
    so the caller must retire their records directly.
    """

    def __init__(self, runner: BackgroundJobRunner, *, thread_name: str, concurrency: int = 8) -> None:
        self.runner = runner
        self.thread_name = thread_name
        self.concurrency = max(1, concurrency)
        self.jobs: queue.SimpleQueue[SupportsReceipt | None] = queue.SimpleQueue()
        self.results: queue.SimpleQueue[SupportsReceipt] = queue.SimpleQueue()
        self.thread: Thread | None = None
        # Lock submit against the change to broken or closed. Thus no job can
        # enter the queue after the final drain and leave its receipt
        # unresolved.
        self.gate = Lock()
        self.broken = False
        self.closed = False

    def submit(self, job: SupportsReceipt) -> bool:
        """Queue one job. Return ``False`` if the job will never run."""
        with self.gate:
            if self.closed or self.broken:
                return False
            if self.thread is None:
                self.thread = Thread(target=self.run_thread, name=self.thread_name, daemon=True)
                self.thread.start()
            self.jobs.put(job)
            return True

    def poll(self) -> list[SupportsReceipt]:
        """Return every finished result without blocking."""
        results: list[SupportsReceipt] = []
        try:
            while True:
                results.append(self.results.get_nowait())
        except queue.Empty:
            return results

    def close(self) -> None:
        """Stop accepting jobs, give running jobs a limited time to finish, and join the thread."""
        with self.gate:
            if self.closed:
                return
            self.closed = True
        if self.thread is None:
            return
        self.jobs.put(None)
        self.thread.join(timeout=CLOSE_GRACE_S + 10.0)

    # ------------------------------------------------------------ worker side

    def run_thread(self) -> None:
        try:
            asyncio.run(self.run_jobs())
        except Exception:
            logger.exception("background job worker %s failed; queued jobs resolve as Failed", self.thread_name)
        finally:
            with self.gate:
                self.broken = True
            # ``broken`` is set, so no submit can add a job during this drain.
            while True:
                try:
                    job = self.jobs.get_nowait()
                except queue.Empty:
                    break
                if job is not None:
                    self.results.put(Failed(job.receipt))

    async def run_jobs(self) -> None:
        # At most ``concurrency`` jobs run at the same time. The semaphore
        # belongs to this event loop, as does the rest of the worker state.
        semaphore = asyncio.Semaphore(self.concurrency)
        loop = asyncio.get_running_loop()
        tasks: set[asyncio.Task] = set()
        while True:
            job = await asyncio.to_thread(self.jobs.get)
            if job is None:
                break
            task = loop.create_task(self.run_one(job, semaphore))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        if tasks:
            # asyncio.run cancels the jobs that do not finish in the grace
            # period. Each cancelled job still resolves its receipt in its
            # finally block.
            await asyncio.wait(tasks, timeout=CLOSE_GRACE_S)

    async def run_one(self, job: SupportsReceipt, semaphore: asyncio.Semaphore) -> None:
        resolved = False
        try:
            async with semaphore:
                self.results.put(await self.runner.run(job))
            resolved = True
        except Exception:
            logger.exception("background job for receipt %s failed; it resolves as Failed", job.receipt)
        finally:
            if not resolved:  # an exception, or a cancellation at shutdown
                self.results.put(Failed(job.receipt))
