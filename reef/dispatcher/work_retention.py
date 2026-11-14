"""Control's sweep of ended work items: the queues keep each ended item for a retention time, then lose it."""

from __future__ import annotations

import logging
import threading
import time

from reef.artifact.repository import RepositoryBackendFactory
from reef.dispatcher.scenario_deletes import DeletedScenarioCleanup
from reef.execution.workers import ServiceWorker
from reef.storage.work_queue import WorkQueue
from reef.train.job_queue import JOBS_QUEUE, discard_ended_content

logger = logging.getLogger(__name__)

#: The seconds between two sweeps.
SWEEP_INTERVAL_SECONDS = 3600.0


class EndedWorkSweeper(ServiceWorker):
    """Deletes the final work items that changed last more than ``retention_seconds`` ago, every
    ``interval_seconds``.

    Values: ``queue`` holds the items; ``backends`` opens the artifact store of a scenario; ``deletions`` cleans up
    the deleted scenarios, or is None for a sweep of ended items only; ``retention_seconds`` and
    ``interval_seconds`` are positive. A failed sweep is logged, and the next one runs after ``interval_seconds``,
    or at once after ``wake``. A sweep first cleans up the deleted scenarios (``DeletedScenarioCleanup.clean_up``),
    so their jobs end and their content goes before the rest of the sweep. Before a sweep deletes the items, it discards the stored content of the checkpoints
    of the training jobs among them (``reef.train.job_queue.discard_ended_content``), so the artifact store keeps
    no checkpoint that no commit took for longer than its job's item. A job item whose content could not be
    discarded stays for the next sweep, for at most one more retention time: then the sweep gives the content up.
    After the items, the sweep frees the storage that discarded content still holds
    (``RepositoryBackendFactory.reclaim_discarded``): a discard only deletes the content's name, so its files stay
    until the next sweep.

    Purpose: the dispatcher of a control instance runs one, so that the queues hold the items that a producer may
    still read (a job's trainer, an evaluation's results) and no item forever, and so that nothing of a deleted
    scenario stays. A delete wakes it.
    """

    def __init__(
        self,
        queue: WorkQueue,
        backends: RepositoryBackendFactory,
        *,
        retention_seconds: float,
        interval_seconds: float = SWEEP_INTERVAL_SECONDS,
        deletions: DeletedScenarioCleanup | None = None,
    ) -> None:
        if retention_seconds <= 0 or interval_seconds <= 0:
            raise ValueError("retention_seconds and interval_seconds are positive")
        self.queue = queue
        self.backends = backends
        self.retention_seconds = retention_seconds
        self.interval_seconds = interval_seconds
        self.deletions = deletions
        self.stopped = threading.Event()
        #: Set by ``wake`` and ``stop``: the thread sweeps at once, or ends.
        self.woken = threading.Event()
        #: Held for one whole sweep, so a sweep that a caller runs waits for the one that the thread runs.
        self.sweep_lock = threading.Lock()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is not None:
            return
        self.stopped.clear()
        self.thread = threading.Thread(target=self.run, name="reef-ended-work-sweep", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stopped.set()
        self.woken.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def run(self) -> None:
        while not self.stopped.is_set():
            self.sweep()
            self.woken.wait(self.interval_seconds)
            self.woken.clear()

    def wake(self) -> None:
        """Let the thread sweep now, not at the end of its interval."""
        self.woken.set()

    def sweep(self) -> int:
        """Clean up the deleted scenarios, delete the items that ended before the retention, then free the storage
        of discarded content; return how many items went, 0 when the store failed."""
        with self.sweep_lock:
            return self.sweep_once()

    def sweep_once(self) -> int:
        """One sweep, as ``sweep`` describes it; the caller holds ``sweep_lock``."""
        if self.deletions is not None:
            try:
                self.deletions.clean_up()
            except Exception:
                logger.exception("could not clean up the deleted scenarios")
        try:
            ended_before = time.time() - self.retention_seconds
            kept = discard_ended_content(
                self.queue,
                self.backends,
                ended_before=ended_before,
                give_up_before=ended_before - self.retention_seconds,
            )
            deleted = self.queue.delete_final(finished_before=ended_before, keep={JOBS_QUEUE: kept})
        except Exception:
            logger.exception("could not delete the ended work items")
            deleted = 0
        try:
            self.backends.reclaim_discarded()
        except Exception:
            logger.exception("could not free the storage of discarded content")
        return deleted


__all__ = ["SWEEP_INTERVAL_SECONDS", "EndedWorkSweeper"]
