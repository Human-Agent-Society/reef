"""Control's sweep of ended work items: the queues keep each ended item for a retention time, then lose it."""

from __future__ import annotations

import logging
import threading
import time

from reef.execution.workers import ServiceWorker
from reef.storage.work_queue import WorkQueue

logger = logging.getLogger(__name__)

#: The seconds between two sweeps.
SWEEP_INTERVAL_SECONDS = 3600.0


class EndedWorkSweeper(ServiceWorker):
    """Deletes the final work items that changed last more than ``retention_seconds`` ago, every
    ``interval_seconds``.

    Values: ``queue`` holds the items; ``retention_seconds`` and ``interval_seconds`` are positive. A failed sweep is
    logged, and the next one runs after ``interval_seconds``.

    Purpose: the dispatcher of a control instance runs one, so that the queues hold the items that a producer may
    still read (a job's trainer, an evaluation's results) and no item forever.
    """

    def __init__(
        self, queue: WorkQueue, *, retention_seconds: float, interval_seconds: float = SWEEP_INTERVAL_SECONDS
    ) -> None:
        if retention_seconds <= 0 or interval_seconds <= 0:
            raise ValueError("retention_seconds and interval_seconds are positive")
        self.queue = queue
        self.retention_seconds = retention_seconds
        self.interval_seconds = interval_seconds
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is not None:
            return
        self.stopped.clear()
        self.thread = threading.Thread(target=self.run, name="reef-ended-work-sweep", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def run(self) -> None:
        while not self.stopped.is_set():
            self.sweep()
            self.stopped.wait(self.interval_seconds)

    def sweep(self) -> int:
        """Delete the items that ended before the retention; return how many, 0 when the store failed."""
        try:
            return self.queue.delete_final(finished_before=time.time() - self.retention_seconds)
        except Exception:
            logger.exception("could not delete the ended work items")
            return 0


__all__ = ["SWEEP_INTERVAL_SECONDS", "EndedWorkSweeper"]
