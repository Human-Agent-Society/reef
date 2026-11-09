"""The log of the attempt of a work item that a worker runs (RFC #1267).

While a worker runs an attempt in ``capture_log``, the context variable ``CURRENT_LOG`` names the attempt's log, and a
handler on the root logger keeps each record logged in that context as a ``WorkLogLine``. The context goes with the
work that the attempt starts: ``asyncio.to_thread`` copies it, and ``BackgroundJobWorker`` runs each job in a copy of
the context of its submit call. So the lines of a stage's background jobs go to the attempt that submitted them, also
after the attempt ended.

One flusher thread writes the kept lines of every log to its queue (``WorkQueue.append_log``) every
``FLUSH_SECONDS``, and an attempt writes its own lines once more when it ends. So a reader sees the lines of a
running attempt, and no write runs inside a transaction of the thread that logged. One attempt keeps at most
``MAX_LOG_LINES`` lines; the line after the last one says that the log was cut.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from reef.storage.work_queue import MAX_LOG_LINES, MAX_LOG_MESSAGE, WorkItem, WorkLogLine, WorkQueue

logger = logging.getLogger(__name__)

#: The seconds between two writes of the kept lines.
FLUSH_SECONDS = 2.0


class AttemptLog:
    """The log lines of one attempt of one work item, until the queue keeps them.

    Values: ``queue`` keeps the lines of attempt ``item.attempt_count`` of ``item``; ``pending`` holds the lines that
    it does not keep yet, in number order; ``count`` is the number of lines so far, at most ``MAX_LOG_LINES``, plus
    one for the line that says that the log was cut.

    Purpose: ``capture_log`` makes one for each attempt; ``WorkLogHandler`` adds lines, and the flusher and the end
    of the attempt write them.
    """

    def __init__(self, queue: WorkQueue, item: WorkItem) -> None:
        self.queue = queue
        self.item = item
        self.pending: list[WorkLogLine] = []
        self.count = 0
        self.lock = threading.Lock()

    def add(self, record: logging.LogRecord) -> None:
        """Keep ``record`` as the next line, or as the line that says that the log was cut, and hand the log to the
        flusher."""
        with self.lock:
            if self.count > MAX_LOG_LINES:
                return
            if self.count == MAX_LOG_LINES:
                level, name, message = "WARNING", __name__, f"the log was cut after {MAX_LOG_LINES} lines"
            else:
                level, name, message = record.levelname, record.name, record_text(record)
            self.pending.append(WorkLogLine(self.count, record.created, level, name, message[:MAX_LOG_MESSAGE]))
            self.count += 1
        FLUSHER.hold(self)

    def flush(self) -> None:
        """Write the pending lines; a failed write is logged and keeps them for the next flush."""
        with self.lock:
            lines, self.pending = self.pending, []
        if not lines:
            return
        try:
            self.queue.append_log(self.item.queue, self.item.id, self.item.attempt_count, lines)
        except Exception:
            logger.exception("could not keep the log of %s", self.item.id)
            with self.lock:
                self.pending = lines + self.pending
            FLUSHER.hold(self)


class Flusher:
    """Writes the pending lines of the logs that it holds every ``FLUSH_SECONDS``, on one daemon thread.

    Values: ``held`` is the set of logs with lines that a flush has not written; a log leaves it at its next flush.

    Purpose: ``AttemptLog.add`` hands it each log that gets a line. It holds the log until then, so a line of a
    background job that ends after its attempt is written too.
    """

    def __init__(self) -> None:
        self.held: set[AttemptLog] = set()
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None

    def hold(self, log: AttemptLog) -> None:
        with self.lock:
            self.held.add(log)
            if self.thread is None:
                self.thread = threading.Thread(target=self.run, name="reef-work-logs", daemon=True)
                self.thread.start()

    def flush(self) -> None:
        """Write the pending lines of every held log."""
        with self.lock:
            logs, self.held = self.held, set()
        for log in logs:
            log.flush()

    def run(self) -> None:
        while True:
            time.sleep(FLUSH_SECONDS)
            self.flush()


FLUSHER = Flusher()


def record_text(record: logging.LogRecord) -> str:
    """The message of ``record``, with its traceback when it has one."""
    text = record.getMessage()
    if record.exc_info:
        text = f"{text}\n{logging.Formatter().formatException(record.exc_info)}"
    return text


#: The log of the attempt whose work runs in this context, or None.
CURRENT_LOG: ContextVar[AttemptLog | None] = ContextVar("reef_work_log", default=None)


class WorkLogHandler(logging.Handler):
    """Gives each record to the log of the attempt whose work logged it (``CURRENT_LOG``)."""

    def emit(self, record: logging.LogRecord) -> None:
        log = CURRENT_LOG.get()
        if log is not None:
            log.add(record)


HANDLER = WorkLogHandler()
INSTALL_LOCK = threading.Lock()


def install_handler() -> None:
    """Add ``HANDLER`` to the root logger once."""
    with INSTALL_LOCK:
        root = logging.getLogger()
        if HANDLER not in root.handlers:
            root.addHandler(HANDLER)


@contextmanager
def capture_log(queue: WorkQueue, item: WorkItem) -> Iterator[None]:
    """Keep the lines that the block's work logs as the log of the attempt of ``item`` that runs now."""
    install_handler()
    log = AttemptLog(queue, item)
    token = CURRENT_LOG.set(log)
    try:
        yield
    finally:
        CURRENT_LOG.reset(token)
        log.flush()


__all__ = [
    "CURRENT_LOG",
    "FLUSHER",
    "FLUSH_SECONDS",
    "AttemptLog",
    "Flusher",
    "WorkLogHandler",
    "capture_log",
    "install_handler",
]
