"""The log of the attempt of a work item that a worker runs (RFC #1267).

While a worker runs an attempt in ``capture_log``, a handler on the root logger keeps each record that the worker's
thread logs, as a ``WorkLogLine``. A flusher thread writes the kept lines to the queue (``WorkQueue.append_log``)
every ``FLUSH_SECONDS``, and once more when the attempt ends. So a reader sees the lines of a running attempt, and
no write runs inside a transaction of the worker's thread.

Lines that other threads log, such as background jobs of a stage, are not kept. One attempt keeps at most
``MAX_LOG_LINES`` lines; the line after the last one says that the log was cut.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from reef.storage.work_queue import MAX_LOG_LINES, MAX_LOG_MESSAGE, WorkItem, WorkLogLine, WorkQueue

logger = logging.getLogger(__name__)

#: The seconds between two writes of the kept lines of a running attempt.
FLUSH_SECONDS = 2.0


class AttemptLog:
    """The log lines of one attempt of one work item, until the queue keeps them.

    Values: ``queue`` keeps the lines of attempt ``item.attempt_count`` of ``item``; ``pending`` holds the lines that
    it does not keep yet, in number order; ``count`` is the number of lines so far, at most ``MAX_LOG_LINES``, plus
    one for the line that says that the log was cut.

    Purpose: ``capture_log`` makes one for each attempt; ``WorkLogHandler`` adds lines and the flusher writes them.
    """

    def __init__(self, queue: WorkQueue, item: WorkItem) -> None:
        self.queue = queue
        self.item = item
        self.pending: list[WorkLogLine] = []
        self.count = 0
        self.lock = threading.Lock()

    def add(self, record: logging.LogRecord) -> None:
        """Keep ``record`` as the next line, or as the line that says that the log was cut."""
        with self.lock:
            if self.count > MAX_LOG_LINES:
                return
            if self.count == MAX_LOG_LINES:
                level, name, message = "WARNING", __name__, f"the log was cut after {MAX_LOG_LINES} lines"
            else:
                level, name, message = record.levelname, record.name, record_text(record)
            self.pending.append(WorkLogLine(self.count, record.created, level, name, message[:MAX_LOG_MESSAGE]))
            self.count += 1

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


def record_text(record: logging.LogRecord) -> str:
    """The message of ``record``, with its traceback when it has one."""
    text = record.getMessage()
    if record.exc_info:
        text = f"{text}\n{logging.Formatter().formatException(record.exc_info)}"
    return text


#: The log of the attempt that runs on this thread, or None.
CURRENT_LOG: ContextVar[AttemptLog | None] = ContextVar("reef_work_log", default=None)


class WorkLogHandler(logging.Handler):
    """Gives each record to the log of the attempt that runs on the thread that logged it (``CURRENT_LOG``)."""

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
    """Keep the lines that this thread logs in the block as the log of the attempt of ``item`` that runs now."""
    install_handler()
    log = AttemptLog(queue, item)
    done = threading.Event()

    def flush_until_done() -> None:
        while not done.wait(FLUSH_SECONDS):
            log.flush()

    flusher = threading.Thread(target=flush_until_done, name=f"reef-log-{item.id}", daemon=True)
    flusher.start()
    token = CURRENT_LOG.set(log)
    try:
        yield
    finally:
        CURRENT_LOG.reset(token)
        done.set()
        flusher.join()
        log.flush()


__all__ = ["CURRENT_LOG", "FLUSH_SECONDS", "AttemptLog", "WorkLogHandler", "capture_log", "install_handler"]
