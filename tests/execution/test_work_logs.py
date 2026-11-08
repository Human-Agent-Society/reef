"""The log of each attempt of a work item that a worker keeps."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import closing

import pytest

from reef.execution import work_logs
from reef.execution.jobs import BackgroundJobRunner, BackgroundJobWorker, SupportsReceipt
from reef.execution.workers import QueueWorker
from reef.storage.sql_coordination import SQLiteCoordinationDatabase, SQLWorkQueue
from reef.storage.work_queue import MAX_LOG_LINES, NewWork

logger = logging.getLogger("reef.test.work_logs")


class LoggingWorker(QueueWorker):
    """Logs ``lines`` lines, one line from another thread, and then raises ``error`` when it is set."""

    keeps_log = True

    def __init__(self, queue, *, lines: int = 2, error: Exception | None = None) -> None:
        super().__init__(queue, "jobs", ["job"], worker_id="w")
        self.lines = lines
        self.error = error

    def run_item(self, item, lease_lost):
        for number in range(self.lines):
            logger.info("line %d", number)
        other = threading.Thread(target=lambda: logger.info("from another thread"))
        other.start()
        other.join()
        if self.error is not None:
            raise self.error
        return {}


@pytest.fixture
def queue():
    with closing(SQLiteCoordinationDatabase(None)) as database:
        queue = SQLWorkQueue(database, max_attempts={"jobs": 2})
        queue.add(NewWork("jobs", "a", "s", "job"))
        yield queue


@pytest.mark.unit
def test_a_worker_keeps_the_lines_that_its_work_logs_in_each_attempt(queue, caplog):
    with caplog.at_level(logging.INFO, logger="reef"):
        assert LoggingWorker(queue, error=RuntimeError("judge down")).run_once()
        assert LoggingWorker(queue).run_once()
    first = queue.read_log("jobs", "a", 1)
    assert [line.message for line in first[:2]] == ["line 0", "line 1"]
    assert first[-1].level == "ERROR" and "RuntimeError: judge down" in first[-1].message
    assert "Traceback" in first[-1].message
    assert all("another thread" not in line.message for line in first)
    second = queue.read_log("jobs", "a", 2)
    assert [line.message for line in second] == ["line 0", "line 1"]


@pytest.mark.unit
def test_an_attempt_keeps_at_most_the_line_limit_and_says_that_it_cut_the_log(queue, caplog, monkeypatch):
    monkeypatch.setattr(work_logs, "MAX_LOG_LINES", 5)
    with caplog.at_level(logging.INFO, logger="reef"):
        assert LoggingWorker(queue, lines=9).run_once()
    lines = queue.read_log("jobs", "a", 1)
    assert [line.number for line in lines] == list(range(6))
    assert lines[-1].message == "the log was cut after 5 lines"


@pytest.mark.unit
def test_a_worker_without_a_log_keeps_nothing(queue, caplog):
    class Quiet(LoggingWorker):
        keeps_log = False

    with caplog.at_level(logging.INFO, logger="reef"):
        assert Quiet(queue).run_once()
    assert queue.read_log("jobs", "a", 1) == ()
    assert MAX_LOG_LINES == 2000


class Judge(BackgroundJobRunner):
    """Logs one line when its job runs, after ``release`` is set."""

    def __init__(self) -> None:
        self.release = threading.Event()

    async def run(self, job):
        await asyncio.to_thread(self.release.wait, 5)
        logger.info("judged %s", job.receipt)
        return job


@pytest.mark.unit
def test_a_background_job_logs_to_the_attempt_that_submitted_it_also_after_the_attempt(queue, caplog):
    judge = Judge()
    background = BackgroundJobWorker(judge, thread_name="judge", concurrency=1)

    class Submitting(LoggingWorker):
        def run_item(self, item, lease_lost):
            logger.info("submit")
            assert background.submit(SupportsReceipt("r1"))
            return {}

    try:
        with caplog.at_level(logging.INFO, logger="reef"):
            assert Submitting(queue).run_once()
            assert [line.message for line in queue.read_log("jobs", "a", 1)] == ["submit"]
            judge.release.set()
            while not background.poll():
                time.sleep(0.01)
            # The job ended after its attempt: the flusher writes its line to that attempt.
            work_logs.FLUSHER.flush()
    finally:
        background.close()
    assert [line.message for line in queue.read_log("jobs", "a", 1)] == ["submit", "judged r1"]
