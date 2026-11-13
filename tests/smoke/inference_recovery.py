"""Observation of a serving runtime's health watcher, for the recovery profile of ``tests.smoke.native_inference``.

The watcher of a serving runtime checks its engines at a fixed interval, and recovers them after a failed check.
Nothing here changes a result: ``ObservedEngines`` lists each health check that the watcher makes, and
``ServingLog`` lists each warning and error that the runtime logs.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from reef.inference.remote.runtime import ComponentEngines
from reef.inference.remote.service import RemoteInferenceService

#: The logger of the serving runtime: a warning for each failed health check, an error for each failed recovery.
SERVING_LOGGER = "reef.inference.serving"
#: How often a wait reads the list of health checks again.
POLL_SECONDS = 0.1


@dataclass(frozen=True)
class HealthCheck:
    """One finished health check of a component's engines.

    Values: ``ended`` is the ``time.monotonic`` reading when the check returned; ``thread`` is the name of the
    thread that made it; ``error`` is None when the check passed, else the type and text of the exception that it
    raised.
    Purpose: ``ObservedEngines.check_health`` makes it; the recovery profile waits for, checks and reports them.
    """

    ended: float
    thread: str
    error: str | None


class ObservedEngines(ComponentEngines):
    """The engines of a remote inference component, which also list the health checks of their runtime.

    Values: ``checks`` lists the finished health checks in order; only the checking thread appends to it.
    ``check_health`` makes the component's own check and passes on its result, a return or an exception.
    Purpose: the recovery profile gives them to its serving runtime, and reads ``checks``.
    """

    def __init__(self, service: RemoteInferenceService) -> None:
        super().__init__(service)
        self.checks: list[HealthCheck] = []

    def check_health(self) -> None:
        error: str | None = None
        try:
            super().check_health()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.checks.append(HealthCheck(time.monotonic(), threading.current_thread().name, error))


async def ended_checks(engines: ObservedEngines, start: int, count: int, timeout_seconds: float) -> list[HealthCheck]:
    """Wait until ``count`` health checks from index ``start`` on have ended, and return them.

    ``TimeoutError`` when fewer ended within ``timeout_seconds``: no watcher checks the engines.
    """
    deadline = time.monotonic() + timeout_seconds
    while len(engines.checks) < start + count:
        if time.monotonic() > deadline:
            ended = len(engines.checks) - start
            raise TimeoutError(f"{ended} of {count} health checks ended within {timeout_seconds:.0f} s")
        await asyncio.sleep(POLL_SECONDS)
    return engines.checks[start : start + count]


async def first_check(engines: ObservedEngines, start: int, *, passed: bool, timeout_seconds: float) -> int:
    """Wait for the first health check from index ``start`` on that passed, or that failed when ``passed`` is
    false, and return its index. ``TimeoutError`` when no such check ended within ``timeout_seconds``."""
    deadline = time.monotonic() + timeout_seconds
    index = start
    while True:
        while index < len(engines.checks):
            if (engines.checks[index].error is None) is passed:
                return index
            index += 1
        if time.monotonic() > deadline:
            outcome = "passed" if passed else "failed"
            raise TimeoutError(f"no health check {outcome} within {timeout_seconds:.0f} s")
        await asyncio.sleep(POLL_SECONDS)


@dataclass(frozen=True)
class ServingLogLine:
    """One warning or error that a serving runtime logged.

    Values: ``error`` is whether its level is ``logging.ERROR`` or above; ``thread`` is the name of the thread
    that logged it; ``message`` is its formatted text; ``cause`` is the type and text of the exception that it
    reports, or None without one.
    Purpose: ``ServingLog.emit`` makes it; the recovery profile reports each line and fails on an error.
    """

    error: bool
    thread: str
    message: str
    cause: str | None


class ServingLog(logging.Handler):
    """A log handler that lists what it receives, and writes nothing.

    Values: ``lines`` has one entry for each warning and error received, in order; a logging thread appends it.
    Purpose: ``serving_log`` adds it to the serving runtime's logger; the recovery profile reads ``lines``.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.lines: list[ServingLogLine] = []

    def emit(self, record: logging.LogRecord) -> None:
        raised = record.exc_info[1] if record.exc_info else None
        cause = None if raised is None else f"{type(raised).__name__}: {raised}"
        self.lines.append(
            ServingLogLine(record.levelno >= logging.ERROR, record.threadName or "", record.getMessage(), cause)
        )


@contextmanager
def serving_log() -> Iterator[ServingLog]:
    """List the warnings and errors that the serving runtimes of this process log during the block."""
    log = ServingLog()
    logger = logging.getLogger(SERVING_LOGGER)
    logger.addHandler(log)
    try:
        yield log
    finally:
        logger.removeHandler(log)
