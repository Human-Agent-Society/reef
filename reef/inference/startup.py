"""The reports of an artifact's startup, in the three fixed phases of ``reef.inference.interfaces.PHASES`` (RFC #1283).

A runtime reports the phases while it serves an artifact (``report_startup``). The serving worker binds each report to
a ``StartupReporter`` of its serving work item and to the artifact that it serves (``reporting_startup``); its reporter
adds the report to the item's progress. The startup status reads the reports back as ``StartupSteps``
(``reported_steps``). An ``InferenceHandler`` reports
``start`` and ``load`` of a ready artifact whose engine starts again from zero (``InferenceHandler.engine_steps``).

The serving worker also binds what a runtime reads while it serves (RFC #1299, ``serving_work``): whether a newer
request replaced the work (``stopping``), and whether a person asked to deploy the engines again (``redeploying``). A
runtime raises ``StartupStopped`` when it stops for ``stopping``, and ``StartupFailed`` for a failure whose message a
person may read.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from reef.core.values import JsonValue
from reef.inference.interfaces import PHASES, StartupPhase, StartupStep, StartupSteps

#: The key of a startup report in a serving work item's progress.
PROGRESS_KEY = "startup"
#: The start of the error of a serving work item that failed with ``StartupFailed``: the status gives the rest.
STARTUP_FAILED_PREFIX = "startup failed: "


class StartupFailed(Exception):
    """The artifact did not start, for a reason that a person can read and fix, such as an engine argument that the
    model refuses. The message must hold no secret: the startup status gives it."""


class StartupStopped(Exception):
    """The runtime stopped serving work because ``stopping`` became true."""


#: The stop event and the redeploy flag of the serving work that runs in this context (``serving_work``).
WORK: ContextVar[tuple[threading.Event, bool] | None] = ContextVar("reef_serving_work", default=None)


@contextmanager
def serving_work(stop: threading.Event, *, redeploy: bool) -> Iterator[None]:
    """Bind ``stopping`` in the block to ``stop``, and ``redeploying`` to ``redeploy``. A thread that the block starts
    sees them only when it runs in a copy of the context (``contextvars.copy_context``)."""
    token = WORK.set((stop, redeploy))
    try:
        yield
    finally:
        WORK.reset(token)


def stopping() -> bool:
    """Whether the serving work that runs in this context should stop: a newer request replaced it, or its worker lost
    it. A runtime reads it in long steps and then raises ``StartupStopped``. Outside a serving work, False."""
    work = WORK.get()
    return work is not None and work[0].is_set()


def redeploying() -> bool:
    """Whether a person asked to deploy the engines of this serving work again, also when they run the current
    settings (``POST /reef/scenarios/{s}/redeploy``). Outside a serving work, False."""
    work = WORK.get()
    return work is not None and work[1]


class StartupReporter(ABC):
    """Keeps the startup reports of one serving work, each a JSON object under ``PROGRESS_KEY``.

    Purpose: the serving worker gives one to ``reporting_startup``, and ``report_startup`` writes to it. A report only
    informs people, so ``add`` must not raise for a failed write.
    """

    @abstractmethod
    def add(self, entry: Mapping[str, JsonValue]) -> None:
        """Keep ``entry``, one report."""


#: The report and the artifact of the serving work that runs in this context, or None.
CURRENT: ContextVar[tuple[StartupReporter, str] | None] = ContextVar("reef_startup_report", default=None)


@contextmanager
def reporting_startup(report: StartupReporter, artifact_id: str) -> Iterator[None]:
    """Bind ``report_startup`` in the block to ``report``, for the artifact ``artifact_id``."""
    token = CURRENT.set((report, artifact_id))
    try:
        yield
    finally:
        CURRENT.reset(token)


def report_startup(phase: StartupPhase, detail: str | None = None, *, estimate_seconds: float | None = None) -> None:
    """Report that the artifact being served is in ``phase`` from now, doing ``detail``. A runtime calls it when the
    phase or its detail changes. Outside a serving work, it does nothing. ``ValueError`` for another phase."""
    if phase not in PHASES:
        raise ValueError(f"a startup phase is one of {PHASES}, not {phase!r}")
    current = CURRENT.get()
    if current is None:
        return
    report, artifact_id = current
    report.add(
        {
            "artifact_id": artifact_id,
            "phase": phase,
            "detail": detail,
            "estimate_seconds": estimate_seconds,
            "at": time.time(),
        }
    )


def reported_steps(
    progress: Sequence[Mapping[str, JsonValue]], artifact_id: str, created_at: float, ended_at: float | None
) -> StartupSteps:
    """The startup of ``artifact_id`` that a serving work reported in its ``progress``: ``prepare`` from the work's
    creation at ``created_at``, then each phase from its first report, each with the detail and the estimate of its last
    report. A step ends where the next one begins, and the last one at ``ended_at``, when the work ended. A report of an
    earlier phase than the current one is kept in the current phase."""
    steps: list[StartupStep] = [StartupStep("prepare", created_at)]
    estimate: float | None = None
    for entries in progress:
        entry = entries.get(PROGRESS_KEY)
        if not isinstance(entry, Mapping) or entry.get("artifact_id") != artifact_id:
            continue
        phase, at = entry.get("phase"), entry.get("at")
        if phase not in PHASES or not isinstance(at, (int, float)):
            continue
        written = entry.get("detail")
        detail = written if isinstance(written, str) else None
        value = entry.get("estimate_seconds")
        estimate = float(value) if isinstance(value, (int, float)) and value >= 0 else None
        last = steps[-1]
        if PHASES.index(phase) > PHASES.index(last.phase):
            at = max(float(at), last.started_at)
            steps[-1] = StartupStep(last.phase, last.started_at, at, last.detail)
            steps.append(StartupStep(phase, at, None, detail))
        else:
            steps[-1] = StartupStep(last.phase, last.started_at, None, detail)
    if ended_at is not None:
        last = steps[-1]
        steps[-1] = StartupStep(last.phase, last.started_at, max(ended_at, last.started_at), last.detail)
    return StartupSteps(tuple(steps), estimate)


__all__ = [
    "PROGRESS_KEY",
    "STARTUP_FAILED_PREFIX",
    "StartupFailed",
    "StartupReporter",
    "StartupStopped",
    "redeploying",
    "report_startup",
    "reported_steps",
    "reporting_startup",
    "serving_work",
    "stopping",
]
