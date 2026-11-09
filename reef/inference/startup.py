"""The reports of an artifact's startup, in the three fixed phases of ``reef.inference.interfaces.PHASES`` (RFC #1283).

A runtime reports the phases while it serves an artifact (``report_startup``). The serving worker binds each report to
a ``StartupReporter`` of its serving work item and to the artifact that it serves (``reporting_startup``); its reporter
adds the report to the item's progress. The startup status reads the reports back as ``StartupSteps``
(``reported_steps``). An ``InferenceHandler`` reports
``start`` and ``load`` of a ready artifact whose engine starts again from zero (``InferenceHandler.engine_steps``).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from reef.core.values import JsonValue
from reef.inference.interfaces import PHASES, StartupPhase, StartupStep, StartupSteps

#: The key of a startup report in a serving work item's progress.
PROGRESS_KEY = "startup"


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
    "StartupReporter",
    "report_startup",
    "reported_steps",
    "reporting_startup",
]
