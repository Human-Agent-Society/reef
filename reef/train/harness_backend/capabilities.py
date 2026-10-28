"""What the service routes may ask a harness backend, beyond the candidate lifecycle.

The routes check each capability with ``isinstance``, so a backend that
lacks one simply does not offer the route: ``StepRecords`` backs the release
pages' record reads, and ``StepProgressReader`` the request page that follows
a step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class StepRecords(ABC):
    """Read a scenario's retained step records.

    Purpose: the release pages read a step's record directory through it.
    """

    @abstractmethod
    def read_step_records(self, directory: str, relative: str | None) -> dict[str, Any]: ...


#: The phases of a running step, in order; only an offline step, which rolls out train tasks, has the first.
STEP_PHASES = ("rolling out", "proposing", "evaluating")


@dataclass(frozen=True)
class StepProgress:
    """Where the running step stands, for a page a person watches.

    Values: ``request_id`` names the instruction the step answers, ``None``
    for an automatic step. ``phase`` is one of
    ``STEP_PHASES``: ``"rolling out"`` while an offline step runs the head
    tree on its train tasks, ``"evaluating"`` while the evaluation's
    episodes run, and ``"proposing"`` at any other time from the moment the
    step claims its record directory until a candidate exists.
    ``started_at`` is the step's start in seconds since the epoch.
    ``step_record`` is the step's record directory, which the step claims
    before it reports any progress.
    ``episodes_total`` ≥ 0 is the episode count of the rollouts or of the
    evaluation once its episodes are laid out, else ``None``. ``activity``
    is what the method did so far, oldest first, each ``{at, kind, text}``
    with ``failed`` on a line that went wrong.
    Purpose: the backend replaces it at each phase change; the request page
    reads it from another thread.
    """

    request_id: str | None
    phase: str
    started_at: float
    step_record: str
    episodes_total: int | None = None
    activity: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.phase not in STEP_PHASES:
            raise ValueError(f"StepProgress.phase must be one of {', '.join(STEP_PHASES)}")
        if not isinstance(self.step_record, str) or not self.step_record:
            raise ValueError("StepProgress.step_record must name the step's record directory")
        total = self.episodes_total
        if total is not None and (isinstance(total, bool) or not isinstance(total, int) or total < 0):
            raise ValueError("StepProgress.episodes_total must be an integer of at least 0 or None")


class StepProgressReader(ABC):
    """Report where the running step stands.

    Purpose: the request page reads ``step_progress`` while a step runs;
    ``None`` when no step runs.
    """

    @property
    @abstractmethod
    def step_progress(self) -> StepProgress | None: ...


__all__ = ["STEP_PHASES", "StepProgress", "StepProgressReader", "StepRecords"]
