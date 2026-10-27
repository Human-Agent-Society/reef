"""What the service routes may ask a harness backend, beyond the candidate lifecycle.

The routes check each capability with ``isinstance``, so a backend that
lacks one simply does not offer the route: ``PatchAdmission`` backs
``POST /reef/harness/proposals``, ``StepRecords`` the release pages' record
reads, and ``StepProgressReader`` the request page that follows a step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.harness.tree import TreePatch, ValidationResult
from reef.train.harness_backend.inbox import ProposalInbox


class PatchAdmission(ABC):
    """Validate a client's patch against the head tree and hold it for the next automatic step.

    Purpose: ``POST /reef/harness/proposals`` validates a patch with
    ``admit`` and files an admitted one in ``proposals``.
    """

    @property
    @abstractmethod
    def proposals(self) -> ProposalInbox | None:
        """The scenario's proposal inbox, ``None`` when the deployment takes no proposals."""

    @abstractmethod
    def admit(self, patch: TreePatch) -> ValidationResult:
        """``validate_patch`` of ``patch`` against the head tree; touches no step state."""


class StepRecords(ABC):
    """Read a scenario's retained step records.

    Purpose: the release pages read a step's record directory through it.
    """

    @abstractmethod
    def read_step_records(self, directory: str, relative: str | None) -> dict[str, Any]: ...


@dataclass(frozen=True)
class StepProgress:
    """Where the running step stands, for a page a person watches.

    Values: ``request_id`` names the instruction the step answers, ``None``
    for an automatic step or a client's proposal. ``phase`` is
    ``"proposing"`` from the moment the step claims its record directory
    until a candidate exists, then ``"evaluating"`` while the episodes run.
    ``started_at`` is the step's start in seconds since the epoch.
    ``step_record`` is the step's record directory, which the step claims
    before it reports any progress.
    ``episodes_total`` ≥ 0 is the evaluation's episode count once the
    episodes are laid out, else ``None``. ``activity`` is what the method did
    so far, oldest first, each ``{at, kind, text}`` with ``failed`` on a line
    that went wrong.
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
        if self.phase not in ("proposing", "evaluating"):
            raise ValueError("StepProgress.phase must be 'proposing' or 'evaluating'")
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


__all__ = ["PatchAdmission", "StepProgress", "StepProgressReader", "StepRecords"]
