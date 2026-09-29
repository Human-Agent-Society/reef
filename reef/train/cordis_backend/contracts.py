"""Optional candidate backend capabilities used by service routes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reef.harness.tree.mutations import Mutation
from reef.train.cordis_backend.proposals import ProposalInbox


class StepRecords(ABC):
    """Read a scenario's retained training step files."""

    @abstractmethod
    def read_step_records(self, directory: str, relative: str | None) -> dict[str, Any]: ...


class ProposalValidator(ABC):
    """Admit proposed mutations and expose their scenario inbox."""

    @property
    @abstractmethod
    def proposals(self) -> ProposalInbox | None: ...

    @abstractmethod
    def admit(
        self, entries: Sequence[Mapping[str, Any]], mutations: Sequence[Mutation]
    ) -> tuple[list[dict[str, Any]], str | None]: ...


@dataclass(frozen=True)
class StepProgress:
    """Where the running step stands, for a page a person watches: its phase, its start, its record directory.

    ``request_id`` names the instruction the step answers, ``None`` for an
    automatic step or an agent's proposal. ``phase`` is ``proposing`` from
    the moment the step claims its directory until a candidate exists, then
    ``evaluating`` while the episodes run; ``episodes_total`` is the evaluation's
    episode count once ``evaluate`` has laid the episodes out, else ``None``.
    ``activity`` is what the proposer has done so far, oldest first, each
    ``{at, kind, text}`` with ``failed`` on a line that went wrong: the model
    calls, and an agent's tool calls, checks and trials as they happen.
    """

    request_id: str | None
    phase: str
    started_at: float
    step_record: str | None
    episodes_total: int | None = None
    activity: tuple[Mapping[str, Any], ...] = ()


class StepProgressReader(ABC):
    """Report where the running step stands; the harness backend does, another need not."""

    @property
    @abstractmethod
    def step_progress(self) -> StepProgress | None: ...


# Compatibility for adapters that implement the earlier interface name.
ProposalGate = ProposalValidator
