"""Record storage interface, results, errors, and retention limits.

Record stores append and replay interaction records. They do not know
about scenario lifecycle, commits, artifact publication, or training. Concrete
backends in this package implement this interface; importing it loads no database adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from reef.core.errors import ReefError
from reef.core.records_types import AgentRecord, RecordLoss, RequestType


class RecordConflict(ReefError):
    """A record id already identifies different content."""


@dataclass(frozen=True)
class AppendResult:
    """The accepted record and whether this append added it to training reads."""

    item: AgentRecord
    inserted: bool


@dataclass(frozen=True)
class StoredRecord:
    """A retained record and its append sequence, independent of consumption."""

    sequence: int
    item: AgentRecord


@dataclass(frozen=True)
class RecordRetention:
    """Deployment-wide budget for all record bodies, regardless of consumption.

    Retry metadata, indexes and database overhead are outside the body budget.
    Capacity alone triggers automatic eviction; records do not expire by age.
    """

    max_bytes: int = 20 * 1024**3

    def __post_init__(self) -> None:
        if isinstance(self.max_bytes, bool) or not isinstance(self.max_bytes, int) or self.max_bytes <= 0:
            raise ValueError("agent_record_retention_max_bytes must be a positive integer")


class RecordStore(ABC):
    """Persist records independently of processors.

    Reads are scoped to the supplied scenario. Record ids are
    unique across the store: retries with identical content are idempotent,
    while different content raises ``RecordConflict``. Retry comparison ignores
    ``created_at``.

    Append sequences increase and must never be reused, including after
    retention deletes a record. Reads include all retained bodies regardless of consumption.
    Implementations serialize conflicting writes so callers
    can append while training and eviction run in other threads.
    """

    @abstractmethod
    def append(self, item: AgentRecord) -> AgentRecord:
        """Accept a record or return its existing receipt on an identical retry."""

    @abstractmethod
    def append_result(self, item: AgentRecord) -> AppendResult:
        """Append with insertion status: ``inserted`` is False for a retry of a stored record."""

    def append_many(self, items: Sequence[AgentRecord]) -> tuple[AppendResult, ...]:
        """Atomically append a bounded batch from one scenario, in input order.

        Any conflict rolls back the entire batch. Identical retries retain
        the same semantics as append_result.
        Adapters without atomic batch support must reject before writing.
        """
        raise NotImplementedError("this record store does not support atomic batch append")

    @abstractmethod
    def existing_receipt(self, item: AgentRecord) -> AgentRecord | None:
        """Validate retry content without appending or changing its receipt."""

    @abstractmethod
    def get(self, scenario: str, agent_record_id: str) -> AgentRecord | None:
        """Read one training-visible record in the supplied scenario."""

    @abstractmethod
    def replay(
        self,
        scenario: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> tuple[AgentRecord, ...]:
        """Read training-visible records in append order with nonnegative bounds."""

    @abstractmethod
    def replay_page(
        self,
        scenario: str,
        *,
        after_sequence: int = 0,
        limit: int = 256,
    ) -> tuple[tuple[int, AgentRecord], ...]:
        """Read at most a positive ``limit`` after a nonnegative append sequence."""

    @abstractmethod
    def count(self, scenario: str, *, request_type: RequestType | None = None, after_sequence: int = 0) -> int:
        """Count training-visible records by optional type and append sequence."""

    @abstractmethod
    def last_sequence(self, scenario: str) -> int:
        """The append sequence of the newest training-visible record of ``scenario``; 0 when it has none."""

    @abstractmethod
    def get_for_audit(self, scenario: str, agent_record_id: str) -> StoredRecord | None:
        """Read a retained record with its append sequence."""

    @abstractmethod
    def audit_page(
        self,
        scenario: str,
        *,
        after_sequence: int = 0,
        limit: int = 256,
        request_type: RequestType | None = None,
    ) -> tuple[StoredRecord, ...]:
        """Read retained records with the same sequence bounds as ``replay_page``, optionally of one type."""

    def trajectory(self, scenario: str, agent_record_id: str) -> tuple[str, ...]:
        """The record IDs of the trajectory that the record ends, oldest first (RFC #1054); ``()`` when the record is
        not stored. A store that keeps no trajectories treats each record as a trajectory of one call."""
        return () if self.get_for_audit(scenario, agent_record_id) is None else (agent_record_id,)

    def loss(self, scenario: str) -> RecordLoss:
        """The records that retention deleted, including after a restart; a store without retention returns zero."""
        return RecordLoss()

    @abstractmethod
    def close(self) -> None:
        """Release resources; repeated closes are harmless."""


__all__ = [
    "AgentRecord",
    "AppendResult",
    "RecordConflict",
    "RecordLoss",
    "RecordRetention",
    "RecordStore",
    "RequestType",
    "StoredRecord",
]
