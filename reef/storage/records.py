"""Record storage interface, results, errors, and retention limits.

Record stores append and replay interaction records. They do not know
about scenario lifecycle, commits, artifact publication, or training. Concrete
backends in this package implement this interface; importing it loads no database adapters.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

from reef.core.errors import ReefError
from reef.core.records_types import AgentRecord, RequestType


class ConsumptionReceipt(TypedDict):
    receipt_id: str
    consumed_ids: tuple[str, ...]
    metadata: dict[str, object]
    recorded_at: float


class RecordConflict(ReefError):
    """A record id or consumption receipt already identifies different content."""


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
class RecordLoss:
    """Durable capacity-eviction totals for one scenario's stored records."""

    record_count: int = 0
    body_bytes: int = 0
    first_sequence: int = 0
    last_sequence: int = 0


@dataclass(frozen=True)
class RecordRetention:
    """Deployment-wide budget for all record bodies, regardless of consumption.

    Retry metadata, indexes and database overhead are outside the body budget.
    ``days`` is accepted for configuration compatibility but no longer expires
    data: capacity alone triggers automatic eviction.
    """

    days: float = 7.0
    max_bytes: int = 20 * 1024**3

    def __post_init__(self) -> None:
        if (
            isinstance(self.days, bool)
            or not isinstance(self.days, (int, float))
            or not math.isfinite(self.days)
            or self.days <= 0
        ):
            raise ValueError("agent_record_retention_days must be positive and finite")
        if isinstance(self.max_bytes, bool) or not isinstance(self.max_bytes, int) or self.max_bytes <= 0:
            raise ValueError("agent_record_retention_max_bytes must be a positive integer")


class RecordStore(ABC):
    """Persist records and consumption receipts independently of processors.

    Reads and consumption receipts are scoped to the supplied scenario. Record ids are
    unique across the store: retries with identical content are idempotent,
    while different content raises ``RecordConflict``. Retry comparison ignores
    ``created_at`` and continues to work after capacity eviction.

    Append sequences increase and must never be reused, including after a
    capacity eviction. Reads include all retained bodies regardless of consumption.
    Implementations serialize conflicting writes so callers
    can append while training and eviction run in other threads.
    """

    @abstractmethod
    def append(self, item: AgentRecord) -> AgentRecord:
        """Accept a record or return its existing receipt on an identical retry."""

    @abstractmethod
    def append_result(self, item: AgentRecord) -> AppendResult:
        """Append with insertion status; retries never restore evicted records.

        Reports referencing evicted records remain outside training reads.
        Their content is still remembered so conflicting retries are rejected.
        """

    def append_many(self, items: Sequence[AgentRecord]) -> tuple[AppendResult, ...]:
        """Atomically append a bounded batch from one scenario, in input order.

        Any conflict rolls back the entire batch. Identical retries retain
        the same semantics as append_result, including evicted records.
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
    def get_for_audit(self, scenario: str, agent_record_id: str) -> StoredRecord | None:
        """Read a retained record with its append sequence."""

    @abstractmethod
    def audit_page(
        self,
        scenario: str,
        *,
        after_sequence: int = 0,
        limit: int = 256,
    ) -> tuple[StoredRecord, ...]:
        """Read retained records with the same sequence bounds as ``replay_page``."""

    @abstractmethod
    def record_consumption(
        self,
        scenario: str,
        agent_record_ids: frozenset[str],
        *,
        receipt_id: str,
        metadata: Mapping[str, object],
    ) -> None:
        """Persist consumption outside a training commit; leave record bodies readable.

        An identical (scenario, receipt id, record IDs) retry is idempotent.
        Different metadata for the same identity raises ``RecordConflict``.
        """

    @abstractmethod
    def consumption_receipts(self, scenario: str) -> tuple[ConsumptionReceipt, ...]:
        """Read durable consumption receipts in recorded order."""

    def loss(self, scenario: str) -> RecordLoss:
        """Capacity losses, including after restart; non-evicting stores return zero."""
        return RecordLoss()

    @abstractmethod
    def close(self) -> None:
        """Release resources; repeated closes are harmless."""


__all__ = [
    "AgentRecord",
    "AppendResult",
    "ConsumptionReceipt",
    "RecordConflict",
    "RecordLoss",
    "RecordRetention",
    "RecordStore",
    "RequestType",
    "StoredRecord",
]
