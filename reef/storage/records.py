"""Record storage interface, results, errors, and retention limits.

Record stores append, replay, and retire interaction records. They do not know
about scenario lifecycle, commits, artifact publication, or training. Concrete
backends in this package implement this interface; importing it loads no database adapters.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from collections.abc import Mapping
from dataclasses import dataclass

from reef.core.errors import ReefError
from reef.core.records_types import AgentRecord, RequestType


class RecordConflict(ReefError):
    """A record id or compaction receipt already identifies different content."""


@dataclass(frozen=True)
class AppendResult:
    """The accepted record and whether this append added it to training reads."""

    item: AgentRecord
    inserted: bool


@dataclass(frozen=True)
class StoredRecord:
    """A retained record and its storage state, for audit reads only.

    ``compacted_at`` marks retirement from training, not proof of learning.
    The commit log's ``consumed_ids`` identifies which records a step consumed.
    """

    sequence: int
    item: AgentRecord
    compacted_at: float | None


@dataclass(frozen=True)
class RecordRetention:
    """Deployment-wide limits on compacted JSON bodies, applied by service maintenance.

    The byte budget counts compacted JSON bodies only. It excludes active
    records, retry metadata, indexes, and storage overhead; it is not a physical
    database size limit. Storage services apply these limits.
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
    """Append, replay, and retire scenario records independently of storage.

    Reads and compaction are scoped to the supplied scenario. Record ids are
    unique across the store: retries with identical content are idempotent,
    while different content raises ``RecordConflict``. Retry comparison ignores
    ``created_at`` and continues to work after compaction or body purging.

    Append sequences increase and must never be reused, including after a
    purge. Training reads hide compacted records; audit reads include their
    retained bodies. Implementations serialize conflicting writes so callers
    can append while training and compaction run in other threads.
    """

    @abstractmethod
    def append(self, item: AgentRecord) -> AgentRecord:
        """Accept a record or return its existing receipt on an identical retry."""

    @abstractmethod
    def append_result(self, item: AgentRecord) -> AppendResult:
        """Append with insertion status; retries never reactivate retired records.

        Reports referencing retired records remain outside training reads.
        Their content is still remembered so conflicting retries are rejected.
        """

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
        """Read a retained record and its compaction state without reactivating it."""

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
    def compact(
        self,
        scenario: str,
        agent_record_ids: frozenset[str],
        *,
        receipt_id: str | None = None,
        receipt_metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Retire records and atomically remember an optional compaction receipt.

        Repeated calls preserve the first retirement time and append retry
        protection. Receipt id and metadata must be supplied together. A receipt
        is identified by scenario, receipt id, and the set of compacted ids;
        reusing that identity with different metadata raises ``RecordConflict``.
        """

    @abstractmethod
    def purge_compacted(self, scenario: str, *, before: float, limit: int = 256) -> int:
        """Delete bounded retired bodies while retaining retry hashes and receipts.

        ``before`` must be a finite Unix timestamp and ``limit`` a positive
        integer. Return the number of removed bodies.
        """

    @abstractmethod
    def compaction_receipts(self, scenario: str) -> tuple[dict[str, object], ...]:
        """Read durable compaction receipts in recorded-time and receipt-id order."""

    @abstractmethod
    def close(self) -> None:
        """Release resources; repeated closes are harmless."""


__all__ = [
    "AgentRecord",
    "AppendResult",
    "RecordConflict",
    "RecordRetention",
    "RecordStore",
    "RequestType",
    "StoredRecord",
]
