from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreparedCommit:
    """Trainer-side state of a prepared commit, captured before compaction.

    Everything a durable commit record needs from the trainer: the post-step
    algorithm state, the record high-water mark (how far the store was
    consumed), the rows the step's batch consumed, and the rows the retention
    policy says are now disposable. ``consumed_ids`` and ``compacted_ids``
    answer different questions — a retention policy may keep a consumed row
    stored (audit-only retention), so recovery needs the consumed set on its
    own to know which retained rows must never re-enter a processor.
    Compaction itself is applied separately so the commit record can be made
    durable first. Persisted as a CommitRecord (reef/scenario/commit_log.py);
    the only construction site is Scenario._append_commit_record.

    ``metrics`` is the preparer's step result, carried opaquely: its schema is
    owned by the processor or backend that produced it; the trainer and commit log
    never interpret it, and the harness manifest republishes it verbatim as
    ``gate``. It rides the commit record because that is the only durable
    version-keyed store, which is what lets provenance survive to serving.
    """

    algorithm_state: Mapping[str, Any]
    high_water_sequence: int
    high_water_offset: int
    compacted_ids: frozenset[str]
    consumed_ids: frozenset[str] = frozenset()
    metrics: Mapping[str, Any] | None = None
    training_job_id: str | None = None
