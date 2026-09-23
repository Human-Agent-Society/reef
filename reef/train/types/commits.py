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
    own to know which retained rows must never re-enter a processor. Fixed
    datasets additionally carry pass progress in metrics, allowing records
    consumed in earlier passes to remain eligible in later passes.
    Compaction itself is applied separately so the commit record can be made
    durable first. Persisted as a CommitRecord (reef/scenario/commit_log.py);
    the only construction site is Scenario._append_commit_record.

    ``metrics`` carries the objective's step result and trainer-owned
    ``dataset_*`` progress fields. Objective metrics remain opaque; the trainer
    reads dataset progress during recovery. The harness manifest republishes it verbatim as
    ``gate``. It rides the commit record because that is the only durable
    version-keyed store, so training metrics remain available when the
    resulting version is served.
    """

    algorithm_state: Mapping[str, Any]
    high_water_sequence: int
    high_water_offset: int
    compacted_ids: frozenset[str]
    consumed_ids: frozenset[str] = frozenset()
    metrics: Mapping[str, Any] | None = None
    training_job_id: str | None = None
