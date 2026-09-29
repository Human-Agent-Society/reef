from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreparedCommit:
    """Trainer state captured before publishing a step.

    ``consumed_ids`` records committed consumption and intentionally skipped
    records for recovery. ``released_ids`` only releases processor memory after
    commit; it never retires stored records.

    ``metrics`` is the objective's step result, carried opaquely: its schema is
    owned by the processor or backend that produced it; the trainer and commit log
    never interpret it, and the harness manifest republishes it verbatim as
    ``gate``. It rides the commit record because that is the only durable
    version-keyed store, so training metrics remain available when the
    resulting version is served.
    """

    algorithm_state: Mapping[str, Any]
    high_water_sequence: int
    high_water_offset: int
    consumed_ids: frozenset[str] = frozenset()
    released_ids: frozenset[str] = frozenset()
    metrics: Mapping[str, Any] | None = None
    training_job_id: str | None = None
    #: The release the step's batch was reserved against; the committer
    #: refuses a result whose base is no longer the served release.
    base_release_id: str | None = None
