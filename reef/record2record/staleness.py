"""The staleness check: find held sets that are too many versions behind to train."""

from __future__ import annotations

from collections.abc import Sequence

from reef.record2record.sets import RecordSet
from reef.runtime.interfaces import RuntimeLoadId


def stale_sets(record_sets: Sequence[RecordSet], max_staleness: int) -> tuple[RecordSet, ...]:
    """Return the sets that are more than ``max_staleness`` versions behind the newest set.

    The reference version is the ``runtime_load_id`` of the set whose source
    record arrived last (``source_index``, or ``arrival_index`` when it is
    None). The pipeline does not see the serving version, so the check
    measures the lag against this reference. A set stays when it has the
    same incarnation as the reference and is 0 to ``max_staleness`` versions
    behind it. With ``max_staleness`` 0, only sets of the reference version
    stay, as exact-version training requires.

    A set that is too far behind the reference is also too far behind the
    serving version, so runtime admission would drop its batch. Runtime
    admission still makes the final decision for the sets that stay.

    If the reference is not a canonical runtime load ID, only sets with the
    same ``runtime_load_id`` stay.
    """
    if not record_sets:
        return ()
    newest = max(
        record_sets,
        key=lambda record_set: (
            record_set.arrival_index if record_set.source_index is None else record_set.source_index
        ),
    )
    reference = canonical_runtime_load_id(newest.runtime_load_id)
    if reference is None:
        return tuple(record_set for record_set in record_sets if record_set.runtime_load_id != newest.runtime_load_id)
    stale: list[RecordSet] = []
    for record_set in record_sets:
        producing = canonical_runtime_load_id(record_set.runtime_load_id)
        # A malformed version, another incarnation (no lag), a future version, or a lag beyond the window.
        lag = None if producing is None else producing.lag_behind(reference)
        if lag is None or not 0 <= lag <= max_staleness:
            stale.append(record_set)
    return tuple(stale)


def canonical_runtime_load_id(value: str | None) -> RuntimeLoadId | None:
    """Parse a canonical runtime load ID. Return None for a missing or non-canonical value."""
    if value is None:
        return None
    try:
        parsed = RuntimeLoadId.parse(value)
    except (TypeError, ValueError):
        return None
    return parsed if str(parsed) == value else None
