"""Checks that find held sets which can no longer train."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from reef.record2record.sets import RecordSet
from reef.runtime.interfaces import RuntimeLoadId


@dataclass(frozen=True)
class IneligibleSet:
    """A held set that a check found unusable, and the reason."""

    record_set: RecordSet
    reason: str


class EligibilityCheck(ABC):
    """Find the held sets that the current update method can no longer use.

    The stage chooses the checks. The pool runs them and removes the sets
    that they return. The stage then releases the records of those sets.
    """

    @abstractmethod
    def ineligible_sets(self, record_sets: Sequence[RecordSet]) -> tuple[IneligibleSet, ...]:
        """Return the sets in ``record_sets`` that cannot train, each with a reason."""


class StalenessCheck(EligibilityCheck):
    """Drop sets that are more than ``max_staleness`` versions behind the newest held set.

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

    def __init__(self, max_staleness: int = 0) -> None:
        if not isinstance(max_staleness, int) or isinstance(max_staleness, bool) or max_staleness < 0:
            raise ValueError("max_staleness must be a non-negative integer")
        self.max_staleness = max_staleness

    def ineligible_sets(self, record_sets: Sequence[RecordSet]) -> tuple[IneligibleSet, ...]:
        if not record_sets:
            return ()
        newest = max(
            record_sets,
            key=lambda record_set: (
                record_set.arrival_index if record_set.source_index is None else record_set.source_index
            ),
        )
        reference = canonical_runtime_load_id(newest.runtime_load_id)
        results: list[IneligibleSet] = []
        for record_set in record_sets:
            if reference is None:
                if record_set.runtime_load_id != newest.runtime_load_id:
                    results.append(IneligibleSet(record_set, "different_runtime_load_id"))
                continue
            producing = canonical_runtime_load_id(record_set.runtime_load_id)
            lag = None if producing is None else producing.lag_behind(reference)
            if producing is None:
                results.append(IneligibleSet(record_set, "malformed_producing_runtime_load_id"))
            elif lag is None:
                results.append(IneligibleSet(record_set, "cross_incarnation"))
            elif lag < 0:
                results.append(IneligibleSet(record_set, "future_producing_runtime_load_id"))
            elif lag > self.max_staleness:
                results.append(IneligibleSet(record_set, "policy_lag_exceeded"))
        return tuple(results)


def canonical_runtime_load_id(value: str | None) -> RuntimeLoadId | None:
    """Parse a canonical runtime load ID. Return None for a missing or non-canonical value."""
    if value is None:
        return None
    try:
        parsed = RuntimeLoadId.parse(value)
    except (TypeError, ValueError):
        return None
    return parsed if str(parsed) == value else None
