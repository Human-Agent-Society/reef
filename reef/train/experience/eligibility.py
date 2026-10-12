"""Checks that find held units which can no longer train."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from reef.runtime.interfaces import RuntimeLoadId
from reef.train.experience.selection import ExperienceUnit


@dataclass(frozen=True)
class IneligibleUnit:
    """A held unit that a check found unusable, and the reason."""

    unit: ExperienceUnit
    reason: str


class EligibilityCheck(ABC):
    """Find the held units that the current update method can no longer use.

    The processor chooses the checks. The buffer runs them and removes the
    units that they return. The processor then releases those units.
    """

    @abstractmethod
    def ineligible_units(self, units: Sequence[ExperienceUnit]) -> tuple[IneligibleUnit, ...]:
        """Return the units in ``units`` that cannot train, each with a reason."""


class StalenessCheck(EligibilityCheck):
    """Drop units that are more than ``max_staleness`` versions behind the newest held unit.

    The reference version is the ``runtime_load_id`` of the unit whose source
    record arrived last (``source_index``, or ``arrival_index`` when it is
    None). The processor does not see the serving version, so the check
    measures the lag against this reference. A unit stays when it has the
    same incarnation as the reference and is 0 to ``max_staleness`` versions
    behind it. With ``max_staleness`` 0, only units of the reference version
    stay, as exact-version training requires.

    A unit that is too far behind the reference is also too far behind the
    serving version, so runtime admission would drop its batch. Runtime
    admission still makes the final decision for the units that stay.

    If the reference is not a canonical runtime load ID, only units with the
    same ``runtime_load_id`` stay.
    """

    def __init__(self, max_staleness: int = 0) -> None:
        if not isinstance(max_staleness, int) or isinstance(max_staleness, bool) or max_staleness < 0:
            raise ValueError("max_staleness must be a non-negative integer")
        self.max_staleness = max_staleness

    def ineligible_units(self, units: Sequence[ExperienceUnit]) -> tuple[IneligibleUnit, ...]:
        if not units:
            return ()
        newest = max(units, key=lambda unit: unit.arrival_index if unit.source_index is None else unit.source_index)
        reference = canonical_runtime_load_id(newest.runtime_load_id)
        results: list[IneligibleUnit] = []
        for unit in units:
            if reference is None:
                if unit.runtime_load_id != newest.runtime_load_id:
                    results.append(IneligibleUnit(unit, "different_runtime_load_id"))
                continue
            producing = canonical_runtime_load_id(unit.runtime_load_id)
            lag = None if producing is None else producing.lag_behind(reference)
            if producing is None:
                results.append(IneligibleUnit(unit, "malformed_producing_runtime_load_id"))
            elif lag is None:
                results.append(IneligibleUnit(unit, "cross_incarnation"))
            elif lag < 0:
                results.append(IneligibleUnit(unit, "future_producing_runtime_load_id"))
            elif lag > self.max_staleness:
                results.append(IneligibleUnit(unit, "policy_lag_exceeded"))
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
