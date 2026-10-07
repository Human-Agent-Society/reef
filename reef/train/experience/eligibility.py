"""Checks that find held units which can no longer train."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

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


class NewestVersionCheck(EligibilityCheck):
    """Keep only the units that the newest runtime load ID produced.

    The newest unit is the unit whose source record arrived last
    (``source_index``, or ``arrival_index`` when it is None). Continual serving
    publishes a new version each step. Thus units from older versions would
    never form a batch with newer units.
    """

    def ineligible_units(self, units: Sequence[ExperienceUnit]) -> tuple[IneligibleUnit, ...]:
        if not units:
            return ()
        newest = max(units, key=lambda unit: unit.arrival_index if unit.source_index is None else unit.source_index)
        return tuple(
            IneligibleUnit(unit, "older_runtime_load_id")
            for unit in units
            if unit.runtime_load_id != newest.runtime_load_id
        )
