"""The units that a processor holds, and the batch that it reserved from them."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

from reef.train.experience.eligibility import EligibilityCheck, IneligibleUnit
from reef.train.experience.selection import ArrivalOrder, ExperienceUnit, SelectionPolicy


class ExperienceBuffer:
    """Hold the units that a processor can batch, and its reserved batch.

    The processor adds and removes units. It also decides what a consumed unit
    releases. The buffer orders units only through its ``SelectionPolicy``, and
    removes unusable units only through its eligibility checks.
    A reservation does not change until the processor consumes it or reserves
    again. Thus the processor can return the same batch until the trainer
    acknowledges it.
    """

    def __init__(self, selection: SelectionPolicy | None = None, checks: Sequence[EligibilityCheck] = ()) -> None:
        self.selection = ArrivalOrder() if selection is None else selection
        self.checks = tuple(checks)
        self.units_by_id: dict[Hashable, ExperienceUnit] = {}
        self.reserved: tuple[ExperienceUnit, ...] | None = None
        self.arrival_count = 0

    def __len__(self) -> int:
        return len(self.units_by_id)

    def __contains__(self, unit_id: Hashable) -> bool:
        return unit_id in self.units_by_id

    def next_arrival_index(self) -> int:
        """Return a new index. It is larger than each index that this buffer returned before."""
        self.arrival_count += 1
        return self.arrival_count

    def put(self, unit: ExperienceUnit) -> None:
        """Add the unit. If the buffer holds a unit with the same ``unit_id``, replace that unit."""
        self.units_by_id[unit.unit_id] = unit

    def remove(self, unit_id: Hashable) -> ExperienceUnit | None:
        """Remove the unit and return it. Return None if the buffer does not hold it."""
        return self.units_by_id.pop(unit_id, None)

    def units(self) -> tuple[ExperienceUnit, ...]:
        """Return all units, in the order that the buffer received them."""
        return tuple(self.units_by_id.values())

    def drop_ineligible(self) -> tuple[IneligibleUnit, ...]:
        """Run the eligibility checks in order, remove the units that fail, and return them.

        Each check sees the units that the earlier checks kept. The processor
        calls this method when no batch is out, and releases the returned units.
        """
        dropped: list[IneligibleUnit] = []
        for check in self.checks:
            for result in check.ineligible_units(self.units()):
                if self.units_by_id.get(result.unit.unit_id) is not result.unit:
                    raise ValueError(f"{type(check).__name__} returned a unit that the buffer does not hold")
                self.units_by_id.pop(result.unit.unit_id)
                dropped.append(result)
        return tuple(dropped)

    def ordered_units(self) -> tuple[ExperienceUnit, ...]:
        """Return all units, in the order of the selection policy."""
        return self.select(len(self.units_by_id))

    def reserved_units(self) -> tuple[ExperienceUnit, ...]:
        """Return the reserved units. Return an empty tuple if no batch is reserved."""
        return () if self.reserved is None else self.reserved

    def reserve(self, max_unit_count: int) -> tuple[ExperienceUnit, ...]:
        """Select at most ``max_unit_count`` units as the next batch, and keep them reserved."""
        if max_unit_count < 0:
            raise ValueError("max_unit_count must not be negative")
        self.reserved = self.select(max_unit_count)
        return self.reserved

    def consume_reserved(self) -> tuple[ExperienceUnit, ...]:
        """Remove the reserved units from the buffer, clear the reservation, and return the units.

        The result also includes a reserved unit that the processor removed
        after the reservation.
        """
        consumed = self.reserved_units()
        for unit in consumed:
            self.units_by_id.pop(unit.unit_id, None)
        self.reserved = None
        return consumed

    def select(self, max_unit_count: int) -> tuple[ExperienceUnit, ...]:
        """Apply the selection policy, and make sure that it returned held units only, each one time."""
        selected = self.selection.select(self.units(), max_unit_count)
        selected_ids = [unit.unit_id for unit in selected]
        if len(selected) > max_unit_count or len(set(selected_ids)) != len(selected_ids):
            raise ValueError(f"{type(self.selection).__name__} returned too many units or a unit more than one time")
        if any(self.units_by_id.get(unit.unit_id) is not unit for unit in selected):
            raise ValueError(f"{type(self.selection).__name__} returned a unit that the buffer does not hold")
        return selected
