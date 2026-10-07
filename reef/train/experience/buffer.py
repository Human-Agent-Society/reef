"""The units that a processor holds, and the batch that it reserved from them."""

from __future__ import annotations

from typing import Generic

from reef.train.experience.selection import ArrivalOrder, ExperienceUnit, KeyT, MemberT, SelectionPolicy


class ExperienceBuffer(Generic[KeyT, MemberT]):
    """Hold the units that a processor can batch, and its reserved batch.

    The processor adds and removes units. It also decides what a consumed unit
    releases. The buffer orders units only through its ``SelectionPolicy``.
    A reservation does not change until the processor consumes it or reserves
    again. Thus the processor can return the same batch until the trainer
    acknowledges it.
    """

    def __init__(self, selection: SelectionPolicy | None = None) -> None:
        self.selection = ArrivalOrder() if selection is None else selection
        self.units_by_id: dict[KeyT, ExperienceUnit[KeyT, MemberT]] = {}
        self.reserved: tuple[ExperienceUnit[KeyT, MemberT], ...] | None = None
        self.arrival_count = 0

    def __len__(self) -> int:
        return len(self.units_by_id)

    def __contains__(self, unit_id: KeyT) -> bool:
        return unit_id in self.units_by_id

    def next_arrival_index(self) -> int:
        """Return a new index. It is larger than each index that this buffer returned before."""
        self.arrival_count += 1
        return self.arrival_count

    def put(self, unit: ExperienceUnit[KeyT, MemberT]) -> None:
        """Add the unit. If the buffer holds a unit with the same ``unit_id``, replace that unit."""
        self.units_by_id[unit.unit_id] = unit

    def remove(self, unit_id: KeyT) -> ExperienceUnit[KeyT, MemberT] | None:
        """Remove the unit and return it. Return None if the buffer does not hold it."""
        return self.units_by_id.pop(unit_id, None)

    def units(self) -> tuple[ExperienceUnit[KeyT, MemberT], ...]:
        """Return all units, in the order that the buffer received them."""
        return tuple(self.units_by_id.values())

    def ordered_units(self) -> tuple[ExperienceUnit[KeyT, MemberT], ...]:
        """Return all units, in the order of the selection policy."""
        return self.selection.select(self.units(), len(self.units_by_id))

    def reserved_units(self) -> tuple[ExperienceUnit[KeyT, MemberT], ...]:
        """Return the reserved units. Return an empty tuple if no batch is reserved."""
        return () if self.reserved is None else self.reserved

    def reserve(self, max_unit_count: int) -> tuple[ExperienceUnit[KeyT, MemberT], ...]:
        """Select at most ``max_unit_count`` units as the next batch, and keep them reserved."""
        if max_unit_count < 0:
            raise ValueError("max_unit_count must not be negative")
        self.reserved = self.selection.select(self.units(), max_unit_count)
        return self.reserved

    def consume_reserved(self) -> tuple[ExperienceUnit[KeyT, MemberT], ...]:
        """Remove the reserved units from the buffer, clear the reservation, and return the units.

        The result also includes a reserved unit that the processor removed
        after the reservation.
        """
        consumed = self.reserved_units()
        for unit in consumed:
            self.units_by_id.pop(unit.unit_id, None)
        self.reserved = None
        return consumed
