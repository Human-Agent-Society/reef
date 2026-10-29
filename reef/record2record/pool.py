"""The record set units that a record stage holds until they are ready to store."""

from __future__ import annotations

from collections.abc import Hashable

from reef.record2record.sets import RecordSetUnit


class RecordSetUnitPool:
    """Hold the ready record set units of a record stage, in the order that the stage put them.

    The stage adds and removes units. It also decides what a consumed unit releases. The pipeline stores the
    samples of each unit that the pool holds and removes the unit (``TrainingPipeline.store_unit``).
    """

    def __init__(self) -> None:
        self.units_by_id: dict[Hashable, RecordSetUnit] = {}
        self.arrival_count = 0

    def __len__(self) -> int:
        return len(self.units_by_id)

    def next_arrival_index(self) -> int:
        """Return a new index. It is larger than each index that this pool returned before."""
        self.arrival_count += 1
        return self.arrival_count

    def put(self, unit: RecordSetUnit) -> None:
        """Add the unit. If the pool holds a unit with the same ``unit_id``, replace that unit."""
        self.units_by_id[unit.unit_id] = unit

    def remove(self, unit_id: Hashable) -> RecordSetUnit | None:
        """Remove the unit and return it. Return None if the pool does not hold it."""
        return self.units_by_id.pop(unit_id, None)

    def units(self) -> tuple[RecordSetUnit, ...]:
        """Return all units, in the order that the pool received them."""
        return tuple(self.units_by_id.values())
