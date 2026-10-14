"""The record sets that a stage holds, and the batch that the pipeline reserved from them."""

from __future__ import annotations

from collections.abc import Hashable

from reef.record2record.selection import ArrivalOrder, SelectionPolicy
from reef.record2record.sets import RecordSet


class RecordSetPool:
    """Hold the record sets that a batch can use, and the reserved batch.

    The stage adds and removes sets. It also decides what a consumed set
    releases. The pool orders sets only through its ``SelectionPolicy``.
    A reservation does not change until the pipeline consumes it or reserves
    again. Thus the pipeline can return the same batch until the trainer
    acknowledges it.
    """

    def __init__(self, selection: SelectionPolicy | None = None) -> None:
        self.selection = ArrivalOrder() if selection is None else selection
        self.sets_by_id: dict[Hashable, RecordSet] = {}
        self.reserved: tuple[RecordSet, ...] | None = None
        self.arrival_count = 0

    def __len__(self) -> int:
        return len(self.sets_by_id)

    def next_arrival_index(self) -> int:
        """Return a new index. It is larger than each index that this pool returned before."""
        self.arrival_count += 1
        return self.arrival_count

    def put(self, record_set: RecordSet) -> None:
        """Add the set. If the pool holds a set with the same ``set_id``, replace that set."""
        self.sets_by_id[record_set.set_id] = record_set

    def remove(self, set_id: Hashable) -> RecordSet | None:
        """Remove the set and return it. Return None if the pool does not hold it."""
        return self.sets_by_id.pop(set_id, None)

    def sets(self) -> tuple[RecordSet, ...]:
        """Return all sets, in the order that the pool received them."""
        return tuple(self.sets_by_id.values())

    def ordered_sets(self) -> tuple[RecordSet, ...]:
        """Return all sets, in the order of the selection policy."""
        return self.select(len(self.sets_by_id))

    def reserved_sets(self) -> tuple[RecordSet, ...]:
        """Return the reserved sets. Return an empty tuple if no batch is reserved."""
        return () if self.reserved is None else self.reserved

    def reserve(self, max_set_count: int) -> tuple[RecordSet, ...]:
        """Select at most ``max_set_count`` sets as the next batch, and keep them reserved."""
        if max_set_count < 0:
            raise ValueError("max_set_count must not be negative")
        self.reserved = self.select(max_set_count)
        return self.reserved

    def release_reservation(self) -> None:
        """Clear the reservation without consuming it. The sets stay held for the next reservation."""
        self.reserved = None

    def consume_reserved(self) -> tuple[RecordSet, ...]:
        """Remove the reserved sets from the pool, clear the reservation, and return the sets.

        The result also includes a reserved set that the stage removed after
        the reservation.
        """
        consumed = self.reserved_sets()
        for record_set in consumed:
            self.sets_by_id.pop(record_set.set_id, None)
        self.reserved = None
        return consumed

    def select(self, max_set_count: int) -> tuple[RecordSet, ...]:
        """Apply the selection policy, and make sure that it returned held sets only, each one time."""
        selected = self.selection.select(self.sets(), max_set_count)
        selected_ids = [record_set.set_id for record_set in selected]
        if len(selected) > max_set_count or len(set(selected_ids)) != len(selected_ids):
            raise ValueError(f"{type(self.selection).__name__} returned too many sets or a set more than one time")
        if any(self.sets_by_id.get(record_set.set_id) is not record_set for record_set in selected):
            raise ValueError(f"{type(self.selection).__name__} returned a set that the pool does not hold")
        return selected
