"""Selection policies: which held batch units go into a batch, and in what order."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, cast

from reef.record2record.sets import RecordSetUnit


class SelectionPolicy(ABC):
    """Choose the units for one batch, and put them in batch order.

    The stage chooses the policy. The pool applies it to the units that it holds.
    """

    @abstractmethod
    def select(self, candidates: Sequence[RecordSetUnit], max_unit_count: int) -> tuple[RecordSetUnit, ...]:
        """Return at most ``max_unit_count`` units from ``candidates``, in batch order."""


class ArrivalOrder(SelectionPolicy):
    """Take the oldest units first."""

    def select(self, candidates: Sequence[RecordSetUnit], max_unit_count: int) -> tuple[RecordSetUnit, ...]:
        return tuple(sorted(candidates, key=lambda unit: unit.arrival_index)[:max_unit_count])


class GroupKeyOrder(SelectionPolicy):
    """Take ungrouped units in arrival order, then groups in ``group_key`` order.

    Group keys must be sortable, such as step indices.
    """

    def select(self, candidates: Sequence[RecordSetUnit], max_unit_count: int) -> tuple[RecordSetUnit, ...]:
        ungrouped = sorted(
            (unit for unit in candidates if unit.group_key is None),
            key=lambda unit: unit.arrival_index,
        )
        # The unit contract requires a hashable group key. This policy also requires a sortable key.
        grouped = sorted(
            (unit for unit in candidates if unit.group_key is not None),
            key=lambda unit: cast(Any, unit.group_key),
        )
        return tuple([*ungrouped, *grouped][:max_unit_count])
