"""Selection policies: which held sets go into a batch, and in what order."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, cast

from reef.record2record.sets import RecordSet


class SelectionPolicy(ABC):
    """Choose the sets for one batch, and put them in batch order.

    The stage chooses the policy. The pool applies it to the sets that it holds.
    """

    @abstractmethod
    def select(self, candidates: Sequence[RecordSet], max_set_count: int) -> tuple[RecordSet, ...]:
        """Return at most ``max_set_count`` sets from ``candidates``, in batch order."""


class ArrivalOrder(SelectionPolicy):
    """Take the oldest sets first."""

    def select(self, candidates: Sequence[RecordSet], max_set_count: int) -> tuple[RecordSet, ...]:
        return tuple(sorted(candidates, key=lambda record_set: record_set.arrival_index)[:max_set_count])


class GroupKeyOrder(SelectionPolicy):
    """Take ungrouped sets in arrival order, then groups in ``group_key`` order.

    Group keys must be sortable, such as step indices.
    """

    def select(self, candidates: Sequence[RecordSet], max_set_count: int) -> tuple[RecordSet, ...]:
        ungrouped = sorted(
            (record_set for record_set in candidates if record_set.group_key is None),
            key=lambda record_set: record_set.arrival_index,
        )
        # The set contract requires a hashable group key. This policy also requires a sortable key.
        grouped = sorted(
            (record_set for record_set in candidates if record_set.group_key is not None),
            key=lambda record_set: cast(Any, record_set.group_key),
        )
        return tuple([*ungrouped, *grouped][:max_set_count])
