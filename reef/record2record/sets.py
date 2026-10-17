"""Record sets: the records that a pipeline uses together, and the fields that the pool reads."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass

from reef.core.batches import TaskItem
from reef.core.records_types import AgentRecord


@dataclass(frozen=True, kw_only=True)
class PoolItem(ABC):
    """An item that a pool holds and orders: a record set, or a saved task waiting to run.

    The pool and the selection policies read only these fields:

    * ``set_id`` identifies the item in the pool.
    * ``arrival_index`` tells when the stage received the item.
    * ``group_key`` names the group of an item that belongs to one.
    """

    set_id: Hashable
    arrival_index: int
    group_key: Hashable | None = None


@dataclass(frozen=True, kw_only=True)
class TaskSet(PoolItem):
    """One saved task waiting to run. ``set_id`` is the task name, and it holds no records."""

    item: TaskItem


@dataclass(frozen=True, kw_only=True)
class RecordSet(PoolItem):
    """Records that are used together, such as one record, a report with its inferences, or a ready group.

    A stage forms record sets, and its pool holds and orders them. The pool
    and the selection policies read only these fields:

    * ``set_id`` identifies the set in the pool.
    * ``arrival_index`` tells when the stage received the set. A group
      uses the index of its oldest member.
    * ``group_key`` names the group of a set that is a group.
    """

    @abstractmethod
    def record_ids(self) -> tuple[str, ...]:
        """Return the IDs of the records in the set."""

    def formed_set_ids(self) -> tuple[Hashable, ...]:
        """Return the IDs of the formed sets whose tasks this set trains on.

        A stage that holds the sets it formed returns ``set_id``. A stage that
        combines formed sets, such as a ready group, returns their IDs in batch order.
        """
        return (self.set_id,)


@dataclass(frozen=True, kw_only=True)
class SingleRecordSet(RecordSet):
    """One record by itself. ``set_id`` is the record's ID."""

    record: AgentRecord

    def record_ids(self) -> tuple[str, ...]:
        return (self.record.agent_record_id,)
