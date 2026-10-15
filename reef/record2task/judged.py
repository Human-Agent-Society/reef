"""Judged records as tasks: one record that the judging stage judged becomes one task.

The record stage (:class:`~reef.record2record.JudgingRecord2Record`) forms a
``JudgedSet`` for each judged record. A recipe's ``JudgedTasks`` assembles it
in ``make_sample``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from reef.core.batches import TrajectoryItem
from reef.core.records_types import AgentRecord
from reef.record2record import JudgedSet, RecordSet
from reef.record2task.base import Record2Task, TaskResult


class JudgedTasks(Record2Task, ABC):
    """Turn one judged record into one task. A recipe implements ``make_sample``."""

    @abstractmethod
    def make_sample(self, record: AgentRecord, judgment: Any) -> TrajectoryItem | None:
        """One judged record into a sample; ``None`` cannot train."""

    def make_tasks(self, record_set: RecordSet) -> TaskResult:
        if not isinstance(record_set, JudgedSet):
            raise TypeError(f"{type(self).__name__} makes tasks from judged sets, not {type(record_set).__name__}")
        sample = self.make_sample(record_set.record, record_set.judgment)
        if sample is None:
            return TaskResult.skipped()
        return TaskResult.ready((sample,))
