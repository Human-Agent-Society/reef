"""Judged records as samples: one record that the judging stage judged becomes one sample.

The record stage (:class:`~reef.record2record.JudgingRecord2Record`) forms a
``JudgedSet`` for each judged record. A recipe's ``JudgedSamples`` assembles it
in ``make_sample``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from reef.core.job_data import TrajectorySample
from reef.core.records_types import AgentRecord
from reef.record2record import JudgedSet, RecordSet
from reef.record2sample.base import ImmediateRecord2Sample, SampleResult


class JudgedSamples(ImmediateRecord2Sample, ABC):
    """Turn one judged record into one sample. A recipe implements ``make_sample``."""

    @abstractmethod
    def make_sample(self, record: AgentRecord, judgment: Any) -> TrajectorySample | None:
        """One judged record into a sample; ``None`` cannot train."""

    def make_samples(self, record_set: RecordSet) -> SampleResult:
        if not isinstance(record_set, JudgedSet):
            raise TypeError(f"{type(self).__name__} makes samples from judged sets, not {type(record_set).__name__}")
        sample = self.make_sample(record_set.record, record_set.judgment)
        if sample is None:
            return SampleResult.skipped()
        return SampleResult.ready((sample,))
