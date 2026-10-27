"""Reports as samples: one valid report and its inference records become one sample.

The record stage (:class:`~reef.record2record.ReportRecord2Record`) forms a
``ReportCandidate`` for each valid report that trains. A recipe's
``ReportSamples`` assembles it in ``make_sample``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

from reef.core.job_data import Sample, TaskSample, TrajectorySample
from reef.record2record import RecordSet, ReportCandidate, ReportContext
from reef.record2sample.base import ImmediateRecord2Sample, SampleResult


@dataclass(frozen=True)
class SkippedReport:
    """``make_sample`` returns this for a valid report that cannot train. The stage releases the report."""

    reason: str


class ReportSamples(ImmediateRecord2Sample, ABC):
    """Turn one valid report and its inference records into one sample.

    A recipe implements ``make_sample``. The stage attaches the report ID and
    its ordered references to the sample's ``source_agent_record_ids``.
    """

    @abstractmethod
    def make_sample(self, context: ReportContext) -> Sample | SkippedReport:
        """Assemble valid feedback into data; raise on a broken training contract.

        Runs on the trainer thread without network or model calls. Return
        ``SkippedReport`` for a valid report that cannot train.
        """

    def make_samples(self, record_set: RecordSet) -> SampleResult:
        if not isinstance(record_set, ReportCandidate):
            raise TypeError(
                f"{type(self).__name__} makes samples from report candidates, not {type(record_set).__name__}"
            )
        context = record_set.context
        sample = self.make_sample(context)
        if isinstance(sample, SkippedReport):
            return SampleResult.skipped()
        if not isinstance(sample, (TrajectorySample, TaskSample)):
            raise TypeError(f"{type(self).__name__}.make_sample must return a Sample")
        report_id = context.report.agent_record_id
        return SampleResult.ready((replace(sample, source_agent_record_ids=(*context.references, report_id)),))
