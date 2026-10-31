"""Trace samples for harness evolution: recorded requests as served, with a reported score or unscored.

A harness proposer reads traces, not training rows: each sample is the
trajectory of the requests a session sent, exactly as Reef served them.
``TraceSamples`` pairs with ``ReportRecord2Record`` and carries the reported
score; ``InferenceSamples`` pairs with ``InferenceRecord2Record`` and lets a
deployment without reports evolve on recorded traffic alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from reef.core.job_data import TrajectorySample
from reef.core.plugins import NoSettings
from reef.core.trajectories import make_trajectory
from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record, RecordSet, ReportContext, SingleRecordSet
from reef.record2sample.assembly import reported_task
from reef.record2sample.base import ImmediateRecord2Sample, SampleResult
from reef.record2sample.reports import ReportSamples


class TraceSamples(ReportSamples):
    """One report's referenced requests, as served, as one trace sample with the reported score.

    Several references become one trajectory sample. Requests are recorded
    post-transform, so a trace shows exactly what the backend served. Harness
    evolution pairs it with ``ReportRecord2Record``: every valid report trains
    by itself. A deployment without outcome reports names
    ``InferenceRecord2Record`` and ``InferenceSamples`` instead, to evolve on
    recorded traffic alone; a measured result beats model self judgment, so a
    deployment with reports keeps the report stages.
    """

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        environ: Mapping[str, str],
    ) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls()

    def make_sample(self, context: ReportContext) -> TrajectorySample:
        sample = make_trajectory(context.inferences, context.require_score(), context.report.payload.get("feedback"))
        task = reported_task(context.report.payload.get("metadata"))
        return sample if task is None else sample.with_metadata(task=task)


class InferenceSamples(ImmediateRecord2Sample):
    """One recorded inference as one unscored trace sample, with ``InferenceRecord2Record``.

    The proposer contract requires handling unscored samples.
    """

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        environ: Mapping[str, str],
    ) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls()

    def make_samples(self, record_set: RecordSet) -> SampleResult:
        if not isinstance(record_set, SingleRecordSet):
            raise TypeError(
                f"{type(self).__name__} makes tasks from single-record sets, not {type(record_set).__name__}"
            )
        return SampleResult.ready((make_trajectory((record_set.record,)),))


__all__ = ["InferenceSamples", "TraceSamples"]
