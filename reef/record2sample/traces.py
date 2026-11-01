"""Trace samples for harness evolution: recorded requests as served, with a reported score or unscored.

A harness proposer reads traces, not training rows: each sample is the
trajectory of the requests a session sent, exactly as Reef served them.
``TraceSamples`` carries the reported score; ``InferenceSamples`` makes one
sample per inference record and lets a deployment without reports evolve on
recorded traffic alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from reef.core.job_data import TrajectorySample
from reef.core.trajectories import make_trajectory
from reef.pipeline_context import PipelineContext
from reef.record2sample.assembly import reported_task
from reef.record2sample.base import InferenceRecordSamples, Record2SampleSettings, SampleResult
from reef.record2sample.reports import ReportContext, ReportSamples
from reef.record2sample.sets import RecordSet, SingleRecordSet


class TraceSamples(ReportSamples):
    """One report's referenced requests, as served, as one trace sample with the reported score.

    Several references become one trajectory sample. Requests are recorded
    post-transform, so a trace shows exactly what the backend served. Every
    valid report trains by itself. A deployment without outcome reports names
    ``InferenceSamples`` instead, to evolve on recorded traffic alone; a
    measured result beats model self judgment, so a deployment with reports
    keeps this stage. Config: ``Record2SampleSettings``.
    """

    def make_sample(self, context: ReportContext) -> TrajectorySample:
        sample = make_trajectory(context.inferences, context.require_score(), context.report.payload.get("feedback"))
        task = reported_task(context.report.payload.get("metadata"))
        return sample if task is None else sample.with_metadata(task=task)


class InferenceSamples(InferenceRecordSamples):
    """One recorded inference as one unscored trace sample.

    The proposer contract requires handling unscored samples. Config: ``Record2SampleSettings``.
    """

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        environ: Mapping[str, str],
    ) -> Self:
        return cls(max_staleness=Record2SampleSettings.parse(config, plugin_class=cls).max_staleness)

    def make_samples(self, record_set: RecordSet) -> SampleResult:
        if not isinstance(record_set, SingleRecordSet):
            raise TypeError(
                f"{type(self).__name__} makes tasks from single-record sets, not {type(record_set).__name__}"
            )
        return SampleResult.ready((make_trajectory((record_set.record,)),))


__all__ = ["InferenceSamples", "TraceSamples"]
