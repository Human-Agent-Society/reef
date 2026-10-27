"""Shared policy pipeline test double: assemble every valid scored report, and batch the samples as they come."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record, ReportContext, ReportGrouping, ReportRecord2Record
from reef.record2sample import Record2Sample, ReportSamples
from reef.record2sample.assembly import SampleAssembly, SampleAssemblySettings
from reef.sample2job import Sample2Job, Sample2JobSettings
from reef.train.pipeline import TrainingPipeline
from reef.train.types import JobData, Sample, TrajectorySample


class ThresholdSamples(ReportSamples):
    """One report becomes one sample, with no score filtering."""

    def __init__(self, settings: SampleAssemblySettings | None = None) -> None:
        super().__init__()
        self.assembly = SampleAssembly(SampleAssemblySettings() if settings is None else settings)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        environ: Mapping[str, str],
    ) -> Self:
        return cls(SampleAssemblySettings.parse(config, plugin_class=cls))

    def make_sample(self, context: ReportContext) -> TrajectorySample:
        return self.assembly.build(context, context.require_score())


class ThresholdJobData(Sample2Job):
    """Batch the tasks of ``ThresholdSamples`` as they come."""

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        record2sample: Record2Sample,
        environ: Mapping[str, str],
    ) -> Self:
        settings = Sample2JobSettings.parse(config, plugin_class=cls)
        return cls(context, settings.buffer_size, settings.job_samples)

    def make_job_data(self, samples: tuple[Sample, ...], data_id: str) -> JobData:
        return JobData(data_id, samples)


def threshold_pipeline(
    context: PipelineContext,
    *,
    buffer_size: int | None = None,
    accept_multi_turn: bool = False,
    record2record: ReportRecord2Record | None = None,
    sample2job: Sample2Job | None = None,
    max_staleness: int = 0,
) -> TrainingPipeline:
    """Every report by itself (or ``record2record``), one sample each, batched by ``ThresholdJobData`` (or ``sample2job``).

    ``max_staleness`` is the staleness limit of the default record stage.
    """
    return TrainingPipeline(
        context,
        record2record=(
            ReportRecord2Record(context, ReportGrouping(), max_staleness=max_staleness)
            if record2record is None
            else record2record
        ),
        record2sample=ThresholdSamples(SampleAssemblySettings(accept_multi_turn_policy_samples=accept_multi_turn)),
        sample2job=ThresholdJobData(context, buffer_size, None) if sample2job is None else sample2job,
    )
