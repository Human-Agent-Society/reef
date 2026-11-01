"""Shared policy pipeline test double: assemble every valid scored report, and batch the samples as they come."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from reef.pipeline_context import PipelineContext
from reef.record2sample import Record2Sample, ReportContext, ReportGrouping, ReportSamples
from reef.record2sample.assembly import SampleAssembly, SampleAssemblySettings
from reef.sample2job import Sample2Job, Sample2JobSettings
from reef.train.pipeline import TrainingPipeline
from reef.train.types import JobData, Sample, TrajectorySample


class ThresholdSamples(ReportSamples):
    """One report becomes one sample, with no score filtering; every report by itself, or in ``grouping``'s groups."""

    def __init__(
        self,
        context: PipelineContext,
        settings: SampleAssemblySettings | None = None,
        *,
        grouping: ReportGrouping | None = None,
    ) -> None:
        settings = SampleAssemblySettings() if settings is None else settings
        super().__init__(context, grouping, max_staleness=settings.max_staleness)
        self.assembly = SampleAssembly(settings)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        environ: Mapping[str, str],
    ) -> Self:
        return cls(context, SampleAssemblySettings.parse(config, plugin_class=cls))

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
    grouping: ReportGrouping | None = None,
    record2sample: Record2Sample | None = None,
    sample2job: Sample2Job | None = None,
    max_staleness: int = 0,
) -> TrainingPipeline:
    """Every report by itself (or in ``grouping``'s groups), one sample each, batched by ``ThresholdJobData`` (or
    ``sample2job``). ``record2sample`` replaces the ``ThresholdSamples`` stage.

    ``max_staleness`` is the staleness limit of the default stage.
    """
    settings = SampleAssemblySettings(accept_multi_turn_policy_samples=accept_multi_turn, max_staleness=max_staleness)
    return TrainingPipeline(
        context,
        record2sample=(
            ThresholdSamples(context, settings, grouping=grouping) if record2sample is None else record2sample
        ),
        sample2job=ThresholdJobData(context, buffer_size, None) if sample2job is None else sample2job,
    )
