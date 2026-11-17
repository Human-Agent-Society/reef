"""Shared policy pipeline test double: assemble every valid scored report, and batch the samples as they come."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

from reef.core.config import config_option
from reef.pipeline_context import PipelineContext
from reef.record2sample import Record2Sample, ReportContext, ReportGrouping, ReportSamples
from reef.record2sample.assembly import SampleAssembly, SampleAssemblySettings
from reef.record2sample.staleness import ServedArtifacts, lags_behind
from reef.sample2job import Sample2Job, Sample2JobSettings
from reef.train.pipeline import TrainingPipeline
from reef.train.types import JobData, Sample, TrajectorySample


@dataclass(frozen=True)
class ThresholdSettings(SampleAssemblySettings):
    """The config of ``ThresholdSamples``: the sample assembly, and ``max_artifact_lag``, a non-negative int: the
    artifacts that the scenario may have served after the one that made a sample, for the sample to train."""

    max_artifact_lag: int = config_option(0, help="The artifacts that a sample may lag behind the served one.")


class ThresholdSamples(ReportSamples):
    """One report becomes one sample, with no score filtering; every report by itself, or in ``grouping``'s groups.
    A sample goes stale when it lags more than ``max_artifact_lag`` artifacts behind the served one."""

    def __init__(
        self,
        context: PipelineContext,
        settings: ThresholdSettings | None = None,
        *,
        grouping: ReportGrouping | None = None,
    ) -> None:
        settings = ThresholdSettings() if settings is None else settings
        super().__init__(context, grouping)
        self.assembly = SampleAssembly(settings)
        self.max_artifact_lag = settings.max_artifact_lag

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        environ: Mapping[str, str],
    ) -> Self:
        return cls(context, ThresholdSettings.parse(config, plugin_class=cls))

    def is_stale(self, samples: Sequence[Sample], served: ServedArtifacts) -> bool:
        return lags_behind(samples, served, self.max_artifact_lag)

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
    job_samples: int | None = None,
    accept_multi_turn: bool = False,
    grouping: ReportGrouping | None = None,
    record2sample: Record2Sample | None = None,
    sample2job: Sample2Job | None = None,
    max_artifact_lag: int = 0,
) -> TrainingPipeline:
    """Every report by itself (or in ``grouping``'s groups), one sample each, batched by ``ThresholdJobData`` (or
    ``sample2job``). ``record2sample`` replaces the ``ThresholdSamples`` stage.

    ``job_samples`` is the job size of the default sample-to-job stage, and
    ``max_artifact_lag`` is the lag bound of the default stage.
    """
    settings = ThresholdSettings(accept_multi_turn_policy_samples=accept_multi_turn, max_artifact_lag=max_artifact_lag)
    return TrainingPipeline(
        context,
        record2sample=(
            ThresholdSamples(context, settings, grouping=grouping) if record2sample is None else record2sample
        ),
        sample2job=ThresholdJobData(context, buffer_size, job_samples) if sample2job is None else sample2job,
    )
