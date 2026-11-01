"""SAO's record-to-sample and sample-to-job stages: one scored rollout, one training unit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from reef.pipeline_context import PipelineContext
from reef.record2sample import Record2Sample, ReportContext, ReportSamples
from reef.record2sample.assembly import SampleAssembly, SampleAssemblySettings
from reef.sample2job import Sample2Job, Sample2JobSettings
from reef.train.types import JobData, Sample, TrajectorySample


class SAOSamples(ReportSamples):
    """One scored rollout as one SAO sample: its trajectory with the score, and ``rollout_created_at``, the time of
    its last call, for the backend's queue-age metric.

    Every rollout trains by itself and owns its inference records. The training component renders the sample, and
    its action mask is the loss mask of the rendered tokens. Config: ``SampleAssemblySettings``.
    """

    def __init__(self, context: PipelineContext, settings: SampleAssemblySettings) -> None:
        super().__init__(context, exclusive_sources=True, max_staleness=settings.max_staleness)
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
        sample = self.assembly.build(context, context.require_score())
        return sample.with_training(rollout_created_at=context.ends[-1].created_at)


class SAOJobData(Sample2Job):
    """Put scored rollouts into job data as independently scheduled SAO samples.

    Single-Rollout Asynchronous Optimization ships each completed rollout on
    its own — no comparison group, no slowest-sample barrier. The pipeline
    collects ``job_samples`` accepted rollouts, from as many prompts, into
    one job. With ``job_samples`` equal to Slime's ``--global-batch-size``,
    the job is one optimizer step; the paper uses 128 (arXiv:2607.07508,
    §4.1), and 1 trains once per accepted rollout for smoke runs.

    SAO reuses ``TrajectorySample`` / ``JobData``: a sample carries
    ``rollout_created_at``, and the training component renders it, with the
    loss mask as its action mask. The training backend validates the rendered
    tensors; malformed training input fails explicitly.

    Config: ``buffer_size`` and ``job_samples`` (``Sample2JobSettings``).
    """

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

    def make_job_data(self, items: tuple[Sample, ...], data_id: str) -> JobData:
        return JobData(data_id, items)
