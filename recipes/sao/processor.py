"""SAO's record-to-sample and sample-to-job stages: one scored rollout, one training unit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from reef.core.records_types import AgentRecord
from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record, ReportContext
from reef.record2sample import Record2Sample, ReportSamples
from reef.record2sample.assembly import SampleAssembly, SampleAssemblySettings
from reef.record2sample.samples import make_policy_trajectory, recorded_training
from reef.sample2job import Sample2Job, Sample2JobSettings
from reef.train.types import JobData, Sample, TrajectorySample


class SAOSampleAssembly(SampleAssembly):
    """The shared assembly with SAO's two extra fields on each single-call sample.

    ``make_policy_trajectory`` reads the shared training fields, so SAO and
    the group-relative recipes resolve every shared field identically,
    including the ``runtime_load_id`` fallback chain the durable runtime
    needs to identify a training job's producing version. SAO then fills the
    two fields that path leaves at their defaults: ``action_mask`` (read from
    ``response.training`` first, the top-level payload second) and
    ``rollout_created_at``, for the backend's queue-age metric.
    """

    def single_sample(self, record: AgentRecord, score: float) -> TrajectorySample:
        base = make_policy_trajectory(record, score)
        action_mask = recorded_training(record.payload).get("action_mask", record.payload.get("action_mask", ()))
        return base.with_training(
            action_mask=[int(value) for value in action_mask], rollout_created_at=record.created_at
        )


class SAOSamples(ReportSamples):
    """One scored rollout as one SAO sample, with ``action_mask`` and ``rollout_created_at``.

    Config: the sample-assembly keys of ``SampleAssemblySettings``.
    """

    def __init__(self, settings: SampleAssemblySettings) -> None:
        super().__init__()
        self.assembly = SAOSampleAssembly(settings)

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
        sample = self.assembly.build(context, context.require_score())
        if not sample.training.get("action_mask", []):
            sample = sample.with_training(action_mask=sample.training.get("loss_mask", []))
        return sample


class SAOJobData(Sample2Job):
    """Put scored rollouts into job data as independently scheduled SAO samples.

    Single-Rollout Asynchronous Optimization ships each completed rollout on
    its own — no comparison group, no slowest-sample barrier. The pipeline
    collects ``job_samples`` accepted rollouts, from as many prompts, into
    one job. With ``job_samples`` equal to Slime's ``--global-batch-size``,
    the job is one optimizer step; the paper uses 128 (arXiv:2607.07508,
    §4.1), and 1 trains once per accepted rollout for smoke runs.

    SAO reuses ``TrajectorySample`` / ``JobData`` and fills ``action_mask``
    and ``rollout_created_at``. The training backend validates required
    tensors; malformed training input fails explicitly.

    Config: ``buffer_size`` and ``job_samples`` (``Sample2JobSettings``).
    """

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

    def make_job_data(self, items: tuple[Sample, ...], data_id: str) -> JobData:
        return JobData(data_id, items)
