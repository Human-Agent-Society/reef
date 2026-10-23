"""SAO's record-to-task and task-to-batch stages: one scored rollout, one training unit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from reef.core.records_types import AgentRecord
from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record, ReportContext
from reef.record2task import Record2Task, ReportTasks
from reef.record2task.assembly import SampleAssembly, SampleAssemblySettings
from reef.record2task.samples import make_policy_trajectory, recorded_training
from reef.task2batch import BufferSettings, Task2Batch
from reef.train.types import TrainDataItem, TrainingBatch, TrajectoryItem


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

    def single_sample(self, record: AgentRecord, score: float) -> TrajectoryItem:
        base = make_policy_trajectory(record, score)
        action_mask = recorded_training(record.payload).get("action_mask", record.payload.get("action_mask", ()))
        return base.with_training(
            action_mask=[int(value) for value in action_mask], rollout_created_at=record.created_at
        )


class SAOTasks(ReportTasks):
    """One scored rollout as one SAO task, with ``action_mask`` and ``rollout_created_at``.

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

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        sample = self.assembly.build(context, context.require_score())
        if not sample.training.get("action_mask", []):
            sample = sample.with_training(action_mask=sample.training.get("loss_mask", []))
        return sample


class SAOBatches(Task2Batch):
    """Batch scored rollouts as independently scheduled SAO samples.

    Single-Rollout Asynchronous Optimization ships each completed rollout on
    its own — no comparison group, no slowest-sample barrier. The pipeline
    collects the backend's batch size (Slime ``--global-batch-size``) of
    accepted rollouts, from as many prompts, into one optimizer step; the
    paper uses 128 (arXiv:2607.07508, §4.1), and 1 trains once per accepted
    rollout for smoke runs.

    SAO reuses ``TrajectoryItem`` / ``TrainingBatch`` and fills ``action_mask``
    and ``rollout_created_at``. The training backend validates required
    tensors; malformed training input fails explicitly.

    Config: ``buffer_size`` (``BufferSettings``).
    """

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        record2task: Record2Task,
        environ: Mapping[str, str],
    ) -> Self:
        return cls(context, BufferSettings.parse(config, plugin_class=cls).buffer_size)

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_id: str) -> TrainingBatch:
        return TrainingBatch(batch_id, items)
