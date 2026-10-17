"""Shared policy pipeline test double: assemble every valid scored report, and batch the samples as they come."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record, ReportContext, ReportGrouping, ReportRecord2Record
from reef.record2task import Record2Task, ReportTasks
from reef.record2task.assembly import SampleAssembly, SampleAssemblySettings
from reef.task2batch import BatchSettings, Task2Batch
from reef.train.pipeline import TrainingPipeline
from reef.train.types import TrainDataItem, TrainingBatch, TrajectoryItem


class ThresholdTasks(ReportTasks):
    """One report becomes one sample, with no score filtering."""

    def __init__(self, settings: SampleAssemblySettings | None = None) -> None:
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

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        return self.assembly.build(context, context.require_score())


class ThresholdBatches(Task2Batch):
    """Batch the tasks of ``ThresholdTasks`` as they come."""

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
        return cls(context, BatchSettings.parse(config, plugin_class=cls).batch_size)

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_id: str) -> TrainingBatch:
        return TrainingBatch(batch_id, tasks)


def threshold_pipeline(
    context: PipelineContext,
    *,
    batch_size: int = 1,
    accept_multi_turn: bool = False,
    record2record: ReportRecord2Record | None = None,
    task2batch: Task2Batch | None = None,
) -> TrainingPipeline:
    """Every report by itself (or ``record2record``), one sample each, batched by ``ThresholdBatches`` (or ``task2batch``)."""
    return TrainingPipeline(
        context,
        record2record=ReportRecord2Record(context, ReportGrouping()) if record2record is None else record2record,
        record2task=ThresholdTasks(SampleAssemblySettings(accept_multi_turn_policy_samples=accept_multi_turn)),
        task2batch=ThresholdBatches(context, batch_size) if task2batch is None else task2batch,
    )
