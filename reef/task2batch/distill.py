"""Batches of distillation tasks; ``reef.record2task.distill`` makes the tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from reef.core.batches import TrainDataItem, TrainingBatch
from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record
from reef.record2task import Record2Task
from reef.task2batch.base import BatchSettings, Task2Batch


class DistillBatches(Task2Batch):
    """Batch tasks as they come, ``batch_size`` record sets to a batch. A subclass sets ``batch_label``, its batches' name."""

    batch_label = "teacher"

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

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:{self.batch_label}:{batch_number}", tasks)
