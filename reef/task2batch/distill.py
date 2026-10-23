"""Batches of distillation tasks; ``reef.record2task.distill`` makes the tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from reef.core.batches import TrainDataItem, TrainingBatch
from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record
from reef.record2task import Record2Task
from reef.task2batch.base import BufferSettings, Task2Batch


class DistillBatches(Task2Batch):
    """Batch the tasks as they come, in batch order."""

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

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_id: str) -> TrainingBatch:
        return TrainingBatch(batch_id, tasks)
