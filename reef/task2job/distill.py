"""Job data of distillation tasks; ``reef.record2task.distill`` makes the tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from reef.core.job_data import JobData, TrainDataItem
from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record
from reef.record2task import Record2Task
from reef.task2job.base import Task2Job, Task2JobSettings


class DistillJobData(Task2Job):
    """Put the tasks into one job's data as they come, in selection order."""

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
        settings = Task2JobSettings.parse(config, plugin_class=cls)
        return cls(context, settings.buffer_size, settings.job_samples)

    def make_job_data(self, tasks: tuple[TrainDataItem, ...], data_id: str) -> JobData:
        return JobData(data_id, tasks)
