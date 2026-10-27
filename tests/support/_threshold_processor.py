"""Shared policy pipeline test double: assemble every valid scored report, and batch the samples as they come."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record, ReportContext, ReportGrouping, ReportRecord2Record
from reef.record2task import Record2Task, ReportTasks
from reef.record2task.assembly import SampleAssembly, SampleAssemblySettings
from reef.task2job import Task2Job, Task2JobSettings
from reef.train.pipeline import TrainingPipeline
from reef.train.types import JobData, TrainDataItem, TrajectoryItem


class ThresholdTasks(ReportTasks):
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

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        return self.assembly.build(context, context.require_score())


class ThresholdJobData(Task2Job):
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
        settings = Task2JobSettings.parse(config, plugin_class=cls)
        return cls(context, settings.buffer_size, settings.job_samples)

    def make_job_data(self, tasks: tuple[TrainDataItem, ...], data_id: str) -> JobData:
        return JobData(data_id, tasks)


def threshold_pipeline(
    context: PipelineContext,
    *,
    buffer_size: int | None = None,
    accept_multi_turn: bool = False,
    record2record: ReportRecord2Record | None = None,
    task2job: Task2Job | None = None,
    max_staleness: int = 0,
) -> TrainingPipeline:
    """Every report by itself (or ``record2record``), one sample each, batched by ``ThresholdJobData`` (or ``task2job``).

    ``max_staleness`` is the staleness limit of the default record stage.
    """
    return TrainingPipeline(
        context,
        record2record=(
            ReportRecord2Record(context, ReportGrouping(), max_staleness=max_staleness)
            if record2record is None
            else record2record
        ),
        record2task=ThresholdTasks(SampleAssemblySettings(accept_multi_turn_policy_samples=accept_multi_turn)),
        task2job=ThresholdJobData(context, buffer_size, None) if task2job is None else task2job,
    )
