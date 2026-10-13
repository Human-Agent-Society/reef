"""Harness evolution: recorded requests as traces, with reported scores or unscored, batched unmodified."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from reef.core import RequestType
from reef.core.plugins import NoSettings
from reef.core.trajectories import make_trajectory
from reef.pipeline_context import PipelineContext
from reef.record2record import (
    Record2Record,
    RecordSet,
    ReportContext,
    ReportGrouping,
    ReportRecord2Record,
    SingleRecordSet,
)
from reef.record2task import Record2Task, ReportTasks, TaskResult
from reef.record2task.assembly import reported_task
from reef.task2batch import BatchSettings, Task2Batch
from reef.train.types import TrainDataItem, TrainingBatch, TrajectoryItem


class TraceRecord2Record(ReportRecord2Record):
    """Traces from reports: every valid report trains by itself, as one set with the requests it references.

    Requests are recorded post-transform, so a trace shows exactly what the
    backend served. The default record stage of harness evolution. A
    deployment without outcome reports names ``InferenceRecord2Record`` and
    ``InferenceTasks`` instead, to evolve on recorded traffic alone; a
    measured result beats model self judgment, so a deployment with reports
    keeps this stage.
    """

    def __init__(self, context: PipelineContext) -> None:
        super().__init__(context, ReportGrouping())

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls(context)


class TraceTasks(ReportTasks):
    """One report's referenced requests, as served, as one trace task with the reported score.

    Several references become one trajectory sample.
    """

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        environ: Mapping[str, str],
    ) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls()

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        sample = make_trajectory(context.inferences, context.require_score(), context.report.payload.get("feedback"))
        task = reported_task(context.report.payload.get("metadata"))
        return sample if task is None else sample.with_metadata(task=task)


class InferenceTasks(Record2Task):
    """One recorded inference as one unscored trace task, with ``InferenceRecord2Record``.

    The proposer contract requires handling unscored samples.
    """

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        environ: Mapping[str, str],
    ) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls()

    def make_tasks(self, record_set: RecordSet) -> TaskResult:
        if not isinstance(record_set, SingleRecordSet):
            raise TypeError(
                f"{type(self).__name__} makes tasks from single-record sets, not {type(record_set).__name__}"
            )
        return TaskResult.ready((make_trajectory((record_set.record,)),))


class TraceBatches(Task2Batch):
    """Batch trace tasks unmodified; the backends consume them without more processing.

    In ``hybrid`` a queued instruction batches with the traces that an
    automatic batch would take next, up to ``batch_size``, so the proposer
    reads the request beside them; in ``manual`` it runs alone.
    """

    supported_training_modes = frozenset({"auto", "manual", "hybrid"})
    required_request_types = frozenset(RequestType)

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
        return TrainingBatch(f"{self.scenario}:harness_evolve:{batch_number}", tasks)
