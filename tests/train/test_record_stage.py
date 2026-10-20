"""Stage contracts: how the training pipeline drives its three stages, and what a commit records."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import pytest
from tests.support._threshold_processor import threshold_pipeline
from tests.train.test_record_stage_batch_parity import CountingBackend, inference, instruction, report

from reef.core import AgentRecord
from reef.record2record import (
    GroupDecision,
    InferenceRecord2Record,
    NoRecordSets,
    RecordSet,
    ReportContext,
    ReportGrouping,
    ReportRecord2Record,
)
from reef.record2task import ImmediateRecord2Task, Record2Task, ReportTasks, SkippedReport, TaskResult
from reef.storage.sqlite import SQLiteRecordStore
from reef.task2batch import Task2Batch
from reef.train import PipelineContext, Trainer
from reef.train.pipeline import TrainingPipeline
from reef.train.types import TaskItem, TrainDataItem, TrainingBatch, TrajectoryItem


class WithholdingRecord2Record(InferenceRecord2Record):
    """Forms a set of each inference record, except records whose ID starts with ``drop``: it ends those itself."""

    def __init__(self) -> None:
        super().__init__()
        self.ended: set[str] = set()
        self.released: set[str] = set()
        self.pending = False
        self.closed = False

    def push(self, record: AgentRecord) -> None:
        if record.agent_record_id.startswith("drop"):
            self.ended.add(record.agent_record_id)
            return
        super().push(record)

    def releasable_record_ids(self) -> frozenset[str]:
        return frozenset(self.ended)

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        self.released |= agent_record_ids
        self.ended -= agent_record_ids

    def background_jobs_running(self) -> bool:
        return self.pending

    def status(self) -> Mapping[str, Any]:
        return {"ended_records": len(self.ended)}

    def close(self) -> None:
        self.closed = True


class RecordingTasks(ImmediateRecord2Task):
    """Keeps the record IDs of every set that it sees, and skips the set."""

    def __init__(self) -> None:
        super().__init__()
        self.assembled: list[str] = []

    def make_tasks(self, record_set: RecordSet) -> TaskResult:
        self.assembled.extend(record_set.record_ids())
        return TaskResult.skipped()


class RecordingBatches(Task2Batch):
    """Takes instructions in every mode."""

    supported_training_modes = frozenset({"auto", "manual", "hybrid"})

    def __init__(self, context: PipelineContext) -> None:
        super().__init__(context, 1)

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_id: str) -> TrainingBatch:
        return TrainingBatch(batch_id, tasks)


class RecordingPipeline(TrainingPipeline):
    """Sends records through ``WithholdingRecord2Record``, ``RecordingTasks``, and ``RecordingBatches``. It never becomes ready."""

    def __init__(self, context: PipelineContext) -> None:
        self.stage = WithholdingRecord2Record()
        self.recording = RecordingTasks()
        self.batches = RecordingBatches(context)
        super().__init__(context, record2record=self.stage, record2task=self.recording, task2batch=self.batches)


class LaterTasks(Record2Task):
    """Returns no result when a set arrives; ``finish`` queues a result that ``pull`` returns."""

    def __init__(self) -> None:
        self.finished: list[tuple[Hashable, TaskResult]] = []

    def push(self, record_set: RecordSet) -> None:
        return

    def finish(self, set_id: Hashable, result: TaskResult) -> None:
        self.finished.append((set_id, result))

    def pull(self) -> tuple[tuple[Hashable, TaskResult], ...]:
        finished, self.finished = tuple(self.finished), []
        return finished


class WaitingPipeline(TrainingPipeline):
    """Holds each inference record as one set whose tasks ``LaterTasks`` makes later."""

    def __init__(self, context: PipelineContext | None = None) -> None:
        context = PipelineContext("s") if context is None else context
        self.later_tasks = LaterTasks()
        self.batches = RecordingBatches(context)
        super().__init__(
            context, record2record=InferenceRecord2Record(), record2task=self.later_tasks, task2batch=self.batches
        )


class DroppingReportRecord2Record(ReportRecord2Record):
    """A report stage that ends every record whose ID starts with ``drop``, so the record is releasable."""

    def push(self, record: AgentRecord) -> None:
        if record.agent_record_id.startswith("drop"):
            self.end_report(record)
            return
        super().push(record)


def dropping_pipeline(context: PipelineContext) -> TrainingPipeline:
    return threshold_pipeline(context, record2record=DroppingReportRecord2Record(context, ReportGrouping()))


class PairGrouping(ReportGrouping):
    """Groups by ``metadata.group`` with slots from ``metadata.slot``; ready at two members."""

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        metadata = context.report.payload.get("metadata", {})
        return metadata.get("group"), metadata.get("slot")

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        return GroupDecision.READY if len(members) >= 2 else GroupDecision.INCOMPLETE


class PairTasks(ReportTasks):
    """Each report becomes a task named after it. Reports whose ID starts with ``skip`` cannot train."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_assembly = False

    def make_sample(self, context: ReportContext) -> TrainDataItem | SkippedReport:
        if self.fail_assembly:
            raise ValueError("broken training data")
        if context.report.agent_record_id.startswith("skip"):
            return SkippedReport("test")
        return TaskItem(Path(context.report.agent_record_id))


class PairBatches(Task2Batch):
    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_id: str) -> TrainingBatch:
        return TrainingBatch(batch_id, tasks)


class PairPipeline(TrainingPipeline):
    """Trains pairs of reports."""

    def __init__(self) -> None:
        context = PipelineContext("s")
        self.pair_tasks = PairTasks()
        super().__init__(
            context,
            record2record=ReportRecord2Record(context, PairGrouping(), exclusive_sources=True),
            record2task=self.pair_tasks,
            task2batch=PairBatches(context, 1),
        )


def slotted(record_id: str, reference: str, slot: str) -> AgentRecord:
    return report(record_id, reference, group="g", slot=slot)


@pytest.mark.unit
def test_the_inference_stage_forms_one_set_per_inference_record_in_arrival_order() -> None:
    stage = InferenceRecord2Record()
    for record_id in ("i1", "i2"):
        stage.push(inference(record_id))
    stage.push(report("r1", "i1"))
    formed = stage.pull()
    assert [record_set.record_ids() for record_set in formed] == [("i1",), ("i2",)]
    assert [record_set.arrival_index for record_set in formed] == [1, 2]
    assert stage.pull() == ()
    assert stage.releasable_record_ids() == {"r1"} and not stage.background_jobs_running()


@pytest.mark.unit
def test_the_pipeline_keeps_instructions_and_sends_other_records_through_its_stage() -> None:
    processor = RecordingPipeline(PipelineContext("s", training_mode="hybrid"))
    processor.push(inference("i1"))
    processor.push(inference("drop-1"))
    processor.push(instruction("train-1"))
    assert processor.recording.assembled == ["i1"]
    assert processor.buffered_requests() == 1
    assert processor.releasable_record_ids() == {"drop-1"}


@pytest.mark.unit
def test_the_pipeline_adds_its_stages_background_work_and_status_and_closes_them() -> None:
    processor = RecordingPipeline(PipelineContext("s", training_mode="hybrid"))
    assert not processor.background_jobs_running()
    processor.stage.pending = True
    assert processor.background_jobs_running()
    assert processor.status() == {
        "ready_units": 0,
        "batch_size": 1,
        "buffered_requests": 0,
        "ended_records": 0,
    }
    processor.close()
    assert processor.stage.closed


@pytest.mark.unit
def test_a_record_the_stage_ends_is_consumed_by_the_next_commit_and_then_released() -> None:
    records = SQLiteRecordStore()
    for record in (inference("i1"), inference("drop-1"), report("r1", "i1")):
        records.append(record)
    trainer = Trainer.build(
        "s",
        records,
        pipeline_factory=dropping_pipeline,
        candidate_backend=CountingBackend(),
    )
    batch = trainer.reserve_training_batch()
    assert batch is not None
    execution = trainer.execute_reserved_step(0)
    assert execution.result is not None
    prepared = trainer.prepare_commit(execution.result)
    assert prepared.consumed_ids == {"i1", "r1", "drop-1"}
    trainer.commit(prepared)
    assert not trainer.pipeline.releasable_record_ids()
    assert "drop-1" in trainer.consumed_record_ids


@pytest.mark.unit
def test_the_trainer_reports_the_stages_background_work() -> None:
    trainer = Trainer.build("s", SQLiteRecordStore(), pipeline_factory=RecordingPipeline)
    processor = trainer.pipeline
    assert isinstance(processor, RecordingPipeline)
    assert not trainer.pipeline.background_jobs_running()
    processor.stage.pending = True
    assert trainer.pipeline.background_jobs_running()


@pytest.mark.unit
def test_a_retry_in_a_taken_slot_is_terminal_and_released() -> None:
    processor = PairPipeline()
    for record_id in ("i1", "i2", "i3"):
        processor.push(inference(record_id))
    processor.push(slotted("r1", "i1", "a"))
    processor.push(slotted("r2", "i2", "a"))
    processor.push(slotted("r3", "i3", "b"))
    batch = processor.build_batch()
    assert [str(item.task_path) for item in batch.items] == ["r1", "r3"]
    assert {"r2", "i2"} <= processor.releasable_record_ids()
    assert "r2" not in processor.tasks


@pytest.mark.unit
def test_a_failed_assembly_keeps_the_report_and_leaves_its_slot_free() -> None:
    processor = PairPipeline()
    for record_id in ("i1", "i2"):
        processor.push(inference(record_id))
    processor.pair_tasks.fail_assembly = True
    with pytest.raises(ValueError, match="broken training data"):
        processor.push(slotted("r1", "i1", "a"))
    assert not processor.releasable_record_ids() & {"r1", "i1"}
    processor.pair_tasks.fail_assembly = False
    processor.push(slotted("r1", "i1", "a"))
    processor.push(slotted("r2", "i2", "b"))
    assert [str(item.task_path) for item in processor.build_batch().items] == ["r1", "r2"]


@pytest.mark.unit
def test_a_skipped_report_is_released_and_never_batched() -> None:
    processor = PairPipeline()
    processor.push(inference("i1"))
    processor.push(slotted("skip-1", "i1", "a"))
    assert {"skip-1", "i1"} <= processor.releasable_record_ids()
    assert not processor.ready()


@pytest.mark.unit
def test_a_batch_holds_the_tasks_of_every_report_in_a_group_in_arrival_order() -> None:
    processor = PairPipeline()
    for record_id in ("i1", "i2"):
        processor.push(inference(record_id))
    processor.push(slotted("r2", "i2", "b"))
    processor.push(slotted("r1", "i1", "a"))
    batch = processor.build_batch()
    assert [str(item.task_path) for item in batch.items] == ["r2", "r1"]
    assert processor.acknowledge(batch.batch_id) == {"i1", "i2", "r1", "r2"}
    assert not processor.tasks


@pytest.mark.unit
def test_the_report_stage_releases_a_report_whose_data_is_behind_the_serving_version() -> None:
    processor = threshold_pipeline(PipelineContext("s"))
    processor.observe_serving_runtime_load_id("e:1")
    processor.push(inference("i1", version="e:1"))
    processor.push(report("r1", "i1"))
    processor.observe_serving_runtime_load_id("e:2")
    processor.push(inference("i2", version="e:2"))
    processor.push(report("r2", "i2"))
    batch = processor.build_batch()
    assert [item.source_agent_record_ids for item in batch.items] == [("i2", "r2")]
    # The stale report ended with the inference that it owns, and its tasks are gone.
    assert {"r1", "i1"} <= processor.releasable_record_ids()
    assert "r1" not in processor.tasks


@pytest.mark.unit
def test_the_default_pipeline_forms_no_record_set_and_makes_no_batch() -> None:
    processor = TrainingPipeline(PipelineContext("s"))
    assert isinstance(processor.record2record, NoRecordSets)
    processor.push(inference("i1"))
    processor.push(report("r1", "i1"))
    assert processor.record2record.pull() == ()
    assert not processor.ready() and not processor.releasable_record_ids()


@pytest.mark.unit
def test_a_waiting_set_holds_its_records_and_settles_when_its_result_comes_back() -> None:
    processor = WaitingPipeline()
    processor.push(inference("i1"))
    processor.push(inference("i2"))
    assert not processor.ready() and processor.background_jobs_running()
    processor.later_tasks.finish("i2", TaskResult.skipped())
    assert not processor.ready() and processor.releasable_record_ids() == {"i2"}
    processor.later_tasks.finish("i1", TaskResult.ready((TaskItem(Path("t1")),)))
    assert processor.ready(), "the result alone makes a batch, without another record"
    assert not processor.background_jobs_running()
    assert [str(item.task_path) for item in processor.build_batch().items] == ["t1"]


@pytest.mark.unit
def test_a_second_result_for_a_settled_set_is_refused() -> None:
    processor = WaitingPipeline()
    processor.push(inference("i1"))
    result = TaskResult.ready((TaskItem(Path("t1")),))
    processor.later_tasks.finish("i1", result)
    assert processor.ready()
    processor.later_tasks.finish("i1", result)
    with pytest.raises(ValueError, match="set 'i1', which no set waits for"):
        processor.ready()


@pytest.mark.unit
def test_the_trainer_reserves_a_batch_when_a_waiting_set_finishes() -> None:
    records = SQLiteRecordStore()
    records.append(inference("i1"))
    trainer = Trainer.build("s", records, pipeline_factory=WaitingPipeline, candidate_backend=CountingBackend())
    assert trainer.reserve_training_batch() is None
    assert trainer.holds_untrained_records(), "the trainer keeps polling while the set waits"
    processor = trainer.pipeline
    assert isinstance(processor, WaitingPipeline)
    processor.later_tasks.finish("i1", TaskResult.ready((TaskItem(Path("t1")),)))
    batch = trainer.reserve_training_batch()
    assert batch is not None and [str(item.task_path) for item in batch.items] == ["t1"]


class InferenceTasks(ImmediateRecord2Task):
    """Each inference set becomes one task named after its record."""

    def make_tasks(self, record_set: RecordSet) -> TaskResult:
        record_ids = record_set.record_ids()
        return TaskResult.ready((TaskItem(Path(record_ids[0]), source_agent_record_ids=record_ids),))


class DataItemBatches(PairBatches):
    accepted_items = frozenset({TrajectoryItem, TaskItem})


TRAJECTORY = TrajectoryItem({"schema_version": "ATIF-v1.4", "agent": {"name": "fixture"}, "steps": [{"step_id": 1}]})


@pytest.mark.unit
def test_uploaded_data_items_go_first_in_a_batch_and_count_as_units() -> None:
    context = PipelineContext("s")
    processor = TrainingPipeline(
        context,
        record2record=InferenceRecord2Record(),
        record2task=InferenceTasks(),
        task2batch=DataItemBatches(context, 2),
    )
    processor.push(inference("i1"))
    assert not processor.ready()
    processor.push_item("t1", TRAJECTORY)
    processor.push_item("k1", TaskItem(Path("k1")))
    assert processor.status()["ready_units"] == 3

    batch = processor.build_batch()
    assert batch.items == (TRAJECTORY, TaskItem(Path("k1")))
    processor.release_batch(batch.batch_id)
    batch = processor.build_batch()
    assert batch.items == (TRAJECTORY, TaskItem(Path("k1")))
    assert processor.acknowledge(batch.batch_id) == {"item:t1", "item:k1"}

    # A consumed data item that the trainer reads again before the commit releases it does not queue again.
    processor.push_item("t1", TRAJECTORY)
    processor.push_item("t2", TRAJECTORY)
    batch = processor.build_batch()
    assert batch.items == (TRAJECTORY, TaskItem(Path("i1"), source_agent_record_ids=("i1",)))
    assert processor.acknowledge(batch.batch_id) == {"item:t2", "i1"}
    assert {"item:t1", "item:k1", "item:t2"} <= processor.releasable_record_ids()
    processor.release_records(frozenset({"item:t1", "item:k1", "item:t2"}))
    assert not processor.consumed_items


@pytest.mark.unit
def test_a_pipeline_refuses_a_data_item_kind_that_its_batch_stage_does_not_accept() -> None:
    context = PipelineContext("s")
    processor = TrainingPipeline(context, task2batch=PairBatches(context, 1))
    with pytest.raises(ValueError, match="does not accept TrajectoryItem uploads"):
        processor.push_item("t1", TRAJECTORY)
