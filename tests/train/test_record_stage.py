"""Stage contracts: how the training pipeline drives its two stages, and what a commit records."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from tests.support._threshold_processor import ThresholdSamples, threshold_pipeline
from tests.support._trajectories import policy_trajectory
from tests.support.records import memory_records
from tests.support.runtime_stubs import STUB_ROOT
from tests.support.samples import data_storage, made_task, open_sample_store, sample_store
from tests.support.samples_service import make_samples
from tests.support.task_service_stubs import task_for
from tests.train.test_record_stage_job_data_parity import counting_jobs, inference, report

from reef.artifact import SnapshotRef
from reef.core import AgentRecord, RequestType
from reef.core.job_data import JobOutcome
from reef.core.tasks import write_harbor_task
from reef.core.training_request import TrainingRequest
from reef.record2sample import (
    GroupDecision,
    InferenceRecordSamples,
    NoSamples,
    Record2Sample,
    RecordSet,
    ReportContext,
    ReportGrouping,
    ReportSamples,
    SampleResult,
    SingleRecordSet,
    SkippedReport,
)
from reef.sample2job import Sample2Job, Sample2JobSettings
from reef.storage.commits import CommitRecord
from reef.storage.data import MAX_QUERY_LIMIT, NewItem
from reef.storage.samples import SampleStore, consumed_sample_id, stage_item_id
from reef.train import PipelineContext, Trainer
from reef.train.pipeline import TrainingPipeline
from reef.train.types import JobData, Sample, TaskSample, TrajectorySample


class RecordingSamples(InferenceRecordSamples):
    """Forms a set of each inference record, except records whose ID starts with ``drop``: it ends those itself.

    It keeps the record IDs of every set that it forms, and skips the set.
    """

    def __init__(self) -> None:
        super().__init__()
        self.assembled: list[str] = []
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

    def make_samples(self, record_set: RecordSet) -> SampleResult:
        self.assembled.extend(record_set.record_ids())
        return SampleResult.skipped()


class RecordingJobData(Sample2Job):
    """Makes job data of the samples as they come; in ``manual`` it takes requests."""

    def __init__(self, context: PipelineContext, job_samples: int | None = None) -> None:
        super().__init__(context, None, job_samples)

    def make_job_data(self, samples: tuple[Sample, ...], data_id: str) -> JobData:
        return JobData(data_id, samples)


class RecordingPipeline(TrainingPipeline):
    """Sends records through ``RecordingSamples`` and ``RecordingJobData``. It never becomes ready."""

    def __init__(self, context: PipelineContext) -> None:
        self.stage = RecordingSamples()
        self.batches = RecordingJobData(context)
        super().__init__(context, record2sample=self.stage, sample2job=self.batches)


class LaterSamples(Record2Sample):
    """Holds each inference record; ``finish`` gives the result of its set, which forms at the next ``advance``, as
    the end of a background job does."""

    def __init__(self) -> None:
        super().__init__()
        self.waiting: dict[str, AgentRecord] = {}
        self.finished: dict[Hashable, SampleResult] = {}

    def push(self, record: AgentRecord) -> None:
        if record.request_type is not RequestType.INFERENCE:
            self.terminal.add(record.agent_record_id)
            return
        self.waiting[record.agent_record_id] = record

    def finish(self, set_id: str, result: SampleResult) -> None:
        self.finished[set_id] = result

    def advance(self, now: float) -> None:
        for set_id in tuple(self.finished):
            self.form(SingleRecordSet(set_id=set_id, record=self.waiting.pop(set_id)))

    def make_samples(self, record_set: RecordSet) -> SampleResult:
        return self.finished.pop(record_set.set_id)

    def background_jobs_running(self) -> bool:
        return bool(self.waiting)


class WaitingPipeline(TrainingPipeline):
    """Holds each inference record as one set whose tasks ``LaterSamples`` makes later; jobs of ``job_samples``."""

    def __init__(self, context: PipelineContext | None = None, job_samples: int | None = None) -> None:
        context = PipelineContext("s") if context is None else context
        self.later_tasks = LaterSamples()
        self.batches = RecordingJobData(context, job_samples)
        super().__init__(context, record2sample=self.later_tasks, sample2job=self.batches)


class DroppingThresholdSamples(ThresholdSamples):
    """A report stage that ends every record whose ID starts with ``drop``, so the record is releasable."""

    def push(self, record: AgentRecord) -> None:
        if record.agent_record_id.startswith("drop"):
            self.end_report(record)
            return
        super().push(record)


def dropping_pipeline(context: PipelineContext) -> TrainingPipeline:
    return threshold_pipeline(context, job_samples=1, record2sample=DroppingThresholdSamples(context))


class PairGrouping(ReportGrouping):
    """Groups by ``metadata.group`` with slots from ``metadata.slot``; ready at two members."""

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        metadata = context.report.payload.get("metadata", {})
        return metadata.get("group"), metadata.get("slot")

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        return GroupDecision.READY if len(members) >= 2 else GroupDecision.INCOMPLETE


class PairSamples(ReportSamples):
    """Trains pairs of reports (``PairGrouping``); each report becomes a task named after it. Reports whose ID starts
    with ``skip`` cannot train."""

    def __init__(self, context: PipelineContext) -> None:
        super().__init__(context, PairGrouping(), exclusive_sources=True)
        self.fail_assembly = False

    def make_sample(self, context: ReportContext) -> Sample | SkippedReport:
        if self.fail_assembly:
            raise ValueError("broken training data")
        if context.report.agent_record_id.startswith("skip"):
            return SkippedReport("test")
        return named_task(context.report.agent_record_id)


def named_task(name: str, **fields: Any) -> TaskSample:
    """A task sample whose metadata holds ``name``, so that its name survives a round trip through the data store."""
    return made_task(name, metadata={"name": name}, **fields)


def task_names(job_data: JobData) -> list[str]:
    return [sample.metadata["name"] for sample in job_data.samples]


class PairJobData(Sample2Job):
    def make_job_data(self, samples: tuple[Sample, ...], data_id: str) -> JobData:
        return JobData(data_id, samples)


class PairPipeline(TrainingPipeline):
    """Trains pairs of reports."""

    def __init__(self) -> None:
        context = PipelineContext("s")
        self.pair_tasks = PairSamples(context)
        super().__init__(context, record2sample=self.pair_tasks, sample2job=PairJobData(context, None, None))


def slotted(record_id: str, reference: str, slot: str) -> AgentRecord:
    return report(record_id, reference, group="g", slot=slot)


@pytest.mark.unit
def test_the_inference_stage_forms_one_set_per_inference_record_in_arrival_order() -> None:
    stage = InferenceSamples()
    for record_id in ("i1", "i2"):
        stage.push(inference(record_id))
    stage.push(report("r1", "i1"))
    assert [unit.set_ids() for unit in stage.take_ready_units()] == [("i1",), ("i2",)]
    assert stage.take_ready_units() == ()
    assert stage.releasable_record_ids() == {"r1"} and not stage.background_jobs_running()


@pytest.mark.unit
def test_the_pipeline_keeps_requests_and_sends_records_through_its_stage() -> None:
    processor = RecordingPipeline(PipelineContext("s", training_mode="manual"))
    processor.push(inference("i1"))
    processor.push(inference("drop-1"))
    processor.queue_requests(
        [TrainingRequest.from_dict({"text": "train-1", "session": "s-1", "artifact_id": "r-1"}, request_id="train-1")]
    )
    assert processor.stage.assembled == ["i1"]
    assert len(processor.training_requests) == 1
    assert processor.record2sample.releasable_record_ids() == {"drop-1"}


@pytest.mark.unit
def test_the_pipeline_adds_its_stages_background_work_and_status_and_closes_them() -> None:
    processor = RecordingPipeline(PipelineContext("s", training_mode="manual"))
    assert not processor.background_jobs_running()
    processor.stage.pending = True
    assert processor.background_jobs_running()
    assert processor.status() == {
        "ready_samples": 0,
        "buffer_size": None,
        "ended_records": 0,
    }
    processor.close()
    assert processor.stage.closed


@pytest.mark.unit
def test_a_record_the_stage_ends_is_consumed_by_the_next_commit_and_then_released() -> None:
    records = memory_records("s")
    for record in (inference("i1"), inference("drop-1"), report("r1", "i1")):
        records.append(record)
    trainer = Trainer.build(
        "s",
        records,
        pipeline_factory=dropping_pipeline,
        job_loop=counting_jobs(),
        root=STUB_ROOT,
    )
    make_samples(trainer)
    result = trainer.advance_job()
    assert result is not None
    prepared = trainer.prepare_commit(result)
    assert prepared.consumed_ids == {"i1", "r1", "drop-1", consumed_sample_id(stage_item_id("s", "r1:0"))}
    trainer.settle_commit(prepared)
    assert "drop-1" in trainer.consumed_record_ids
    assert "drop-1" in trainer.pipeline.record2sample.releasable_record_ids(), "the commit does not reach the stage"
    make_samples(trainer)
    assert not trainer.pipeline.record2sample.releasable_record_ids()


@pytest.mark.unit
def test_the_trainer_reports_the_stages_background_work() -> None:
    trainer = Trainer.build("s", memory_records("s"), pipeline_factory=RecordingPipeline)
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
    batch = processor.build_job_data(1)
    assert task_names(batch) == ["r1", "r3"]
    assert {"r2", "i2"} <= processor.record2sample.releasable_record_ids()
    assert "r2" not in processor.record2sample.set_samples


@pytest.mark.unit
def test_a_failed_assembly_keeps_the_report_and_leaves_its_slot_free() -> None:
    processor = PairPipeline()
    for record_id in ("i1", "i2"):
        processor.push(inference(record_id))
    processor.pair_tasks.fail_assembly = True
    with pytest.raises(ValueError, match="broken training data"):
        processor.push(slotted("r1", "i1", "a"))
    assert not processor.record2sample.releasable_record_ids() & {"r1", "i1"}
    processor.pair_tasks.fail_assembly = False
    processor.push(slotted("r1", "i1", "a"))
    processor.push(slotted("r2", "i2", "b"))
    assert task_names(processor.build_job_data(1)) == ["r1", "r2"]


@pytest.mark.unit
def test_a_skipped_report_is_released_and_never_batched() -> None:
    processor = PairPipeline()
    processor.push(inference("i1"))
    processor.push(slotted("skip-1", "i1", "a"))
    assert {"skip-1", "i1"} <= processor.record2sample.releasable_record_ids()
    assert not processor.ready(1)


@pytest.mark.unit
def test_a_batch_holds_the_tasks_of_every_report_in_a_group_in_arrival_order() -> None:
    processor = PairPipeline()
    for record_id in ("i1", "i2"):
        processor.push(inference(record_id))
    processor.push(slotted("r2", "i2", "b"))
    processor.push(slotted("r1", "i1", "a"))
    batch = processor.build_job_data(1)
    assert task_names(batch) == ["r2", "r1"]
    # The ready group's records are consumed when its samples are stored; the job data consumes the samples.
    assert {"i1", "i2", "r1", "r2"} <= processor.record2sample.releasable_record_ids()
    assert processor.acknowledge(batch.data_id) == {
        consumed_sample_id(stage_item_id("s", "r2:0")),
        consumed_sample_id(stage_item_id("s", "r1:0")),
    }
    assert not processor.record2sample.set_samples


@pytest.mark.unit
def test_the_report_stage_releases_a_report_whose_data_is_behind_the_serving_version() -> None:
    processor = threshold_pipeline(PipelineContext("s"))
    processor.observe_serving_artifact_id("e:1")
    processor.push(inference("i1", version="e:1"))
    processor.push(report("r1", "i1"))
    processor.observe_serving_artifact_id("e:2")
    processor.push(inference("i2", version="e:2"))
    processor.push(report("r2", "i2"))
    batch = processor.build_job_data(1)
    assert [item.source_agent_record_ids for item in batch.samples] == [("i2", "r2")]
    # The stale report ended with the inference that it owns, and its tasks are gone.
    assert {"r1", "i1"} <= processor.record2sample.releasable_record_ids()
    assert "r1" not in processor.record2sample.set_samples


@pytest.mark.unit
def test_the_default_pipeline_makes_no_sample_and_no_batch() -> None:
    processor = TrainingPipeline(PipelineContext("s"))
    assert isinstance(processor.record2sample, NoSamples)
    processor.push(inference("i1"))
    processor.push(report("r1", "i1"))
    assert processor.record2sample.take_ready_units() == ()
    assert not processor.ready(1) and not processor.record2sample.releasable_record_ids()


@pytest.mark.unit
def test_a_waiting_set_holds_its_records_and_settles_when_its_result_comes_back() -> None:
    processor = WaitingPipeline()
    processor.push(inference("i1"))
    processor.push(inference("i2"))
    assert not processor.ready(1) and processor.background_jobs_running()
    processor.later_tasks.finish("i2", SampleResult.skipped())
    processor.produce()
    assert not processor.ready(1) and processor.record2sample.releasable_record_ids() == {"i2"}
    processor.later_tasks.finish("i1", SampleResult.ready((named_task("t1"),)))
    processor.produce()
    assert processor.ready(1), "the result alone makes a batch, without another record"
    assert not processor.background_jobs_running()
    assert task_names(processor.build_job_data(1)) == ["t1"]


@pytest.mark.unit
def test_the_trainer_reserves_a_batch_when_a_waiting_set_finishes() -> None:
    records = memory_records("s")
    records.append(inference("i1"))
    trainer = Trainer.build(
        "s",
        records,
        pipeline_factory=partial(WaitingPipeline, job_samples=1),
        job_loop=counting_jobs(),
        root=STUB_ROOT,
    )
    make_samples(trainer)
    assert trainer.advance_job() is None
    assert trainer.holds_untrained_records(), "the trainer keeps polling while the set waits"
    processor = trainer.pipeline
    assert isinstance(processor, WaitingPipeline)
    processor.later_tasks.finish("i1", SampleResult.ready((named_task("t1"),)))
    make_samples(trainer)
    assert trainer.advance_job() is not None
    batch = trainer.pending_job_data
    assert batch is not None and task_names(batch) == ["t1"]


class InferenceSamples(InferenceRecordSamples):
    """Each inference set becomes one task named after its record."""

    def make_samples(self, record_set: RecordSet) -> SampleResult:
        record_ids = record_set.record_ids()
        return SampleResult.ready((named_task(record_ids[0], source_agent_record_ids=record_ids),))


class UploadJobData(PairJobData):
    accepts_trajectories = True


TRAJECTORY = TrajectorySample({"schema_version": "ATIF-v1.4", "agent": {"name": "fixture"}, "steps": [{"step_id": 1}]})


def upload(samples: SampleStore, *trajectory_ids: str) -> None:
    """Upload ``TRAJECTORY`` under each of ``trajectory_ids`` for the training of the scenario."""
    metadata = {"kind": "trajectory", "scenarios": {samples.scenario: "train"}}
    samples.data.create(
        samples.owner, [NewItem(item_id, TRAJECTORY.trajectory, metadata) for item_id in trajectory_ids]
    )


@pytest.mark.unit
def test_uploaded_trajectories_train_in_store_order_and_count_as_one_sample_each() -> None:
    context = PipelineContext("s")
    processor = TrainingPipeline(
        context,
        record2sample=InferenceSamples(),
        sample2job=UploadJobData(context, None, None),
    )
    upload(processor.store.store, "t1", "t2")
    processor.push(inference("i1"))
    assert processor.ready(2)
    # The task sample of i1 is another kind of sample, so it waits for a job of its own.
    assert processor.status()["ready_samples"] == 2

    batch = processor.build_job_data(2)
    assert batch.samples == (TRAJECTORY, TRAJECTORY)
    processor.release_job_data(batch.data_id)
    batch = processor.build_job_data(2)
    assert batch.samples == (TRAJECTORY, TRAJECTORY)
    assert processor.acknowledge(batch.data_id) == {"sample:t1", "sample:t2"}

    # A trained trajectory that is uploaded again does not queue again; job data takes the samples in store order.
    upload(processor.store.store, "t1", "t3")
    batch = processor.build_job_data(1)
    assert task_names(batch) == ["i1"] and batch.samples[0].source_agent_record_ids == ("i1",)
    assert processor.acknowledge(batch.data_id) == {"sample:s.i1:0"}
    batch = processor.build_job_data(1)
    assert batch.samples == (TRAJECTORY,)
    assert processor.acknowledge(batch.data_id) == {"sample:t3"}
    assert "i1" in processor.record2sample.releasable_record_ids()
    assert not processor.ready(1)


def upload_pipeline(context: PipelineContext) -> TrainingPipeline:
    return TrainingPipeline(context, sample2job=UploadJobData(context, None, 1))


@pytest.mark.unit
def test_each_trainer_skips_only_the_samples_that_its_own_component_trained() -> None:
    storage = data_storage()
    session = storage.open("s")
    samples = sample_store(storage, "s", session)
    upload(samples, "t1", "t2")
    session.commit_step(
        expected_step=0,
        commit=CommitRecord(
            scenario="s",
            step=1,
            snapshot=SnapshotRef("release:1", "base"),
            checkpoint=True,
            algorithm_state={},
            high_water_sequence=0,
            high_water_offset=0,
            consumed_ids=frozenset({consumed_sample_id("t1")}),
            component="a",
        ),
    )
    consumed = {}
    for component in ("a", "b"):
        trainer = Trainer.build(
            "s",
            session.records,
            pipeline_factory=upload_pipeline,
            job_loop=counting_jobs(),
            samples=samples.for_trainer(component),
            root=STUB_ROOT,
        )
        consumed[component] = trainer.prepare_commit(trainer.advance_job()).consumed_ids
    # Component a trained t1, so its next job takes t2; component b has not trained t1 yet.
    assert consumed == {"a": {consumed_sample_id("t2")}, "b": {consumed_sample_id("t1")}}
    session.close()
    storage.close()


@pytest.mark.unit
def test_a_pipeline_does_not_train_a_trajectory_that_its_task_to_job_stage_does_not_accept() -> None:
    context = PipelineContext("s")
    processor = TrainingPipeline(context, sample2job=PairJobData(context, None, None))
    upload(processor.store.store, "t1")
    assert not processor.ready(1)
    assert processor.status()["ready_samples"] == 0


def ready_tasks(*names: str) -> SampleResult:
    return SampleResult.ready(tuple(named_task(name) for name in names))


@pytest.mark.unit
def test_producing_collects_later_results_without_reserving_or_replacing_a_batch() -> None:
    processor = WaitingPipeline(PipelineContext("s"))
    processor.push(inference("i1"))
    processor.later_tasks.finish("i1", ready_tasks("a", "b"))
    processor.produce()
    processor.advance()
    assert processor.status()["ready_samples"] == 2
    assert not processor.ready(3)
    batch = processor.build_job_data(2)

    processor.push(inference("i2"))
    processor.later_tasks.finish("i2", ready_tasks("c"))
    processor.produce()
    processor.advance()
    assert processor.build_job_data(2) is batch
    assert processor.acknowledge(batch.data_id) == {"sample:s.i1:0", "sample:s.i1:1"}
    assert processor.status()["ready_samples"] == 1
    assert task_names(processor.build_job_data(1)) == ["c"]


@pytest.mark.unit
def test_samples_beyond_the_buffer_size_wait_and_stale_samples_leave_the_buffer() -> None:
    context = PipelineContext("s")
    later_tasks = LaterSamples()
    processor = TrainingPipeline(
        context,
        record2sample=later_tasks,
        sample2job=UploadJobData(context, 1, None),
    )
    processor.push(inference("i1"))
    processor.push(inference("i2"))
    later_tasks.finish("i1", SampleResult.ready((TRAJECTORY,)))
    later_tasks.finish("i2", SampleResult.ready((TRAJECTORY,)))
    processor.produce()
    processor.advance()
    # Both units are stored and their records consumed; the buffer takes one sample, and the other waits.
    assert processor.status()["ready_samples"] == 1
    assert len(processor.store.page()) == 2
    assert processor.record2sample.releasable_record_ids() == {"i1", "i2"}
    # Trajectories without a producing artifact become stale once the served artifact is known.
    processor.observe_serving_artifact_id("e:1")
    processor.advance()
    assert processor.status()["ready_samples"] == 0
    assert len(processor.store.page()) == 2, "a stale sample stays in the store untrained"
    assert not processor.ready(1)


@pytest.mark.unit
def test_the_records_of_an_artifact_served_beside_train_while_it_serves() -> None:
    context = PipelineContext("s")
    later_tasks = LaterSamples()
    processor = TrainingPipeline(context, record2sample=later_tasks, sample2job=UploadJobData(context, 1, None))
    processor.push(inference("i1"))
    later_tasks.finish(
        "i1", SampleResult.ready((policy_trajectory("i1", [1], [1], [0.0], 1.0, artifact_id="beside"),))
    )
    processor.produce()
    processor.observe_serving_artifact_id("active")
    assert auto_ids(processor) == (), "an artifact outside the served history is stale"
    processor.served_artifacts.restore_beside(("beside",))
    assert len(auto_ids(processor)) == 1
    processor.served_artifacts.restore_beside(())
    assert auto_ids(processor) == ()


@pytest.mark.unit
def test_the_pipeline_is_ready_when_its_held_units_hold_the_asked_sample_count() -> None:
    processor = WaitingPipeline(PipelineContext("s"))
    for record_id in ("i1", "i2", "i3"):
        processor.push(inference(record_id))
    processor.later_tasks.finish("i1", ready_tasks("a", "b"))
    processor.later_tasks.finish("i2", SampleResult.skipped())
    processor.produce()
    assert not processor.ready(3), "one unit of two tasks and one skipped set hold two samples"
    assert processor.status()["ready_samples"] == 2
    processor.later_tasks.finish("i3", ready_tasks("c"))
    processor.produce()
    assert processor.ready(3)
    assert processor.status()["ready_samples"] == 3


@pytest.mark.unit
def test_a_batch_takes_the_buffer_size_in_store_order_and_the_rest_of_each_unit_it_touches() -> None:
    context = PipelineContext("s")
    later_tasks = LaterSamples()
    processor = TrainingPipeline(
        context,
        record2sample=later_tasks,
        sample2job=PairJobData(context, 2, None),
    )
    for record_id in ("i1", "i2", "i3"):
        processor.push(inference(record_id))
    later_tasks.finish("i3", ready_tasks("c"))
    later_tasks.finish("i2", ready_tasks("b1", "b2"))
    later_tasks.finish("i1", ready_tasks("a1", "a2"))
    processor.produce()
    batch = processor.build_job_data(2)
    # Units go to the store in the order they became ready. The second unit passes the buffer size; the batch
    # keeps it whole and leaves the third unit held.
    assert task_names(batch) == ["c", "b1", "b2"]
    assert processor.acknowledge(batch.data_id) == {"sample:s.i3:0", "sample:s.i2:0", "sample:s.i2:1"}
    assert processor.status()["ready_samples"] == 2
    assert task_names(processor.build_job_data(2)) == ["a1", "a2"]


@pytest.mark.unit
def test_samples_beyond_the_buffer_size_wait_for_the_next_job_data() -> None:
    context = PipelineContext("s")
    processor = TrainingPipeline(
        context,
        record2sample=InferenceSamples(),
        sample2job=UploadJobData(context, 2, None),
    )
    processor.push(inference("i1"))
    reserved = processor.build_job_data(1)
    for record_id in ("i2", "i3", "i4"):
        processor.push(inference(record_id))
    upload(processor.store.store, "t1", "t2", "t3")
    processor.push(inference("i5"))
    processor.advance()
    # The reserved job data stays, nothing is dropped, and the buffer holds buffer_size samples of one kind.
    assert processor.build_job_data(1) is reserved
    assert {"i1", "i2", "i3", "i4", "i5"} <= processor.record2sample.releasable_record_ids()
    assert processor.status()["ready_samples"] == 2
    assert processor.acknowledge(reserved.data_id) == {"sample:s.i1:0"}
    # Each later job data takes the next buffer_size samples in store order; the uploads wait for their own kind.
    jobs = []
    while processor.ready(1):
        batch = processor.build_job_data(1)
        jobs.append(sorted(processor.acknowledge(batch.data_id)))
    assert jobs == [
        ["sample:s.i2:0", "sample:s.i3:0"],
        ["sample:s.i4:0", "sample:s.i5:0"],
        ["sample:t1", "sample:t2"],
        ["sample:t3"],
    ]


@pytest.mark.unit
def test_the_pipeline_keeps_the_newest_unit_even_when_it_alone_is_over_the_buffer_size() -> None:
    context = PipelineContext("s")
    later_tasks = LaterSamples()
    processor = TrainingPipeline(
        context,
        record2sample=later_tasks,
        sample2job=UploadJobData(context, 2, None),
    )
    processor.push(inference("i1"))
    later_tasks.finish("i1", ready_tasks("a", "b", "c"))
    processor.produce()
    assert processor.ready(1)
    assert task_names(processor.build_job_data(1)) == ["a", "b", "c"]


@pytest.mark.unit
def test_the_stage_sizes_are_its_own() -> None:
    sized = PairJobData(PipelineContext("s"), None, 3)
    assert (sized.job_samples, sized.buffer_size) == (3, 3), "the buffer holds the data of one job"
    configured = PairJobData(PipelineContext("s"), 5, 2)
    assert (configured.job_samples, configured.buffer_size) == (2, 5)
    unsized = PairJobData(PipelineContext("s"), None, None)
    assert (unsized.job_samples, unsized.buffer_size) == (None, None), "no job size holds every sample"
    with pytest.raises(ValueError, match="job_samples must be None or a positive int"):
        PairJobData(PipelineContext("s"), None, 0)
    with pytest.raises(ValueError, match="job_samples must be None or a positive int"):
        Sample2JobSettings(job_samples=0)


class MadeSamples(RecordingSamples):
    """A sample stage that also makes samples from no record set (``made``) and keeps each job end that it reads."""

    def __init__(self) -> None:
        super().__init__()
        self.made: tuple[tuple[str, Sample], ...] = ()
        self.ended: list[JobOutcome] = []

    def made_samples(self) -> tuple[tuple[str, Sample], ...]:
        made, self.made = self.made, ()
        return made

    def job_ended(self, outcome: JobOutcome) -> None:
        self.ended.append(outcome)


def auto_ids(pipeline: TrainingPipeline) -> tuple[str, ...]:
    """The IDs of the samples that the pipeline's next automatic job takes."""
    return tuple(stored.sample_id for stored, _ in pipeline.auto_entries())


def training_request(*samples: str) -> TrainingRequest:
    """A queued request that names ``samples``."""
    payload = {"text": "train", "session": "s-1", "artifact_id": "r-1", "parameters": {"rollout": {}}}
    return TrainingRequest.from_dict({**payload, **({"samples": list(samples)} if samples else {})}, request_id="t")


@pytest.mark.unit
def test_samples_that_a_stage_made_from_no_record_set_are_stored_and_train() -> None:
    context = PipelineContext("s")
    samples = MadeSamples()
    processor = TrainingPipeline(context, record2sample=samples, sample2job=RecordingJobData(context))
    assert not processor.ready(1)
    samples.made = (("made/a", named_task("a")),)
    processor.produce()
    assert processor.ready(1)
    samples.made = (("made/a", named_task("a")),)
    processor.produce()
    assert auto_ids(processor) == ("s.made.a",), "a sample made again is stored once"
    (stored,) = processor.store.page()
    assert (stored.sample_id, stored.component, stored.group) == ("s.made.a", "", None)
    assert stored.sample.source_agent_record_ids == ()
    built = processor.build_job_data(1)
    assert task_names(built) == ["a"]
    assert processor.acknowledge(built.data_id) == {consumed_sample_id("s.made.a")}
    assert not processor.ready(1)
    outcome = JobOutcome("job", "stub", "finished")
    processor.job_ended(outcome)
    assert samples.ended == [outcome], "the job's end goes to the sample stage"


@pytest.mark.unit
@pytest.mark.parametrize("legacy_stripped", [False, True], ids=["captured", "legacy-stripped"])
@pytest.mark.parametrize("replay_before_commit", [False, True], ids=["live", "replay"])
@pytest.mark.parametrize("references", [("aborted",), ("aborted", "other")], ids=["single", "mixed"])
def test_aborted_report_references_are_released_and_stay_excluded_after_replay(
    replay_before_commit: bool, references: tuple[str, ...], legacy_stripped: bool
) -> None:
    records = memory_records("s")
    aborted = inference("aborted")
    aborted = replace(
        aborted,
        payload={
            **aborted.payload,
            "response": {"training": {"finish_reason": "abort", "tokens": [10, 20], "loss_mask": [1]}},
        },
    )
    if legacy_stripped:
        aborted = replace(
            aborted,
            payload={
                "response": {
                    "stop_reason": "end_turn",
                    "stream_delivery": {"complete": False, "error": "upstream engine aborted the generation"},
                }
            },
        )
    rejected_report = report("aborted-report", *references)
    for record in (aborted, inference("other"), rejected_report, inference("valid"), report("valid-report", "valid")):
        records.append(record)
    trainer = Trainer.build(
        "s",
        records,
        pipeline_factory=partial(threshold_pipeline, job_samples=1),
        job_loop=counting_jobs(),
        root=STUB_ROOT,
    )
    if replay_before_commit:
        trainer.restore_consumed_records(frozenset())
    make_samples(trainer)
    result = trainer.advance_job()
    assert result is not None
    batch = trainer.pending_job_data
    assert batch is not None
    assert len(batch.samples) == 1
    assert batch.samples[0].source_agent_record_ids == ("valid", "valid-report")
    prepared = trainer.prepare_commit(result)
    assert prepared.consumed_ids == {
        "valid",
        "valid-report",
        "aborted-report",
        *references,
        consumed_sample_id(stage_item_id("s", "valid-report:0")),
    }
    trainer.settle_commit(prepared)
    make_samples(trainer)
    assert not trainer.pipeline.record2sample.releasable_record_ids()

    restored = Trainer.build(
        "s",
        records,
        pipeline_factory=partial(threshold_pipeline, job_samples=1),
        job_loop=counting_jobs(),
        root=STUB_ROOT,
    )
    restored.restore_consumed_records(prepared.consumed_ids)
    records.append(report("later-report", "aborted"))
    make_samples(restored)
    assert restored.advance_job() is None
    assert "later-report" in restored.skipped_record_ids
    # Consumption changes training eligibility, not the stored diagnostic payload.
    assert records.get("s", "aborted") == aborted


@pytest.mark.unit
def test_a_request_names_stored_samples_of_one_kind_that_the_trainer_can_train() -> None:
    context = PipelineContext("s", training_mode="manual")
    processor = TrainingPipeline(context, sample2job=UploadJobData(context, None, None))
    upload(processor.store.store, "upload")
    processor.store.add_record_samples(None, ("r1",), (("r1:0", named_task("task")),))
    processor.store.store.for_trainer("other").add_record_samples(None, ("r2",), (("other:r2:0", TRAJECTORY),))
    processor.check_request(training_request("upload"))
    with pytest.raises(ValueError, match="one kind"):
        processor.check_request(training_request("upload", "s.r1:0"))
    with pytest.raises(ValueError, match="cannot train the samples"):
        processor.check_request(training_request("s.other:r2:0"))
    with pytest.raises(ValueError, match="holds no samples"):
        processor.check_request(training_request("missing"))
    with pytest.raises(ValueError, match="names no samples"):
        processor.check_request(training_request())
    # A sample that left the scenario after the request was filed is left out of its job.
    assert [stored.sample_id for stored, _ in processor.request_entries(training_request("upload", "gone"))] == [
        "upload"
    ]


@pytest.mark.unit
def test_a_request_reads_its_samples_in_batches_and_a_poll_does_not_read_them_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = PipelineContext("s", training_mode="manual")
    processor = TrainingPipeline(context, sample2job=UploadJobData(context, None, None))
    names = [f"upload-{index}" for index in range(MAX_QUERY_LIMIT + 1)]
    upload(processor.store.store, *names)
    data = processor.store.store.data
    read_many, reads = data.read_many, []

    def counted(owner: str, item_ids: Any, **options: Any) -> Any:
        reads.append(len(item_ids))
        return read_many(owner, item_ids, **options)

    monkeypatch.setattr(data, "read_many", counted)
    request = training_request(*names)
    processor.check_request(request)
    assert reads == [MAX_QUERY_LIMIT, 1]
    processor.queue_requests([request])
    assert processor.holds_job_data(None) and processor.holds_job_data(None)
    assert reads == [MAX_QUERY_LIMIT, 1] * 2, "the pipeline decides once whether it takes a queued request"


class TaskJobData(RecordingJobData):
    """Recording job data that also trains the scenario's imported train tasks."""

    accepts_imported_tasks = True


@pytest.mark.unit
def test_imported_train_tasks_are_buffer_samples_of_a_stage_that_trains_them(tmp_path: Path) -> None:
    _session, samples = open_sample_store(tmp_path / "records", "s")
    tasks = {"task-a": ("record-1", 9000, "train"), "held": ("record-2", 9001, "eval")}
    samples.data.create(
        samples.owner,
        [
            NewItem(
                name,
                write_harbor_task(task_for(name, record, port), tmp_path),
                {"kind": "task", "scenarios": {"s": split}},
            )
            for name, (record, port, split) in tasks.items()
        ],
    )

    def pipeline(stage: type[RecordingJobData], **context: Any) -> TrainingPipeline:
        built = PipelineContext("s", samples=samples.for_trainer(""), **context)
        return TrainingPipeline(built, record2sample=RecordingSamples(), sample2job=stage(built))

    trains = pipeline(TaskJobData)
    assert trains.ready(1) and auto_ids(trains) == ("task-a",), "an eval task never trains"
    assert not pipeline(RecordingJobData).ready(1), "a stage that trains no imported task skips them"
    # A kind that the recipe gives no parameters trains only for a request that gives them.
    manual = pipeline(TaskJobData, job_parameters={"trajectory": {}}, training_mode="manual")
    manual.set_training_mode("auto")
    assert auto_ids(manual) == ()
    manual.set_training_mode("manual")
    manual.queue_requests([training_request("task-a")])
    data = manual.build_job_data(1)
    assert manual.sample2job.job_parameters(data) == {"rollout": {}}
    with pytest.raises(ValueError, match="cannot train the samples"):
        manual.check_request(training_request("held"))
