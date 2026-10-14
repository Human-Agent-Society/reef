"""Focused unit tests for the computed-feedback pipeline's state machine.

tests.recipes.openclawrl.test_openclawrl_contracts pins the real recipe's behavior; these tests pin the
mechanics of a pipeline with a ``JudgingRecord2Record`` and ``JudgedTasks`` with a
toy recipe, where recipe policy cannot blur the state machine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pytest
from tests.support._trajectories import policy_trajectory

from reef.background_jobs import BackgroundJobRunner, BackgroundJobWorker, Failed, SupportsReceipt
from reef.core import AgentRecord, RequestType
from reef.core.trajectories import source_record_id
from reef.pipeline_context import PipelineContext
from reef.record2record import JudgingRecord2Record, RecordJudging
from reef.record2task import JudgedTasks
from reef.task2batch import Task2Batch
from reef.train.pipeline import TrainingPipeline
from reef.train.types import TrainDataItem, TrainingBatch, TrajectoryItem

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _Job(SupportsReceipt):
    receipt: str


@dataclass(frozen=True)
class _Judgment(SupportsReceipt):
    receipt: str
    good: bool = True


class _FakeWorker:
    def __init__(self) -> None:
        self.jobs: list[_Job] = []
        self.closed = False
        self._judgments: list[object] = []

    def submit(self, job) -> bool:
        self.jobs.append(job)
        return True

    def poll(self):
        judgments, self._judgments = self._judgments, []
        return judgments

    def close(self) -> None:
        self.closed = True

    def push(self, judgment) -> None:
        self._judgments.append(judgment)


class _ToyJudging(RecordJudging):
    """A record tracks itself; a record whose payload names ``completes`` finishes that receipt."""

    def observe(self, item: AgentRecord, now: float, stage: JudgingRecord2Record) -> None:
        completes = item.payload.get("completes")
        if completes:
            stage.dispatch(_Job(completes))
        if item.payload.get("track"):
            stage.track(item)
        else:
            stage.retire(item.agent_record_id)

    async def judge(self, job: SupportsReceipt) -> SupportsReceipt:
        return _Judgment(job.receipt)


class _ToyTasks(JudgedTasks):
    """One-token stub samples; a judgment that is not good cannot train."""

    def make_sample(self, record: AgentRecord, judgment: _Judgment) -> TrajectoryItem | None:
        if not judgment.good:
            return None
        return policy_trajectory(
            source_agent_record_id=record.agent_record_id,
            tokens=(1, 2),
            loss_mask=(1,),
            rollout_log_probs=(-0.1,),
            reward=1.0,
            runtime_load_id=record.payload.get("version", "v1"),
        )


class ToyBatches(Task2Batch):
    """Batches the judged samples in the order that the stage selected them."""

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:toy:{batch_number}", tasks)


class _ToyProcessor(TrainingPipeline):
    """Trains the judged records of ``_ToyJudging``. Samples are one-token stubs."""

    def __init__(self, context: PipelineContext, *, worker: Any, batch_size: int = 1) -> None:
        super().__init__(
            context,
            record2record=JudgingRecord2Record(context, _ToyJudging(), worker),
            record2task=_ToyTasks(),
            task2batch=ToyBatches(context, batch_size),
        )


def _record(agent_record_id: str, **payload) -> AgentRecord:
    return AgentRecord.create(
        scenario="s", request_type=RequestType.INFERENCE, payload=payload, agent_record_id=agent_record_id
    )


def _engine(batch_size: int = 1, worker=None) -> tuple[_ToyProcessor, _FakeWorker]:
    worker = worker if worker is not None else _FakeWorker()
    processor = _ToyProcessor(PipelineContext("s"), worker=worker, batch_size=batch_size)
    return processor, worker


def test_track_complete_judge_batch_acknowledge() -> None:
    processor, worker = _engine()
    processor.ingest(_record("r1", track=True))
    assert not processor.background_jobs_running()
    processor.ingest(_record("r2", completes="r1", track=True))
    assert [job.receipt for job in worker.jobs] == ["r1"]
    assert processor.background_jobs_running()

    worker.push(_Judgment("r1"))
    assert processor.ready()
    batch = processor.build_batch()
    assert source_record_id(batch.items[0]) == "r1"
    processor.acknowledge(batch.batch_id)
    decision = processor.releasable_record_ids()
    assert "r1" in decision
    assert "r2" not in decision  # still tracked


def test_untracked_records_are_terminal_on_sight() -> None:
    processor, worker = _engine()
    processor.ingest(_record("noise"))
    assert "noise" in processor.releasable_record_ids()
    assert not worker.jobs


def test_failed_and_declined_judgments_retire_the_record() -> None:
    processor, worker = _engine()
    processor.ingest(_record("a", track=True))
    processor.ingest(_record("b", track=True))
    processor.ingest(_record("done", completes="a"))
    processor.ingest(_record("done2", completes="b"))
    worker.push(Failed("a"))
    worker.push(_Judgment("b", good=False))
    assert not processor.ready()
    releasable = processor.releasable_record_ids()
    assert {"a", "b"} <= set(releasable)


def test_refused_submission_retires_immediately() -> None:
    class _Refusing(_FakeWorker):
        def submit(self, job) -> bool:
            return False

    processor, _ = _engine(worker=_Refusing())
    processor.ingest(_record("r1", track=True))
    processor.ingest(_record("r2", completes="r1"))
    assert "r1" in processor.releasable_record_ids()


def test_stale_runtime_load_ids_drop_by_record_arrival() -> None:
    processor, worker = _engine(batch_size=2)
    processor.ingest(_record("old", track=True, version="v1"))
    processor.ingest(_record("done1", completes="old"))
    processor.ingest(_record("new", track=True, version="v2"))
    processor.ingest(_record("done2", completes="new"))
    # Judgments land in the OPPOSITE order of generation.
    worker.push(_Judgment("new"))
    worker.push(_Judgment("old"))
    assert not processor.ready()  # the stale candidate was dropped
    assert "old" in processor.releasable_record_ids()
    assert "new" not in processor.releasable_record_ids()


def test_candidates_within_max_staleness_of_the_newest_record_stay() -> None:
    worker = _FakeWorker()
    processor = _ToyProcessor(PipelineContext("s", max_staleness=1), worker=worker, batch_size=3)
    for name, version in (("v3", "e:3"), ("v4", "e:4"), ("v5", "e:5")):
        processor.ingest(_record(name, track=True, version=version))
        processor.ingest(_record(f"done-{name}", completes=name))
        worker.push(_Judgment(name))
    assert not processor.ready()  # e:3 is two versions behind e:5
    assert "v3" in processor.releasable_record_ids()
    assert {"v4", "v5"}.isdisjoint(processor.releasable_record_ids())


def test_correlate_only_mode_without_a_worker() -> None:
    processor = _ToyProcessor(PipelineContext("s"), worker=None)
    processor.ingest(_record("r1", track=True))
    processor.ingest(_record("r2", completes="r1"))
    assert "r1" in processor.releasable_record_ids()
    processor.close()


def test_close_propagates_to_the_worker() -> None:
    processor, worker = _engine()
    processor.close()
    assert worker.closed


class _Answers(BackgroundJobRunner):
    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        return _Judgment(job.receipt)


class _Crashes(BackgroundJobRunner):
    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        raise RuntimeError("boom")


def test_real_worker_runs_the_judge_coroutine() -> None:
    worker = BackgroundJobWorker(_Answers(), thread_name="test-jobs")
    assert worker.submit(_Job("r1"))

    deadline = time.monotonic() + 10.0
    judgments: list = []
    while not judgments and time.monotonic() < deadline:
        judgments = worker.poll()
        time.sleep(0.01)
    assert [judgment.receipt for judgment in judgments] == ["r1"]
    worker.close()


def test_real_worker_resolves_a_crashing_judge_as_failed() -> None:
    worker = BackgroundJobWorker(_Crashes(), thread_name="test-jobs")
    assert worker.submit(_Job("r1"))

    deadline = time.monotonic() + 10.0
    judgments: list = []
    while not judgments and time.monotonic() < deadline:
        judgments = worker.poll()
        time.sleep(0.01)
    assert judgments == [Failed("r1")]
    worker.close()
