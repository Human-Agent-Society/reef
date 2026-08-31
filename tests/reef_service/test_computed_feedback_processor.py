"""Focused unit tests for the computed-feedback processor's state machine.

test_openclawrl pins the real recipe's behavior; these tests pin the
engine mechanics themselves with a toy recipe, where recipe policy cannot
blur the state machine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

from reef.core import AgentRecord, RequestType
from reef.train.processors.computed import ComputedFeedbackProcessor, Failed, JudgingWorker
from reef.train.types import PolicyBatch, PolicySample, ProcessorContext

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _Job:
    receipt: str


@dataclass(frozen=True)
class _Judgment:
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


class _ToyProcessor(ComputedFeedbackProcessor):
    """A record tracks itself; a record whose payload names ``completes``
    finishes that receipt. Samples are one-token stubs."""

    def ingest(self, item: AgentRecord) -> None:
        self.catch_up(time.monotonic())
        completes = item.payload.get("completes")
        if completes:
            self.dispatch(_Job(completes))
        if item.payload.get("track"):
            self.track(item)
        else:
            self.retire(item.agent_record_id)

    async def judge(self, job: _Job) -> _Judgment:
        return _Judgment(job.receipt)

    def make_sample(self, record: AgentRecord, judgment: _Judgment) -> PolicySample | None:
        if not judgment.good:
            return None
        return PolicySample(
            source_agent_record_id=record.agent_record_id,
            tokens=(1, 2),
            loss_mask=(1,),
            rollout_log_probs=(-0.1,),
            reward=1.0,
            weight_version=record.payload.get("version", "v1"),
        )

    def make_batch(self, samples: tuple[PolicySample, ...], batch_number: int) -> PolicyBatch:
        return PolicyBatch(f"{self.scenario}:toy:{batch_number}", samples)


def _record(agent_record_id: str, **payload) -> AgentRecord:
    return AgentRecord.create(
        scenario="s", request_type=RequestType.INFERENCE, payload=payload, agent_record_id=agent_record_id
    )


def _engine(batch_size: int = 1, worker=None) -> tuple[_ToyProcessor, _FakeWorker]:
    worker = worker if worker is not None else _FakeWorker()
    processor = _ToyProcessor(ProcessorContext("s", {"batch_size": batch_size}), worker=worker)
    return processor, worker


def test_track_complete_judge_batch_acknowledge() -> None:
    processor, worker = _engine()
    processor.ingest(_record("r1", track=True))
    assert processor.derivation_pending()
    processor.ingest(_record("r2", completes="r1", track=True))
    assert [job.receipt for job in worker.jobs] == ["r1"]

    worker.push(_Judgment("r1"))
    assert processor.ready()
    batch = processor.build_batch()
    assert batch.samples[0].source_agent_record_id == "r1"
    processor.acknowledge(batch.batch_id)
    decision = processor.retention_decision()
    assert "r1" in decision.releasable_agent_record_ids
    assert "r2" in decision.protected_agent_record_ids  # still tracked


def test_untracked_records_are_terminal_on_sight() -> None:
    processor, worker = _engine()
    processor.ingest(_record("noise"))
    assert "noise" in processor.retention_decision().releasable_agent_record_ids
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
    releasable = processor.retention_decision().releasable_agent_record_ids
    assert {"a", "b"} <= set(releasable)


def test_refused_submission_retires_immediately() -> None:
    class _Refusing(_FakeWorker):
        def submit(self, job) -> bool:
            return False

    processor, _ = _engine(worker=_Refusing())
    processor.ingest(_record("r1", track=True))
    processor.ingest(_record("r2", completes="r1"))
    assert "r1" in processor.retention_decision().releasable_agent_record_ids


def test_stale_weight_versions_drop_by_record_arrival() -> None:
    processor, worker = _engine(batch_size=2)
    processor.ingest(_record("old", track=True, version="v1"))
    processor.ingest(_record("done1", completes="old"))
    processor.ingest(_record("new", track=True, version="v2"))
    processor.ingest(_record("done2", completes="new"))
    # Judgments land in the OPPOSITE order of generation.
    worker.push(_Judgment("new"))
    worker.push(_Judgment("old"))
    assert not processor.ready()  # the stale candidate was dropped
    assert "old" in processor.retention_decision().releasable_agent_record_ids
    assert "new" in processor.retention_decision().protected_agent_record_ids


def test_correlate_only_mode_without_a_worker() -> None:
    processor = _ToyProcessor(ProcessorContext("s", {"batch_size": 1}), worker=None)
    processor.ingest(_record("r1", track=True))
    processor.ingest(_record("r2", completes="r1"))
    assert "r1" in processor.retention_decision().releasable_agent_record_ids
    processor.close()


def test_close_propagates_to_the_worker() -> None:
    processor, worker = _engine()
    processor.close()
    assert worker.closed


def test_real_worker_runs_the_judge_coroutine() -> None:
    async def judge(job):
        return _Judgment(job.receipt)

    worker = JudgingWorker(judge)
    assert worker.submit(_Job("r1"))
    import time

    deadline = time.monotonic() + 10.0
    judgments: list = []
    while not judgments and time.monotonic() < deadline:
        judgments = worker.poll()
        time.sleep(0.01)
    assert [judgment.receipt for judgment in judgments] == ["r1"]
    worker.close()


def test_real_worker_resolves_a_crashing_judge_as_failed() -> None:
    async def judge(job):
        raise RuntimeError("boom")

    worker = JudgingWorker(judge)
    assert worker.submit(_Job("r1"))
    import time

    deadline = time.monotonic() + 10.0
    judgments: list = []
    while not judgments and time.monotonic() < deadline:
        judgments = worker.poll()
        time.sleep(0.01)
    assert judgments == [Failed("r1")]
    worker.close()
