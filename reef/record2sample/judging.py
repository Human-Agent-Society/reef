"""Judged records as samples: records that a model judges asynchronously, after later traffic completes them.

A computed-feedback method makes its training signal from the traffic itself.
A record only becomes judgeable when some later record completes it, and the
judgment calls models, so it runs on a background worker. ``JudgedSamples``
owns the record lifecycle around that: tracked (awaiting later traffic), in
flight (judging), then a judged set that becomes one sample, or terminal. A
recipe writes the correlation and the judge as a ``RecordJudging``, and the
sample of one judged record as ``JudgedSamples.make_sample``.

Everything here is reconstructible state: after a crash the trainer replays the
unacknowledged tail and judging runs again, with a cost bounded by the
training backlog. A record whose judgment was in flight is lost, which is one
record's exposure.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

from reef.core.job_data import TrajectorySample
from reef.core.records_types import AgentRecord
from reef.execution.jobs import BackgroundJobRunner, BackgroundJobWorker, Failed, SupportsReceipt
from reef.record2sample.base import Record2Sample, SampleResult
from reef.record2sample.sets import RecordSet


class RecordJudging(BackgroundJobRunner, ABC):
    """A computed-feedback method's correlation and judge.

    ``observe`` and ``expire`` run on the trainer thread and must not block.
    ``judge`` runs on the worker thread, so it may call models and take
    minutes. State that it shares with the trainer thread must be safe to use from both threads.
    """

    @abstractmethod
    def observe(self, record: AgentRecord, now: float, stage: JudgedSamples) -> None:
        """Correlate one record, with the stage's verbs: ``dispatch``, ``track``, ``retire``, and ``tracked_record``.

        The stage has already retired expired records and absorbed finished
        judgments. ``now`` is the monotonic time of this read.
        """

    @abstractmethod
    async def judge(self, job: SupportsReceipt) -> SupportsReceipt:
        """The method's judgment for one job. It carries the ``receipt`` of the record that it judges."""

    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        """Run one job on the worker: the method's ``judge``."""
        return await self.judge(job)

    def expire(self, now: float) -> tuple[str, ...]:
        """Return tracked receipts whose wait for later traffic is over. They retire. The default expires nothing."""
        return ()


@dataclass(frozen=True, kw_only=True)
class JudgedSet(RecordSet):
    """One judged record that references no other record, with its judgment. ``set_id`` is the record's receipt."""

    record: AgentRecord
    judgment: SupportsReceipt

    @property
    def receipt(self) -> str:
        return self.record.agent_record_id

    def records(self) -> tuple[AgentRecord, ...]:
        return (self.record,)


class JudgedSamples(Record2Sample, ABC):
    """Judge records on a background worker, and make one sample of each judged record (``make_sample``).

    Every record is in exactly one state: tracked, in flight, formed into a
    judged set, or terminal. Terminal records are released from memory and never train. Judged sets form in the
    order that their judgments arrived.

    Without a worker, records track and expire, and nothing is ever judged.
    """

    def __init__(self, judging: RecordJudging, worker: BackgroundJobWorker | None, *, max_staleness: int = 0) -> None:
        super().__init__(max_staleness=max_staleness)
        self.judging = judging
        self.worker = worker
        self.tracked: dict[str, AgentRecord] = {}
        self.in_flight: dict[str, AgentRecord] = {}

    @abstractmethod
    def make_sample(self, record: AgentRecord, judgment: SupportsReceipt) -> TrajectorySample | None:
        """One judged record into a sample; ``None`` cannot train."""

    def make_samples(self, record_set: RecordSet) -> SampleResult:
        if not isinstance(record_set, JudgedSet):
            raise TypeError(f"{type(self).__name__} makes samples from judged sets, not {type(record_set).__name__}")
        sample = self.make_sample(record_set.record, record_set.judgment)
        return SampleResult.skipped() if sample is None else SampleResult.ready((sample,))

    # ------------------------------------------------------------- formation

    def push(self, record: AgentRecord) -> None:
        now = time.monotonic()
        # Expiry runs before the record is read, so an expired context can never capture it.
        self.advance(now)
        self.judging.observe(record, now, self)

    def advance(self, now: float) -> None:
        """Retire expired records, then form a set of each judgment that the worker finished."""
        for receipt in self.judging.expire(now):
            self.abandon(receipt)
        if self.worker is None:
            return
        for judgment in self.worker.poll():
            # Only records still in flight resolve; anything else was already released or closed.
            record = self.in_flight.pop(judgment.receipt, None)
            if record is None:
                continue
            if isinstance(judgment, Failed):
                self.retire(judgment.receipt)
                continue
            self.form(JudgedSet(set_id=judgment.receipt, record=record, judgment=judgment))

    # ------------------------------------------------------- the verbs for observe

    def track(self, record: AgentRecord) -> None:
        """This record awaits later traffic."""
        self.tracked[record.agent_record_id] = record

    def tracked_record(self, receipt: str) -> AgentRecord | None:
        """Return the record still awaiting completion under ``receipt``, if any."""
        return self.tracked.get(receipt)

    def dispatch(self, job: SupportsReceipt) -> None:
        """A tracked record was completed: give its judgment to the worker. ``job.receipt`` names the record."""
        record = self.tracked.pop(job.receipt, None)
        if record is None:
            return
        if self.worker is not None and self.worker.submit(job):
            self.in_flight[job.receipt] = record
        else:
            self.retire(job.receipt)

    def retire(self, receipt: str) -> None:
        """This receipt is terminal: released from memory, never trained."""
        self.terminal.add(receipt)

    def abandon(self, receipt: str) -> None:
        """A tracked record's later traffic never came: stop awaiting it, and retire it."""
        self.tracked.pop(receipt, None)
        self.retire(receipt)

    # --------------------------------------------------------- background work

    def background_jobs_running(self) -> bool:
        """Return True while judgments run. An expiry only retires records, so it waits for the next ``ready``."""
        return bool(self.in_flight)

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()

    # ---------------------------------------------------------------- release

    def releasable_record_ids(self) -> frozenset[str]:
        return super().releasable_record_ids() - frozenset(self.tracked) - frozenset(self.in_flight)

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        super().release_records(agent_record_ids)
        for agent_record_id in agent_record_ids:
            self.tracked.pop(agent_record_id, None)
            self.in_flight.pop(agent_record_id, None)

    def operational_metrics(self) -> Mapping[str, float | int]:
        return {"tracked_records": len(self.tracked), "judging_records": len(self.in_flight)}
