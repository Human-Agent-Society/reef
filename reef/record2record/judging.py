"""Judged sets: records that a model judges asynchronously, after later traffic completes them.

A computed-feedback method makes its training signal from the traffic itself.
A record only becomes judgeable when some later record completes it, and the
judgment calls models, so it runs on a background worker. ``JudgingRecord2Record``
owns the record lifecycle around that: tracked (awaiting later traffic), in
flight (judging), then a judged set, or terminal. A recipe writes the
correlation and the judge as a ``RecordJudging``.

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

from reef.background_jobs import BackgroundJobRunner, BackgroundJobWorker, Failed, SupportsReceipt
from reef.core.records_types import AgentRecord
from reef.pipeline_context import PipelineContext
from reef.record2record.base import Record2Record
from reef.record2record.selection import ArrivalOrder
from reef.record2record.sets import RecordSet
from reef.record2record.staleness import stale_sets


class RecordJudging(BackgroundJobRunner, ABC):
    """A computed-feedback method's correlation and judge.

    ``observe`` and ``expire`` run on the trainer thread and must not block.
    ``judge`` runs on the worker thread, so it may call models and take
    minutes. State that it shares with the trainer thread must be safe to use from both threads.
    """

    @abstractmethod
    def observe(self, record: AgentRecord, now: float, stage: JudgingRecord2Record) -> None:
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
    """One judged record. ``set_id`` is the record's receipt; ``source_index`` is its arrival order."""

    record: AgentRecord
    judgment: SupportsReceipt

    @property
    def receipt(self) -> str:
        return self.record.agent_record_id

    def record_ids(self) -> tuple[str, ...]:
        return (self.receipt,)


def judged_sets(pool_sets: tuple[RecordSet, ...]) -> tuple[JudgedSet, ...]:
    """Return pool sets as judged sets. A judging stage puts only judged sets in its pool."""
    selected = tuple(record_set for record_set in pool_sets if isinstance(record_set, JudgedSet))
    if len(selected) != len(pool_sets):
        raise TypeError("the judging stage's pool must hold only judged sets")
    return selected


class JudgingRecord2Record(Record2Record):
    """Judge records on a background worker, and hold the judged sets until a batch uses them.

    Every record is in exactly one state: tracked, in flight, held in the
    pool, or terminal. Terminal records are released from memory and never
    batch. Judged sets wait in the order that their judgments arrived. The
    staleness check (``stale_sets``) drops sets that are more than ``max_staleness``
    versions behind the newest one (the context's ``max_staleness``), by record arrival, while no batch is out:
    continual serving advances the version every step, and stale sets would
    otherwise block the queue.

    Without a worker, records track and expire, and nothing is ever judged.
    """

    def __init__(self, context: PipelineContext, judging: RecordJudging, worker: BackgroundJobWorker | None) -> None:
        super().__init__(ArrivalOrder())
        self.max_staleness = context.max_staleness
        self.judging = judging
        self.worker = worker
        self.tracked: dict[str, AgentRecord] = {}
        self.in_flight: dict[str, AgentRecord] = {}
        # Record-arrival order per live receipt: the staleness check needs
        # generation recency, which judgment completion order is not.
        self.arrival_order: dict[str, int] = {}
        self.arrival_count = 0

    # ------------------------------------------------------------- formation

    def ingest(self, record: AgentRecord) -> None:
        now = time.monotonic()
        # Expiry runs before the record is read, so an expired context can never capture it.
        self.poll(now)
        self.judging.observe(record, now, self)

    def poll(self, now: float) -> None:
        """Retire expired records, then absorb the judgments that the worker finished."""
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
            self.formed.append(
                JudgedSet(
                    set_id=judgment.receipt,
                    arrival_index=self.pool.next_arrival_index(),
                    source_index=self.arrival_order[judgment.receipt],
                    record=record,
                    judgment=judgment,
                )
            )

    def accept(self, record_set: RecordSet, runtime_load_id: str | None) -> None:
        super().accept(record_set, runtime_load_id)
        # The staleness check never runs under a reserved batch: its samples reference held sets.
        if self.pool.reserved is None:
            for judged_set in judged_sets(stale_sets(self.pool.sets(), self.max_staleness)):
                self.pool.remove(judged_set.set_id)
                self.retire(judged_set.receipt)
                self.ended_set_ids.add(judged_set.set_id)

    def reject(self, record_set: RecordSet) -> None:
        self.retire(judged_sets((record_set,))[0].receipt)

    # ------------------------------------------------------- the verbs for observe

    def track(self, record: AgentRecord) -> None:
        """This record awaits later traffic."""
        self.arrival_count += 1
        self.arrival_order[record.agent_record_id] = self.arrival_count
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
        """This receipt is terminal: released from memory, never batched."""
        self.terminal.add(receipt)
        self.arrival_order.pop(receipt, None)

    def abandon(self, receipt: str) -> None:
        """A tracked record's later traffic never came: stop awaiting it, and retire it."""
        self.tracked.pop(receipt, None)
        self.retire(receipt)

    # ------------------------------------------------------------ consumption

    def consume(self, record_sets: tuple[RecordSet, ...], *, trained: bool) -> frozenset[str]:
        consumed = super().consume(judged_sets(record_sets), trained=trained)
        for receipt in consumed:
            self.arrival_order.pop(receipt, None)
        return consumed

    # --------------------------------------------------------- background work

    def background_jobs_running(self) -> bool:
        """Return True while judgments run. An expiry only retires records, so it waits for the next ``ready``."""
        return bool(self.in_flight)

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()

    # ---------------------------------------------------------------- release

    def releasable_record_ids(self) -> frozenset[str]:
        held = frozenset(judged_set.receipt for judged_set in judged_sets(self.pool.sets()))
        protected = frozenset(self.tracked) | frozenset(self.in_flight) | held
        return super().releasable_record_ids() - protected

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        super().release_records(agent_record_ids)
        for agent_record_id in agent_record_ids:
            self.tracked.pop(agent_record_id, None)
            self.in_flight.pop(agent_record_id, None)
            self.pool.remove(agent_record_id)
            self.arrival_order.pop(agent_record_id, None)

    def operational_metrics(self) -> Mapping[str, float | int]:
        reserved = len(self.pool.reserved_sets())
        return {
            "tracked_records": len(self.tracked),
            "judging_records": len(self.in_flight),
            "unreserved_candidates": len(self.pool) - reserved,
            "reserved_candidates": reserved,
        }
