"""Computed-feedback engine: judge records asynchronously, batch the results.

The sibling of :class:`~reef.train.processors.reported.ReportedFeedbackProcessor`
for the other way feedback arrives. A reported-feedback recipe is handed
explicit reports; a computed-feedback recipe produces its signal from the traffic itself — a
record only becomes judgeable when some LATER record completes it, and
the judgment itself calls models, so it runs asynchronously on a private
worker. This engine owns everything between those facts and the trainer:
the record lifecycle, the judging worker, the batch cycle, retention, and
crash-replay semantics. A recipe subclasses it and writes four methods:

* ``ingest(record)`` — the method's own correlation, written with the
  engine's verbs: start with ``catch_up``, ``dispatch`` each completed
  record's job, then ``track`` the record or ``retire`` it;
* ``judge(job)`` — an async coroutine producing the recipe's judgment,
  run on the worker's thread;
* ``make_sample(record, judgment)`` — one judged record into a
  :class:`PolicySample`, or ``None`` for a record that cannot train;
* ``make_batch(samples, batch_number)`` — the selected samples into the
  recipe's batch type.

The ingest stays recipe code on purpose: for a computed-feedback method the
correlation IS the method, so its whole line reads in the recipe file.

Everything here is reconstructible state: after a crash the trainer replays the
un-acknowledged tail and judging re-runs, recompute bounded by the
training backlog; a record whose judgment was in flight is lost — one
record's exposure.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from threading import Lock, Thread
from typing import Any, Protocol

from reef.core.records_types import AgentRecord
from reef.train.processors.base import DataProcessor, RetentionDecision
from reef.train.types import PolicySample, ProcessorContext, TrainingBatch

logger = logging.getLogger(__name__)

#: How long ``close()`` lets in-flight judgments finish before cancelling.
CLOSE_GRACE_S = 5.0


class SupportsReceipt(Protocol):
    """What the engine needs of a recipe's job and of its judgment alike:
    the receipt of the tracked record they concern.

    The recipe defines both types; the engine only ever reads ``receipt``, so
    this is the whole seam between them. :class:`Failed` satisfies it too.
    """

    @property
    def receipt(self) -> str: ...


@dataclass(frozen=True)
class Failed:
    """The worker's answer for a job whose judgment never finished."""

    receipt: str


class JudgingWorker:
    """Judgments off the trainer thread: jobs in, judgments out, both queues.

    The worker owns only execution — one daemon thread, an event loop, and
    the two queues; what a judgment *is* stays the recipe's ``judge``
    coroutine, handed in at construction. Faults degrade, never hang: a
    failed or cancelled judgment resolves as :class:`Failed`, and a worker
    whose event loop dies marks itself broken, resolves everything queued as
    :class:`Failed`, and rejects new work so the engine retires records
    directly.
    """

    def __init__(self, judge: Callable[[Any], Awaitable[SupportsReceipt]], *, concurrency: int = 8) -> None:
        self._judge = judge
        self._concurrency = max(1, concurrency)
        self._jobs: queue.SimpleQueue[SupportsReceipt | None] = queue.SimpleQueue()
        self._judgments: queue.SimpleQueue[SupportsReceipt] = queue.SimpleQueue()
        self._thread: Thread | None = None
        # Serializes submit against the broken/closed transitions so a job
        # can never slip in after the final drain and leave its receipt
        # unresolved.
        self._gate = Lock()
        self._broken = False
        self._closed = False

    def submit(self, job: SupportsReceipt) -> bool:
        """Queue one job for judgment; ``False`` means it will never run."""
        with self._gate:
            if self._closed or self._broken:
                return False
            if self._thread is None:
                self._thread = Thread(target=self._run, name="computed-feedback-judging", daemon=True)
                self._thread.start()
            self._jobs.put(job)
            return True

    def poll(self) -> list[SupportsReceipt]:
        """Drain every finished judgment without blocking."""
        judgments: list[SupportsReceipt] = []
        try:
            while True:
                judgments.append(self._judgments.get_nowait())
        except queue.Empty:
            return judgments

    def close(self) -> None:
        """Stop accepting work, give in-flight judgments a bounded grace, join."""
        with self._gate:
            if self._closed:
                return
            self._closed = True
        if self._thread is None:
            return
        self._jobs.put(None)
        self._thread.join(timeout=CLOSE_GRACE_S + 10.0)

    # ------------------------------------------------------------ worker side

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception:
            logger.exception("judging worker failed; queued jobs resolve untrainable")
        finally:
            with self._gate:
                self._broken = True
            while True:  # no submit can race this drain: _broken is set
                try:
                    job = self._jobs.get_nowait()
                except queue.Empty:
                    break
                if job is not None:
                    self._judgments.put(Failed(job.receipt))

    async def _main(self) -> None:
        # At most ``concurrency`` jobs judge at once; the semaphore lives on
        # this loop, like the rest of the worker's state.
        semaphore = asyncio.Semaphore(self._concurrency)
        loop = asyncio.get_running_loop()
        tasks: set[asyncio.Task] = set()
        while True:
            job = await asyncio.to_thread(self._jobs.get)
            if job is None:
                break
            task = loop.create_task(self._judge_one(job, semaphore))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        if tasks:
            # Whatever the grace period does not finish, asyncio.run cancels
            # on the way out — and each judgment still resolves its receipt
            # from its finally.
            await asyncio.wait(tasks, timeout=CLOSE_GRACE_S)

    async def _judge_one(self, job: SupportsReceipt, semaphore: asyncio.Semaphore) -> None:
        resolved = False
        try:
            async with semaphore:
                self._judgments.put(await self._judge(job))
            resolved = True
        except Exception:
            logger.exception("judgment for receipt %s failed; record resolves untrainable", job.receipt)
        finally:
            if not resolved:  # exception or teardown cancellation
                self._judgments.put(Failed(job.receipt))


class ComputedFeedbackProcessor(DataProcessor, ABC):
    """Record lifecycle and batch cycle for computed-feedback recipes.

    Every ingested receipt sits in exactly one state: tracked
    (awaiting future traffic) → in-flight (judging) → candidate | terminal
    | trained. Terminal records are released for compaction instead of
    ever entering a batch. Only the newest weight version (by record
    arrival, not judgment arrival) ever batches: continual serving
    advances the version every step, and stale pending candidates would
    deadlock the FIFO.

    The line, in reading order: the recipe's ``ingest`` (catch_up →
    dispatch → track), its ``judge`` on the worker, then here —
    ``_collect_judgments`` absorbs what landed, ``build_batch`` calls the
    recipe's ``make_batch``, ``acknowledge`` releases.
    """

    def __init__(self, context: ProcessorContext, *, worker: JudgingWorker | None) -> None:
        super().__init__(context)
        #: No worker: records track and expire, nothing is ever judged,
        #: everything stays compactable.
        self._worker = worker

        # Every ingested receipt is in exactly one state.
        self._tracked: dict[str, AgentRecord] = {}
        self._in_flight: dict[str, AgentRecord] = {}
        self._candidates: dict[str, PolicySample] = {}
        #: Terminal receipts: retired or trained, both releasable and never
        #: distinguished by anything that reads them.
        self._terminal: set[str] = set()
        # Record-arrival order per live receipt: the stale-version policy
        # needs generation recency, which judgment completion order is not.
        self._arrival_order: dict[str, int] = {}
        self._next_order = 0
        self._pending_receipts: tuple[str, ...] = ()

    # ------------------------------------------------------- the recipe hooks

    @abstractmethod
    def ingest(self, item: AgentRecord) -> None:
        """The method's own correlation, one record at a time, written with
        the engine's verbs (``catch_up`` first, then ``dispatch`` /
        ``track`` / ``retire``)."""

    @abstractmethod
    async def judge(self, job: Any) -> SupportsReceipt:
        """The recipe's judgment for one job.

        Awaited on the worker thread after your ``ingest`` dispatched the
        job — never on the trainer's — so it may call models and take
        minutes. It carries the ``receipt`` of the record it judges; that
        is all the engine reads of it.
        """

    @abstractmethod
    def make_sample(self, record: AgentRecord, judgment: Any) -> PolicySample | None:
        """One judged record into a sample; ``None`` cannot train."""

    @abstractmethod
    def make_batch(self, samples: tuple[PolicySample, ...], batch_number: int) -> TrainingBatch:
        """Shape the selected samples into this recipe's batch type."""

    def expire(self, now: float) -> tuple[str, ...]:
        """Tracked receipts whose wait for future traffic is over; they
        retire terminal. The default recipe never expires anything."""
        return ()

    # ------------------------------------------- the engine's verbs for ingest

    def catch_up(self, now: float) -> None:
        """Retire expired records, absorb finished judgments. Every ingest
        starts here — expiry runs before the record is read, so an expired
        context can never capture it."""
        for receipt in self.expire(now):
            self.abandon(receipt)
        self._collect_judgments()

    def track(self, item: AgentRecord) -> None:
        """This record awaits future traffic."""
        self._next_order += 1
        self._arrival_order[item.agent_record_id] = self._next_order
        self._tracked[item.agent_record_id] = item

    def tracked_record(self, receipt: str) -> AgentRecord | None:
        """The record still awaiting completion under ``receipt``, if any."""
        return self._tracked.get(receipt)

    def dispatch(self, job: SupportsReceipt) -> None:
        """A tracked record was completed: hand its judgment to the worker.
        ``job.receipt`` names the tracked record it judges."""
        record = self._tracked.pop(job.receipt, None)
        if record is None:
            return
        if self._worker is not None and self._worker.submit(job):
            self._in_flight[job.receipt] = record
        else:
            self.retire(job.receipt)

    def retire(self, receipt: str) -> None:
        """This receipt is terminal: released for compaction, never batched."""
        self._terminal.add(receipt)
        self._arrival_order.pop(receipt, None)

    def abandon(self, receipt: str) -> None:
        """A tracked record's future never came: stop awaiting it, retire it.
        (``retire`` alone would leave the record protected as tracked.)"""
        self._tracked.pop(receipt, None)
        self.retire(receipt)

    def _collect_judgments(self) -> None:
        """Absorb judgments the worker finished since the last entry."""
        if self._worker is None:
            return
        for judgment in self._worker.poll():
            # 1. Only records still in flight resolve; anything else was
            #    already retired by compaction or teardown.
            record = self._in_flight.pop(judgment.receipt, None)
            if record is None:
                continue
            # 2. A failed or declined judgment retires the record; so does a
            #    sample the recipe cannot train on — never a failed training
            #    step.
            sample = None if isinstance(judgment, Failed) else self.make_sample(record, judgment)
            if sample is None:
                self.retire(judgment.receipt)
                continue
            # 3. The record is a batch candidate now. Only the newest weight
            #    version (by record arrival) keeps its place — but never
            #    reshuffle under an emitted-but-unacknowledged batch, whose
            #    samples reference the pending candidates.
            self._candidates[judgment.receipt] = sample
            if self._pending is None:
                newest = max(self._candidates, key=self._arrival_order.__getitem__)
                newest_version = self._candidates[newest].weight_version
                for receipt in [r for r, s in self._candidates.items() if s.weight_version != newest_version]:
                    self._candidates.pop(receipt)
                    self.retire(receipt)

    # ----------------------------------------------------------- batch cycle
    #
    # A unit is one judged record's sample; the engine's half of the shared
    # cycle in base.py is the three methods below. ``ready`` also absorbs
    # whatever the worker finished, since judgments land without a record
    # ever arriving to trigger it.

    def ready(self) -> bool:
        self.catch_up(time.monotonic())
        return super().ready()

    def _ready_count(self) -> int:
        return len(self._candidates)

    def _make_pending(self, batch_number: int) -> TrainingBatch:
        receipts = tuple(list(self._candidates)[: self._batch_size])
        self._pending_receipts = receipts
        return self.make_batch(tuple(self._candidates[receipt] for receipt in receipts), batch_number)

    def _consume_pending(self) -> frozenset[str]:
        consumed = frozenset(self._pending_receipts)
        for receipt in self._pending_receipts:
            self._candidates.pop(receipt, None)
            self._arrival_order.pop(receipt, None)
            self._terminal.add(receipt)
        self._pending_receipts = ()
        return consumed

    # -------------------------------------------------------- background work

    def derivation_pending(self) -> bool:
        """In-flight judgments resolve and tracked records expire without a
        new record ever waking the training worker; poll while either exists."""
        return bool(self._in_flight or self._tracked)

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()

    # -------------------------------------------------------------- retention

    def retention_decision(self) -> RetentionDecision:
        protected = frozenset(self._tracked) | frozenset(self._in_flight) | frozenset(self._candidates)
        return RetentionDecision(
            protected_agent_record_ids=protected,
            releasable_agent_record_ids=frozenset(self._terminal) - protected,
        )

    def compaction_applied(self, agent_record_ids: frozenset[str]) -> None:
        self._terminal -= agent_record_ids
        for agent_record_id in agent_record_ids:
            self._tracked.pop(agent_record_id, None)
            self._in_flight.pop(agent_record_id, None)
            self._candidates.pop(agent_record_id, None)
            self._arrival_order.pop(agent_record_id, None)
