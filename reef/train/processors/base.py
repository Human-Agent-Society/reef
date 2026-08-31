"""The contract every processor implements, and its retention type.

The two processors that implement it for recipes live beside this module:
``reported`` (feedback received in a report) and ``computed`` (feedback
computed from traffic).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.core.records_types import AgentRecord, RequestType
from reef.observability import ExperimentLogger
from reef.train.types import PolicyBatch, ProcessorContext, TrainingBatch


@dataclass(frozen=True)
class RetentionDecision:
    """Processor-owned semantic decision about stored records.

    A record is compactable only when it is explicitly releasable.
    Protected records document the processor's current dependencies
    and take precedence.
    """

    protected_agent_record_ids: frozenset[str] = frozenset()
    releasable_agent_record_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        overlap = self.protected_agent_record_ids & self.releasable_agent_record_ids
        if overlap:
            raise ValueError(f"retention decision cannot protect and release the same records: {sorted(overlap)!r}")


class DataProcessor:
    """The processor contract: turn records into typed training batches.

    Which base a recipe builds on is one question — how does its feedback
    arrive?

    * As **reports** referencing inference records → subclass
      :class:`~reef.train.processors.reported.ReportedFeedbackProcessor` and write
      ``judge`` (what counts) and ``make_batch`` (what a batch looks like).
      The engine owns everything between them — including ``ingest``, which
      is where it calls your ``judge``, on the trainer's thread: a plain
      method, so keep it a pure decision on data already in hand.
    * **Computed from the traffic itself** — correlated across records,
      judged by a model, landing asynchronously → subclass
      :class:`~reef.train.processors.computed.ComputedFeedbackProcessor` and write
      ``ingest`` (the method's own correlation, built from the engine's
      ``catch_up``/``dispatch``/``track``/``retire`` verbs), ``judge``,
      ``make_sample``, and ``make_batch``; bulky machinery stays in the
      method package. Here ``judge`` is an ``async def``: your ``ingest`` hands a
      job to ``dispatch`` and a private worker thread awaits it, so it may
      call models and take minutes without blocking serving.

    That is the whole difference between the two processors: feedback is
    either reported explicitly or computed from correlated traffic. The
    reported path calls synchronous ``judge`` inside ``ingest``; the computed
    path awaits ``async def judge`` on its worker after recipe code dispatches
    a job.

    ``DataProcessor`` itself is never a recipe's processor. Instantiated
    bare it is the no-update default: it ingests records for audit
    (retaining only their ids) but never becomes ready and never produces a
    batch. The tradeoff of folding that default into the base (rather than
    keeping it abstract with a separate ``NoUpdateProcessor``) is that a
    half-written subclass that forgets to override ``build_batch`` silently
    becomes a no-op instead of failing at construction. Recipes that go
    through :class:`WeightTrainingRecipe.build` are still guarded — it raises
    ``TypeError`` unless the recipe declares a concrete ``processor``.

    Whatever implements this contract, the trainer-facing surface stays
    synchronous: ``ingest``/``ready``/``build_batch`` run on the trainer's
    thread under its lock and must never block on network or model
    latency. Judging that calls models therefore runs on a background
    thread — the computed-feedback processor owns one, starts it on demand, exchanges
    results through thread-safe queues, and releases it in :meth:`close`,
    the teardown hook the owning scenario guarantees to call when the
    processor is dropped (dispatcher shutdown and durable reload both go
    through it). State that lives only in processor memory is rebuilt by
    replaying ingest after a restart: the trainer replays un-acknowledged
    records, so the recompute cost is bounded by the training backlog, not
    history.
    """

    required_request_types: frozenset[RequestType] = frozenset(RequestType)

    def __init__(self, context: ProcessorContext) -> None:
        self._context = context
        self._scenario = context.scenario
        # No-update default: retain only ids for retention; never build a batch.
        self._agent_record_ids: set[str] = set()
        self._batch_size = int(context.config.get("batch_size", 1))
        if self._batch_size <= 0:
            raise ValueError("batch_size must be positive")
        #: The batch handed out and not yet acknowledged. While it exists the
        #: processor is ready, hands out the same object, and must not
        #: reshuffle what it references.
        self._pending: TrainingBatch | None = None
        self._batch_number = 0

    @property
    def context(self) -> ProcessorContext:
        return self._context

    @property
    def scenario(self) -> str:
        return self._scenario

    @property
    def experiment_logger(self) -> ExperimentLogger:
        """The scenario logger shared by its recipe, processor, and backend."""
        return self._context.experiment_logger

    #: The batch type ``build_batch`` returns; the trainer validates it.
    output_schema: type[TrainingBatch] = PolicyBatch

    def ingest(self, item: AgentRecord) -> None:
        self._agent_record_ids.add(item.agent_record_id)

    # ------------------------------------------------------------ batch cycle
    #
    # The shape is the same for every processor: batch when enough units are
    # held, hand the same batch out until it is acknowledged, then release
    # what it consumed. An engine fills in the three things that differ —
    # what a unit is, how the selected ones become a batch, and what
    # consuming them releases.

    def ready(self) -> bool:
        return self._pending is not None or self._ready_count() >= self._batch_size

    def build_batch(self) -> TrainingBatch:
        if self._pending is None:
            if not self.ready():
                raise RuntimeError(f"{type(self).__name__} batch is not ready")
            self._batch_number += 1
            self._pending = self._make_pending(self._batch_number)
        return self._pending

    def acknowledge(self, batch_id: str) -> frozenset[str]:
        if self._pending is None or self._pending.batch_id != batch_id:
            raise ValueError(f"unknown batch_id {batch_id!r}")
        consumed = self._consume_pending()
        self._pending = None
        return consumed

    def _ready_count(self) -> int:
        """How many batch-ready units are held.

        Zero is the no-update default, and it is what makes a bare
        ``DataProcessor`` ingest for audit without ever becoming ready.
        """
        return 0

    def _make_pending(self, batch_number: int) -> TrainingBatch:
        """Select this batch's units and shape them through ``make_batch``."""
        raise RuntimeError(f"{type(self).__name__} never produces a training batch")

    def _consume_pending(self) -> frozenset[str]:
        """Release what the acknowledged batch consumed and name its records.

        The returned ids ride the step's commit record so recovery can skip
        them when rebuilding processor memory: a record a committed batch
        consumed must never train twice, even when retention keeps it stored.
        The no-update default consumes nothing.
        """
        return frozenset()

    def retention_decision(self) -> RetentionDecision:
        """Return the records the processor currently protects or releases.

        The no-update default protects every ingested id (audit-only retention).
        Subclasses with real pairing semantics override this to derive
        protected/releasable sets from their own state.
        """
        return RetentionDecision(protected_agent_record_ids=frozenset(self._agent_record_ids))

    def compaction_applied(self, agent_record_ids: frozenset[str]) -> None:
        """Forget semantic markers whose positioned records were deleted."""
        self._agent_record_ids -= agent_record_ids

    def derivation_pending(self) -> bool:
        """Whether background derivation could flip ``ready`` without records.

        The training worker sleeps until the next accepted record; a
        processor whose judgments land asynchronously (or whose sessions
        flush on a TTL) returns ``True`` here so the worker polls on a
        bounded interval instead. Read on the training thread between
        drains — implementations must not block.
        """
        return False

    def status(self) -> Mapping[str, Any]:
        """Return JSON-safe state that callers need while waiting.

        Most processors have no caller-visible state. A processor may
        override this for a terminal outcome that cannot become a training
        batch, allowing a bounded external wait to fail explicitly.
        """
        return {}

    def close(self) -> None:
        """Release resources the processor owns; safe to call more than once.

        The no-update default owns nothing. Processors with background
        derivation work override this to signal their workers and join them;
        after ``close`` returns, no thread of the processor may touch shared
        state or deliver further results.
        """
