"""The contract every processor implements, and its retention type.

The two processors that implement it for recipes live beside this module:
``reported`` (feedback received in a report) and ``computed`` (feedback
computed from traffic).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from reef.core.records_types import AgentRecord, RequestType
from reef.core.training_request import TrainingRequest
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

    required_request_types: frozenset[RequestType] = frozenset({RequestType.INFERENCE, RequestType.REPORT})
    supported_training_modes: frozenset[str] = frozenset({"auto"})

    def __init__(self, context: ProcessorContext) -> None:
        self._context = context
        # The two modes that take an instruction go together: a recipe that cannot run one cannot run it in either.
        if not context.config.get("manual_enabled", True):
            self.supported_training_modes = self.supported_training_modes - {"manual", "hybrid"}
        self.set_training_mode(context.training_mode)
        self._training_requests: dict[str, TrainingRequest] = {}
        self._consumed_requests: set[str] = set()
        # The error of each buffered instruction whose step failed; its next batch is a skip row, not a run.
        self._request_failures: dict[str, str] = {}
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
    def training_mode(self) -> str:
        """The batching policy selected for this processor."""
        return self._context.training_mode

    def set_training_mode(self, training_mode: str) -> None:
        """Select future batches while preserving shared buffers and reservations."""
        if training_mode not in ("auto", "manual", "hybrid"):
            raise ValueError("training_mode must be 'auto', 'manual' or 'hybrid'")
        if training_mode not in self.supported_training_modes:
            raise NotImplementedError(f"{type(self).__name__} does not implement training_mode={training_mode!r}")
        self._context = replace(self._context, training_mode=training_mode)

    def buffered_requests(self) -> int:
        """How many instructions are read into memory and not yet consumed."""
        return len(self._training_requests)

    def request_failure(self, request_id: str) -> str | None:
        """What the instruction's failed step said, when it failed at all."""
        return self._request_failures.get(request_id)

    def request_failures(self) -> Mapping[str, str]:
        """The failed instructions still buffered, by id, with what their step said."""
        return dict(self._request_failures)

    def mark_request_failed(self, request_id: str, error: str) -> None:
        """Record that the instruction's step failed; its next batch is consumed with a skip row."""
        self._request_failures[request_id] = error

    def set_request_failures(self, failures: Mapping[str, str]) -> None:
        """Carry the failed instructions of a replaced processor into this one."""
        self._request_failures = dict(failures)

    @property
    def experiment_logger(self) -> ExperimentLogger:
        """The scenario logger shared by its recipe, processor, and backend."""
        return self._context.experiment_logger

    #: The batch type ``build_batch`` returns; the trainer validates it.
    output_schema: type[TrainingBatch] = PolicyBatch

    def ingest(self, item: AgentRecord) -> None:
        if item.request_type is RequestType.TRAIN:
            if item.scenario != self.scenario:
                raise ValueError("training records must belong to the processor's scenario")
            request = replace(TrainingRequest.from_dict(item.payload), id=item.agent_record_id)
            if request.id not in self._consumed_requests:
                self._training_requests.setdefault(request.id, request)
        else:
            self._agent_record_ids.add(item.agent_record_id)

    # ------------------------------------------------------------ batch cycle
    #
    # The shape is the same for every processor: batch when enough units are
    # held (auto, hybrid) or an instruction is queued (manual, hybrid), hand the
    # same batch out until it is acknowledged, then release what it consumed.
    # An engine fills in the three things that differ: what a unit is, how
    # the selected ones become a batch, and what consuming them releases.

    def ready(self) -> bool:
        if self._pending is not None:
            return True
        if self.training_mode == "manual":
            return bool(self._training_requests)
        if self.training_mode == "hybrid" and self._training_requests:
            return True
        return self._ready_count() >= self._batch_size

    def build_batch(self) -> TrainingBatch:
        if self._pending is None:
            if not self.ready():
                raise RuntimeError(f"{type(self).__name__} batch is not ready")
            self._batch_number += 1
            # Manual and hybrid run the oldest queued instruction first; auto leaves the queue to a mode that takes it.
            request = (
                next(iter(self._training_requests.values()))
                if self.training_mode != "auto" and self._training_requests
                else None
            )
            self._pending = self.make_training_batch(self._batch_number, request)
            if request is not None:
                self._pending = replace(
                    self._pending, batch_id=f"{self.scenario}:instruction:{request.id}", request=request
                )
        return self._pending

    def make_training_batch(self, batch_number: int, request: TrainingRequest | None) -> TrainingBatch:
        """Select inputs for one batch; in ``manual`` and ``hybrid`` a queued instruction arrives as ``request``.

        Override this single assembly hook to take instructions. Ingestion,
        acknowledgement and retention operate on the same state in every mode.
        With a request the hook's own batch id is replaced by
        ``<scenario>:instruction:<request id>`` and the request is attached.
        """
        if request is not None:
            raise NotImplementedError(f"{type(self).__name__} does not implement instruction batch assembly")
        return self._make_pending(batch_number)

    def acknowledge(self, batch_id: str) -> frozenset[str]:
        if self._pending is None or self._pending.batch_id != batch_id:
            raise ValueError(f"unknown batch_id {batch_id!r}")
        consumed = self._consume_pending()
        if self._pending.request is not None:
            request_id = self._pending.request.id
            self._training_requests.pop(request_id)
            self._request_failures.pop(request_id, None)
            self._consumed_requests.add(request_id)
            consumed = consumed | {request_id}
        self._pending = None
        return consumed

    def release_batch(self, batch_id: str) -> None:
        """Forget the handed out batch without consuming anything; the next ``build_batch`` selects again."""
        if self._pending is None or self._pending.batch_id != batch_id:
            raise ValueError(f"unknown batch_id {batch_id!r}")
        self._pending = None

    def discard_request(self, request_id: str) -> frozenset[str]:
        """Consume one queued instruction without a batch; the committed row that names it is what recovery skips."""
        if self._pending is not None and self._pending.request is not None and self._pending.request.id == request_id:
            raise ValueError(f"training request {request_id!r} is reserved; release its batch first")
        if request_id not in self._training_requests:
            raise ValueError(f"unknown training request {request_id!r}")
        self._training_requests.pop(request_id)
        self._request_failures.pop(request_id, None)
        self._consumed_requests.add(request_id)
        return frozenset({request_id})

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
        return RetentionDecision(
            protected_agent_record_ids=frozenset(self._agent_record_ids | self._training_requests.keys()),
            releasable_agent_record_ids=frozenset(self._consumed_requests),
        )

    def compaction_applied(self, agent_record_ids: frozenset[str]) -> None:
        """Forget semantic markers whose positioned records were deleted."""
        self._agent_record_ids -= agent_record_ids
        self._consumed_requests -= agent_record_ids

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
        if self.supported_training_modes & {"manual", "hybrid"}:
            return {"buffered_requests": self.buffered_requests()}
        return {}

    def close(self) -> None:
        """Release resources the processor owns; safe to call more than once.

        The no-update default owns nothing. Processors with background
        derivation work override this to signal their workers and join them;
        after ``close`` returns, no thread of the processor may touch shared
        state or deliver further results.
        """
