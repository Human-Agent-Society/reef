"""Harness-evolution processor: pair recorded requests with reported scores, unmodified."""

from __future__ import annotations

from reef.core import AgentRecord, RequestType
from reef.train.processors.base import DataProcessor, RetentionDecision
from reef.train.processors.reported import (
    NEVER,
    BatchUnit,
    Outcome,
    ReportContext,
    ReportDecision,
    ReportedFeedbackProcessor,
)
from reef.train.types import ProcessorContext, TraceBatch, TraceSample


class CordisProcessor(ReportedFeedbackProcessor):
    """Pair recorded requests with reported scores and batch them unmodified.

    Requests are recorded post-transform, so a trace shows exactly what the
    backend served. A score window in config selects which traces batch —
    harness evolution's ``max_score`` bound keeps only failures — and reports
    outside it are terminal and release their records. A report may reference
    one request or a whole run's worth; several references become one
    trajectory sample. The backends consume
    the resulting trace batches without adding processor logic.
    """

    output_schema = TraceBatch

    def __init__(self, context: ProcessorContext) -> None:
        self._min_score = float(context.config.get("min_score", float("-inf")))
        self._max_score = float(context.config.get("max_score", float("inf")))
        if self._min_score > self._max_score:
            raise ValueError("min_score must not exceed max_score")
        super().__init__(context)

    def judge(self, context: ReportContext) -> ReportDecision:
        # 1. Reports that can never train are terminal on sight. The shared
        #    gate is also what refuses a non-finite score: an open window
        #    would otherwise admit inf, which no comparison can order.
        gate = context.eligibility()
        if gate is not None and gate.outcome is Outcome.NEVER:
            return gate
        score = context.score
        if score is None:
            raise RuntimeError("eligible harness report has no score")
        # 2. A trace outside the window, or one referencing nothing, is
        #    terminal before any reference resolves — the records release
        #    immediately.
        if not context.references or not self._min_score <= score <= self._max_score:
            return NEVER
        # 3. Park until every referenced request record arrives.
        if gate is not None:
            return gate
        if context.inferences is None:
            raise RuntimeError("resolved harness report has no inference")
        # 4. The recorded requests themselves are the sample, unmodified: one
        #    reference is a single-exchange sample, several are one trajectory
        #    sample in reference order.
        last = context.inferences[-1]
        return ReportDecision.train(
            TraceSample(
                source_agent_record_id=last.agent_record_id,
                payload=last.payload,
                score=score,
                feedback=context.report.payload.get("feedback"),
                trajectory=(
                    tuple(record.payload for record in context.inferences) if len(context.inferences) > 1 else ()
                ),
            )
        )

    def make_batch(self, units: tuple[BatchUnit, ...], batch_number: int) -> TraceBatch:
        return TraceBatch(
            f"{self.scenario}:harness_evolve:{batch_number}",
            tuple(unit.candidates[0].value for unit in units),
        )


class RecordDrivenTraceProcessor(DataProcessor):
    """Batch recorded inference traffic every ``batch_size`` requests, unscored.

    The report-free half of harness evolution: a deployment that only serves
    still evolves. Each recorded inference is one unit, in arrival order;
    when ``batch_size`` have accumulated they batch as trace samples with
    ``score=None``, and the proposer contract requires handling unscored
    samples. Reports that arrive under this policy are released untouched;
    a deployment with real outcome signal selects the reported policy
    instead, because a measured result beats model self judgment.
    """

    output_schema = TraceBatch

    def __init__(self, context: ProcessorContext) -> None:
        super().__init__(context)
        self._records: list[AgentRecord] = []
        self._released: set[str] = set()

    def ingest(self, item: AgentRecord) -> None:
        if item.request_type is RequestType.INFERENCE:
            self._records.append(item)
        else:
            self._released.add(item.agent_record_id)

    def _ready_count(self) -> int:
        return len(self._records)

    def _make_pending(self, batch_number: int) -> TraceBatch:
        selected = self._records[: self._batch_size]
        return TraceBatch(
            f"{self.scenario}:harness_evolve:{batch_number}",
            tuple(
                TraceSample(
                    source_agent_record_id=record.agent_record_id,
                    payload=record.payload,
                    score=None,
                )
                for record in selected
            ),
        )

    def _consume_pending(self) -> frozenset[str]:
        if self._pending is None or not isinstance(self._pending, TraceBatch):
            raise RuntimeError("no pending trace batch to consume")
        consumed = frozenset(sample.source_agent_record_id for sample in self._pending.samples)
        self._records = [record for record in self._records if record.agent_record_id not in consumed]
        self._released |= consumed
        return consumed

    def retention_decision(self) -> RetentionDecision:
        return RetentionDecision(
            protected_agent_record_ids=frozenset(record.agent_record_id for record in self._records),
            releasable_agent_record_ids=frozenset(self._released),
        )

    def compaction_applied(self, agent_record_ids: frozenset[str]) -> None:
        self._released -= agent_record_ids
