"""Harness-evolution processor: pair recorded requests with reported scores, unmodified."""

from __future__ import annotations

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
    outside it are terminal and release their records. The backends consume
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
        # 2. A trace outside the window, or one claiming more than a single
        #    request, is terminal before its reference resolves — so both
        #    records release immediately.
        if len(context.references) != 1 or not self._min_score <= score <= self._max_score:
            return NEVER
        # 3. Park until the referenced request record arrives.
        if gate is not None:
            return gate
        if context.inferences is None:
            raise RuntimeError("resolved harness report has no inference")
        inference = context.inferences[0]
        # 4. The recorded request itself is the sample, unmodified.
        return ReportDecision.train(
            TraceSample(
                source_agent_record_id=inference.agent_record_id,
                payload=inference.payload,
                score=score,
            )
        )

    def make_batch(self, units: tuple[BatchUnit, ...], batch_number: int) -> TraceBatch:
        return TraceBatch(
            f"{self.scenario}:harness_evolve:{batch_number}",
            tuple(unit.candidates[0].value for unit in units),
        )
