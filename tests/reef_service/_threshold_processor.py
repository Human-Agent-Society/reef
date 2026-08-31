"""The reported-feedback processor's test double, shared by five test modules.

The engine tests exercise the batch cycle, retention, compaction, and crash
recovery, which needs one concrete processor carrying no method of its own.
A score bar is the least arbitrary judge that still reaches both outcomes the
engine routes: at or above it a report trains, below it the report is
terminal and its records release.

Nothing under ``reef/`` imports this. No shipped recipe has this shape.
"""

from __future__ import annotations

from reef.train.processors.reported import (
    NEVER,
    Outcome,
    ReportContext,
    ReportDecision,
    ReportedFeedbackProcessor,
    SampleAssembly,
)
from reef.train.types import PolicyBatch, ProcessorContext


class ThresholdProcessor(ReportedFeedbackProcessor):
    """Keep only reports at or above ``min_score``; one report, one sample.

    A single-call report below ``min_score`` is terminal on arrival; a
    multi-call report is still assembled first, so retention answers do not
    depend on which record arrived last.
    """

    output_schema = PolicyBatch

    def __init__(self, context: ProcessorContext) -> None:
        self._min_score = float(context.config.get("min_score", float("-inf")))
        self._assembly = SampleAssembly.from_config(context)
        super().__init__(context)

    def judge(self, context: ReportContext) -> ReportDecision:
        # 1. Reports that can never train are terminal on sight.
        gate = context.eligibility()
        if gate is not None and gate.outcome is Outcome.NEVER:
            return gate
        score = context.score
        assert score is not None
        # 2. A single-call report below min_score is terminal before its
        #    reference resolves; owning no source, it releases only itself.
        if len(context.references) == 1 and score < self._min_score:
            return NEVER
        # 3. Park until every referenced inference has arrived.
        if gate is not None:
            return gate
        # 4. Assemble one sample from the resolved records.
        sample = self._assembly.build(context, score)
        if sample is None or score < self._min_score:
            return NEVER
        return ReportDecision.train(sample)

    def make_batch(self, units, batch_number: int) -> PolicyBatch:
        return PolicyBatch(
            f"{self.scenario}:threshold:{batch_number}",
            tuple(unit.candidates[0].value for unit in units),
        )
