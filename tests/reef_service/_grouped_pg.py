"""Shared test-only grouped-pg processor and step preparer.

Group-relative pg is the shape a cookbook grouped method would register: a
processor that groups reports by ``metadata.comparison_set`` and a preparer
that turns the resulting ``GroupedPolicyBatch`` into group-relative
advantages. Both live here rather than in ``reef`` because no bundled recipe
uses them — the grouping machinery they exercise (group keys, slots, the
``decide_group`` barrier, ``GroupedPolicyBatch`` through the trainer and the
runtime) IS production code, and these suites are what keep it pinned.

``GROUPED_PG_PREPARER`` is the dotted ``module:callable`` spelling the
resolver (``reef.train.algos.registry.resolve_preparer``) accepts, so tests exercise
the same custom-preparer path a reader's package would.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

from reef.core.records_types import AgentRecord
from reef.train.algos import StepSignal
from reef.train.algos.helpers import next_steps
from reef.train.processors.reported import (
    NEVER,
    BatchUnit,
    Candidate,
    GroupDecision,
    Outcome,
    ReportContext,
    ReportDecision,
    ReportedFeedbackProcessor,
    SampleAssembly,
)
from reef.train.types import GroupedPolicyBatch, ProcessorContext, TrainingBatch

GROUPED_PG_PREPARER = "reef_service._grouped_pg:prepare_grouped_pg"


def prepare_grouped_pg(batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
    if not isinstance(batch, GroupedPolicyBatch):
        raise TypeError(f"grouped-pg requires GroupedPolicyBatch, got {type(batch).__name__}")
    advantages: list[float] = []
    for comparison_set in batch.comparison_sets:
        rewards = [sample.reward for sample in comparison_set]
        mean = sum(rewards) / len(rewards)
        std = (sum((reward - mean) ** 2 for reward in rewards) / len(rewards)) ** 0.5
        advantages.extend((reward - mean) / std if std else 0.0 for reward in rewards)
    steps = next_steps(state)
    normalized = tuple(advantages)
    return StepSignal("train", "pg", {"steps": steps}, {"advantages": normalized}, normalized)


def _comparison_set_id(report: AgentRecord) -> str | None:
    metadata = report.payload.get("metadata", {})
    set_id = metadata.get("comparison_set") if isinstance(metadata, Mapping) else None
    return set_id if isinstance(set_id, str) and set_id else None


class GroupedPolicyProcessor(ReportedFeedbackProcessor):
    output_schema = GroupedPolicyBatch

    def __init__(self, context: ProcessorContext) -> None:
        self._assembly = SampleAssembly.from_config(context)
        super().__init__(context)

    def judge(self, context: ReportContext) -> ReportDecision:
        # 1. Reports that can never train are terminal on sight.
        gate = context.eligibility()
        if gate is not None and gate.outcome is Outcome.NEVER:
            return gate
        # 2. A single-call report with no comparison set can never group —
        #    terminal before its reference even resolves.
        set_id = _comparison_set_id(context.report)
        if len(context.references) == 1 and set_id is None:
            return NEVER
        # 3. Park until every referenced inference has arrived.
        if gate is not None:
            return gate
        score = context.score
        assert score is not None
        # 4. Assemble the sample and address it into its comparison group.
        sample = self._assembly.build(context, score)
        if sample is None or set_id is None:
            return NEVER
        return ReportDecision.train(sample, group_key=set_id)

    def decide_group(self, key: Hashable, candidates: tuple[Candidate, ...]) -> GroupDecision:
        del key
        return GroupDecision.READY if len(candidates) >= 2 else GroupDecision.INCOMPLETE

    def make_batch(self, units: tuple[BatchUnit, ...], batch_number: int) -> GroupedPolicyBatch:
        return GroupedPolicyBatch(
            f"{self.scenario}:grouped:{batch_number}",
            tuple(tuple(candidate.value for candidate in unit.candidates) for unit in units),
        )
