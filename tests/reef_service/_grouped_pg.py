"""Shared test-only grouped-pg processor and training objective.

Group-relative pg is the shape a cookbook grouped method would register: a
processor that groups reports by ``metadata.comparison_set`` and an objective
that turns the resulting ``TrainingBatch`` into group-relative
advantages. Both live here rather than in ``reef`` because no bundled recipe
uses them — the grouping machinery they exercise (group keys, slots, the
``decide_group`` barrier, ``TrainingBatch`` through the trainer and the
runtime) IS production code, and these suites are what keep it pinned.

``GROUPED_PG_OBJECTIVE`` is the dotted ``module:Objective`` spelling the
resolver (``reef.train.algos.registry.resolve_objective``) accepts, so tests exercise
the same custom-objective path a reader's package would.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import replace
from typing import Any

from reef.core.records_types import AgentRecord
from reef.core.trajectories import trajectory_reward
from reef.train.algos import StepSignal, TrainingObjective
from reef.train.algos.helpers import next_steps
from reef.train.processors.reported import GroupDecision, ReportContext, ReportedFeedbackProcessor, SampleAssembly
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem, trajectory_groups

GROUPED_PG_OBJECTIVE = "reef_service._grouped_pg:GroupedPolicyObjective"


class GroupedPolicyObjective(TrainingObjective):
    name = "test-grouped-pg"
    loss_family = "pg"

    def prepare(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        advantages: list[float] = []
        for comparison_set in trajectory_groups(batch):
            rewards = [trajectory_reward(sample) for sample in comparison_set]
            mean = sum(rewards) / len(rewards)
            std = (sum((reward - mean) ** 2 for reward in rewards) / len(rewards)) ** 0.5
            advantages.extend((reward - mean) / std if std else 0.0 for reward in rewards)
        steps = next_steps(state)
        normalized = tuple(advantages)
        return StepSignal("train", {"steps": steps}, {"advantages": normalized}, normalized)


def _comparison_set_id(report: AgentRecord) -> str | None:
    metadata = report.payload.get("metadata", {})
    set_id = metadata.get("comparison_set") if isinstance(metadata, Mapping) else None
    return set_id if isinstance(set_id, str) and set_id else None


class GroupedPolicyProcessor(ReportedFeedbackProcessor):
    output_schema = TrainingBatch

    def __init__(self, context: ProcessorContext) -> None:
        self._assembly = SampleAssembly.from_config(context)
        super().__init__(context)

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        set_id = _comparison_set_id(context.report)
        if set_id is None:
            raise ValueError("grouped training requires metadata.comparison_set")
        return replace(self._assembly.build(context, context.require_score()), group_id=set_id)

    def grouping(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        return _comparison_set_id(context.report), None

    def decide_group(self, key: Hashable, items: tuple[TrainDataItem, ...]) -> GroupDecision:
        del key
        return GroupDecision.READY if len(items) >= 2 else GroupDecision.INCOMPLETE

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:grouped:{batch_number}", items)
