"""Shared test-only grouped-pg processor and training objective.

Group-relative pg is the shape a cookbook grouped method would register: a
processor that groups reports by ``metadata.comparison_set`` and an objective
that turns the resulting ``TrainingBatch`` into group-relative
advantages. Both live here rather than in ``reef`` because no bundled recipe
uses them — the grouping machinery they exercise (group keys, slots, the
``ReportGrouping.decide`` barrier, ``TrainingBatch`` through the trainer and the
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
from reef.pipeline_context import PipelineContext
from reef.record2record import GroupDecision, ReportContext, ReportGrouping, ReportRecord2Record
from reef.record2task import ReportTasks
from reef.record2task.assembly import SampleAssembly, SampleAssemblySettings
from reef.task2batch import Task2Batch
from reef.train.algos import StepSignal, TrainingObjective
from reef.train.algos.helpers import next_steps
from reef.train.pipeline import TrainingPipeline
from reef.train.types import TrainDataItem, TrainingBatch, TrajectoryItem, trajectory_groups

GROUPED_PG_OBJECTIVE = "tests.support._grouped_pg:GroupedPolicyObjective"


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


class ComparisonSetGrouping(ReportGrouping):
    """Reports with one ``metadata.comparison_set`` form one group, ready at two members."""

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        return _comparison_set_id(context.report), None

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        return GroupDecision.READY if len(members) >= 2 else GroupDecision.INCOMPLETE


class ComparisonSetTasks(ReportTasks):
    """One report as one sample in its comparison set."""

    def __init__(self, settings: SampleAssemblySettings | None = None) -> None:
        self.assembly = SampleAssembly(SampleAssemblySettings() if settings is None else settings)

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        set_id = _comparison_set_id(context.report)
        if set_id is None:
            raise ValueError("grouped training requires metadata.comparison_set")
        return replace(self.assembly.build(context, context.require_score()), group_id=set_id)


class GroupedPolicyBatches(Task2Batch):
    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_id: str) -> TrainingBatch:
        return TrainingBatch(batch_id, tasks)


def grouped_policy_pipeline(context: PipelineContext, batch_size: int = 1) -> TrainingPipeline:
    """Reports grouped by ``metadata.comparison_set``, two to a group."""
    return TrainingPipeline(
        context,
        record2record=ReportRecord2Record(context, ComparisonSetGrouping()),
        record2task=ComparisonSetTasks(),
        task2batch=GroupedPolicyBatches(context, batch_size),
    )
