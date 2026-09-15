"""The Reasoning Agent's training data: the task player's reports grouped by task, a batch of complete groups.

Every plain episode the task player reports names its task under ``metadata.task``; the processor
groups the episodes of one task, a group is complete at ``rollouts_per_task`` episodes, and a batch
holds ``tasks_per_step`` complete groups, in arrival order. The preparer turns each group into
group relative advantages, so the Reasoning Agent learns from the tasks the Designer wrote for it.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import replace

from reef.core import AgentRecord
from reef.train.processors.reported import GroupDecision, ReportContext, ReportedFeedbackProcessor, SampleAssembly
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem

DEFAULT_TASKS_PER_STEP = 4
DEFAULT_ROLLOUTS_PER_TASK = 4
# Qwen3 with thinking off ends the generation prompt with an empty think block of four tokens.
DEFAULT_SCAFFOLD_TOLERANCE = 8


def reported_task_name(report: AgentRecord) -> str | None:
    """The task a report names under ``metadata.task.name``, the way the task player sends it."""
    metadata = report.payload.get("metadata")
    task = metadata.get("task") if isinstance(metadata, Mapping) else None
    name = task.get("name") if isinstance(task, Mapping) else None
    return name if isinstance(name, str) and name else None


class SpadeProcessor(ReportedFeedbackProcessor):
    """Groups the Reasoning Agent's episodes by the task they played; a batch is ``tasks_per_step`` complete groups."""

    output_schema = TrainingBatch
    exclusive_sources = True

    def __init__(self, context: ProcessorContext) -> None:
        config = dict(context.config)
        self.tasks_per_step = int(config.get("tasks_per_step", DEFAULT_TASKS_PER_STEP))
        self.rollouts_per_task = int(config.get("rollouts_per_task", DEFAULT_ROLLOUTS_PER_TASK))
        if self.tasks_per_step <= 0:
            raise ValueError("tasks_per_step must be positive")
        if self.rollouts_per_task < 2:
            raise ValueError("rollouts_per_task must be at least two: a group of one has no relative reward")
        # An agent's episode is many model calls that extend one conversation; the sample spans them.
        config.setdefault("accept_multi_turn_policy_samples", True)
        config.setdefault("scaffold_tolerance", DEFAULT_SCAFFOLD_TOLERANCE)
        self._assembly = SampleAssembly.from_config(context.with_config(config))
        # One unit of the batch is one complete task group.
        super().__init__(context.with_config({**config, "batch_size": self.tasks_per_step}))

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        task_name = reported_task_name(context.report)
        if task_name is None:
            raise ValueError("SPADE training requires metadata.task.name on the report, as the task player sends it")
        sample = self._assembly.build(context, context.require_score())
        return replace(sample, group_id=task_name)

    def grouping(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        return reported_task_name(context.report), None

    def decide_group(self, key: Hashable, items: tuple[TrainDataItem, ...]) -> GroupDecision:
        del key
        return GroupDecision.READY if len(items) >= self.rollouts_per_task else GroupDecision.INCOMPLETE

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:spade:{batch_number}", items)
