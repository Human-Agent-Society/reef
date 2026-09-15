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

HINT_ARM = "hint"
ROUND_KEY = "round:"
DEFAULT_TASKS_PER_STEP = 4
DEFAULT_ROLLOUTS_PER_TASK = 4
# Qwen3 with thinking off ends the generation prompt with an empty think block of four tokens.
DEFAULT_SCAFFOLD_TOLERANCE = 8


def reported_arm(report: AgentRecord) -> str | None:
    """The arm a report names under ``metadata.arm``; the hint arm is measured, never trained on."""
    metadata = report.payload.get("metadata")
    arm = metadata.get("arm") if isinstance(metadata, Mapping) else None
    return arm if isinstance(arm, str) and arm else None


def reported_round(report: AgentRecord) -> tuple[str, int] | None:
    """The round a report belongs to and how many plain plays that round reports, when the driver stamped them."""
    metadata = report.payload.get("metadata")
    if not isinstance(metadata, Mapping) or "round" not in metadata:
        return None
    label, plays = metadata.get("round"), metadata.get("round_plays")
    if not isinstance(label, str) or not label:
        raise ValueError("SPADE training requires metadata.round to be a non-empty label")
    if isinstance(plays, bool) or not isinstance(plays, int) or plays < 1:
        raise ValueError("SPADE training requires metadata.round_plays, the round's plain plays, at least 1")
    return label, plays


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
        stamped = reported_round(context.report)
        if stamped is not None:
            sample = sample.with_metadata(round=stamped[0], round_plays=stamped[1])
        return replace(sample, group_id=task_name)

    def grouping(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        # A hint arm report is its own group, discarded at once: that releases the episode's records.
        if reported_arm(context.report) == HINT_ARM:
            return (HINT_ARM, context.report.agent_record_id), None
        # A round's plays are one unit, so one step trains on all of them before the weights move.
        stamped = reported_round(context.report)
        if stamped is not None:
            return f"{ROUND_KEY}{stamped[0]}", None
        return reported_task_name(context.report), None

    def decide_group(self, key: Hashable, items: tuple[TrainDataItem, ...]) -> GroupDecision:
        if isinstance(key, tuple) and key and key[0] == HINT_ARM:
            return GroupDecision.DISCARD
        if isinstance(key, str) and key.startswith(ROUND_KEY):
            plays = int(items[0].metadata["round_plays"])
            return GroupDecision.READY if len(items) >= plays else GroupDecision.INCOMPLETE
        return GroupDecision.READY if len(items) >= self.rollouts_per_task else GroupDecision.INCOMPLETE

    def ready(self) -> bool:
        # A complete round is a batch on its own, whatever tasks_per_step says.
        if any(isinstance(key, str) and key.startswith(ROUND_KEY) for key in self.ready_group_keys()):
            return True
        return super().ready()

    def status(self) -> Mapping[str, object]:
        return {**super().status(), **self.group_status()}

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        # The preparer groups contiguous samples by task, so a round's interleaved plays are ordered by task.
        ordered = tuple(sorted(items, key=lambda item: str(item.group_id)))
        return TrainingBatch(f"{self.scenario}:spade:{batch_number}", ordered)
