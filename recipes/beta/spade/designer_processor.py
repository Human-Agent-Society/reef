"""The Designer's training data: one sample per proposal, a generation's proposals as one group, regret as the reward.

Every report a generation sends against a Designer receipt names its generation under ``metadata.generation``
and the generation's proposal count under ``metadata.proposals``; a group is complete when that many reports
arrived, and a batch holds ``generations_per_step`` complete groups, in arrival order. A refused proposal
scores the refusal floor and is a member like any other, so the preparer's group relative advantages tell the Designer which
proposals of one generation landed at the Reasoning Agent's frontier.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import replace

from reef.core import AgentRecord
from reef.train.processors.reported import GroupDecision, ReportContext, ReportedFeedbackProcessor, SampleAssembly
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem

DEFAULT_GENERATIONS_PER_STEP = 1
# What the sample keeps from the report, so a step's record names the versions a regret was measured against.
COPIED_METADATA_KEYS = ("generation", "proposals", "designer_version", "opponent")


def reported_skill(report: AgentRecord) -> str | None:
    """The skill a Designer report names under ``metadata.skill``; proposals of one skill share one prompt."""
    metadata = report.payload.get("metadata")
    skill = metadata.get("skill") if isinstance(metadata, Mapping) else None
    return skill if isinstance(skill, str) and skill else None


def reported_generation(report: AgentRecord) -> tuple[int, int]:
    """The generation a Designer report belongs to and how many proposals that generation made."""
    metadata = report.payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("SPADE Designer training requires metadata.generation and metadata.proposals on the report")
    values = []
    for key, floor in (("generation", 0), ("proposals", 1)):
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < floor:
            raise ValueError(
                f"SPADE Designer training requires metadata.{key} on the report, an integer of at least {floor}"
            )
        values.append(value)
    return values[0], values[1]


def generation_label(generation: int, skill: str | None = None) -> str:
    """The comparison group: one generation's proposals of one skill, named like the generation's manifest."""
    return f"generation-{generation:05d}" if skill is None else f"generation-{generation:05d}-{skill}"


class SpadeDesignerProcessor(ReportedFeedbackProcessor):
    """Groups the Designer's proposals by generation; a batch is ``generations_per_step`` complete generations."""

    output_schema = TrainingBatch
    exclusive_sources = True

    def __init__(self, context: ProcessorContext) -> None:
        config = dict(context.config)
        self.generations_per_step = int(config.get("generations_per_step", DEFAULT_GENERATIONS_PER_STEP))
        if self.generations_per_step <= 0:
            raise ValueError("generations_per_step must be positive")
        # A proposal is one chat call, so the sample is one turn and needs no realignment.
        self._assembly = SampleAssembly.from_config(context.with_config(config))
        # One unit of the batch is one complete generation.
        super().__init__(context.with_config({**config, "batch_size": self.generations_per_step}))

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        generation, _ = reported_generation(context.report)
        metadata = context.report.payload["metadata"]
        copied = {key: metadata[key] for key in COPIED_METADATA_KEYS if key in metadata}
        sample = self._assembly.build(context, context.require_score()).with_metadata(**copied)
        return replace(sample, group_id=generation_label(generation, reported_skill(context.report)))

    def grouping(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        # A generation is one unit, so one step trains on all of its proposals before the weights move.
        generation, _ = reported_generation(context.report)
        return generation, None

    def decide_group(self, key: Hashable, items: tuple[TrainDataItem, ...]) -> GroupDecision:
        del key
        proposals = int(items[0].metadata["proposals"])
        return GroupDecision.READY if len(items) >= proposals else GroupDecision.INCOMPLETE

    def status(self) -> Mapping[str, object]:
        return {**super().status(), **self.group_status()}

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        # Each skill has its own prompt, so proposals compare within a generation and a skill: the group id.
        ordered = tuple(sorted(items, key=lambda item: str(item.group_id)))
        return TrainingBatch(f"{self.scenario}:spade-designer:{batch_number}", ordered)
