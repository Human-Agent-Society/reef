"""CORAL attempts -> grouped policy batches (the training-semantics piece).

CORAL's attempt tree already has the shape reef's grouped relative-reward
training wants: every attempt names its ``parent_hash``, so the siblings of
one parent form one comparison group — attempts that started from the same
code and diverged. The record stage (``CoralRecord2Record`` in
:mod:`recipes.beta.coral.recipe`) groups reports by parent commit and releases
a group as one training unit once ``group_size`` scored siblings accrued.

The stages consume exactly the reports that :mod:`recipes.beta.coral.reporter` emits:
``score``, ordered inference ``references``, and ``metadata.coral`` with
``agent_id``/``commit_hash``/``parent_hash``. Root attempts (no parent)
compare against each other under a sentinel group — they all diverged from
the task seed.

Token/loss-mask/logprob materialization is reef's ``SampleAssembly`` over the
referenced INFERENCE records; a CORAL attempt is a multi-call trajectory, so
``CoralTasks`` turns multi-turn assembly on by default.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Self

from reef.core.config import config_option
from reef.core.plugins import NoSettings
from reef.core.trajectories import trajectory_reward
from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record, ReportContext
from reef.record2task import Record2Task, ReportTasks
from reef.record2task.assembly import SampleAssembly, SampleAssemblySettings
from reef.task2batch import Task2Batch
from reef.train.types import TrainDataItem, TrainingBatch, TrajectoryItem, trajectories

logger = logging.getLogger(__name__)

#: Group key for attempts without a parent commit: first-generation attempts
#: all diverged from the task seed, so they are each other's siblings.
ROOT_GROUP = "__coral_root__"


def coral_metadata(context: ReportContext) -> Mapping[str, Any] | None:
    """The report's ``metadata.coral``, or None when it has no commit hash."""
    metadata = context.report.payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    coral = metadata.get("coral")
    if not isinstance(coral, Mapping):
        return None
    if not isinstance(coral.get("commit_hash"), str) or not coral["commit_hash"]:
        return None
    return coral


def sibling_group(context: ReportContext) -> tuple[str, str]:
    """The attempt's sibling group (its parent commit, or ``ROOT_GROUP``) and its own commit hash."""
    coral = coral_metadata(context)
    if coral is None:
        raise ValueError("CoralBatches requires metadata.coral with a commit_hash")
    parent = coral.get("parent_hash")
    return (parent if isinstance(parent, str) and parent else ROOT_GROUP), coral["commit_hash"]


def attempt_release_ids(context: ReportContext) -> set[str]:
    """The releases that served the attempt's inference records."""
    return {
        inference.artifact_ref.release_id for inference in context.inferences if inference.artifact_ref is not None
    }


@dataclass(frozen=True)
class CoralAssemblySettings(SampleAssemblySettings):
    """The config of ``CoralTasks``: the sample-assembly keys, with multi-turn samples on by default."""

    accept_multi_turn_policy_samples: bool = config_option(
        True, help="Train on an attempt that references several model calls as one multi-turn sample."
    )


class CoralTasks(ReportTasks):
    """One scored attempt as one task, with its CORAL metadata, sibling group, and release.

    Config: the sample-assembly keys of ``CoralAssemblySettings``.
    """

    def __init__(self, settings: CoralAssemblySettings) -> None:
        self.assembly = SampleAssembly(settings)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        environ: Mapping[str, str],
    ) -> Self:
        return cls(CoralAssemblySettings.parse(config, plugin_class=cls))

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        coral = coral_metadata(context)
        if coral is None:
            raise ValueError("CoralBatches requires metadata.coral with a commit_hash")
        sample = self.assembly.build(context, context.require_score())
        release_ids = attempt_release_ids(context)
        if len(release_ids) > 1:
            raise ValueError(f"attempt {coral['commit_hash'][:12]} spans releases {sorted(release_ids)}")
        return sample.with_metadata(
            coral={**coral, "group": sibling_group(context)[0], "release_id": next(iter(release_ids), None)}
        )


class CoralBatches(Task2Batch):
    """One CORAL sibling group, one batch: the siblings of one parent are one relative-reward group.

    The record stage decides when a group is complete (``group_size`` in
    ``pipeline.record2record.config``). This stage takes no config.
    """

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        record2task: Record2Task,
        environ: Mapping[str, str],
    ) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls(context)

    def __init__(self, context: PipelineContext) -> None:
        # One batch is one sibling group.
        super().__init__(context, 1)

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_id: str) -> TrainingBatch:
        parents = {item.metadata["coral"]["group"] for item in items}
        if len(parents) != 1:
            raise RuntimeError("CoralBatches batches exactly one sibling group per step")
        parent = next(iter(parents))
        batch = TrainingBatch(batch_id, items)
        group = trajectories(batch)
        rewards = tuple(trajectory_reward(sample) for sample in group)
        if all(reward == rewards[0] for reward in rewards[1:]):
            # Constant-reward group: keep it for a well-defined zero-gradient
            # step rather than starving the barrier (mirrors TTTD's fallback).
            logger.info("CORAL group %s has constant reward %s", parent, rewards[0])
        self.experiment_logger.log(
            {
                "parent": parent,
                "siblings": len(group),
                "reward_min": min(rewards),
                "reward_max": max(rewards),
            },
            namespace="coral",
        )
        return replace(batch, items=tuple(replace(item, group_id=str(parent)) for item in group))
