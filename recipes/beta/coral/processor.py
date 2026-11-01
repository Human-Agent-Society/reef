"""CORAL attempts -> grouped policy batches (the training-semantics piece).

CORAL's attempt tree already has the shape reef's grouped relative-reward
training wants: every attempt names its ``parent_hash``, so the siblings of
one parent form one comparison group — attempts that started from the same
code and diverged. ``CoralSamples`` groups reports by parent commit
(``CoralGrouping``) and releases a group as one training unit once
``group_size`` scored siblings accrued.

The stages consume exactly the reports that :mod:`recipes.beta.coral.reporter` emits:
``score``, ordered inference ``references``, and ``metadata.coral`` with
``agent_id``/``commit_hash``/``parent_hash``. Root attempts (no parent)
compare against each other under a sentinel group — they all diverged from
the task seed.

Token/loss-mask/logprob materialization is reef's ``SampleAssembly`` over the
referenced INFERENCE records; a CORAL attempt is a multi-call trajectory, so
``CoralSamples`` turns multi-turn assembly on by default.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Self

from reef.core.config import config_option
from reef.core.trajectories import trajectory_reward
from reef.pipeline_context import PipelineContext
from reef.record2sample import GroupDecision, Record2Sample, ReportContext, ReportGrouping, ReportSamples
from reef.record2sample.assembly import SampleAssembly, SampleAssemblySettings
from reef.sample2job import Sample2Job, Sample2JobSettings
from reef.train.types import JobData, Sample, TrajectorySample, trajectories

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
        raise ValueError("CoralJobData requires metadata.coral with a commit_hash")
    parent = coral.get("parent_hash")
    return (parent if isinstance(parent, str) and parent else ROOT_GROUP), coral["commit_hash"]


def attempt_release_ids(context: ReportContext) -> set[str]:
    """The releases that served the attempt's inference records."""
    return {
        inference.artifact_ref.release_id for inference in context.inferences if inference.artifact_ref is not None
    }


class CoralGrouping(ReportGrouping):
    """The siblings of one parent commit form one group, and each attempt's commit hash is its slot.

    A group is ready once ``group_size`` siblings are present. A group whose
    attempts span releases is discarded and listed in ``discarded_groups``.
    """

    def __init__(self, group_size: int) -> None:
        self.group_size = group_size
        self.mixed_release_groups: dict[str, tuple[str, ...]] = {}

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        return sibling_group(context)

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        if len(members) < self.group_size:
            return GroupDecision.INCOMPLETE
        versions = {release_id for member in members for release_id in attempt_release_ids(member)}
        if len(versions) <= 1:
            return GroupDecision.READY
        ordered = tuple(sorted(versions))
        self.mixed_release_groups[str(key)] = ordered
        logger.error(
            "CORAL sibling group %s discarded: %d attempts span releases %s",
            key,
            len(members),
            list(ordered),
        )
        return GroupDecision.DISCARD

    def status(self) -> Mapping[str, Any]:
        return {
            "discarded_groups": [
                {"parent": parent, "reason": "mixed_release_ids", "release_ids": list(versions)}
                for parent, versions in sorted(self.mixed_release_groups.items())
            ],
        }


@dataclass(frozen=True)
class CoralSettings(SampleAssemblySettings):
    """The config of ``CoralSamples``: the size of a sibling group, the sample assembly, and the staleness limit.

    Values: ``group_size`` is an int of at least two; the other keys are as in ``SampleAssemblySettings``, with
    multi-turn samples on by default. The constructor refuses other values.
    Purpose: ``CoralSamples.from_config`` parses it from ``pipeline.record2sample.config``.
    """

    group_size: int = config_option(
        4,
        help=(
            "The scored siblings of one parent commit that complete its group. "
            "A parent with fewer scored children never trains."
        ),
    )
    accept_multi_turn_policy_samples: bool = config_option(
        True, help="Train on an attempt that references several model calls as one multi-turn sample."
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.group_size < 2:
            raise ValueError("group_size must be at least two (relative rewards need contrast)")


class CoralSamples(ReportSamples):
    """One scored attempt as one sample, with its CORAL metadata, sibling group, and release.

    Sibling groups train in the order that they close (``CoralGrouping``); a terminal report owns its sources.
    CORAL sibling counts are dynamic, so the barrier ``group_size`` is a recipe setting, not a CORAL invariant.
    Config: ``CoralSettings``.
    """

    def __init__(self, context: PipelineContext, settings: CoralSettings) -> None:
        super().__init__(
            context, CoralGrouping(settings.group_size), exclusive_sources=True, max_staleness=settings.max_staleness
        )
        self.assembly = SampleAssembly(settings)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        environ: Mapping[str, str],
    ) -> Self:
        return cls(context, CoralSettings.parse(config, plugin_class=cls))

    def make_sample(self, context: ReportContext) -> TrajectorySample:
        coral = coral_metadata(context)
        if coral is None:
            raise ValueError("CoralJobData requires metadata.coral with a commit_hash")
        sample = self.assembly.build(context, context.require_score())
        release_ids = attempt_release_ids(context)
        if len(release_ids) > 1:
            raise ValueError(f"attempt {coral['commit_hash'][:12]} spans releases {sorted(release_ids)}")
        return sample.with_metadata(
            coral={**coral, "group": sibling_group(context)[0], "release_id": next(iter(release_ids), None)}
        )


class CoralJobData(Sample2Job):
    """Complete CORAL sibling groups, one job's data: the siblings of one parent are one relative-reward group.

    ``CoralSamples`` decides when a group is complete (``group_size`` in
    ``pipeline.record2sample.config``); a group is fixed when it is first ready, and later siblings form a new
    group. Job data takes up to ``buffer_size`` samples and the rest of each group that those touch. Config:
    ``buffer_size`` and ``job_samples``.
    """

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2sample: Record2Sample,
        environ: Mapping[str, str],
    ) -> Self:
        settings = Sample2JobSettings.parse(config, plugin_class=cls)
        return cls(context, settings.buffer_size, settings.job_samples)

    def make_job_data(self, items: tuple[Sample, ...], data_id: str) -> JobData:
        data = JobData(data_id, items)
        # The pipeline names each stored group (``group_id``); a parent's later siblings form another group.
        groups: dict[str, list[TrajectorySample]] = {}
        for sample in trajectories(data):
            groups.setdefault(coral_group_id(sample), []).append(sample)
        for parent, group in groups.items():
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
        return replace(
            data,
            samples=tuple(replace(item, group_id=coral_group_id(item)) for item in trajectories(data)),
        )


def coral_group_id(sample: TrajectorySample) -> str:
    """The comparison group of ``sample``: the group that the pipeline stored it in, else its parent."""
    return sample.group_id if sample.group_id is not None else str(sample.metadata["coral"]["group"])
