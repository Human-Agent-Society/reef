"""CORAL attempts -> grouped policy batches (the training-semantics piece).

CORAL's attempt tree already has the shape reef's grouped relative-reward
training wants: every attempt names its ``parent_hash``, so the siblings of
one parent form one comparison group — attempts that started from the same
code and diverged. This processor groups reports by parent commit and
releases a group as one training unit once ``group_size`` scored siblings
accrued.

The reports it consumes are exactly what :mod:`recipes.beta.coral.reporter` emits:
``score``, ordered inference ``references``, and ``metadata.coral`` with
``agent_id``/``commit_hash``/``parent_hash``. Root attempts (no parent)
compare against each other under a sentinel group — they all diverged from
the task seed.

Token/loss-mask/logprob materialization is reef's ``SampleAssembly`` over the
referenced INFERENCE records; a CORAL attempt is a multi-call trajectory, so
multi-turn assembly is on by default.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping
from dataclasses import replace
from typing import Any

from reef.core.trajectories import trajectory_reward
from reef.train.processors.reported import GroupDecision, ReportContext, ReportedFeedbackProcessor, SampleAssembly
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem, trajectories

logger = logging.getLogger(__name__)

#: Group key for attempts without a parent commit: first-generation attempts
#: all diverged from the task seed, so they are each other's siblings.
ROOT_GROUP = "__coral_root__"


class CoralProcessor(ReportedFeedbackProcessor):
    """One CORAL sibling group = one grouped relative-reward training unit.

    Config:

    - ``group_size`` (default 4, min 2): scored siblings required before a
      parent's group trains. CORAL sibling counts are dynamic, so this is a
      recipe-level barrier, not a CORAL invariant; parents that never accrue
      enough scored children simply never train.
    - ``group_by`` (default ``parent``): ``parent`` groups scored siblings of
      one parent commit; ``release`` groups scored attempts produced under one
      policy release, which is what a single-agent linear evolution needs to
      ever form a group.
    """

    output_schema = TrainingBatch
    exclusive_sources = True
    ordered_groups = False  # sibling groups close in whatever order grading lands

    def __init__(self, context: ProcessorContext) -> None:
        config = dict(context.config)
        self.group_size = int(config.get("group_size", 4))
        if self.group_size < 2:
            raise ValueError("group_size must be at least two (relative rewards need contrast)")
        self.group_by = str(config.get("group_by", "parent"))
        if self.group_by not in ("parent", "release"):
            raise ValueError("group_by must be 'parent' or 'release'")
        config.setdefault("accept_multi_turn_policy_samples", True)
        assembly_config = context.with_config(config)
        self._assembly = SampleAssembly.from_config(assembly_config)
        self._mixed_release_groups: dict[str, tuple[str, ...]] = {}
        self._terminal_call_fallbacks = 0
        super().__init__(context.with_config({**config, "batch_size": 1}))

    @staticmethod
    def _coral_metadata(context: ReportContext) -> Mapping[str, Any] | None:
        metadata = context.report.payload.get("metadata")
        if not isinstance(metadata, Mapping):
            return None
        coral = metadata.get("coral")
        if not isinstance(coral, Mapping):
            return None
        if not isinstance(coral.get("commit_hash"), str) or not coral["commit_hash"]:
            return None
        return coral

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        coral = self._coral_metadata(context)
        if coral is None:
            raise ValueError("CoralProcessor requires metadata.coral with a commit_hash")
        release_ids = {
            inference.artifact_ref.release_id for inference in context.inferences if inference.artifact_ref is not None
        }
        if len(release_ids) > 1:
            raise ValueError(f"attempt {coral['commit_hash'][:12]} spans releases {sorted(release_ids)}")
        score = context.require_score()
        try:
            sample = self._assembly.build(context, score)
        except ValueError:
            if len(context.inferences) < 2:
                raise
            # A coding-agent episode is not always one linear prompt
            # extension: the runtime compacts or re-renders history between
            # calls, and the shared assembler treats that as a fork. The last
            # call is the one that produced the graded submission, so train on
            # it alone rather than dropping the attempt.
            sample = self._assembly.build(replace(context, inferences=context.inferences[-1:]), score)
            self._terminal_call_fallbacks += 1
        release_id = next(iter(release_ids), None)
        return sample.with_metadata(
            coral={**coral, "group": self._group_key(coral, release_id), "release_id": release_id}
        )

    def _group_key(self, coral: Mapping[str, Any], release_id: str | None) -> str:
        if self.group_by == "release":
            # A single agent evolving one lineage produces a chain, not a
            # tree: every attempt has a different parent, so parent groups
            # never reach group_size. Grouping by the policy release that
            # produced the attempts gives the TTT-Discover comparison set
            # instead: scored attempts at the same problem under one policy.
            return f"release:{release_id or 'none'}"
        parent = coral.get("parent_hash")
        return parent if isinstance(parent, str) and parent else ROOT_GROUP

    def grouping(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        coral = self._coral_metadata(context)
        if coral is None:
            raise ValueError("CoralProcessor requires metadata.coral with a commit_hash")
        release_ids = {
            inference.artifact_ref.release_id
            for inference in (context.inferences or ())
            if inference.artifact_ref is not None
        }
        return self._group_key(coral, next(iter(release_ids), None)), coral["commit_hash"]

    def decide_group(self, key: Hashable, items: tuple[TrainDataItem, ...]) -> GroupDecision:
        if len(items) < self.group_size:
            return GroupDecision.INCOMPLETE
        versions = {
            item.metadata["coral"]["release_id"] for item in items if item.metadata["coral"]["release_id"] is not None
        }
        if len(versions) <= 1:
            return GroupDecision.READY
        ordered = tuple(sorted(versions))
        self._mixed_release_groups[str(key)] = ordered
        logger.error(
            "CORAL sibling group %s discarded: %d attempts span releases %s",
            key,
            len(items),
            list(ordered),
        )
        return GroupDecision.DISCARD

    def status(self) -> Mapping[str, Any]:
        return {
            "discarded_groups": [
                {"parent": parent, "reason": "mixed_release_ids", "release_ids": list(versions)}
                for parent, versions in sorted(self._mixed_release_groups.items())
            ],
            # Attempts whose call sequence could not be linearized and were
            # trained on their terminal call instead.
            "terminal_call_fallbacks": self._terminal_call_fallbacks,
        }

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        parents = {item.metadata["coral"]["group"] for item in items}
        if len(parents) != 1:
            raise RuntimeError("CoralProcessor batches exactly one sibling group per step")
        parent = next(iter(parents))
        batch = TrainingBatch(f"{self.scenario}:coral:{parent}:{batch_number}", items)
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
