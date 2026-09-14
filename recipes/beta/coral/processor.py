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
      ever form a group; ``agent`` groups one agent's attempts under one
      policy release, so multi-agent runs compare each agent's own lineage
      instead of mixing agents that started from different commits.
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
        if self.group_by not in ("parent", "release", "agent"):
            raise ValueError("group_by must be 'parent', 'release' or 'agent'")
        config.setdefault("accept_multi_turn_policy_samples", True)
        # Qwen3-style templates re-render the previous assistant turn a few
        # tokens differently (empty think block); that is masked scaffold,
        # not a fork.
        config.setdefault("scaffold_tolerance", 8)
        assembly_config = context.with_config(config)
        self._assembly = SampleAssembly.from_config(assembly_config)
        self._mixed_release_groups: dict[str, tuple[str, ...]] = {}
        self._terminal_call_fallbacks = 0
        self._fallback_calls_kept = 0
        self._fallback_calls_total = 0
        self._release_truncations = 0
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
            # The policy was updated while this attempt was running. Its
            # calls under the earlier release cannot share an importance
            # ratio with the later ones, and a data error here would stop the
            # scenario's training, so keep the calls made under the release
            # that produced the graded submission (a suffix: calls are
            # time-ordered) and train on those.
            latest = context.inferences[-1].artifact_ref.release_id if context.inferences[-1].artifact_ref else None
            kept = tuple(
                inference
                for inference in context.inferences
                if inference.artifact_ref is not None and inference.artifact_ref.release_id == latest
            )
            context = replace(context, inferences=kept)
            release_ids = {latest}
            self._release_truncations += 1
        score = context.require_score()
        try:
            sample = self._assembly.build(context, score)
        except ValueError:
            if len(context.inferences) < 2:
                raise
            sample = self._longest_linear_suffix(context, score)
        release_id = next(iter(release_ids), None)
        return sample.with_metadata(
            coral={**coral, "group": self._group_key(coral, release_id), "release_id": release_id}
        )

    def _longest_linear_suffix(self, context: ReportContext, score: float) -> TrajectoryItem:
        """Assemble the longest run of calls, ending at the graded one, that is one linear episode.

        A coding agent's call sequence is not always one prompt extension: the
        runtime compacts history, re-renders the system prompt, and interleaves
        small helper calls, and the shared assembler treats each of those as a
        fork. Whether a suffix assembles is monotone in its length, so binary
        search the longest one. The final call always assembles alone.
        """
        inferences = context.inferences
        low, high = 1, len(inferences)  # low assembles (single call); high does not (just failed)
        while high - low > 1:
            mid = (low + high) // 2
            try:
                self._assembly.build(replace(context, inferences=inferences[-mid:]), score)
                low = mid
            except ValueError:
                high = mid
        sample = self._assembly.build(replace(context, inferences=inferences[-low:]), score)
        self._terminal_call_fallbacks += 1
        self._fallback_calls_kept += low
        self._fallback_calls_total += len(inferences)
        return sample

    def _group_key(self, coral: Mapping[str, Any], release_id: str | None) -> str:
        if self.group_by == "release":
            # A single agent evolving one lineage produces a chain, not a
            # tree: every attempt has a different parent, so parent groups
            # never reach group_size. Grouping by the policy release that
            # produced the attempts gives the TTT-Discover comparison set
            # instead: scored attempts at the same problem under one policy.
            return f"release:{release_id or 'none'}"
        if self.group_by == "agent":
            return f"agent:{coral.get('agent_id', 'unknown')}:{release_id or 'none'}"
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
            # Attempts whose call sequence could not be linearized as a whole
            # and were trained on their longest linear suffix instead, and
            # how many of those attempts' calls that suffix kept.
            "terminal_call_fallbacks": self._terminal_call_fallbacks,
            "fallback_calls_kept": self._fallback_calls_kept,
            "fallback_calls_total": self._fallback_calls_total,
            # Attempts that spanned a weight update and were cut to the calls
            # made under the release that produced the graded submission.
            "release_truncations": self._release_truncations,
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
