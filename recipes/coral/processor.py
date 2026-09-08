"""CORAL attempts -> grouped policy batches (the training-semantics piece).

CORAL's attempt tree already has the shape reef's grouped relative-reward
training wants: every attempt names its ``parent_hash``, so the siblings of
one parent form one comparison group — attempts that started from the same
code and diverged. This processor groups reports by parent commit and
releases a group as one training unit once ``group_size`` scored siblings
accrued.

The reports it consumes are exactly what :mod:`reef_coral.reporter` emits:
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
from typing import Any

from reef.train.processors.reported import (
    BatchUnit,
    GroupDecision,
    ReportContext,
    ReportDecision,
    ReportedFeedbackProcessor,
    SampleAssembly,
)
from reef.train.types import GroupedPolicyBatch, ProcessorContext, policy_row_violation

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
    """

    output_schema = GroupedPolicyBatch
    exclusive_sources = True
    ordered_groups = False  # sibling groups close in whatever order grading lands

    def __init__(self, context: ProcessorContext) -> None:
        config = dict(context.config)
        self.group_size = int(config.get("group_size", 4))
        if self.group_size < 2:
            raise ValueError("group_size must be at least two (relative rewards need contrast)")
        config.setdefault("accept_multi_turn_policy_samples", True)
        assembly_config = context.with_config(config)
        self._assembly = SampleAssembly.from_config(assembly_config)
        self._mixed_release_groups: dict[str, tuple[str, ...]] = {}
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

    def judge(self, context: ReportContext) -> ReportDecision:
        coral = self._coral_metadata(context)
        if coral is None:
            return ReportDecision.never("CoralProcessor requires metadata.coral with a commit_hash")
        if (gate := context.eligibility()) is not None:
            return gate
        score = context.score
        if score is None or context.inferences is None:
            raise RuntimeError("eligible CORAL report is not fully resolved")
        try:
            sample = self._assembly.build(context, score)
        except (TypeError, ValueError) as error:
            return ReportDecision.never(f"sample assembly failed: {error}")
        if sample is None or policy_row_violation(sample.tokens, sample.loss_mask, sample.rollout_log_probs):
            return ReportDecision.never("policy tensor contract violation")
        release_ids = {
            inference.artifact_ref.release_id for inference in context.inferences if inference.artifact_ref is not None
        }
        if len(release_ids) > 1:
            # One attempt spanning a weight update cannot carry one coherent
            # set of rollout log probs.
            return ReportDecision.never(f"attempt {coral['commit_hash'][:12]} spans releases {sorted(release_ids)}")
        parent = coral.get("parent_hash")
        group_key = parent if isinstance(parent, str) and parent else ROOT_GROUP
        return ReportDecision.train(
            _CoralRow(sample, next(iter(release_ids), None)),
            group_key=group_key,
            slot=coral["commit_hash"],  # one attempt = one commit; regrade retries collapse
        )

    def decide_group(self, key: Hashable, candidates: tuple[Any, ...]) -> GroupDecision:
        if len(candidates) < self.group_size:
            return GroupDecision.INCOMPLETE
        versions = {candidate.value.release_id for candidate in candidates if candidate.value.release_id is not None}
        if len(versions) <= 1:
            return GroupDecision.READY
        ordered = tuple(sorted(versions))
        self._mixed_release_groups[str(key)] = ordered
        logger.error(
            "CORAL sibling group %s discarded: %d attempts span releases %s",
            key,
            len(candidates),
            list(ordered),
        )
        return GroupDecision.DISCARD

    def status(self) -> Mapping[str, Any]:
        return {
            "discarded_groups": [
                {"parent": parent, "reason": "mixed_release_ids", "release_ids": list(versions)}
                for parent, versions in sorted(self._mixed_release_groups.items())
            ],
            # Why reports did not train — the first thing to look at when a
            # group never releases on a live deployment.
            "never_reasons": dict(self.never_reasons),
        }

    def make_batch(self, units: tuple[BatchUnit, ...], batch_number: int) -> GroupedPolicyBatch:
        if len(units) != 1:
            raise RuntimeError("CoralProcessor batches exactly one sibling group per step")
        unit = units[0]
        group = tuple(candidate.value.sample for candidate in unit.candidates)
        rewards = tuple(sample.reward for sample in group)
        if all(reward == rewards[0] for reward in rewards[1:]):
            # Constant-reward group: keep it for a well-defined zero-gradient
            # step rather than starving the barrier (mirrors TTTD's fallback).
            logger.info("CORAL group %s has constant reward %s", unit.group_key, rewards[0])
        self.experiment_logger.log(
            {
                "parent": unit.group_key,
                "siblings": len(group),
                "reward_min": min(rewards),
                "reward_max": max(rewards),
            },
            namespace="coral",
        )
        return GroupedPolicyBatch(f"{self.scenario}:coral:{unit.group_key}:{batch_number}", (group,))


class _CoralRow:
    """One accepted attempt: its assembled sample plus producing release."""

    __slots__ = ("release_id", "sample")

    def __init__(self, sample: Any, release_id: str | None) -> None:
        self.sample = sample
        self.release_id = release_id
