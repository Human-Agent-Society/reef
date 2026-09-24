"""TTT-Discover's exact step barrier and grouped policy-data processor."""

from __future__ import annotations

import logging
import math
from collections.abc import Hashable, Mapping
from dataclasses import replace
from typing import Any

from reef.core.trajectories import trajectory_reward
from reef.train.processors.reported import GroupDecision, ReportContext, ReportedFeedbackProcessor, SampleAssembly
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem, trajectories

logger = logging.getLogger(__name__)


class TTTDProcessor(ReportedFeedbackProcessor):
    """Wait for one complete TTTD step before exposing a training batch.

    TTT-Discover samples ``groups_per_step`` parents and exactly
    ``rollouts_per_group`` siblings from each parent.  A partially received
    group is not a valid relative-reward group, and groups from different
    policy steps must never be mixed.  The whole step is therefore one engine
    group — its barrier is the ``decide_group`` rule, its (group, rollout)
    coordinates are candidate slots (a retried report at an occupied
    coordinate is terminal; the first durable report wins), and a step whose
    reports span releases is discarded wholesale with every row
    released.

    Constant-reward groups are removed only after the complete step is
    present, matching ``remove_constant_reward_groups`` in the reference
    implementation. When every group is constant, the first group is kept so
    the backend performs a well-defined zero-gradient step, also matching the
    reference behavior.
    """

    output_schema = TrainingBatch
    exclusive_sources = True
    ordered_groups = True  # steps train in order

    def __init__(self, context: ProcessorContext) -> None:
        config = dict(context.config)
        self.groups_per_step = int(config.get("groups_per_step", 8))
        self.rollouts_per_group = int(config.get("rollouts_per_group", 64))
        if self.groups_per_step <= 0:
            raise ValueError("groups_per_step must be positive")
        if self.rollouts_per_group < 2:
            raise ValueError("rollouts_per_group must be at least two")
        self._assembly = SampleAssembly.from_config(context)
        self._failed_step_versions: dict[int, tuple[str, ...]] = {}
        super().__init__(context.with_config({**config, "batch_size": 1}))

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        parsed: Any = context.parsed_report
        if parsed is None:
            raise ValueError("TTTDProcessor requires the recipe's report schema")
        if parsed.groups_per_step != self.groups_per_step or parsed.rollouts_per_group != self.rollouts_per_group:
            raise ValueError(
                f"report announces a {parsed.groups_per_step}x{parsed.rollouts_per_group} grid; "
                f"this scenario trains on {self.groups_per_step}x{self.rollouts_per_group}"
            )
        sample = self._assembly.build(context, context.require_score())
        inference = context.inferences[0]
        release_id = inference.artifact_ref.release_id if inference.artifact_ref is not None else None
        return sample.with_metadata(
            tttd={"step": parsed.step, "group": parsed.group, "rollout": parsed.rollout, "release_id": release_id}
        )

    def grouping(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        parsed: Any = context.parsed_report
        return parsed.step, (parsed.group, parsed.rollout)

    def decide_group(self, key: Hashable, items: tuple[TrainDataItem, ...]) -> GroupDecision:
        # 1. The barrier: a step batches only when every one of its
        #    groups_per_step x rollouts_per_group slots is filled.
        expected_count = self.groups_per_step * self.rollouts_per_group
        if len(items) != expected_count:
            return GroupDecision.INCOMPLETE
        # 2. A step whose reports span releases is discarded
        #    wholesale — mixed-version rewards are not comparable.
        versions = {
            item.metadata["tttd"]["release_id"] for item in items if item.metadata["tttd"]["release_id"] is not None
        }
        if len(versions) <= 1:
            return GroupDecision.READY
        if not isinstance(key, int):
            raise TypeError(f"TTTD step key must be an integer, got {key!r}")
        step = key
        ordered_versions = tuple(sorted(versions))
        self._failed_step_versions[step] = ordered_versions
        logger.error(
            "TTTD step %d failed because its %d reports span releases %s",
            step,
            expected_count,
            list(ordered_versions),
        )
        return GroupDecision.DISCARD

    def status(self) -> Mapping[str, Any]:
        """Expose mixed-version steps that terminally violated TTTD's invariant."""
        return {
            "failed_steps": [
                {
                    "step": step,
                    "reason": "mixed_release_ids",
                    "release_ids": list(versions),
                }
                for step, versions in sorted(self._failed_step_versions.items())
            ]
        }

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        steps = {item.metadata["tttd"]["step"] for item in items}
        if len(steps) != 1:
            raise RuntimeError("TTTDProcessor creates exactly one complete step per batch")
        step = next(iter(steps))
        batch = TrainingBatch(f"{self.scenario}:tttd:{step}", items)
        # Lay the accepted rollouts back out as the step's grid.
        samples = {
            (item.metadata["tttd"]["group"], item.metadata["tttd"]["rollout"]): item for item in trajectories(batch)
        }
        all_groups = tuple(
            tuple(samples[(group, rollout)] for rollout in range(self.rollouts_per_group))
            for group in range(self.groups_per_step)
        )
        # 2. Drop constant-reward groups (no relative signal); when every
        #    group is constant, keep the first for a well-defined
        #    zero-gradient step — both matching the reference implementation.
        non_constant = tuple(
            group
            for group in all_groups
            if any(trajectory_reward(sample) != trajectory_reward(group[0]) for sample in group[1:])
        )
        training_groups = non_constant or all_groups[:1]
        rewards = tuple(trajectory_reward(sample) for group in all_groups for sample in group)
        reward_scale = max(1.0, *(abs(reward) for reward in rewards))
        normalized_rewards = tuple(reward / reward_scale for reward in rewards)
        normalized_mean = math.fsum(normalized_rewards) / len(normalized_rewards)
        reward_mean = normalized_mean * reward_scale
        constant_groups = len(all_groups) - len(non_constant)
        self.experiment_logger.log(
            {
                "step": step,
                "grid_groups": len(all_groups),
                "grid_rollouts": len(rewards),
                "reward_min": min(rewards),
                "reward_max": max(rewards),
                "reward_mean": reward_mean,
                "reward_std": math.sqrt(
                    math.fsum((reward - normalized_mean) ** 2 for reward in normalized_rewards) / len(rewards)
                )
                * reward_scale,
                "reward_zero_fraction": sum(reward == 0 for reward in rewards) / len(rewards),
                "constant_groups": constant_groups,
                "constant_groups_removed": constant_groups - (1 if not non_constant else 0),
                "training_groups": len(training_groups),
                "training_rollouts": sum(len(group) for group in training_groups),
            },
            namespace="tttd",
        )
        return replace(
            batch,
            items=tuple(
                replace(sample, group_id=str(index)) for index, group in enumerate(training_groups) for sample in group
            ),
        )
