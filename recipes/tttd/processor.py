"""TTT-Discover's pipeline stages: the step barrier, the grouped policy samples, and the step data.

``TTTDRecord2Record`` owns the step grid (``groups_per_step`` x
``rollouts_per_group``). ``TTTDSamples`` and ``TTTDJobData`` read the grid from
the record stage that they receive, so the deployment writes it once, in
``pipeline.record2record.config``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Self

from reef.core.config import config_option
from reef.core.plugins import PluginConfigError
from reef.core.trajectories import trajectory_reward
from reef.pipeline_context import PipelineContext
from reef.record2record import (
    GroupDecision,
    Record2Record,
    Record2RecordSettings,
    ReportContext,
    ReportGrouping,
    ReportRecord2Record,
)
from reef.record2sample import Record2Sample, ReportSamples
from reef.record2sample.assembly import SampleAssembly, SampleAssemblySettings
from reef.sample2job import Sample2Job, Sample2JobSettings
from reef.train.types import JobData, Sample, TrajectorySample, trajectories

logger = logging.getLogger(__name__)


def inference_release_id(context: ReportContext) -> str | None:
    """The release that served the report's first inference, if the record names one."""
    inference = context.inferences[0]
    return inference.artifact_ref.release_id if inference.artifact_ref is not None else None


@dataclass(frozen=True)
class TTTDGridSettings(Record2RecordSettings):
    """The config of ``TTTDRecord2Record``: the step grid, and the staleness limit.

    Values: ``groups_per_step`` is a positive int; ``rollouts_per_group`` is an int of at least two;
    ``max_staleness`` is as in ``Record2RecordSettings``. The constructor refuses other values.
    Purpose: ``TTTDRecord2Record.from_config`` parses it from ``pipeline.record2record.config``.
    """

    groups_per_step: int = config_option(8, help="The comparison groups in one step.")
    rollouts_per_group: int = config_option(
        64, help="The sibling rollouts in each group; at least two, since a group of one has no relative reward."
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.groups_per_step <= 0:
            raise ValueError("groups_per_step must be positive")
        if self.rollouts_per_group < 2:
            raise ValueError("rollouts_per_group must be at least two")


class TTTDGrouping(ReportGrouping):
    """One TTTD step is one group, and its (group, rollout) coordinates are the slots.

    A step is ready when every one of its ``groups_per_step`` x
    ``rollouts_per_group`` slots is filled. A complete step whose reports span
    releases is discarded wholesale, because mixed-version rewards are not
    comparable.
    """

    def __init__(self, groups_per_step: int, rollouts_per_group: int) -> None:
        self.groups_per_step = groups_per_step
        self.rollouts_per_group = rollouts_per_group
        self.failed_step_versions: dict[int, tuple[str, ...]] = {}

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        parsed: Any = context.parsed_report
        return parsed.step, (parsed.group, parsed.rollout)

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        # 1. The barrier: a step batches only when every one of its
        #    groups_per_step x rollouts_per_group slots is filled.
        expected_count = self.groups_per_step * self.rollouts_per_group
        if len(members) != expected_count:
            return GroupDecision.INCOMPLETE
        # 2. A step whose reports span releases is discarded
        #    wholesale — mixed-version rewards are not comparable.
        versions = {release_id for member in members if (release_id := inference_release_id(member)) is not None}
        if len(versions) <= 1:
            return GroupDecision.READY
        if not isinstance(key, int):
            raise TypeError(f"TTTD step key must be an integer, got {key!r}")
        step = key
        ordered_versions = tuple(sorted(versions))
        self.failed_step_versions[step] = ordered_versions
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
                for step, versions in sorted(self.failed_step_versions.items())
            ]
        }


class TTTDRecord2Record(ReportRecord2Record):
    """TTTD's record stage: one group per step, in step order; a terminal report owns its sources.

    Config: ``groups_per_step`` and ``rollouts_per_group``, the step grid, and ``max_staleness``.
    The later stages read the grid from this stage.
    """

    def __init__(self, context: PipelineContext, settings: TTTDGridSettings) -> None:
        #: The comparison groups in one step.
        self.groups_per_step = settings.groups_per_step
        #: The sibling rollouts in each group.
        self.rollouts_per_group = settings.rollouts_per_group
        super().__init__(
            context,
            TTTDGrouping(settings.groups_per_step, settings.rollouts_per_group),
            exclusive_sources=True,
            ordered_groups=True,
            max_staleness=settings.max_staleness,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        return cls(context, TTTDGridSettings.parse(config, plugin_class=cls))


class TTTDSamples(ReportSamples):
    """One rollout as one sample, checked against the record stage's step grid, with its grid coordinates.

    Config: the sample-assembly keys of ``SampleAssemblySettings``.
    """

    def __init__(self, settings: SampleAssemblySettings, record2record: TTTDRecord2Record) -> None:
        super().__init__()
        self.groups_per_step = record2record.groups_per_step
        self.rollouts_per_group = record2record.rollouts_per_group
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
        settings = SampleAssemblySettings.parse(config, plugin_class=cls)
        if not isinstance(record2record, TTTDRecord2Record):
            raise PluginConfigError(
                f"{cls.__name__} needs TTTDRecord2Record as record2record, not {type(record2record).__name__}"
            )
        return cls(settings, record2record)

    def make_sample(self, context: ReportContext) -> TrajectorySample:
        parsed: Any = context.parsed_report
        if parsed is None:
            raise ValueError("TTTD requires the recipe's report schema")
        if parsed.groups_per_step != self.groups_per_step or parsed.rollouts_per_group != self.rollouts_per_group:
            raise ValueError(
                f"report announces a {parsed.groups_per_step}x{parsed.rollouts_per_group} grid; "
                f"this scenario trains on {self.groups_per_step}x{self.rollouts_per_group}"
            )
        sample = self.assembly.build(context, context.require_score())
        return sample.with_metadata(
            tttd={
                "step": parsed.step,
                "group": parsed.group,
                "rollout": parsed.rollout,
                "release_id": inference_release_id(context),
            }
        )


class TTTDJobData(Sample2Job):
    """One complete TTTD step, one job's data: the step's grid, without its constant-reward groups.

    TTT-Discover samples ``groups_per_step`` parents and exactly
    ``rollouts_per_group`` siblings from each parent.  A partially received
    group is not a valid relative-reward group, and groups from different
    policy steps must never be mixed.  The whole step is therefore one record
    set — its barrier is the recipe's ``TTTDGrouping``, its (group, rollout)
    coordinates are candidate slots (a retried report at an occupied
    coordinate is terminal; the first durable report wins), and a step whose
    reports span releases is discarded wholesale with every row
    released.

    Constant-reward groups are removed only after the complete step is
    present, matching ``remove_constant_reward_groups`` in the reference
    implementation. When every group is constant, the first group is kept so
    the backend performs a well-defined zero-gradient step, also matching the
    reference behavior.

    The grid comes from the record stage. ``max_job_samples`` is the grid,
    so the trainer refuses a larger ``job_samples``, and one job's data is
    exactly one step. Config: ``buffer_size`` and ``job_samples``
    (``Sample2JobSettings``); set ``job_samples`` to the grid.
    """

    def __init__(
        self,
        context: PipelineContext,
        record2record: TTTDRecord2Record,
        buffer_size: int | None,
        job_samples: int | None,
    ) -> None:
        self.groups_per_step = record2record.groups_per_step
        self.rollouts_per_group = record2record.rollouts_per_group
        super().__init__(
            context, buffer_size, job_samples, max_job_samples=self.groups_per_step * self.rollouts_per_group
        )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        record2sample: Record2Sample,
        environ: Mapping[str, str],
    ) -> Self:
        settings = Sample2JobSettings.parse(config, plugin_class=cls)
        if not isinstance(record2record, TTTDRecord2Record):
            raise PluginConfigError(
                f"{cls.__name__} needs TTTDRecord2Record as record2record, not {type(record2record).__name__}"
            )
        return cls(context, record2record, settings.buffer_size, settings.job_samples)

    def make_job_data(self, items: tuple[Sample, ...], data_id: str) -> JobData:
        steps = {item.metadata["tttd"]["step"] for item in items}
        if len(steps) != 1:
            raise RuntimeError("TTTDJobData creates exactly one complete step per job")
        step = next(iter(steps))
        data = JobData(data_id, items)
        # Lay the accepted rollouts back out as the step's grid.
        samples = {
            (item.metadata["tttd"]["group"], item.metadata["tttd"]["rollout"]): item for item in trajectories(data)
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
            data,
            samples=tuple(
                replace(sample, group_id=str(index)) for index, group in enumerate(training_groups) for sample in group
            ),
        )
