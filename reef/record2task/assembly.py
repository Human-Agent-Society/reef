"""Reported feedback: assemble a report and its inference records into a policy sample.

The record-to-record stage (:class:`~reef.record2record.ReportRecord2Record`) forms a
set for each valid report that trains, and a recipe's
:class:`~reef.record2task.ReportTasks` turns it into a task. ``SampleAssembly``
is the shared assembly that those recipes use in ``make_sample``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.core.batches import TrajectoryItem
from reef.core.config import config_option
from reef.core.plugins import PluginSettings
from reef.core.records_types import AgentRecord
from reef.record2record import ReportContext
from reef.record2task.samples import make_multi_turn_policy_trajectory, make_policy_trajectory

logger = logging.getLogger(__name__)


def reported_task(metadata: object) -> dict[str, Any] | None:
    """The task a report names, so a recipe can group samples by task; None when the report names none.

    ``metadata.task`` is a mapping with at least a ``name`` (the task player also sends ``path`` and
    ``digest``); a Harbor report from the shipped harness names the task as ``metadata.harbor.task_name``.
    """
    if not isinstance(metadata, Mapping):
        return None
    task = metadata.get("task")
    if isinstance(task, Mapping) and isinstance(task.get("name"), str) and task["name"]:
        return dict(task)
    harbor = metadata.get("harbor")
    if isinstance(harbor, Mapping) and isinstance(harbor.get("task_name"), str) and harbor["task_name"]:
        return {"name": harbor["task_name"]}
    return None


@dataclass(frozen=True)
class SampleAssemblySettings(PluginSettings):
    """The sample-assembly keys of a record-to-task stage's config. A stage's settings extend this class."""

    accept_multi_turn_policy_samples: bool = config_option(
        False, help="Train on a report that references several model calls as one multi-turn sample."
    )
    realign_threshold: int = config_option(
        1024, help="Tokens a multi-turn sample may realign where a later prompt rewrote an earlier response."
    )
    scaffold_tolerance: int = config_option(
        0, help="Masked tokens the assembly may realign before a response, for a thinking template's scaffold."
    )

    def __post_init__(self) -> None:
        if self.realign_threshold < 0:
            raise ValueError("realign_threshold must be non-negative")
        if self.scaffold_tolerance < 0:
            raise ValueError("scaffold_tolerance must be non-negative")


class SampleAssembly:
    """Shape a resolved report into a policy sample, without recipe policy.

    One model call becomes one sample (``single_sample``); an ordered
    multi-reference report becomes one assembled multi-turn sample. The
    settings' ``accept_multi_turn_policy_samples`` gates consumption only;
    assembly always runs first. An unsupported or unassemblable trajectory
    raises a training data error and retains its source records. A recipe
    whose single sample carries more fields overrides ``single_sample``.
    """

    def __init__(self, settings: SampleAssemblySettings) -> None:
        self.settings = settings

    def single_sample(self, record: AgentRecord, score: float) -> TrajectoryItem:
        """One model call as one sample: the shared tensor reader."""
        return make_policy_trajectory(record, score)

    def build(self, context: ReportContext, score: float) -> TrajectoryItem:
        """Build training data, raising if the recorded trajectory cannot be used."""
        inferences = context.inferences
        if len(inferences) == 1:
            sample: TrajectoryItem | None = self.single_sample(inferences[0], score)
        else:
            sample = make_multi_turn_policy_trajectory(
                inferences,
                score,
                source_agent_record_id=context.report.agent_record_id,
                realign_threshold=self.settings.realign_threshold,
                scaffold_tolerance=self.settings.scaffold_tolerance,
            )
        if sample is None:
            raise ValueError("training data cannot assemble the recorded multi-turn trajectory")
        if sample.training.get("turn_count", 1) > 1 and not self.settings.accept_multi_turn_policy_samples:
            raise ValueError("training data requires accept_multi_turn_policy_samples for this trajectory")
        fields: dict[str, Any] = {
            "feedback": context.report.payload.get("feedback"),
            "report_agent_record_id": context.report.agent_record_id,
            "references": list(context.references),
        }
        task = reported_task(context.report.payload.get("metadata"))
        if task is not None:
            fields["task"] = task
        return sample.with_metadata(**fields)
