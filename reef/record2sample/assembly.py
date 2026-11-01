"""Reported feedback: assemble a report and its inference records into a policy sample.

:class:`~reef.record2sample.ReportSamples` forms a set for each valid report that
trains, and a recipe's ``make_sample`` turns it into a sample. ``SampleAssembly``
is the shared assembly that those recipes use in ``make_sample``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.core.config import config_option
from reef.core.job_data import TrajectorySample
from reef.core.records_types import AgentRecord
from reef.core.trajectories import make_trajectory
from reef.record2sample.base import Record2SampleSettings
from reef.record2sample.reports import ReportContext
from reef.record2sample.samples import make_multi_turn_policy_trajectory, make_policy_trajectory

logger = logging.getLogger(__name__)


def reported_task(metadata: object) -> dict[str, Any] | None:
    """The task a report names, so a recipe can group samples by task; None when the report names none.

    ``metadata.task`` is a mapping with at least a ``name`` (the rollout runner also sends ``path`` and
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
class SampleAssemblySettings(Record2SampleSettings):
    """The sample-assembly keys of a record-to-sample stage's config, with ``max_staleness``. A stage's settings
    extend this class."""

    accept_multi_turn_policy_samples: bool = config_option(
        False,
        help=(
            "Train a report's trajectory of several model calls as one multi-turn sample; otherwise the report "
            "trains only the last call of its trajectory."
        ),
    )
    realign_threshold: int = config_option(
        1024, help="Tokens a multi-turn sample may realign where a later prompt rewrote an earlier response."
    )
    scaffold_tolerance: int = config_option(
        0, help="Masked tokens the assembly may realign before a response, for a thinking template's scaffold."
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.realign_threshold < 0:
            raise ValueError("realign_threshold must be non-negative")
        if self.scaffold_tolerance < 0:
            raise ValueError("scaffold_tolerance must be non-negative")


class SampleAssembly:
    """Shape a resolved report into a policy sample, without recipe policy.

    ``build`` makes the ATIF sample of a report that serving recorded, which the training component renders: the
    trajectory's last call, or with ``accept_multi_turn_policy_samples`` every call. ``assemble`` builds the tensor
    sample of the calls of a training component's own task rollout, which carry the tokens that its engine sampled:
    one call becomes one sample (``single_sample``), and the ordered calls of a trajectory one multi-turn sample,
    realigned within ``realign_threshold`` and ``scaffold_tolerance``. An unassemblable trajectory raises a training
    data error. A recipe whose single sample carries more fields overrides ``single_sample``.
    """

    def __init__(self, settings: SampleAssemblySettings) -> None:
        self.settings = settings

    def assemble(self, inferences: tuple[AgentRecord, ...], score: float, *, source_id: str) -> TrajectorySample:
        """Assemble ordered model calls and their score into one sample, raising if they cannot be used.

        ``source_id`` names what the calls came from: a report, or a rollout, whose calls are
        in-memory values that Reef never stored.
        """
        if len(inferences) == 1:
            sample: TrajectorySample | None = self.single_sample(inferences[0], score)
        else:
            sample = make_multi_turn_policy_trajectory(
                inferences,
                score,
                source_agent_record_id=source_id,
                realign_threshold=self.settings.realign_threshold,
                scaffold_tolerance=self.settings.scaffold_tolerance,
            )
        if sample is None:
            raise ValueError("training data cannot assemble the recorded multi-turn trajectory")
        if sample.training.get("turn_count", 1) > 1 and not self.settings.accept_multi_turn_policy_samples:
            raise ValueError("training data requires accept_multi_turn_policy_samples for this trajectory")
        return sample

    def single_sample(self, record: AgentRecord, score: float) -> TrajectorySample:
        """One model call as one sample: the shared tensor reader."""
        return make_policy_trajectory(record, score)

    def build(self, context: ReportContext, score: float) -> TrajectorySample:
        """The report's trajectory as one ATIF sample with the reward ``score``, which the training component renders.

        Without ``accept_multi_turn_policy_samples``, the sample is the trajectory's last call: its earlier turns are
        context (``is_copied_context``), and only its reply trains. With it, the sample is every call of the report's
        trajectories, and every reply trains. ``ValueError``: a report of several trajectories without
        ``accept_multi_turn_policy_samples``.
        """
        if self.settings.accept_multi_turn_policy_samples:
            calls = context.inferences
        elif len(context.ends) == 1:
            calls = context.ends
        else:
            raise ValueError(
                "training data requires accept_multi_turn_policy_samples for a report of several trajectories"
            )
        return self.with_report(make_trajectory(calls, score), context)

    def with_report(self, sample: TrajectorySample, context: ReportContext) -> TrajectorySample:
        """``sample`` with the report's feedback, ID, references and task in its metadata."""
        fields: dict[str, Any] = {
            "feedback": context.report.payload.get("feedback"),
            "report_agent_record_id": context.report.agent_record_id,
            "references": list(context.references),
        }
        task = reported_task(context.report.payload.get("metadata"))
        if task is not None:
            fields["task"] = task
        return sample.with_metadata(**fields)
