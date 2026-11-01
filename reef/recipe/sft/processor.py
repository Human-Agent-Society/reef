"""The SFT stages: each report becomes one sample of a recorded request and its target response.

``SFTSamples`` makes each report its own record set, and its sample the recorded request with the target as the final
agent turn, an ATIF trajectory: the training component renders it with the model's chat template, and the loss mask
selects the target tokens. ``SFTJobData`` puts the samples into job data, together with uploaded ATIF trajectories,
which train with every agent turn that is not copied context.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

from reef.core.chat_rendering import trained_turns
from reef.core.config import config_option
from reef.core.job_data import Sample, TrajectorySample
from reef.core.plugins import PluginSettings
from reef.core.reports import TeacherContextReport
from reef.core.trajectories import make_trajectory
from reef.pipeline_context import PipelineContext
from reef.record2sample import ReportContext, ReportSamples
from reef.sample2job import DistillJobData

#: ``demonstration`` trains the report's ``teacher_context``. ``response`` trains the recorded response.
SFT_TARGETS = ("demonstration", "response")


@dataclass(frozen=True)
class SFTSettings(PluginSettings):
    """The config of ``SFTSamples``: the target, and the maximum sample length.

    Values: ``target`` is one of ``SFT_TARGETS``; ``max_tokens`` is a non-negative int, 0 for no limit. The
    constructor refuses other values.
    Purpose: ``SFTSamples.from_config`` parses it from ``pipeline.record2sample.config``.
    """

    target: str = config_option(
        "demonstration",
        help="The response to train: demonstration (the report's teacher_context) or response (the recorded response).",
    )
    max_tokens: int = config_option(
        0, help="The maximum tokens in a sample; a longer sample does not train. 0 disables the limit."
    )

    def __post_init__(self) -> None:
        if self.target not in SFT_TARGETS:
            raise ValueError(f"target must be one of: {', '.join(SFT_TARGETS)}")
        if self.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative (0 disables the limit)")


class SFTSamples(ReportSamples):
    """Make one supervised sample from one report and its recorded request.

    The sample is the report's last call as an ATIF trajectory: its request, then the target as the final agent turn,
    the only turn that trains. With ``target: demonstration``, the target is the report's ``teacher_context`` and the
    stage ignores the recorded response. With ``target: response``, the target is the recorded response with its tool
    calls, for example an imported dataset example. The training component renders the sample with the model's chat
    template; a sample longer than ``max_tokens`` does not train (``extra.reef.max_tokens``).

    Each valid report trains by itself and owns its inference records. The stage never drops a sample as stale:
    the staleness check drops tokens that an older model version sampled, and an SFT target is fixed text, not
    sampled tokens. Thus ``SFTSettings`` has no ``max_staleness``.
    """

    def __init__(self, context: PipelineContext, settings: SFTSettings) -> None:
        super().__init__(context, exclusive_sources=True)
        self.target = settings.target
        self.max_tokens = settings.max_tokens

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        environ: Mapping[str, str],
    ) -> Self:
        return cls(context, SFTSettings.parse(config, plugin_class=cls))

    def is_stale(self, samples: Sequence[Sample]) -> bool:
        return False

    def limited(self, sample: TrajectorySample) -> TrajectorySample:
        """``sample`` with the stage's ``max_tokens``, which the training component applies, when it sets one."""
        return sample.with_metadata(max_tokens=self.max_tokens) if self.max_tokens else sample

    def uploaded_sample(self, trajectory: TrajectorySample) -> TrajectorySample:
        """An uploaded ATIF conversation, which trains with every agent turn that is not copied context.

        ``ValueError``: the trajectory has no turn that trains (``trained_turns``).
        """
        trained_turns(trajectory.trajectory)
        return self.limited(trajectory)

    def make_sample(self, context: ReportContext) -> TrajectorySample:
        parsed = context.parsed_report
        if not isinstance(parsed, TeacherContextReport):
            raise ValueError(f"{type(self).__name__} requires the TeacherContextReport schema")
        if len(context.ends) != 1:
            raise ValueError(
                f"an SFT report must reference one recorded request; report "
                f"{context.report.agent_record_id} references {len(context.ends)}"
            )
        # A reported score is only metadata. All samples train with the same weight.
        sample = make_trajectory(context.ends, 0.0 if context.score is None else context.score)
        steps = list(sample.trajectory["steps"])
        if steps[-1]["source"] != "agent" or steps[-1].get("is_copied_context"):
            raise ValueError("target response requires the recorded response message of the inference")
        if self.target == "demonstration":
            if not parsed.teacher_context:
                raise ValueError("target demonstration requires the report's teacher_context")
            # The demonstration replaces the recorded reply as the turn that trains.
            steps[-1] = {"step_id": steps[-1]["step_id"], "source": "agent", "message": parsed.teacher_context}
            sample = TrajectorySample(
                {**sample.trajectory, "steps": steps},
                group_id=sample.group_id,
                source_agent_record_ids=sample.source_agent_record_ids,
            )
        return self.limited(sample)


class SFTJobData(DistillJobData):
    """``DistillJobData`` that also takes uploaded ATIF trajectories, which the training component renders, and
    imported train tasks, which train only for an instruction that gives their job's parameters."""

    accepts_trajectories = True
    accepts_imported_tasks = True


__all__ = ["SFT_TARGETS", "SFTJobData", "SFTSamples", "SFTSettings"]
