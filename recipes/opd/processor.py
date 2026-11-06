"""OPD's stages: the teacher scores the exact student sequence, without a privileged teacher prefix."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from reef.core.chat_rendering import TeacherRequest
from reef.core.config import config_option
from reef.core.job_data import TrajectorySample
from reef.core.reports import TeacherContextReport
from reef.pipeline_context import PipelineContext
from reef.record2sample import ReportContext, ReportSamples
from reef.record2sample.assembly import SampleAssembly, SampleAssemblySettings
from reef.record2sample.distill import student_sample


@dataclass(frozen=True)
class OPDSettings(SampleAssemblySettings):
    """The config of ``OPDSamples``: the teacher's window and the sample assembly.

    Values: ``max_teacher_tokens`` is a non-negative int, 0 for no limit; the other keys are as in
    ``SampleAssemblySettings``. The constructor refuses other values.
    Purpose: ``OPDSamples.from_config`` parses it from ``pipeline.record2sample.config``.
    """

    max_teacher_tokens: int = config_option(
        0, help="The longest teacher sequence the trainer scores; a longer sample does not train. 0 is no limit."
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_teacher_tokens < 0:
            raise ValueError("max_teacher_tokens must be non-negative (0 disables the limit)")


class OPDSamples(ReportSamples):
    """One report, one sample: the student's call, whose own sequence the teacher scores.

    The sample is the student's call as an ATIF trajectory with a teacher request that names no messages
    (``TeacherRequest``): the training component renders the student's sequence with the model's chat template and
    gives the teacher that same sequence. A sequence longer than ``max_teacher_tokens`` does not train. Each report
    trains by itself and owns its inference records.
    """

    def __init__(self, context: PipelineContext, settings: OPDSettings) -> None:
        super().__init__(context, exclusive_sources=True, max_staleness=settings.max_staleness)
        self.assembly = SampleAssembly(settings)
        self.max_teacher_tokens = settings.max_teacher_tokens

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        environ: Mapping[str, str],
    ) -> Self:
        return cls(context, OPDSettings.parse(config, plugin_class=cls))

    def make_sample(self, context: ReportContext) -> TrajectorySample:
        report = context.parsed_report
        if not isinstance(report, TeacherContextReport):
            raise ValueError("OPD requires TeacherContextReport")
        if report.teacher_context:
            raise ValueError("OPD teacher_context must be empty: the teacher reads the student's sequence")
        # The teacher's distribution is the signal; the reported score is not a reward.
        sample = student_sample(self.assembly, context, 0.0)
        return sample.with_metadata(teacher=TeacherRequest(max_tokens=self.max_teacher_tokens).as_json())


__all__ = ["OPDSamples", "OPDSettings"]
