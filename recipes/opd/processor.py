"""OPD's stages: the teacher scores the exact student sequence, without a privileged teacher prefix."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from reef.core.batches import TrajectoryItem
from reef.core.config import config_option
from reef.core.reports import TeacherContextReport
from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record, ReportContext
from reef.record2task import ReportTasks, SkippedReport
from reef.record2task.assembly import SampleAssembly, SampleAssemblySettings
from reef.record2task.distill import recorded_sample, teacher_overflow


@dataclass(frozen=True)
class OPDSettings(SampleAssemblySettings):
    """The config of ``OPDTasks``: the teacher's window and the sample assembly. OPD loads no tokenizer."""

    max_teacher_tokens: int = config_option(
        0, help="The longest teacher sequence the trainer scores; a longer report is skipped. 0 is no limit."
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_teacher_tokens < 0:
            raise ValueError("max_teacher_tokens must be non-negative (0 disables the limit)")


class OPDTasks(ReportTasks):
    """One report, one task: the student's policy tensors, and the same captured ids as the teacher sequence.

    Re-rendering the messages can change the sampled prefix, for example a
    thinking switch or a prefill. A teacher with the student's tokenizer
    instead reads the exact prompt and response ids that the inference
    engine recorded, so this stage loads no tokenizer. A sequence longer
    than ``max_teacher_tokens`` is skipped and counted in
    ``teacher_overflow_reports``, as in ``DistillTasks``.
    """

    def __init__(self, settings: OPDSettings) -> None:
        self.assembly = SampleAssembly(settings)
        self.max_teacher_tokens = settings.max_teacher_tokens
        self.overflow_count = 0

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        environ: Mapping[str, str],
    ) -> Self:
        return cls(OPDSettings.parse(config, plugin_class=cls))

    def operational_metrics(self) -> Mapping[str, float | int]:
        return {"teacher_overflow_reports": self.overflow_count}

    def make_sample(self, context: ReportContext) -> TrajectoryItem | SkippedReport:
        report = context.parsed_report
        if not isinstance(report, TeacherContextReport):
            raise ValueError("OPD requires TeacherContextReport")
        if report.teacher_context:
            raise ValueError("OPD teacher_context must be empty: the teacher reads the student's sequence")
        # The teacher's distribution is the signal; the reported score is not a reward.
        sample = recorded_sample(self.assembly, context, 0.0)
        teacher_tokens = list(sample.training["tokens"])
        if teacher_overflow(context, teacher_tokens, self.max_teacher_tokens):
            self.overflow_count += 1
            return SkippedReport("teacher_overflow")
        return sample.with_training(teacher_tokens=teacher_tokens)


__all__ = ["OPDSettings", "OPDTasks"]
