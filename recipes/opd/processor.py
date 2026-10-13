"""OPD scores the exact student sequence, without a privileged teacher prefix."""

from reef.core.reports import TeacherContextReport
from reef.train.processors import DistillProcessor
from reef.train.processors.reported import ReportContext
from reef.train.types import TrajectoryItem


class OPDProcessor(DistillProcessor):
    """Keep the captured prompt and response IDs, including thinking switches and prefill.

    Re-rendering the messages can change the sampled prefix. A same-tokenizer
    teacher instead reads the exact sequence the inference engine recorded, so
    this processor loads no tokenizer. Overflow accounting and report
    consumption use the shared processor.
    """

    batch_label = "opd"
    renders_teacher_prompt = False

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        report = context.parsed_report
        if not isinstance(report, TeacherContextReport):
            raise ValueError("OPD requires TeacherContextReport")
        if report.teacher_context:
            raise ValueError("OPD teacher_context must be empty: the teacher reads the student's sequence")
        sample = self.recorded_sample(context, 0.0)
        teacher_tokens = list(sample.training["tokens"])
        self.check_teacher_length(context, teacher_tokens)
        return sample.with_training(teacher_tokens=teacher_tokens)
