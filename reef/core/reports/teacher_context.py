"""The report contract of the self-distillation recipes: one rollout and the text its teacher sees."""

from dataclasses import dataclass

from reef.core.reports.base import ReportBase, ReportValidationError

__all__ = ["TeacherContextReport"]


@dataclass(frozen=True)
class TeacherContextReport(ReportBase):
    """One recorded request and the privileged text added to its teacher prompt.

    The self-distillation recipes make the served model its own teacher, and
    ``context`` is the only difference between the teacher and the student:
    for SDFT a demonstration of the response, for SDPO the environment
    feedback the rollout produced. ``score`` is optional metadata a harness
    may record beside the context; the recipes distil the teacher and never
    train on it.
    """

    context: str
    score: float | None = None

    def validate(self) -> None:
        if not self.context.strip():
            raise ReportValidationError("metadata.context must be non-empty text")
