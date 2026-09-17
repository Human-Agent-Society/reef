"""The report contract of the distilling recipes: one rollout and the text its teacher sees."""

from dataclasses import dataclass

from reef.core.reports.base import ReportBase

__all__ = ["TeacherContextReport"]


@dataclass(frozen=True)
class TeacherContextReport(ReportBase):
    """One recorded request and the privileged text added to its teacher prompt.

    The distilling recipes make a teacher score the student's own sample,
    and ``context`` is what the teacher reads beyond the student's request:
    for SDFT a demonstration of the response, for SDPO the environment
    feedback the rollout produced. On-policy distillation from a separate
    teacher reads no privileged text and leaves it empty. ``score`` is optional metadata a
    harness may record beside the context; the recipes distil the teacher
    and never train on it.
    """

    context: str = ""
    score: float | None = None
