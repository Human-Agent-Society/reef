"""SDPO's report coordinates: one feedback-bearing rollout in a sampling step."""

from dataclasses import dataclass

from reef.core.reports import ReportValidationError, TeacherContextReport


@dataclass(frozen=True, kw_only=True)
class SDPOReport(TeacherContextReport):
    """Feedback for one question (``group``) and attempt (``rollout``) in ``step``."""

    step: int
    group: int
    rollout: int

    def validate(self) -> None:
        if self.step < 0 or self.group < 0 or self.rollout < 0:
            raise ReportValidationError("SDPO step, group and rollout must be non-negative")


__all__ = ["SDPOReport"]
