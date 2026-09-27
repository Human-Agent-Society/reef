"""Build SDPO teacher contexts after every rollout of a question has been graded."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from reef.core.reports import TeacherContextReport

SOLUTION_TEMPLATE = "\nCorrect solution:\n\n{successful_previous_attempt}"
FEEDBACK_TEMPLATE = "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}"


@dataclass(frozen=True)
class SDPOAttempt:
    """One graded on-policy response and its Reef inference receipt."""

    question_id: str
    inference_id: str
    artifact_version: str
    response: str
    score: float
    feedback: str | None = None

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.question_id, self.inference_id, self.artifact_version)
        ):
            raise ValueError("every SDPO attempt needs a question id, inference id and artifact version")
        if not isinstance(self.response, str) or (self.feedback is not None and not isinstance(self.feedback, str)):
            raise ValueError("SDPO response and optional feedback must be text")
        if not isinstance(self.score, Real) or isinstance(self.score, bool) or not math.isfinite(self.score):
            raise ValueError("SDPO attempt score must be finite")


@dataclass(frozen=True)
class SDPOPreparedReport:
    """A report paired with the exact inference it will train."""

    question_id: str
    inference_id: str
    artifact_version: str
    teacher_context: str
    score: float
    demonstration_id: str | None
    used_feedback: bool

    def payload(self) -> dict[str, Any]:
        """A TeacherContextReport wire body with the group's source information."""
        body = TeacherContextReport(teacher_context=self.teacher_context, score=self.score).to_dict(
            references=(self.inference_id,)
        )
        body.setdefault("metadata", {})["sdpo"] = {
            "question_id": self.question_id,
            "artifact_version": self.artifact_version,
            "demonstration_id": self.demonstration_id,
            "used_feedback": self.used_feedback,
            "active": bool(self.teacher_context),
        }
        return body


def prepare_group(
    attempts: list[SDPOAttempt],
    *,
    success_threshold: float = 0.5,
    expected_rollouts: int | None = None,
    exclude_self_success: bool = True,
    include_environment_feedback: bool = True,
    feedback_only_without_solution: bool = True,
    remove_thinking_from_demonstration: bool = True,
) -> list[SDPOPreparedReport]:
    """Create one report per attempt, marking the uninformed ones inactive.

    The caller must pass a complete, single-version rollout group. Successful
    demonstrations use the first eligible response in group order, matching
    the reference's selection rule. A full group still emits its full number
    of reports, so Reef can make one fixed-size step and a zero weight excludes
    attempts for which neither a solution nor feedback was available.
    """
    if not attempts:
        raise ValueError("SDPO group must contain at least one attempt")
    if expected_rollouts is not None:
        if (
            not isinstance(expected_rollouts, Integral)
            or isinstance(expected_rollouts, bool)
            or expected_rollouts <= 0
        ):
            raise ValueError("expected_rollouts must be a positive integer")
        if len(attempts) != expected_rollouts:
            raise ValueError(f"SDPO group needs exactly {expected_rollouts} graded attempts")
    if not math.isfinite(success_threshold):
        raise ValueError("success_threshold must be finite")
    question_id = attempts[0].question_id
    version = attempts[0].artifact_version
    if any(item.question_id != question_id or item.artifact_version != version for item in attempts):
        raise ValueError("SDPO group must contain one question and one artifact version")
    if len({item.inference_id for item in attempts}) != len(attempts):
        raise ValueError("SDPO group contains duplicate inference receipts")

    successes = [item for item in attempts if item.score >= success_threshold and item.response.strip()]
    prepared: list[SDPOPreparedReport] = []
    for item in attempts:
        candidate = next(
            (
                success
                for success in successes
                if not exclude_self_success or success.inference_id != item.inference_id
            ),
            None,
        )
        feedback = item.feedback.strip() if include_environment_feedback and item.feedback else ""
        use_feedback = bool(feedback) and (not feedback_only_without_solution or candidate is None)
        context = ""
        if candidate is not None:
            demonstration = candidate.response
            if remove_thinking_from_demonstration:
                demonstration = re.sub(r"<think>.*?</think>\s*", "", demonstration, flags=re.DOTALL)
            context += SOLUTION_TEMPLATE.replace("{successful_previous_attempt}", demonstration)
        if use_feedback:
            context += FEEDBACK_TEMPLATE.replace("{feedback_raw}", feedback)
        prepared.append(
            SDPOPreparedReport(
                question_id=question_id,
                inference_id=item.inference_id,
                artifact_version=version,
                teacher_context=context,
                score=item.score,
                demonstration_id=candidate.inference_id if candidate is not None else None,
                used_feedback=use_feedback,
            )
        )
    return prepared


__all__ = ["SDPOAttempt", "SDPOPreparedReport", "prepare_group"]
