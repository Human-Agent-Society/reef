"""The learning-rate schedule progress a training backend keeps with its training state.

A job's ``TrainingMethod`` may select a ``LearningRateSchedule``
(``reef.runtime.interfaces``). The backend keeps the active schedule and the
optimizer steps it has completed (:class:`LearningRateScheduleState`) with its
weights and optimizer state, resolves each job's request against it
(:func:`resolve_learning_rate_schedule`), and reports the rate it trained with
(:func:`learning_rate_metrics`). A retry or a restart therefore never repeats
a warmup, and starting a schedule does not reset optimizer moments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from reef.runtime.interfaces import LearningRateSchedule, checked_step_count


@dataclass(frozen=True)
class LearningRateScheduleState:
    """The active schedule and the optimizer steps it has completed, kept with the training state."""

    schedule: LearningRateSchedule
    completed_steps: int = 0

    def __post_init__(self) -> None:
        checked_step_count(self.completed_steps, "LearningRateScheduleState.completed_steps")

    def learning_rates(self, optimizer_steps: int) -> tuple[float, ...]:
        """The rates of the next ``optimizer_steps`` steps, in order."""
        return tuple(self.schedule.learning_rate(self.completed_steps + step) for step in range(optimizer_steps))

    def advanced(self, optimizer_steps: int) -> LearningRateScheduleState:
        """This state after ``optimizer_steps`` more steps."""
        return replace(self, completed_steps=self.completed_steps + optimizer_steps)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LearningRateScheduleState:
        if not isinstance(value, Mapping):
            raise ValueError("a learning-rate schedule state must be an object")
        return cls(LearningRateSchedule.from_dict(value["schedule"]), value["completed_steps"])


def resolve_learning_rate_schedule(
    active: LearningRateScheduleState | None, requested: LearningRateSchedule | None
) -> LearningRateScheduleState | None:
    """The schedule state a job trains with, given the active state and the job's request.

    No request keeps the active state (``None``: the backend's configured
    rate); the active schedule continues; any other schedule starts at step 0.
    """
    if requested is None or (active is not None and active.schedule == requested):
        return active
    return LearningRateScheduleState(requested)


def learning_rate_metrics(
    learning_rates: Sequence[float], schedule: LearningRateScheduleState | None
) -> dict[str, Any]:
    """The metrics every backend reports for a job's rate: its last optimizer step's, and the schedule's progress."""
    metrics: dict[str, Any] = {"learning_rate": learning_rates[-1]}
    if schedule is not None:
        metrics["learning_rate_schedule"] = {
            "name": schedule.schedule.name,
            "completed_steps": schedule.completed_steps,
        }
    return metrics


__all__ = ["LearningRateScheduleState", "learning_rate_metrics", "resolve_learning_rate_schedule"]
