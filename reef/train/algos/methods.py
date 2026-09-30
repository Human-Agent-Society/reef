"""How a weight recipe chooses the training method of each job.

A recipe returns a :class:`TrainingMethodSelector` from
``WeightTrainingRecipe.training_method_selector``. Reef calls it once for each
job, before the backend prepares the batch, with the batch and the committed
algorithm state; retries of a job reuse the batch and that state, so they
select the same method. The default :class:`FixedTrainingMethod` trains every
job with the recipe's ``training_spec().objective``.

When to switch is recipe logic: a selector may read the batch or the
committed state, such as the ``steps`` counter every shipped objective keeps
(``reef.train.algos.helpers.next_steps``). It does not list its objectives in
advance; the backend resolves and validates each selected objective before
that job trains, and refuses one it cannot train.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from reef.core.batches import TrainingBatch
from reef.core.training_method import LearningRateScheduleState, TrainingMethod


class TrainingMethodSelector(ABC):
    """Select the objective and learning-rate schedule of one training job."""

    @abstractmethod
    def select(self, batch: TrainingBatch, algorithm_state: Mapping[str, Any]) -> TrainingMethod:
        """The method ``batch``'s job trains with, given the committed ``algorithm_state``.

        Must be a function of its arguments: a retry calls it again and has to
        get the same method, since the method is part of the job's identity.
        """

    def experiment_config(self) -> Mapping[str, Any]:
        """Non-secret description attached to experiment runs."""
        return {"training_method_selector": f"{type(self).__module__}.{type(self).__qualname__}"}


class FixedTrainingMethod(TrainingMethodSelector):
    """Train every job with one method: a single-method recipe."""

    def __init__(self, method: TrainingMethod) -> None:
        if not isinstance(method, TrainingMethod):
            raise TypeError(f"FixedTrainingMethod requires a TrainingMethod, got {type(method).__name__}")
        self.method = method

    def select(self, batch: TrainingBatch, algorithm_state: Mapping[str, Any]) -> TrainingMethod:
        return self.method

    def experiment_config(self) -> Mapping[str, Any]:
        schedule = self.method.learning_rate_schedule
        return {
            "objective": self.method.objective,
            **({"learning_rate_schedule": schedule.to_dict()} if schedule is not None else {}),
        }


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


__all__ = ["FixedTrainingMethod", "TrainingMethodSelector", "learning_rate_metrics"]
