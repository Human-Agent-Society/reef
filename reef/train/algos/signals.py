"""What an objective computes from one job's data, and what a training component prepares from it.

``TrainingSignal`` is the objective's backend-neutral result. ``PreparedJob``
is that result as one backend's payload. How a component cuts the job data into
optimizer steps is part of neither: the component reads it from the job's
``scheduling`` parameter (``reef.train.algos.schedule.StepScheduling``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class TrainingSignal:
    """Algorithm output before any backend's wire payload is materialized."""

    action: Literal["train", "skip"]
    next_algorithm_state: Mapping[str, Any]
    metrics: Mapping[str, Any] = field(default_factory=dict)
    advantages: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class PreparedJob:
    """What a training component prepared from one job's data: the objective's signal as its backend's payload.

    ``action`` is ``train`` exactly when ``payload`` is given. A ``skip`` step
    (the objective skipped the job data) trains nothing. ``next_algorithm_state``
    is the objective's state after the job data, which the component keeps with
    the checkpoint it saves; ``metrics`` is the objective's telemetry.
    """

    action: Literal["train", "skip"]
    next_algorithm_state: Mapping[str, Any]
    metrics: Mapping[str, Any]
    payload: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.action not in ("train", "skip"):
            raise ValueError(f"a prepared job's action must be 'train' or 'skip', got {self.action!r}")
        if (self.action == "train") != (self.payload is not None):
            raise ValueError("a train job, and only a train job, carries a backend payload")
        if not isinstance(self.next_algorithm_state, Mapping) or not isinstance(self.metrics, Mapping):
            raise TypeError("a prepared job's next_algorithm_state and metrics must be mappings")
