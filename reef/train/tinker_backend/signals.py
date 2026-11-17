"""What a Tinker objective computes from one job's data.

How the component cuts the job data into optimizer steps is no part of it: the
component reads that from its job parameters
(``reef.train.tinker_backend.schedule.StepScheduling``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class TrainingSignal:
    """What a Tinker objective computes from one job's data, before the rows of a Tinker loss are made.

    ``action`` is ``train`` or ``skip``. ``next_algorithm_state`` is the objective's state after the job data.
    ``metrics`` is the objective's telemetry. ``advantages`` has one value per trajectory, or is None for a loss
    family that reads none. ``TinkerObjective.prepare`` returns the value; ``prepare_tinker_job`` reads it.
    """

    action: Literal["train", "skip"]
    next_algorithm_state: Mapping[str, Any]
    metrics: Mapping[str, Any] = field(default_factory=dict)
    advantages: tuple[float, ...] | None = None
