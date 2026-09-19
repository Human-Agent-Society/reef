"""Backend-neutral output contract for one prepared training step.

How the runtime cuts the batch into optimizer steps is not part of the signal:
the recipe binds a :class:`~reef.core.batches.StepScheduling` beside its objective.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class StepSignal:
    """Algorithm output before any backend's wire payload is materialized."""

    action: Literal["train", "skip"]
    next_algorithm_state: Mapping[str, Any]
    metrics: Mapping[str, Any] = field(default_factory=dict)
    advantages: tuple[float, ...] | None = None
