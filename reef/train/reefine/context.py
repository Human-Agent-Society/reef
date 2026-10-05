"""Immutable step inputs and the optional served-composition capability."""

from dataclasses import dataclass

from reef.core.training_request import TrainingRequest
from reef.train.cordis_backend.contracts import ServedComposition


@dataclass(frozen=True)
class EvaluationContext:
    request: TrainingRequest | None
    current: ServedComposition
