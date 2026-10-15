"""Loss families of the MLX backend.

A loss family turns reserved samples into rows and drives one optimizer
step through :class:`~reef.train.mlx_backend.training.MLXTrainer`. Its
settings come from ``training.options``: ``kl_coef`` and the
``loss_options`` mapping, which the family's settings type parses.

Methods register their loss in the shared loss-family table with
``register_loss_family_ref(name, "package.module:CLASS", backend="mlx")``.
The engine imports the reference the first time it resolves the name, so
Reef never imports the method package.
"""

from __future__ import annotations

import importlib
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from reef.core.batches import TrajectoryItem
from reef.core.config import Settings, config_option
from reef.train.algos.registry import loss_family_refs
from reef.train.mlx_backend.contracts import LossOptions
from reef.train.mlx_backend.training import MLXTrainer

#: The backend key under which methods register their loss.
BACKEND = "mlx"


@dataclass(frozen=True)
class MLXLossSettings(Settings):
    """Settings every MLX loss family reads; a family extends them with its own."""

    kl_coef: float = config_option(0.0, help="Frozen-base KL coefficient.")

    def __post_init__(self) -> None:
        if not math.isfinite(self.kl_coef) or self.kl_coef < 0:
            raise ValueError("kl_coef must be finite and non-negative")

    @classmethod
    def from_options(cls, options: LossOptions) -> Self:
        """Parse the deployment's loss options; ``loss_options.kl_coef`` overrides ``training.options.kl_coef``."""
        values = {str(key).replace("-", "_"): value for key, value in options.options.items()}
        values.setdefault("kl_coef", options.kl_coef)
        return cls.from_config(values, setting="training.options.loss_options")


class MLXLossStep(ABC):
    """One prepared optimizer step of a loss family."""

    @abstractmethod
    def train(self, trainer: MLXTrainer) -> dict[str, float | int]:
        """Run the step on the engine thread and return its metrics."""


class MLXLoss(ABC):
    """One loss family on the MLX backend."""

    def settings(self, options: LossOptions) -> MLXLossSettings:
        """Parse ``options``; raise ``ValueError`` for an option the family does not accept."""
        return MLXLossSettings.from_options(options)

    @abstractmethod
    def step(
        self, samples: Sequence[TrajectoryItem], advantages: Sequence[float], options: LossOptions
    ) -> MLXLossStep:
        """Build the rows of one step from reserved samples and one advantage per sample.

        Raises ``ValueError`` when a sample lacks what the family trains on.
        """


resolved_losses: dict[str, MLXLoss] = {}


def resolve_mlx_loss(loss_family: str) -> MLXLoss:
    """Import the loss a method registered for MLX under ``loss_family``; raise ``ValueError`` if there is none."""
    loss = resolved_losses.get(loss_family)
    if loss is not None:
        return loss
    references = loss_family_refs(BACKEND)
    reference = references.get(loss_family)
    if reference is None:
        raise ValueError(
            f"unsupported MLX loss family {loss_family!r}; registered: {', '.join(sorted(references)) or 'none'}"
        )
    module_name, _, attribute = reference.partition(":")
    candidate = vars(importlib.import_module(module_name)).get(attribute)
    if isinstance(candidate, type) and issubclass(candidate, MLXLoss):
        candidate = candidate()
    if not isinstance(candidate, MLXLoss):
        raise TypeError(f"MLX loss family {reference!r} is not an MLXLoss")
    resolved_losses[loss_family] = candidate
    return candidate


__all__ = ["BACKEND", "MLXLoss", "MLXLossSettings", "MLXLossStep", "resolve_mlx_loss"]
