"""Build optional scenario experiment providers from deployment configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.observability.base import ExperimentTracker, NullExperimentTracker
from reef.observability.wandb import WandbConfig, WandbExperimentTracker


def build_experiment_tracker(
    wandb: object,
    *,
    model: str | None,
    training_config: Mapping[str, Any],
) -> ExperimentTracker:
    config = WandbConfig.from_mapping(wandb)
    if not config.active:
        return NullExperimentTracker()
    return WandbExperimentTracker(config, model=model, training_config=training_config)


__all__ = ["build_experiment_tracker"]
