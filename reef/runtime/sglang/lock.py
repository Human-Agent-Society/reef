"""Compatibility Ray actor for Reef's backend-independent weight-update lock."""

import ray

from reef.runtime.weight_update import WeightUpdateLock

ReefRolloutLock = ray.remote(WeightUpdateLock)

__all__ = ["ReefRolloutLock"]
