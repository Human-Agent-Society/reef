"""Artifact checkpoint-cadence policies selected by individual scenarios.

A deliberate leaf module: it decides only when a published version becomes a
durable artifact checkpoint, so it imports nothing from the rest of Reef at
runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class CheckpointStrategy(ABC):
    """Decision contract for a scenario's artifact checkpoint cadence."""

    @abstractmethod
    def should_checkpoint(
        self,
        scenario: str,
        version: int,
        result: Any,  # TrainStepResult — leaf module, no reef import
    ) -> bool: ...


@dataclass(frozen=True)
class EveryNVersions(CheckpointStrategy):
    n: int

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("checkpoint interval must be positive")

    def should_checkpoint(
        self,
        scenario: str,
        version: int,
        result: Any,  # TrainStepResult — leaf module, no reef import
    ) -> bool:
        return version % self.n == 0
