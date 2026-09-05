"""Component-declared execution needs, independent of a scheduler backend."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionRequirements:
    """Local worker needs only; calling a remote model does not request its GPUs."""

    workers: int = 1
    gpus_per_worker: float = 0
    cluster: bool = False
    supported_backends: tuple[str, ...] = ("uni", "mp", "local", "ray")

    def __post_init__(self) -> None:
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1:
            raise ValueError("execution workers must be a positive integer")
        if (
            isinstance(self.gpus_per_worker, bool)
            or not isinstance(self.gpus_per_worker, (int, float))
            or not math.isfinite(self.gpus_per_worker)
            or self.gpus_per_worker < 0
        ):
            raise ValueError("gpus_per_worker must be a finite nonnegative number")
        if not isinstance(self.cluster, bool):
            raise ValueError("execution cluster must be a boolean")
        if not self.supported_backends or any(
            backend not in ("uni", "mp", "local", "ray") for backend in self.supported_backends
        ):
            raise ValueError("supported_backends must list supported built-in worker launchers")
