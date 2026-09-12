"""Configuration declarations shared by training runtime adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.core.config import config_metadata, config_option


@dataclass(frozen=True)
class TrainingRuntimeSettings:
    """Connection and request settings; model-worker topology stays with the backend."""

    inference_url: str | None = config_option(None, help="Inference worker URL; omitted uses the coordinator.")
    inference_timeout_s: float = config_option(300.0, help="Inference request timeout in seconds.")
    train_timeout_s: float | None = config_option(None, help="Training request timeout; omitted follows inference.")
    max_staleness: int = config_option(0, help="Maximum admitted training-version lag.")
    inference_backend_config: Mapping[str, Any] = field(
        default_factory=dict, metadata=config_metadata("Inference adapter-owned options.")
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("inference_timeout_s", self.inference_timeout_s),
            ("train_timeout_s", self.train_timeout_s),
        ):
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
                raise ValueError(f"runtime.{name} must be a positive finite number")
        if isinstance(self.max_staleness, bool) or self.max_staleness < 0:
            raise ValueError("runtime.max_staleness must be a non-negative integer")
