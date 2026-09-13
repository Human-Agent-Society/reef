"""CPU-only configuration for the Tinker integration."""

from __future__ import annotations

import math
from dataclasses import dataclass

from reef.core.config import config_option


@dataclass(frozen=True)
class TinkerConfig:
    state_dir: str = config_option(help="Persistent local directory for Tinker checkpoint manifests.")
    lora_rank: int = config_option(32, help="Rank of the remote LoRA adapter.")
    learning_rate: float = config_option(1e-4, help="Adam learning rate.")
    kl_coef: float = config_option(0.0, help="Frozen-base KL coefficient for loss adapters that support it.")
    batch_size: int = config_option(
        1, help="Comparison sets per optimizer step when the recipe uses configured sizing."
    )
    seed: int = config_option(0, help="Initial adapter seed.")
    api_key_env: str = config_option("TINKER_API_KEY", help="Environment variable containing the Tinker API key.")
    project_id: str | None = config_option(None, help="Optional Tinker project ID.")
    inference_timeout_s: float = config_option(300.0)
    train_timeout_s: float | None = config_option(None)
    max_staleness: int = config_option(0)

    def __post_init__(self) -> None:
        if not self.state_dir.strip() or not self.api_key_env.strip():
            raise ValueError("Tinker requires non-empty state_dir and api_key_env")
        if self.lora_rank <= 0 or self.batch_size <= 0:
            raise ValueError("Tinker lora_rank and batch_size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Tinker learning_rate must be finite and positive")
        if not math.isfinite(self.kl_coef) or self.kl_coef < 0:
            raise ValueError("Tinker kl_coef must be finite and non-negative")
        if self.max_staleness != 0:
            raise ValueError("Tinker currently requires max_staleness=0")
        if self.inference_timeout_s <= 0 or (self.train_timeout_s is not None and self.train_timeout_s <= 0):
            raise ValueError("Tinker timeouts must be positive")
