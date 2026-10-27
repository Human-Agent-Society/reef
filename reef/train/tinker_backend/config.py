"""CPU-only configuration for the Tinker integration."""

from __future__ import annotations

import math
from dataclasses import dataclass

from reef.core.config import config_option


@dataclass(frozen=True)
class TinkerConfig:
    """The ``training.options`` of the Tinker training component.

    Values: ``state_dir`` and ``api_key_env`` are non-empty; ``lora_rank`` and ``batch_size`` are positive;
    ``learning_rate`` is finite and positive and ``kl_coef`` finite and non-negative, the defaults of each job's
    parameters (``TinkerJobParameters``); ``seed`` is an int or None; ``base_url`` is an http(s) URL or None
    for hosted Tinker; ``inference_timeout_s`` is positive and ``train_timeout_s`` positive or None.
    Purpose: ``tinker_settings`` parses it from a deployment; the component, its SDK client and its sampler read
    it, and ``TinkerDeployment.trainer_serving_config`` derives Tinker serving from it.
    """

    state_dir: str = config_option(help="Persistent local directory for Tinker checkpoint manifests.")
    lora_rank: int = config_option(32, help="Rank of the remote LoRA adapter.")
    learning_rate: float = config_option(1e-4, help="Adam learning rate.")
    kl_coef: float = config_option(0.0, help="Frozen-base KL coefficient for loss adapters that support it.")
    batch_size: int = config_option(
        1, help="Rollouts per optimizer step; the component cuts each job's data into steps of this size."
    )
    seed: int | None = config_option(
        0, help="Initial adapter seed; null lets the service choose (Spindle's multi-LoRA trainer refuses a seed)."
    )
    api_key_env: str = config_option("TINKER_API_KEY", help="Environment variable containing the Tinker API key.")
    base_url: str | None = config_option(
        None,
        help=(
            "Base URL of the Tinker-compatible service that trains and holds the checkpoints, such as a Spindle "
            "deployment; unset is Thinking Machines' hosted Tinker."
        ),
    )
    export_adapter: bool = config_option(
        False,
        help=(
            "Save each checkpoint's PEFT adapter in its release, so inference engines can serve it, and Tinker's "
            "adapter as Tinker saved it, which tinker-cookbook merges into full weights. "
            "Without it, a release holds only the manifest that Tinker's sampler serves."
        ),
    )
    project_id: str | None = config_option(None, help="Optional Tinker project ID.")
    inference_timeout_s: float = config_option(
        300.0, help="Seconds that one sample may take: a request's, and a task job's rollout sampling."
    )
    train_timeout_s: float | None = config_option(
        None, help="Seconds that one training call to Tinker may take; unset waits as long as the SDK does."
    )

    def __post_init__(self) -> None:
        if not self.state_dir.strip() or not self.api_key_env.strip():
            raise ValueError("Tinker requires non-empty state_dir and api_key_env")
        if self.base_url is not None and not self.base_url.startswith(("https://", "http://")):
            raise ValueError("Tinker base_url must be an http(s) URL")
        if self.lora_rank <= 0 or self.batch_size <= 0:
            raise ValueError("Tinker lora_rank and batch_size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Tinker learning_rate must be finite and positive")
        if not math.isfinite(self.kl_coef) or self.kl_coef < 0:
            raise ValueError("Tinker kl_coef must be finite and non-negative")
        if self.inference_timeout_s <= 0 or (self.train_timeout_s is not None and self.train_timeout_s <= 0):
            raise ValueError("Tinker timeouts must be positive")
