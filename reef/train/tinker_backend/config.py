"""CPU-only configuration for the training components on Tinker-compatible services."""

from __future__ import annotations

import math
from dataclasses import dataclass

from reef.artifact.sources import HuggingFaceRevision
from reef.core.config import config_option


@dataclass(frozen=True)
class TinkerServiceConfig:
    """The ``training.options`` that every training component on a Tinker-compatible service reads.

    Values: ``state_dir`` and ``api_key_env`` are non-empty; ``batch_size`` is positive; ``learning_rate`` is finite
    and positive and ``kl_coef`` finite and non-negative, the defaults of each job's parameters
    (``TinkerJobParameters``); ``base_url`` is an http(s) URL or None for hosted Tinker; ``inference_timeout_s`` is
    positive and ``train_timeout_s`` positive or None. Purpose: the base of ``TinkerConfig`` and
    ``SpindleFullConfig``; the components, their SDK client and their sampler read these fields.
    """

    state_dir: str = config_option(help="Persistent local directory for Tinker checkpoint manifests.")
    learning_rate: float = config_option(1e-4, help="Adam learning rate.")
    kl_coef: float = config_option(0.0, help="Frozen-base KL coefficient for loss adapters that support it.")
    batch_size: int = config_option(
        1, help="Rollouts per optimizer step; the component cuts each job's data into steps of this size."
    )
    api_key_env: str = config_option("TINKER_API_KEY", help="Environment variable containing the Tinker API key.")
    base_url: str | None = config_option(
        None,
        help=(
            "Base URL of the Tinker-compatible service that trains and holds the checkpoints, such as a Spindle "
            "deployment; unset is Thinking Machines' hosted Tinker."
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
        if self.batch_size <= 0:
            raise ValueError("Tinker batch_size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Tinker learning_rate must be finite and positive")
        if not math.isfinite(self.kl_coef) or self.kl_coef < 0:
            raise ValueError("Tinker kl_coef must be finite and non-negative")
        if self.inference_timeout_s <= 0 or (self.train_timeout_s is not None and self.train_timeout_s <= 0):
            raise ValueError("Tinker timeouts must be positive")


@dataclass(frozen=True)
class TinkerConfig(TinkerServiceConfig):
    """The ``training.options`` of the Tinker training component, which trains a LoRA.

    Values: those of ``TinkerServiceConfig``; ``lora_rank`` is positive and ``seed`` an int or None. Purpose:
    ``tinker_settings`` parses it from a deployment; the component, its SDK client and its sampler read it, and
    ``TinkerDeployment.trainer_serving_config`` derives Tinker serving from it.
    """

    lora_rank: int = config_option(32, help="Rank of the remote LoRA adapter.")
    seed: int | None = config_option(
        0, help="Initial adapter seed; null lets the service choose (Spindle's multi-LoRA trainer refuses a seed)."
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.lora_rank <= 0:
            raise ValueError("Tinker lora_rank must be positive")


@dataclass(frozen=True, kw_only=True)
class SpindleFullConfig(TinkerServiceConfig):
    """The ``training.options`` of the Spindle full-parameter training component (RFC #996).

    Values: those of ``TinkerServiceConfig``, with ``base_url`` set to the Spindle deployment; ``base`` is
    ``hf://<repo>@<commit>`` with a full 40-digit commit SHA (``HuggingFaceRevision``), the weights that the
    deployment's full-parameter engine for ``<repo>`` trains. Purpose: ``spindle_full_settings`` parses it; the
    component, its SDK client and its sampler read it, and ``SpindleFullDeployment`` derives serving from it.
    """

    base: str = config_option(
        help="hf://<repo>@<commit>: the Hugging Face revision that the Spindle full-parameter engine trains."
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.base_url is None:
            raise ValueError("Spindle full-parameter training requires base_url, the Spindle deployment")
        HuggingFaceRevision.parse(self.base)

    @property
    def base_revision(self) -> HuggingFaceRevision:
        """``base`` as a revision."""
        return HuggingFaceRevision.parse(self.base)
