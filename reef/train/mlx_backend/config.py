"""Settings of the MLX backend; importable without MLX."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.core.config import config_option

#: Template arguments the engine owns. They decide what a prompt is, so a
#: caller that sets them could serve one token sequence and train another.
#: ``tools`` has its own request field.
RESERVED_TEMPLATE_KWARGS = ("tokenize", "add_generation_prompt", "conversation", "messages", "tools")


def check_template_kwargs(template_kwargs: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if ``template_kwargs`` sets an argument the engine owns."""
    reserved = sorted(name for name in RESERVED_TEMPLATE_KWARGS if name in template_kwargs)
    if reserved:
        raise ValueError(f"chat_template_kwargs may not set {reserved}: the engine renders the prompt")


@dataclass(frozen=True, kw_only=True)
class MLXEngineConfig:
    """The model, adapter, optimizer and memory settings of one MLX engine."""

    lora_layers: int = config_option(8, help="Transformer blocks that get a LoRA adapter, counted from the last.")
    lora_rank: int = config_option(8, help="LoRA rank.")
    lora_scale: float = config_option(2.0, help="LoRA scale.")
    lora_dropout: float = config_option(0.0, help="LoRA dropout.")
    lora_keys: tuple[str, ...] = config_option(
        ("self_attn.q_proj", "self_attn.v_proj"), help="Module paths, as mlx-lm names them, that get an adapter."
    )
    learning_rate: float = config_option(1e-5, help="AdamW learning rate.")
    weight_decay: float = config_option(0.01, help="AdamW weight decay.")
    max_tokens: int = config_option(256, help="Completion tokens of a request that sets no limit.")
    max_context_tokens: int = config_option(
        0, help="Prompt and completion tokens of one request; 0 sets no limit. It also bounds each training row."
    )
    temperature: float = config_option(1.0, help="Sampling temperature of a request that sets none.")
    top_p: float = config_option(1.0, help="Nucleus sampling threshold of every request.")
    top_k: int = config_option(0, help="Candidate tokens sampled from in every request; 0 keeps every token.")
    seed: int = config_option(0, help="Seed of the MLX random generator.")
    micro_batch_size: int = config_option(8, help="Rows in one backward pass; a step accumulates over passes.")
    capture_topk: int = config_option(
        0, help="Candidate tokens recorded per generated token; distillation losses need them."
    )
    cache_limit_gib: float = config_option(
        0.0, help="Upper bound in GiB of MLX's buffer cache; 0 keeps the MLX default."
    )
    prefill_step_size: int = config_option(
        0, help="Prompt tokens per cache-filling pass; 0 runs the whole sequence in one pass."
    )
    log_probs_chunk_size: int = config_option(
        0, help="Response positions per pass through the output head; 0 scores all positions in one pass."
    )
    recurrence_chunk_size: int = config_option(
        64, help="Tokens per chunk of a GatedDeltaNet recurrence in training; a power of two, or 0 for mlx-lm's loop."
    )
    checkpoint_layers: bool = config_option(
        False, help="Recompute each adapted layer in the backward pass instead of keeping its activations."
    )
    chat_template_kwargs: Mapping[str, Any] = config_option(
        default_factory=dict, help="Chat-template arguments of every request; a request may override each one."
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("lora_layers", self.lora_layers),
            ("lora_rank", self.lora_rank),
            ("max_tokens", self.max_tokens),
            ("micro_batch_size", self.micro_batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("max_context_tokens", self.max_context_tokens),
            ("top_k", self.top_k),
            ("capture_topk", self.capture_topk),
            ("prefill_step_size", self.prefill_step_size),
            ("log_probs_chunk_size", self.log_probs_chunk_size),
            ("recurrence_chunk_size", self.recurrence_chunk_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.recurrence_chunk_size & (self.recurrence_chunk_size - 1):
            raise ValueError("recurrence_chunk_size must be a power of two")
        for name, rate in (("learning_rate", self.learning_rate), ("temperature", self.temperature)):
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        for name, amount in (
            ("lora_scale", self.lora_scale),
            ("weight_decay", self.weight_decay),
            ("cache_limit_gib", self.cache_limit_gib),
        ):
            if not math.isfinite(amount) or amount < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.lora_keys or any(not isinstance(key, str) or not key for key in self.lora_keys):
            raise ValueError("lora_keys must name at least one module path")
        check_template_kwargs(self.chat_template_kwargs)
        # A cached prompt is a constant of the backward pass. That is exact only
        # when the prompt's cached state depends on no trained weight: every
        # adapted module is in the last layer, and none of them writes keys,
        # values or a recurrent state.
        cache_writers = [key for key in self.lora_keys if "k_proj" in key or "v_proj" in key or "linear_attn" in key]
        if self.prefill_step_size and (self.lora_layers != 1 or cache_writers):
            raise ValueError(
                "prefill_step_size caches the prompt, which is exact only with lora_layers 1 and no adapted "
                f"key, value or linear_attn projection; got lora_layers {self.lora_layers} and "
                f"lora_keys {list(self.lora_keys)}. Set prefill_step_size to 0."
            )


@dataclass(frozen=True, kw_only=True)
class MLXConfig(MLXEngineConfig):
    """``training.options`` of an MLX deployment: the engine settings and the runtime's own."""

    checkpoint_dir: str = config_option(help="Directory that receives each trained candidate's adapter.")
    adapter_name: str = config_option("reef-mlx", help="Name under which the engine serves its adapter.")
    kl_coef: float = config_option(0.0, help="Frozen-base KL coefficient of the selected loss family.")
    loss_options: Mapping[str, Any] = config_option(
        default_factory=dict, help="Settings of the selected loss family; its settings type validates them."
    )
    inference_timeout_s: float = config_option(300.0, help="Seconds to wait for admitted requests to finish.")
    max_train_tokens: int = config_option(
        0, help="Tokens of one training row, at most; longer rows are left out of the step. 0 sets no limit."
    )
    min_train_rows: int = config_option(
        1, help="Rows a step needs after max_train_tokens; a batch with fewer is skipped."
    )
    train_timeout_s: float | None = config_option(None, help="Unsupported; training runs in this process.")

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.checkpoint_dir.strip():
            raise ValueError("the mlx backend requires training.options.checkpoint_dir")
        if not self.adapter_name.strip():
            raise ValueError("adapter_name must be non-empty")
        if not math.isfinite(self.kl_coef) or self.kl_coef < 0:
            raise ValueError("kl_coef must be finite and non-negative")
        if not math.isfinite(self.inference_timeout_s) or self.inference_timeout_s <= 0:
            raise ValueError("inference_timeout_s must be finite and positive")
        if (
            isinstance(self.max_train_tokens, bool)
            or not isinstance(self.max_train_tokens, int)
            or self.max_train_tokens < 0
        ):
            raise ValueError("max_train_tokens must be a non-negative integer")
        if (
            isinstance(self.min_train_rows, bool)
            or not isinstance(self.min_train_rows, int)
            or self.min_train_rows < 1
        ):
            raise ValueError("min_train_rows must be a positive integer")
        if self.train_timeout_s is not None:
            raise ValueError("the mlx backend trains in the service process and has no train timeout")


__all__ = ["MLXConfig", "MLXEngineConfig", "check_template_kwargs"]
