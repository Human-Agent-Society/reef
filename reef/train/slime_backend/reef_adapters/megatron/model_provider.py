"""Megatron model provider that every Reef actor and critic builds through.

It builds the model through Slime or the user's chained provider, installs
the checkpoint's YaRN rotary scaling when its HF config declares one, and
applies Reef LoRA when a rank is set. Without either it returns Slime's
model unchanged. Slime adds the critic's value head after it returns.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def yarn_mscale(factor: float, mscale: float) -> float:
    """YaRN's attention scaling term: ``0.1 * mscale * ln(factor) + 1`` for ``factor > 1``, else 1."""
    if factor <= 1:
        return 1.0
    return 0.1 * mscale * math.log(factor) + 1.0


@dataclass(frozen=True)
class YarnRopeScaling:
    """The YaRN rotary scaling a checkpoint's HF ``config.json`` declares.

    Values: ``factor`` is a finite float > 1; ``original_max_position_embeddings``
    an int > 0; ``beta_fast`` and ``beta_slow`` finite floats with
    ``beta_fast > beta_slow > 0``; ``mscale`` and ``mscale_all_dim`` finite
    floats >= 0; ``truncate`` a bool; ``rope_theta`` a finite float > 0.
    Purpose: :func:`read_yarn_rope_scaling` makes one from the config's
    ``rope_scaling`` (or ``rope_parameters``) when its type is ``yarn``;
    :func:`install_yarn_rotary_embedding` reads it to build Megatron's
    ``YarnRotaryEmbedding`` with the same frequencies and attention factor as
    the Hugging Face model.
    """

    factor: float
    original_max_position_embeddings: int
    beta_fast: float
    beta_slow: float
    mscale: float
    mscale_all_dim: float
    truncate: bool
    rope_theta: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.factor) or self.factor <= 1:
            raise ValueError("yarn factor must be a finite number > 1")
        if self.original_max_position_embeddings <= 0:
            raise ValueError("yarn original_max_position_embeddings must be positive")
        if (
            not (math.isfinite(self.beta_fast) and math.isfinite(self.beta_slow))
            or not self.beta_fast > self.beta_slow > 0
        ):
            raise ValueError("yarn requires beta_fast > beta_slow > 0")
        if (
            not (math.isfinite(self.mscale) and math.isfinite(self.mscale_all_dim))
            or min(self.mscale, self.mscale_all_dim) < 0
        ):
            raise ValueError("yarn mscale and mscale_all_dim must be finite and >= 0")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be a finite positive number")

    @property
    def attention_factor(self) -> float:
        """The factor on cosine and sine, as Hugging Face derives it from the mscale terms."""
        return yarn_mscale(self.factor, self.mscale) / yarn_mscale(self.factor, self.mscale_all_dim)


def read_yarn_rope_scaling(hf_checkpoint: str) -> YarnRopeScaling | None:
    """The YaRN scaling of ``hf_checkpoint/config.json``, or ``None`` for a default rotary embedding.

    Reads ``rope_parameters`` (newer transformers) or ``rope_scaling``, typed by
    ``rope_type`` or ``type``. Refuses a type other than ``default`` or ``yarn``,
    and a yarn ``attention_factor`` that the mscale terms do not reproduce,
    because Megatron's embedding takes the terms, not the factor.
    """
    config = json.loads((Path(hf_checkpoint) / "config.json").read_text(encoding="utf-8"))
    rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    if rope_type == "default":
        return None
    if rope_type != "yarn":
        raise ValueError(f"Reef's model provider supports default and yarn rotary scaling, not {rope_type!r}")
    rope_theta = rope.get("rope_theta", config.get("rope_theta"))
    if "factor" not in rope or "original_max_position_embeddings" not in rope or rope_theta is None:
        raise ValueError("yarn rope scaling needs factor, original_max_position_embeddings and rope_theta")
    scaling = YarnRopeScaling(
        factor=float(rope["factor"]),
        original_max_position_embeddings=int(rope["original_max_position_embeddings"]),
        beta_fast=float(rope.get("beta_fast", 32.0)),
        beta_slow=float(rope.get("beta_slow", 1.0)),
        mscale=float(rope.get("mscale", 1.0)),
        mscale_all_dim=float(rope.get("mscale_all_dim", 0.0)),
        truncate=bool(rope.get("truncate", True)),
        rope_theta=float(rope_theta),
    )
    attention_factor = rope.get("attention_factor")
    if attention_factor is not None and not math.isclose(
        float(attention_factor), scaling.attention_factor, rel_tol=1e-6
    ):
        raise ValueError(
            f"yarn attention_factor {attention_factor} is not the mscale-derived {scaling.attention_factor:.6f}; "
            "Megatron's YaRN embedding takes mscale and mscale_all_dim, not an explicit factor"
        )
    return scaling


def install_yarn_rotary_embedding(model, args, scaling: YarnRopeScaling) -> None:
    """Switch a Megatron model built with ``rope`` to YaRN, as Hugging Face serves the checkpoint.

    The pinned Megatron accepts no ``yarn`` position embedding on its command
    line, so the model is built with ``rope`` and switched here: the YaRN
    fields go onto its config, ``position_embedding_type`` becomes ``yarn``,
    the fused RoPE kernel is off because it does not scale cosine and sine,
    and the rotary embedding is rebuilt. ``--rotary-base`` must equal the
    config's ``rope_theta``.
    """
    from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding

    if not math.isclose(float(args.rotary_base), scaling.rope_theta, rel_tol=1e-9):
        raise ValueError(
            f"--rotary-base {args.rotary_base} differs from the checkpoint's rope_theta {scaling.rope_theta}"
        )
    config = model.config
    config.yarn_rotary_scaling_factor = scaling.factor
    config.yarn_original_max_position_embeddings = scaling.original_max_position_embeddings
    config.yarn_beta_fast = scaling.beta_fast
    config.yarn_beta_slow = scaling.beta_slow
    config.yarn_mscale = scaling.mscale
    config.yarn_mscale_all_dim = scaling.mscale_all_dim
    config.yarn_correction_range_round_to_int = scaling.truncate
    config.apply_rope_fusion = False
    model.position_embedding_type = "yarn"
    model.rotary_pos_emb = YarnRotaryEmbedding(
        kv_channels=config.kv_channels,
        rotary_percent=args.rotary_percent,
        rotary_interleaved=config.rotary_interleaved,
        rotary_base=args.rotary_base,
        scaling_factor=scaling.factor,
        original_max_position_embeddings=scaling.original_max_position_embeddings,
        beta_fast=scaling.beta_fast,
        beta_slow=scaling.beta_slow,
        mscale=scaling.mscale,
        mscale_all_dim=scaling.mscale_all_dim,
        correction_range_round_to_int=scaling.truncate,
        use_cpu_initialization=config.use_cpu_initialization,
        cp_group=model.pg_collection.cp,
    )
    logger.info(
        "Reef installed YaRN rotary scaling from the checkpoint config: factor %s, original context %d, "
        "attention factor %.4f, RoPE fusion off",
        scaling.factor,
        scaling.original_max_position_embeddings,
        scaling.attention_factor,
    )


def provide_actor_model(pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None):
    """Build through Slime (or a chained provider), install the checkpoint's YaRN, then apply Reef LoRA."""
    from megatron.training.global_vars import get_args
    from slime.backends.megatron_utils.model_provider import _get_model_provider_func
    from slime.utils.misc import load_function

    from reef.train.slime_backend.reef_adapters.megatron.lora import apply_megatron_lora

    args = get_args()
    chained_path = getattr(args, "reef_chained_model_provider_path", None)
    if chained_path:
        chained = load_function(chained_path)
        kwargs: dict[str, object] = {"pre_process": pre_process, "post_process": post_process}
        if "vp_stage" in inspect.signature(chained).parameters:
            kwargs["vp_stage"] = vp_stage
        model = chained(**kwargs)
    else:
        current_path = args.custom_model_provider_path
        args.custom_model_provider_path = None
        try:
            provider = _get_model_provider_func(args, role="actor")
            model = provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        finally:
            args.custom_model_provider_path = current_path
    scaling = read_yarn_rope_scaling(args.hf_checkpoint)
    if scaling is not None:
        install_yarn_rotary_embedding(model, args, scaling)
    return apply_megatron_lora(model, args)


__all__ = [
    "YarnRopeScaling",
    "install_yarn_rotary_embedding",
    "provide_actor_model",
    "read_yarn_rope_scaling",
    "yarn_mscale",
]
