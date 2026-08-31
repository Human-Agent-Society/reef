"""Megatron-Bridge bridge for Qwen3.5 as ``slime_plugins.models.qwen3_5`` builds it.

Megatron-Bridge (>= 0.4) registers ``Qwen3_5ForConditionalGeneration`` to its own
``Qwen3VLModel`` layout: parameters live under ``language_model.decoder.*`` and
the GatedDeltaNet layers are megatron-core modules named
``self_attention.in_proj`` / ``out_norm`` / ``conv1d``. slime builds Qwen3.5
differently — a plain ``GPTModel`` whose linear-attention layers drop in an HF
mirror module (``self_attention.linear_attn.in_proj_qkv`` and friends, with the
HF tensor layouts verbatim) — so Bridge's tasks would never match the actor's
parameter names and adapter export would produce nothing.

This bridge maps slime's layout. It exists for weight *export* (LoRA adapter
publication and, in colocated bridge mode, base weights); slime remains the
model provider, so ``provider_bridge`` is deliberately unimplemented.

Registering it overrides Bridge's own registration for the same HF class —
which is the point: inside Reef the slime layout is the only Qwen3.5 layout a
Megatron actor ever has.
"""

from __future__ import annotations

import logging
import warnings

logger = logging.getLogger(__name__)

QWEN35_DENSE_HF_CLASS_NAME = "Qwen3_5ForConditionalGeneration"

_HF_LAYER = "model.language_model.layers.*"
_MEGATRON_LAYER = "decoder.layers.*"

# Megatron parameter -> HF parameter, one tensor each, on megatron-core modules
# whose tensor-parallel layout AutoMapping infers from the module class.
DIRECT_MAPPINGS: dict[str, str] = {
    "embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
    "output_layer.weight": "lm_head.weight",
    "decoder.final_layernorm.weight": "model.language_model.norm.weight",
    # Every layer: dense MLP with the pre-MLP norm fused into linear_fc1.
    f"{_MEGATRON_LAYER}.mlp.linear_fc1.layer_norm_weight": f"{_HF_LAYER}.post_attention_layernorm.weight",
    f"{_MEGATRON_LAYER}.mlp.linear_fc2.weight": f"{_HF_LAYER}.mlp.down_proj.weight",
    # Full-attention layers (every ``full_attention_interval``-th layer):
    # standard megatron-core gated attention with the input norm fused into linear_qkv.
    f"{_MEGATRON_LAYER}.self_attention.linear_qkv.layer_norm_weight": f"{_HF_LAYER}.input_layernorm.weight",
    f"{_MEGATRON_LAYER}.self_attention.q_layernorm.weight": f"{_HF_LAYER}.self_attn.q_norm.weight",
    f"{_MEGATRON_LAYER}.self_attention.k_layernorm.weight": f"{_HF_LAYER}.self_attn.k_norm.weight",
    f"{_MEGATRON_LAYER}.self_attention.linear_proj.weight": f"{_HF_LAYER}.self_attn.o_proj.weight",
}

# Linear-attention (GatedDeltaNet) layers: slime's ``Attention`` module keeps the
# HF module tree and tensor layouts one-to-one under ``self_attention``, on plain
# torch modules that are replicated across tensor-parallel ranks.
REPLICATED_MAPPINGS: dict[str, str] = {
    f"{_MEGATRON_LAYER}.self_attention.input_layernorm.weight": f"{_HF_LAYER}.input_layernorm.weight",
    **{
        f"{_MEGATRON_LAYER}.self_attention.linear_attn.{name}": f"{_HF_LAYER}.linear_attn.{name}"
        for name in (
            "in_proj_qkv.weight",
            "in_proj_z.weight",
            "in_proj_b.weight",
            "in_proj_a.weight",
            "conv1d.weight",
            "out_proj.weight",
            "norm.weight",
            "A_log",
            "dt_bias",
        )
    },
}

QKV_MAPPING = {
    "megatron_param": f"{_MEGATRON_LAYER}.self_attention.linear_qkv.weight",
    "q": f"{_HF_LAYER}.self_attn.q_proj.weight",
    "k": f"{_HF_LAYER}.self_attn.k_proj.weight",
    "v": f"{_HF_LAYER}.self_attn.v_proj.weight",
}

GATED_MLP_MAPPING = {
    "megatron_param": f"{_MEGATRON_LAYER}.mlp.linear_fc1.weight",
    "gate": f"{_HF_LAYER}.mlp.gate_proj.weight",
    "up": f"{_HF_LAYER}.mlp.up_proj.weight",
}

_registered = False


def register_slime_qwen35_bridge() -> None:
    """Register the slime-layout Qwen3.5 bridge, replacing Bridge's VL registration."""

    global _registered
    if _registered:
        return

    # Importing the package registers Bridge's own bridges; ours must come after
    # so the dispatch resolves the HF class to it.
    import megatron.bridge  # noqa: F401
    from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
    from megatron.bridge.models.conversion.param_mapping import (
        AutoMapping,
        GatedMLPMapping,
        QKVMapping,
        ReplicatedMapping,
    )
    from megatron.core.models.gpt.gpt_model import GPTModel

    class SlimeQwen35Bridge(MegatronModelBridge):
        """Export-only bridge for slime's Qwen3.5 dense layout."""

        def provider_bridge(self, hf_pretrained):
            raise NotImplementedError(
                "Reef builds Qwen3.5 through slime_plugins.models.qwen3_5; this bridge only "
                "converts weights and does not provide a Megatron model"
            )

        def mapping_registry(self) -> MegatronMappingRegistry:
            mappings = [
                AutoMapping(megatron_param=megatron_param, hf_param=hf_param)
                for megatron_param, hf_param in DIRECT_MAPPINGS.items()
            ]
            mappings.extend(
                ReplicatedMapping(megatron_param=megatron_param, hf_param=hf_param)
                for megatron_param, hf_param in REPLICATED_MAPPINGS.items()
            )
            mappings.append(QKVMapping(**QKV_MAPPING))
            mappings.append(GatedMLPMapping(**GATED_MLP_MAPPING))
            return MegatronMappingRegistry(*mappings)

    with warnings.catch_warnings():
        # plum reports the deliberate override of Bridge's own registration.
        warnings.simplefilter("ignore")
        MegatronModelBridge.register_bridge(
            source=QWEN35_DENSE_HF_CLASS_NAME,
            target=GPTModel,
            model_type="qwen3_5",
        )(SlimeQwen35Bridge)
    _registered = True
    logger.info("Registered the slime-layout Megatron-Bridge bridge for %s", QWEN35_DENSE_HF_CLASS_NAME)


__all__ = [
    "DIRECT_MAPPINGS",
    "GATED_MLP_MAPPING",
    "QKV_MAPPING",
    "QWEN35_DENSE_HF_CLASS_NAME",
    "REPLICATED_MAPPINGS",
    "register_slime_qwen35_bridge",
]
