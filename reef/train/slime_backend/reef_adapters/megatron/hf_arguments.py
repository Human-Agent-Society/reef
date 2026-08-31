"""Derive Megatron architecture CLI flags from a Hugging Face config.

The runtime validates the two configurations but intentionally leaves model
architecture flags to the launcher. Reef deployments already name the exact
HF checkpoint, so this adapter adds only missing flags before Slime parses the
command line. Explicit launcher values always win and remain subject to
Slime's normal mismatch validation.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import Any


def explicit_flag_names(arguments: Sequence[str]) -> set[str]:
    return {token.split("=", 1)[0] for token in arguments if token.startswith("--")}


def _option_value(arguments: Sequence[str], name: str) -> str | None:
    for index, token in enumerate(arguments):
        if token.startswith(f"{name}="):
            return token.split("=", 1)[1]
        if token == name and index + 1 < len(arguments):
            return arguments[index + 1]
    return None


def _language_config(config: Any) -> Any:
    return getattr(config, "text_config", config)


def _is_moe_config(config: Any) -> bool:
    return any(
        hasattr(config, name)
        for name in ("moe_intermediate_size", "num_experts", "n_routed_experts", "num_local_experts")
    )


def _has_dense_moe_layers(arguments: Sequence[str]) -> bool:
    value = _option_value(arguments, "--moe-layer-freq")
    if value is None:
        return True
    try:
        frequencies = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return "0" in value
    if isinstance(frequencies, int | str):
        return int(frequencies) == 0
    return any(int(frequency) == 0 for frequency in frequencies)


def hf_architecture_arguments(arguments: Sequence[str], config: Any) -> list[str]:
    """Return ``arguments`` plus architecture flags missing from the CLI."""
    result = list(arguments)
    explicit = explicit_flag_names(result)
    language = _language_config(config)
    fields = [
        ("num_hidden_layers", "--num-layers", lambda value: value),
        ("hidden_size", "--hidden-size", lambda value: value),
        ("intermediate_size", "--ffn-hidden-size", lambda value: value),
        ("num_attention_heads", "--num-attention-heads", lambda value: value),
        ("num_key_value_heads", "--num-query-groups", lambda value: value),
        ("head_dim", "--kv-channels", lambda value: value),
        ("rms_norm_eps", "--norm-epsilon", lambda value: value),
        ("rms_norm_eps", "--layernorm-epsilon", lambda value: value),
        ("moe_intermediate_size", "--moe-ffn-hidden-size", lambda value: value),
        ("shared_expert_intermediate_size", "--moe-shared-expert-intermediate-size", lambda value: value),
    ]
    validate_dense_ffn = not _is_moe_config(language) or _has_dense_moe_layers(result)
    for attribute, flag, transform in fields:
        if flag in explicit or not hasattr(language, attribute):
            continue
        if attribute == "intermediate_size" and not validate_dense_ffn:
            continue
        value = transform(getattr(language, attribute))
        if value is not None:
            result.append(f"{flag}={value}")
            explicit.add(flag)

    if "--group-query-attention" not in explicit:
        kv_heads = getattr(language, "num_key_value_heads", None)
        attention_heads = getattr(language, "num_attention_heads", None)
        if kv_heads is not None and attention_heads is not None and kv_heads < attention_heads:
            result.append("--group-query-attention")

    rope = getattr(language, "rope_parameters", None)
    rope_theta = rope.get("rope_theta") if isinstance(rope, dict) else getattr(language, "rope_theta", None)
    if "--rotary-base" not in explicit and rope_theta is not None:
        result.append(f"--rotary-base={int(rope_theta)}")

    if (
        "--untie-embeddings-and-output-weights" not in explicit
        and getattr(language, "tie_word_embeddings", True) is False
    ):
        result.append("--untie-embeddings-and-output-weights")

    if "--num-experts" not in explicit:
        for attribute in ("num_experts", "n_routed_experts", "num_local_experts"):
            value = getattr(language, attribute, None)
            if value is not None:
                result.append(f"--num-experts={value}")
                break
    return result


def add_hf_architecture_arguments(arguments: Sequence[str]) -> list[str]:
    checkpoint = _option_value(arguments, "--hf-checkpoint")
    if not checkpoint:
        return list(arguments)
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    return hf_architecture_arguments(arguments, config)


__all__ = ["add_hf_architecture_arguments", "explicit_flag_names", "hf_architecture_arguments"]
