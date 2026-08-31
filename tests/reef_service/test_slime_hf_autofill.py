"""Unit tests for the HF architecture argument adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from reef.train.slime_backend.reef_adapters.megatron.hf_arguments import explicit_flag_names, hf_architecture_arguments


def _dense_hf_config(**overrides):
    base = SimpleNamespace(
        num_hidden_layers=28,
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-6,
        rope_theta=1000000,
        tie_word_embeddings=False,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _flags(arguments):
    return {token.split("=", 1)[0]: token.split("=", 1)[1] if "=" in token else True for token in arguments}


@pytest.mark.unit
def test_explicit_flag_names_records_flags_ignoring_values() -> None:
    names = explicit_flag_names(["--num-layers", "28", "--hidden-size=4096", "--lr", "1e-6", "positional"])
    assert names == {"--num-layers", "--hidden-size", "--lr"}


@pytest.mark.unit
def test_autofill_fills_all_architecture_fields_from_hf_config() -> None:
    flags = _flags(hf_architecture_arguments([], _dense_hf_config()))

    assert flags["--num-layers"] == "28"
    assert flags["--hidden-size"] == "4096"
    assert flags["--ffn-hidden-size"] == "11008"
    assert flags["--num-attention-heads"] == "32"
    assert flags["--num-query-groups"] == "8"
    # Fewer KV heads than query heads -> grouped-query attention must be enabled
    # so Megatron actually applies num_query_groups (otherwise it sizes K/V for
    # the full head count and the QKV weight fails to load).
    assert flags["--group-query-attention"] is True
    assert flags["--norm-epsilon"] == "1e-06"
    assert flags["--layernorm-epsilon"] == "1e-06"
    assert flags["--untie-embeddings-and-output-weights"] is True
    assert flags["--rotary-base"] == "1000000"


@pytest.mark.unit
def test_autofill_leaves_gqa_off_for_multi_head_attention() -> None:
    # num_key_value_heads == num_attention_heads is plain multi-head attention;
    # group_query_attention must stay off so Megatron keeps full-head K/V.
    flags = _flags(hf_architecture_arguments([], _dense_hf_config(num_attention_heads=32, num_key_value_heads=32)))

    assert flags["--num-query-groups"] == "32"
    assert "--group-query-attention" not in flags


@pytest.mark.unit
def test_autofill_respects_explicit_group_query_attention_flag() -> None:
    # If the caller passed --group-query-attention, auto-fill must not override
    # the parsed value (leaving mismatches for validation, like other fields).
    arguments = ["--group-query-attention"]
    result = hf_architecture_arguments(arguments, _dense_hf_config(num_attention_heads=32, num_key_value_heads=8))

    assert result.count("--group-query-attention") == 1


@pytest.mark.unit
def test_autofill_does_not_touch_explicitly_passed_flags() -> None:
    result = hf_architecture_arguments(["--num-layers", "99"], _dense_hf_config())
    flags = _flags(result)

    assert result[:2] == ["--num-layers", "99"]
    assert result.count("--num-layers") == 1
    assert flags["--hidden-size"] == "4096"
    assert flags["--num-attention-heads"] == "32"


@pytest.mark.unit
def test_autofill_handles_multimodal_text_config_nesting() -> None:
    text_config = _dense_hf_config(num_hidden_layers=12, hidden_size=2048)
    hf = SimpleNamespace(text_config=text_config, architectures=["Qwen3VLForCausalLM"])
    flags = _flags(hf_architecture_arguments([], hf))

    assert flags["--num-layers"] == "12"
    assert flags["--hidden-size"] == "2048"


@pytest.mark.unit
def test_autofill_prefers_rope_parameters_dict_over_attribute() -> None:
    hf = _dense_hf_config()
    # rope_parameters dict wins over a stale top-level rope_theta.
    hf.rope_parameters = {"rope_theta": 500000}
    hf.rope_theta = 1  # should be ignored
    flags = _flags(hf_architecture_arguments([], hf))

    assert flags["--rotary-base"] == "500000"


@pytest.mark.unit
def test_autofill_inverts_tie_word_embeddings_and_skips_when_absent() -> None:
    # tie=False -> untie=True
    flags = _flags(hf_architecture_arguments([], _dense_hf_config(tie_word_embeddings=False)))
    assert flags["--untie-embeddings-and-output-weights"] is True

    # tie=True -> untie=False (tied embeddings)
    flags = _flags(hf_architecture_arguments([], _dense_hf_config(tie_word_embeddings=True)))
    assert "--untie-embeddings-and-output-weights" not in flags


@pytest.mark.unit
def test_autofill_fills_moe_fields_and_expert_count() -> None:
    hf = SimpleNamespace(
        num_hidden_layers=28,
        hidden_size=2048,
        num_attention_heads=16,
        moe_intermediate_size=1408,
        shared_expert_intermediate_size=2048,
        n_routed_experts=64,
        rms_norm_eps=1e-6,
        rope_theta=10000,
    )
    flags = _flags(hf_architecture_arguments([], hf))

    assert flags["--moe-ffn-hidden-size"] == "1408"
    assert flags["--moe-shared-expert-intermediate-size"] == "2048"
    assert flags["--num-experts"] == "64"


@pytest.mark.unit
def test_autofill_skips_dense_ffn_size_for_pure_moe_models() -> None:
    # Pure-MoE layout (no dense layer): every layer has moe_layer_freq 1.
    hf = SimpleNamespace(
        num_hidden_layers=4,
        hidden_size=2048,
        intermediate_size=4096,  # present but must NOT be used for dense FFN
        num_attention_heads=16,
        moe_intermediate_size=1408,
        n_routed_experts=8,
        rms_norm_eps=1e-6,
        rope_theta=10000,
    )
    flags = _flags(hf_architecture_arguments(["--moe-layer-freq=[1,1,1,1]"], hf))

    assert "--ffn-hidden-size" not in flags
    assert flags["--moe-ffn-hidden-size"] == "1408"
