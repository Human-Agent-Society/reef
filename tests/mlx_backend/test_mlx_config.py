"""Validation of the MLX engine and deployment settings; no MLX needed."""

from __future__ import annotations

import pytest

from reef.train.mlx_backend.config import MLXConfig, MLXEngineConfig

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("lora_layers", "lora_keys"),
    [
        (4, ("self_attn.q_proj",)),
        (1, ("self_attn.q_proj", "self_attn.v_proj")),
        (1, ("self_attn.k_proj",)),
        (1, ("linear_attn.in_proj_qkv",)),
    ],
    ids=["lower-layers", "values", "keys", "recurrent-state"],
)
def test_a_cached_prompt_is_refused_when_its_cache_depends_on_trained_weights(
    lora_layers: int, lora_keys: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="prefill_step_size"):
        MLXEngineConfig(lora_layers=lora_layers, lora_keys=lora_keys, prefill_step_size=2048)


def test_a_cached_prompt_is_allowed_when_only_the_last_layer_is_adapted_outside_its_cache() -> None:
    lora_keys = ("self_attn.q_proj", "self_attn.o_proj", "mlp.down_proj")

    config = MLXEngineConfig(lora_layers=1, lora_keys=lora_keys, prefill_step_size=2048)

    assert config.prefill_step_size == 2048


@pytest.mark.parametrize("name", ["add_generation_prompt", "tokenize", "tools"])
def test_the_engine_keeps_the_template_arguments_that_decide_the_prompt(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        MLXEngineConfig(chat_template_kwargs={name: False})


def test_a_deployment_can_default_a_template_flag() -> None:
    config = MLXEngineConfig(chat_template_kwargs={"enable_thinking": False})

    assert config.chat_template_kwargs == {"enable_thinking": False}


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"recurrence_chunk_size": -1}, "recurrence_chunk_size"),
        ({"recurrence_chunk_size": 48}, "power of two"),
        ({"lora_rank": 0}, "lora_rank"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"top_p": 1.5}, "top_p"),
        ({"lora_keys": ()}, "lora_keys"),
    ],
    ids=["negative-chunk", "chunk-not-power-of-two", "rank", "learning-rate", "top-p", "no-keys"],
)
def test_an_invalid_engine_setting_is_refused(options: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MLXEngineConfig(**options)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"train_timeout_s": 30.0}, "no train timeout"),
        ({"kl_coef": -0.1}, "kl_coef"),
        ({"checkpoint_dir": " "}, "checkpoint_dir"),
    ],
    ids=["train-timeout", "kl-coef", "checkpoint-dir"],
)
def test_an_invalid_deployment_setting_is_refused(options: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MLXConfig(**{"checkpoint_dir": "work/checkpoints", **options})
