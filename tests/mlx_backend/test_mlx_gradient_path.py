"""The gradient path through GatedDeltaNet layers, and the chunkwise recurrence that makes it trainable.

mlx-lm's GatedDeltaNet uses a Metal kernel without a vjp in eval mode and a
differentiable loop in training mode. The trainer switches only the layers
on the gradient path to training mode, for the span of a backward pass. The
model here is a synthetic ``qwen3_5`` hybrid of a few tiny layers, so no
model is downloaded.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core", reason="the MLX trainer needs the optional mlx extra")
pytest.importorskip("mlx_lm", reason="the MLX trainer needs the optional mlx extra")

import mlx.nn as nn
from mlx_lm.models import gated_delta, qwen3_5
from mlx_lm.tuner import linear_to_lora_layers

from reef.train.mlx_backend.config import MLXEngineConfig
from reef.train.mlx_backend.gated_delta import (
    SEQUENTIAL_GATED_DELTA_OPS,
    chunkwise_gated_delta_ops,
    use_chunkwise_gated_delta,
)
from reef.train.mlx_backend.training import (
    MLXEngineError,
    MLXTrainer,
    ModelParts,
    PackedRows,
    ResponseTokens,
    TokenObjective,
    gradient_path_layers,
)

pytestmark = pytest.mark.unit

LAYERS = 8
#: Every fourth layer is full attention, as in the real model; layers 3 and 7 here.
INTERVAL = 4
#: Both kinds of layer are adapted, so the gradient reaches a GatedDeltaNet's own projections.
LORA_KEYS = ("self_attn.q_proj", "linear_attn.in_proj_qkv", "mlp.down_proj")


def tiny_hybrid() -> nn.Module:
    text = {
        "hidden_size": 64,
        "num_hidden_layers": LAYERS,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "full_attention_interval": INTERVAL,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "attn_output_gate": True,
        "partial_rotary_factor": 0.25,
        "max_position_embeddings": 4096,
    }
    return qwen3_5.Model(qwen3_5.ModelArgs(model_type="qwen3_5", text_config=text))


def adapted_parts(lora_layers: int = 6) -> ModelParts:
    mx.random.seed(0)
    model = tiny_hybrid()
    model.eval()
    model.freeze()
    linear_to_lora_layers(model, lora_layers, {"rank": 4, "scale": 2.0, "dropout": 0.0, "keys": list(LORA_KEYS)})
    return ModelParts.of(model)


def trainer_for(parts: ModelParts, **options) -> MLXTrainer:
    return MLXTrainer(parts, MLXEngineConfig(lora_layers=6, lora_rank=4, lora_keys=LORA_KEYS, **options), 0)


def packed(trainer: MLXTrainer) -> PackedRows:
    return trainer.pack(
        [ResponseTokens(tuple((index * 7 + position) % 200 + 1 for position in range(24)), 16) for index in range(2)]
    )


class ObservedLogProbs(TokenObjective):
    """The summed response log probabilities; records what the objective observes while the trainer differentiates."""

    def __init__(self, packed_rows: PackedRows, parts: ModelParts) -> None:
        self._targets = packed_rows.targets
        self._mask = packed_rows.response_mask
        self._parts = parts
        self.unwrapped_calls = {type(layer): type(layer).__call__ for layer in parts.layers}
        self.layer_modes: list[list[bool]] = []
        self.chunkwise_recurrence: list[bool] = []
        self.wrapped_calls: list[bool] = []

    def loss(self, logits: mx.array, begin: int, stop: int) -> mx.array:
        layers = self._parts.layers
        self.layer_modes.append([layer.training for layer in layers])
        self.chunkwise_recurrence.append(gated_delta.gated_delta_ops is not SEQUENTIAL_GATED_DELTA_OPS)
        self.wrapped_calls.append(type(layers[-1]).__call__ is not self.unwrapped_calls[type(layers[-1])])
        log_probs = -nn.losses.cross_entropy(logits, self._targets[:, begin:stop], reduction="none")
        return (log_probs * self._mask[:, begin:stop]).sum()


def layer_index(name: str) -> int:
    parts = name.split(".")
    return int(parts[parts.index("layers") + 1])


def test_the_gradient_path_starts_at_the_lowest_adapted_layer() -> None:
    parts = adapted_parts(lora_layers=6)

    assert trainer_for(parts).gradient_path == parts.layers[LAYERS - 6 :]


def test_a_model_without_an_adapter_has_no_gradient_path() -> None:
    model = tiny_hybrid()
    model.freeze()

    assert gradient_path_layers(ModelParts.of(model)) == []


def test_a_model_that_is_not_an_mlx_lm_text_model_is_refused() -> None:
    class Opaque(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)

    with pytest.raises(MLXEngineError, match="decoder layers"):
        ModelParts.of(Opaque())


def test_the_backward_crosses_gated_delta_layers_and_only_the_path_leaves_eval_mode() -> None:
    parts = adapted_parts()
    trainer = trainer_for(parts)
    rows = packed(trainer)
    objective = ObservedLogProbs(rows, parts)

    _, gradients = trainer.gradients(rows, objective)

    moved = {layer_index(name) for name, gradient in gradients.items() if float(mx.abs(gradient).sum()) > 0}
    # Layers 2 to 7: GatedDeltaNet at 2, 4, 5 and 6, full attention at 3 and 7.
    assert moved == set(range(LAYERS - 6, LAYERS))
    assert objective.layer_modes == [[False] * (LAYERS - 6) + [True] * 6]
    assert [layer.training for layer in parts.layers] == [False] * LAYERS
    assert objective.chunkwise_recurrence == [True]
    assert gated_delta.gated_delta_ops is SEQUENTIAL_GATED_DELTA_OPS


def test_in_eval_mode_the_gated_delta_kernel_has_no_backward() -> None:
    """The reason the trainer switches the path to training mode; this fails once mlx-lm gives the kernel a vjp."""
    parts = adapted_parts()
    tokens = mx.array([[(position * 7) % 200 + 1 for position in range(24)]])

    def loss(model: nn.Module) -> mx.array:
        return model(tokens).sum()

    with pytest.raises(ValueError, match="vjp"):
        mx.eval(nn.value_and_grad(parts.model, loss)(parts.model))


def test_checkpointed_layers_give_the_same_gradients_and_are_unwrapped_afterwards() -> None:
    parts = adapted_parts()
    plain = trainer_for(parts)
    rows = packed(plain)
    _, expected = plain.gradients(rows, ObservedLogProbs(rows, parts))
    objective = ObservedLogProbs(rows, parts)

    _, actual = trainer_for(parts, checkpoint_layers=True).gradients(rows, objective)

    assert objective.wrapped_calls == [True]
    assert all(type(layer).__call__ is objective.unwrapped_calls[type(layer)] for layer in parts.layers)
    for name, gradient in expected.items():
        assert relative_deviation(gradient, actual[name]) < 1e-5


# -- The chunkwise recurrence


def recurrence_inputs(length: int):
    mx.random.seed(0)
    batch, key_heads, value_heads, key_dim, value_dim = 2, 2, 4, 32, 32
    # Queries and keys as the model passes them: RMS-normalised and scaled by
    # 1/sqrt(Dk). The delta rule is stable only for such keys.
    scale = key_dim**-0.5
    q = scale**2 * mx.fast.rms_norm(mx.random.normal((batch, length, key_heads, key_dim)), None, 1e-6)
    k = scale * mx.fast.rms_norm(mx.random.normal((batch, length, key_heads, key_dim)), None, 1e-6)
    v = mx.random.normal((batch, length, value_heads, value_dim))
    g = mx.random.uniform(shape=(batch, length, value_heads)) * 0.5 + 0.5
    beta = mx.random.uniform(shape=(batch, length, value_heads))
    state = mx.random.normal((batch, value_heads, value_dim, key_dim)) * 0.1
    mask = mx.array([[1] * length, [1] * (length - 17) + [0] * 17]).astype(mx.bool_)
    return (q, k, v, g, beta, state), mask


def relative_deviation(expected: mx.array, actual: mx.array) -> float:
    return float(mx.max(mx.abs(expected - actual)) / mx.maximum(mx.max(mx.abs(expected)), mx.array(1e-12)))


def value_and_gradients(recurrence, inputs, mask):
    def loss(q, k, v, g, beta, state):
        y, final = recurrence(q, k, v, g, beta, state, mask)
        # The chunkwise form emits zeros at masked positions; the loop does
        # not. Those outputs are padding, so the loss leaves them out.
        weight = 1.0 if mask is None else mask[..., None, None].astype(y.dtype)
        return ((y * weight) ** 2).sum() + (final**2).sum()

    value, gradients = mx.value_and_grad(loss, argnums=(0, 1, 2, 3, 4, 5))(*inputs)
    mx.eval(value, gradients)
    return value, gradients


@pytest.mark.parametrize("chunk_size", [64, 16])
@pytest.mark.parametrize("masked", [False, True])
def test_the_chunkwise_recurrence_matches_mlx_lm_in_value_and_gradient(masked: bool, chunk_size: int) -> None:
    # 77 tokens: whole chunks and a padded tail; two key heads to four value
    # heads; a non-zero starting state; a mask that freezes one row partway.
    inputs, mask = recurrence_inputs(77)
    mask = mask if masked else None

    reference_loss, reference_gradients = value_and_gradients(SEQUENTIAL_GATED_DELTA_OPS, inputs, mask)
    loss, gradients = value_and_gradients(chunkwise_gated_delta_ops(chunk_size), inputs, mask)

    assert abs(float(loss) - float(reference_loss)) / abs(float(reference_loss)) < 1e-4
    for expected, actual in zip(reference_gradients, gradients, strict=True):
        assert relative_deviation(expected, actual) < 1e-3


def test_the_chunkwise_recurrence_is_stable_on_keys_that_barely_change_between_tokens() -> None:
    (q, _, v, _, _, state), _ = recurrence_inputs(77)
    mx.random.seed(1)
    direction = mx.random.normal((1, 1, 2, 32))
    k = 32**-0.5 * mx.fast.rms_norm(direction + 0.02 * mx.random.normal((2, 77, 2, 32)), None, 1e-6)
    # Nearly whole writes and nearly no decay, so nothing damps a cancellation.
    beta = 0.9 + 0.1 * mx.random.uniform(shape=(2, 77, 4))
    g = 0.95 + 0.05 * mx.random.uniform(shape=(2, 77, 4))
    inputs = (q, k, v, g, beta, state)

    reference_loss, reference_gradients = value_and_gradients(SEQUENTIAL_GATED_DELTA_OPS, inputs, None)
    loss, gradients = value_and_gradients(chunkwise_gated_delta_ops(64), inputs, None)

    assert abs(float(loss) - float(reference_loss)) / abs(float(reference_loss)) < 1e-4
    for expected, actual in zip(reference_gradients, gradients, strict=True):
        assert relative_deviation(expected, actual) < 1e-3


def test_per_column_gating_uses_mlx_lm_s_loop_also_while_the_recurrence_is_rerouted() -> None:
    (q, k, v, _, beta, state), mask = recurrence_inputs(40)
    g = mx.random.uniform(shape=(2, 40, 4, 32)) * 0.5 + 0.5
    expected = SEQUENTIAL_GATED_DELTA_OPS(q, k, v, g, beta, state, mask)

    with use_chunkwise_gated_delta(64):
        actual = gated_delta.gated_delta_ops(q, k, v, g, beta, state, mask)
        mx.eval(actual)

    mx.eval(expected)
    for want, got in zip(expected, actual, strict=True):
        assert float(mx.max(mx.abs(want - got))) == 0.0
    assert gated_delta.gated_delta_ops is SEQUENTIAL_GATED_DELTA_OPS


def test_a_chunk_size_of_zero_leaves_mlx_lm_unchanged() -> None:
    with use_chunkwise_gated_delta(0):
        assert gated_delta.gated_delta_ops is SEQUENTIAL_GATED_DELTA_OPS


def test_the_backward_keeps_one_state_per_chunk_rather_than_several_per_token() -> None:
    # The head shape of the real model, so per-token states dominate the
    # loop's backward. The bound is loose: it fails on a design regression,
    # not on allocator noise.
    key_heads, value_heads, key_dim, value_dim, length = 4, 8, 128, 128, 512
    state_bytes = value_heads * value_dim * key_dim * 4

    def peak_of(recurrence) -> int:
        mx.random.seed(0)
        scale = key_dim**-0.5
        q = scale**2 * mx.fast.rms_norm(mx.random.normal((1, length, key_heads, key_dim)), None, 1e-6)
        k = scale * mx.fast.rms_norm(mx.random.normal((1, length, key_heads, key_dim)), None, 1e-6)
        v = mx.random.normal((1, length, value_heads, value_dim))
        g = mx.random.uniform(shape=(1, length, value_heads)) * 0.5 + 0.5
        beta = mx.random.uniform(shape=(1, length, value_heads))
        mx.eval(q, k, v, g, beta)

        def loss(q, k, v, g, beta):
            y, _ = recurrence(q, k, v, g, beta, None, None)
            return (y**2).sum()

        mx.reset_peak_memory()
        base = mx.get_active_memory()
        value, gradients = mx.value_and_grad(loss, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
        mx.eval(value, gradients)
        return mx.get_peak_memory() - base

    loop = peak_of(SEQUENTIAL_GATED_DELTA_OPS)
    chunkwise = peak_of(chunkwise_gated_delta_ops(64))
    assert loop > length * state_bytes
    assert chunkwise < loop / 4


def test_a_chunk_size_that_is_not_a_power_of_two_is_refused() -> None:
    with pytest.raises(ValueError, match="power of two"):
        chunkwise_gated_delta_ops(48)
