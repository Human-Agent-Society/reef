"""Qwen3.5's fused Q/K norm, RoPE, and gate kernel with M-RoPE positions.

``docker/patch/sglang_qwen3_5_mrope_rope.patch`` backports sgl-project/sglang#34446
to the SGLang pin. Before it, the kernel rotated image tokens with the temporal
position only. The tests skip when torch, Triton or SGLang is missing. On an
SGLang without the patch, they fail.

The tests use internals of the pin, such as the kernel's module path. Remove
them with the patch when the pin has the fix. Upstream tests then cover it.

Each check runs in a new process with ``TRITON_INTERPRET=1``. Triton chooses the
interpreter when ``@triton.jit`` runs, which is when a kernel module is imported.
The pytest process can import SGLang's kernels before this module does.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("sglang")

pytestmark = pytest.mark.unit

# Qwen3.5 full-attention layer shapes and rotary settings.
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 256
ROTARY_DIM = 64
MROPE_SECTION = [11, 11, 10]
ROPE_THETA = 10_000_000
EPS = 1e-6
# The Triton CPU interpreter converts float32 to bf16 by truncation. Torch, and
# the kernel on GPU, round to nearest. The kernel converts twice: after the norm
# and after the rotation. So the result can differ from the reference by two bf16
# steps. A bf16 step is the gap between two adjacent bf16 values. It is 0.03125
# for values in [4, 8).
ROUNDING_TOLERANCE = 0.0625


@pytest.mark.parametrize(
    "check_name", ["axis_map", "weight_checker_reset", "image_positions", "strided_positions", "text_positions"]
)
def test_patched_sglang(check_name):
    result = subprocess.run(
        [sys.executable, __file__, check_name],
        # Hide GPUs. On SM 90 and later the kernel enables PDL, which the interpreter cannot run.
        env={**os.environ, "TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def cpu_rotary_embedding(head_size, rotary_dim, mrope_section, interleaved=True, interleaved_glm=False):
    from sglang.srt.layers.rotary_embedding import base as rotary_base
    from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    # Use the CPU rotary path. Float32 keeps the cos/sin cache in float32, as on CUDA.
    rotary_base._is_cpu = True
    return MRotaryEmbedding(
        head_size,
        rotary_dim,
        4096,
        ROPE_THETA,
        True,
        torch.float32,
        mrope_section=mrope_section,
        mrope_interleaved=interleaved,
        mrope_interleaved_glm=interleaved_glm,
    )


def attention_layer():
    """A stand-in with the attributes ``forward_prepare_cuda_fused`` reads."""
    from sglang.srt.layers.fused_qk_rmsnorm_rope_gate import fused_qk_gemma_rmsnorm_rope_gate
    from sglang.srt.models import qwen3_5

    # The model imports the kernel only on CUDA.
    qwen3_5.fused_qk_gemma_rmsnorm_rope_gate = fused_qk_gemma_rmsnorm_rope_gate
    generator = torch.Generator().manual_seed(0)
    return SimpleNamespace(
        # The hidden states in these checks are already the projected Q+gate, K, and V.
        qkv_proj=lambda hidden_states: (hidden_states, None),
        attn_output_gate=True,
        q_size=NUM_Q_HEADS * HEAD_DIM,
        kv_size=NUM_KV_HEADS * HEAD_DIM,
        q_norm=SimpleNamespace(
            weight=0.1 * torch.randn(HEAD_DIM, generator=generator).bfloat16(), variance_epsilon=EPS
        ),
        k_norm=SimpleNamespace(
            weight=0.1 * torch.randn(HEAD_DIM, generator=generator).bfloat16(), variance_epsilon=EPS
        ),
        rotary_emb=cpu_rotary_embedding(HEAD_DIM, ROTARY_DIM, MROPE_SECTION),
        num_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
    )


def prepare_attention(layer, positions, hidden_states):
    from sglang.srt.models.qwen3_5 import Qwen3_5AttentionDecoderLayer

    return Qwen3_5AttentionDecoderLayer.forward_prepare_cuda_fused(layer, positions, hidden_states)


def projected_states(token_count):
    generator = torch.Generator().manual_seed(1)
    width = (2 * NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
    return (2 * torch.randn(token_count, width, generator=generator)).bfloat16()


def image_prompt_positions():
    """Positions of 8 text tokens, a 6 x 7 merged image, and 8 more text tokens."""
    text_before, grid_height, grid_width, text_after = 8, 6, 7, 8
    image_rows = torch.stack(
        [
            torch.full((grid_height * grid_width,), text_before),
            torch.arange(grid_height).repeat_interleave(grid_width) + text_before,
            torch.arange(grid_width).repeat(grid_height) + text_before,
        ]
    )
    after_start = text_before + max(grid_height, grid_width)
    positions = torch.cat(
        [
            torch.arange(text_before).expand(3, -1),
            image_rows,
            torch.arange(after_start, after_start + text_after).expand(3, -1),
        ],
        dim=1,
    ).contiguous()
    is_image = torch.zeros(positions.shape[1], dtype=torch.bool)
    is_image[text_before : text_before + grid_height * grid_width] = True
    return positions, is_image


def gemma_rmsnorm(states, weight):
    states = states.float()
    states = states * torch.rsqrt(states.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return (states * (1.0 + weight.float())).bfloat16()


def unfused_reference(layer, hidden_states, positions):
    """Normalize per head, then rotate in float32 with each lane's own position row."""
    from sglang.srt.layers.rotary_embedding.mrope import apply_interleaved_rope

    token_count = hidden_states.shape[0]
    q_gate, k, _ = hidden_states.split([2 * layer.q_size, layer.kv_size, layer.kv_size], dim=-1)
    q = gemma_rmsnorm(q_gate.view(token_count, NUM_Q_HEADS, 2 * HEAD_DIM)[..., :HEAD_DIM], layer.q_norm.weight)
    k = gemma_rmsnorm(k.view(token_count, NUM_KV_HEADS, HEAD_DIM), layer.k_norm.weight)
    cos, sin = layer.rotary_emb.cos_sin_cache[positions].float().chunk(2, dim=-1)
    cos = apply_interleaved_rope(cos, MROPE_SECTION)[:, None, :]
    sin = apply_interleaved_rope(sin, MROPE_SECTION)[:, None, :]

    def rotate(states):
        first, second = states[..., : ROTARY_DIM // 2].float(), states[..., ROTARY_DIM // 2 : ROTARY_DIM].float()
        rotated = torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)
        return torch.cat([rotated.bfloat16(), states[..., ROTARY_DIM:]], dim=-1).reshape(token_count, -1)

    return rotate(q), rotate(k)


def max_difference(left, right, rows):
    return float((left[rows].float() - right[rows].float()).abs().max())


def check_axis_map():
    from sglang.srt.layers.rotary_embedding.mrope import apply_interleaved_rope

    for section, rotary_dim in (([11, 11, 10], 64), ([24, 20, 20], 128)):
        rope = cpu_rotary_embedding(rotary_dim, rotary_dim, section)
        axis_rows = torch.arange(3).view(3, 1, 1).expand(3, 1, rotary_dim // 2)
        assert rope.axis_map.tolist() == apply_interleaved_rope(axis_rows, section).flatten().tolist()
        # The older rope kernels still get a map only for the GLM layout.
        assert rope._legacy_axis_map is None
    contiguous = cpu_rotary_embedding(128, 128, [24, 20, 20], interleaved=False)
    assert contiguous.axis_map.tolist() == [0] * 24 + [1] * 20 + [2] * 20
    assert contiguous._legacy_axis_map is None
    glm = cpu_rotary_embedding(64, 64, [8, 12, 12], interleaved_glm=True)
    assert glm.axis_map.tolist() == [0, 1, 2] * 8 + [1, 1, 2, 1, 1, 2, 2, 2]
    assert glm._legacy_axis_map is glm.axis_map


def check_weight_checker_reset():
    """The weight checker's reset must leave the axis map alone. A weight update does not restore it."""
    from sglang.srt.utils.weight_checker import WeightChecker

    model = torch.nn.Module()
    model.rotary_emb = cpu_rotary_embedding(HEAD_DIM, ROTARY_DIM, MROPE_SECTION)
    model.projection = torch.nn.Linear(4, 4)
    model.requires_grad_(False)
    axis_map = model.rotary_emb.axis_map.clone()
    projection_weight = model.projection.weight.clone()
    checker = WeightChecker(SimpleNamespace(model=model))

    checker.handle("snapshot")
    checker.handle("reset_tensors")

    assert torch.equal(model.rotary_emb.axis_map, axis_map)
    # The reset still changes the weights it checks.
    assert not torch.equal(model.projection.weight, projection_weight)


def check_image_positions():
    layer = attention_layer()
    positions, is_image = image_prompt_positions()
    hidden_states = projected_states(positions.shape[1])

    q, k, v, gate = prepare_attention(layer, positions, hidden_states)

    want_q, want_k = unfused_reference(layer, hidden_states, positions)
    assert max_difference(q, want_q, is_image) <= ROUNDING_TOLERANCE
    assert max_difference(k, want_k, is_image) <= ROUNDING_TOLERANCE
    q_gate = hidden_states[:, : 2 * layer.q_size].view(-1, NUM_Q_HEADS, 2 * HEAD_DIM)
    assert torch.equal(gate, q_gate[..., HEAD_DIM:].reshape(gate.shape))
    assert torch.equal(v, hidden_states[:, -layer.kv_size :])
    # A kernel that reads only the temporal row misses the reference by far more.
    temporal_q, temporal_k = unfused_reference(layer, hidden_states, positions[:1].expand(3, -1))
    assert max_difference(temporal_q, want_q, is_image) > 10 * ROUNDING_TOLERANCE
    assert max_difference(temporal_k, want_k, is_image) > 10 * ROUNDING_TOLERANCE


def check_strided_positions():
    """CUDA-graph decode passes a slice of a wider position buffer, so stride(0) is not T."""
    layer = attention_layer()
    positions, _ = image_prompt_positions()
    token_count = positions.shape[1]
    hidden_states = projected_states(token_count)
    position_buffer = torch.zeros(3, 4 * token_count, dtype=positions.dtype)
    position_buffer[:, :token_count] = positions
    strided_positions = position_buffer[:, :token_count]
    assert strided_positions.stride(0) != token_count

    contiguous_outputs = prepare_attention(layer, positions, hidden_states)
    strided_outputs = prepare_attention(layer, strided_positions, hidden_states)

    for contiguous_output, strided_output in zip(contiguous_outputs, strided_outputs, strict=True):
        assert torch.equal(contiguous_output, strided_output)


def check_text_positions():
    """Identical rows give the same bits as 1-D positions, the path the patch keeps."""
    layer = attention_layer()
    temporal_positions = torch.arange(24)
    hidden_states = projected_states(temporal_positions.numel())

    mrope_outputs = prepare_attention(layer, temporal_positions.expand(3, -1).contiguous(), hidden_states)
    one_dimensional_outputs = prepare_attention(layer, temporal_positions, hidden_states)

    for mrope_output, one_dimensional_output in zip(mrope_outputs, one_dimensional_outputs, strict=True):
        assert torch.equal(mrope_output, one_dimensional_output)


if __name__ == "__main__":
    {
        "axis_map": check_axis_map,
        "weight_checker_reset": check_weight_checker_reset,
        "image_positions": check_image_positions,
        "strided_positions": check_strided_positions,
        "text_positions": check_text_positions,
    }[sys.argv[1]]()
