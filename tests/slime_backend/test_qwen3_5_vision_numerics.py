"""Slime's Qwen3.5 vision tower with SGLang's numerics.

``docker/Dockerfile.reef`` copies ``docker/patch/slime_qwen3_5_vision_numerics.py`` into Slime and applies
``docker/patch/slime_qwen3_5_vl_vision_numerics.patch``. These tests load the module from ``docker/patch/``.
The tests that need transformers skip when it is missing or not 5.12 or 5.13. The tests that compare with SGLang
skip when SGLang is missing. They compare the replaced patch embedding, position embedding and attention with
SGLang's own functions bit for bit. The patch and position embeddings use the widths of Qwen3.5-0.8B, because a
smaller model can give the same bits with and without the change. On CPU, the eager and compiled RoPE apply give
the same bits, and so do the bool and float attention masks. So these tests cannot show those two changes. In a GPU
check in #749, the compiled RoPE apply gave SGLang's bits and the eager one did not. For the mask, that check
compared only no mask with an all-True bool mask. No GPU run compared the bool mask with a float mask.

Remove these tests with the module and the patch when the Slime pin has the change.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.unit

PATCH_DIR = Path(__file__).resolve().parents[2] / "docker" / "patch"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "slime_qwen3_5_vision_numerics", PATCH_DIR / "slime_qwen3_5_vision_numerics.py"
)
numerics = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(numerics)

GRIDS = [[1, 4, 6], [1, 6, 4], [2, 4, 4]]


def supported_transformers():
    transformers = pytest.importorskip("transformers")
    if not numerics.transformers_api_supported():
        pytest.skip(f"the module supports transformers 5.12 and 5.13, found {transformers.__version__}")


def tiny_vision_model(dtype):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    config = Qwen3_5VisionConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        patch_size=4,
        out_hidden_size=32,
        num_position_embeddings=16,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return Qwen3_5VisionModel._from_config(config).eval().to(dtype=dtype)


def vision_model_with_08b_widths():
    """One block with the patch size, hidden size and position table of Qwen3.5-0.8B."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=768,
        intermediate_size=64,
        num_heads=12,
        patch_size=16,
        out_hidden_size=32,
        num_position_embeddings=2304,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return Qwen3_5VisionModel._from_config(config).eval().to(dtype=torch.bfloat16)


def random_pixels(model, dtype):
    config = model.config
    rows = sum(t * h * w for t, h, w in GRIDS)
    columns = config.in_channels * config.temporal_patch_size * config.patch_size**2
    return torch.randn(rows, columns, generator=torch.Generator().manual_seed(1)).to(dtype)


def test_patch_applies_to_the_slime_pin(tmp_path):
    if importlib.util.find_spec("slime_plugins") is None:
        pytest.skip("Slime is not installed")
    installed = Path(importlib.util.find_spec("slime_plugins.models.qwen3_5_vl").origin)
    if "match_sglang_vision_numerics" in installed.read_text():
        # The image build already applied the patch.
        patched = installed.read_text()
    else:
        shutil.copy(installed, tmp_path / "qwen3_5_vl.py")
        subprocess.run(
            ["git", "apply", "-p3", str(PATCH_DIR / "slime_qwen3_5_vl_vision_numerics.patch")],
            cwd=tmp_path,
            check=True,
        )
        patched = (tmp_path / "qwen3_5_vl.py").read_text()
    assert "from .qwen3_5_vision_numerics import match_sglang_vision_numerics" in patched
    assert "match_sglang_vision_numerics(vision_model)" in patched


def test_local_rope_apply_matches_sglang():
    # Other test modules can put a stub sglang package in sys.modules. Import the real function.
    rotary_utils = pytest.importorskip("sglang.srt.layers.rotary_embedding.utils")
    apply_rotary_pos_emb_native = rotary_utils.apply_rotary_pos_emb_native

    generator = torch.Generator().manual_seed(2)
    q = torch.randn(24, 4, 16, generator=generator).to(torch.bfloat16)
    k = torch.randn(24, 4, 16, generator=generator).to(torch.bfloat16)
    angles = torch.randn(24, 8, generator=generator) * 10
    angles = torch.cat((angles, angles), dim=-1)
    local_q, local_k = numerics.get_local_rope_apply()(q, k, angles.cos(), angles.sin())
    sglang_q, sglang_k = apply_rotary_pos_emb_native(q, k, angles.cos(), angles.sin())
    assert torch.equal(local_q, sglang_q)
    assert torch.equal(local_k, sglang_k)


def test_transformers_version_check():
    assert numerics.transformers_api_supported("5.12.1")
    assert numerics.transformers_api_supported("5.13.0")
    assert not numerics.transformers_api_supported("5.14.1")
    assert not numerics.transformers_api_supported("5.11.0")
    assert not numerics.transformers_api_supported("5.15.0")


def test_unsupported_transformers_leaves_model_unchanged(monkeypatch):
    # The warning names the installed transformers version.
    pytest.importorskip("transformers")
    monkeypatch.setattr(numerics, "transformers_api_supported", lambda: False)
    model = torch.nn.Module()
    assert not numerics.match_sglang_vision_numerics(model, rope_apply=numerics.get_local_rope_apply())
    assert "forward" not in model.__dict__


def test_block_diagonal_mask():
    mask = numerics.block_diagonal_mask(torch.tensor([0, 2, 5]), torch.device("cpu"))
    expected = torch.zeros(1, 5, 5, dtype=torch.bool)
    expected[0, :2, :2] = True
    expected[0, 2:, 2:] = True
    assert torch.equal(mask, expected)


def test_bilinear_weights_follow_hf_order():
    supported_transformers()
    from transformers.vision_utils import get_vision_bilinear_indices_and_weights

    grid = torch.tensor(GRIDS)
    _, hf_weights = get_vision_bilinear_indices_and_weights(grid, num_grid_per_side=4, spatial_merge_size=2)
    weights = numerics.sglang_bilinear_weights(grid, num_grid_per_side=4, spatial_merge_size=2)
    assert weights.dtype == torch.bfloat16
    torch.testing.assert_close(weights.float(), hf_weights.float(), atol=2**-8, rtol=0)


def test_rotary_uses_fp32_inv_freq_in_bf16_model():
    supported_transformers()
    model = tiny_vision_model(torch.bfloat16)
    rotary = model.rotary_pos_emb
    assert numerics.match_sglang_vision_numerics(model, rope_apply=numerics.get_local_rope_apply())
    positions = torch.arange(32).view(16, 2)
    inv_freq = 1.0 / (rotary.theta ** (torch.arange(0, rotary.dim, 2, dtype=torch.float) / rotary.dim))
    assert torch.equal(rotary(positions), (positions.unsqueeze(-1) * inv_freq).flatten(1))


def test_tower_matches_hf_in_fp32():
    supported_transformers()
    reference = tiny_vision_model(torch.float32)
    patched = tiny_vision_model(torch.float32)
    assert numerics.match_sglang_vision_numerics(patched, rope_apply=numerics.get_local_rope_apply())
    pixels = random_pixels(reference, torch.float32)
    with torch.no_grad():
        expected = reference(pixels, grid_thw=torch.tensor(GRIDS)).pooler_output
        actual = patched(pixels, grid_thw=torch.tensor(GRIDS)).pooler_output
    # Only rounding differs. The bilinear weights are rounded to bf16, so the tolerance is loose.
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)


def test_tower_calls_the_hf_forward_once_per_image_with_sglang_weights(monkeypatch):
    supported_transformers()
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    calls = []
    hf_forward = Qwen3_5VisionModel.forward

    def recording_forward(self, hidden_states, grid_thw, **kwargs):
        calls.append((hidden_states.shape[0], grid_thw.tolist(), kwargs["bilinear_weights"]))
        return hf_forward(self, hidden_states, grid_thw=grid_thw, **kwargs)

    monkeypatch.setattr(Qwen3_5VisionModel, "forward", recording_forward)
    model = tiny_vision_model(torch.bfloat16)
    assert numerics.match_sglang_vision_numerics(model, rope_apply=numerics.get_local_rope_apply())
    pixels = random_pixels(model, torch.bfloat16)
    with torch.no_grad():
        model(pixels, grid_thw=torch.tensor(GRIDS))
    assert [(rows, grid) for rows, grid, _ in calls] == [(t * h * w, [[t, h, w]]) for t, h, w in GRIDS]
    for _, grid, weights in calls:
        expected = numerics.sglang_bilinear_weights(
            torch.tensor(grid), model.num_grid_per_side, model.config.spatial_merge_size
        )
        assert weights.dtype == torch.bfloat16
        assert torch.equal(weights, expected)
    with pytest.raises(ValueError, match="pixel rows"):
        model(pixels, grid_thw=torch.tensor(GRIDS[:2]))
    with pytest.raises(ValueError, match="pixel rows"):
        model(pixels[:-8], grid_thw=torch.tensor(GRIDS))


def test_default_rope_apply_is_sglang_function():
    rotary_utils = pytest.importorskip("sglang.srt.layers.rotary_embedding.utils")
    assert numerics.get_rope_apply() is rotary_utils.apply_rotary_pos_emb_native


def test_patch_embedding_matches_sglang():
    supported_transformers()
    conv = pytest.importorskip("sglang.srt.layers.conv")

    model = vision_model_with_08b_widths()
    assert numerics.match_sglang_vision_numerics(model, rope_apply=numerics.get_local_rope_apply())
    patch_embed = model.patch_embed
    pixels = torch.randn(80, 3 * 2 * 16 * 16, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    sglang_conv = SimpleNamespace(
        kernel_size=patch_embed.proj.kernel_size,
        weight=patch_embed.proj.weight,
        bias=patch_embed.proj.bias,
        out_channels=patch_embed.proj.out_channels,
    )
    with torch.no_grad():
        expected = conv.Conv3dLayer._forward_mulmat(sglang_conv, pixels.view(-1, 3, 2, 16, 16)).reshape(-1, 768)
        assert torch.equal(patch_embed(pixels), expected)


def test_position_embedding_matches_sglang():
    supported_transformers()
    qwen3_vl = pytest.importorskip("sglang.srt.models.qwen3_vl")
    from transformers.vision_utils import get_vision_bilinear_indices_and_weights

    model = vision_model_with_08b_widths()
    merge = model.config.spatial_merge_size
    grids = [*GRIDS, [1, 34, 86]]
    grid = torch.tensor(grids)
    indices, _ = get_vision_bilinear_indices_and_weights(
        grid, num_grid_per_side=model.num_grid_per_side, spatial_merge_size=merge
    )
    weights = numerics.sglang_bilinear_weights(grid, model.num_grid_per_side, merge)
    sglang_model = SimpleNamespace(
        num_grid_per_side=model.num_grid_per_side,
        spatial_merge_size=merge,
        pos_embed=model.pos_embed,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        # The module passes these indices and weights to the HF forward, which sums the weighted embeddings.
        patched = (model.pos_embed(indices) * weights[:, :, None]).sum(0)
        expected = qwen3_vl.Qwen3VLMoeVisionModel.fast_pos_embed_interpolate_from_list(sglang_model, grids)
    assert torch.equal(patched, expected)


def test_block_attention_matches_sglang():
    supported_transformers()
    vision_attention = pytest.importorskip("sglang.srt.layers.attention.vision")
    rotary_utils = pytest.importorskip("sglang.srt.layers.rotary_embedding.utils")

    model = tiny_vision_model(torch.bfloat16)
    assert numerics.match_sglang_vision_numerics(model)
    attention = model.blocks[0].attn
    generator = torch.Generator().manual_seed(3)
    cu_seqlens = torch.tensor([0, 24, 48, 80], dtype=torch.int32)
    hidden_states = torch.randn(80, model.config.hidden_size, generator=generator).to(torch.bfloat16)
    angles = torch.randn(80, 8, generator=generator) * 10
    angles = torch.cat((angles, angles), dim=-1)
    with torch.no_grad():
        q, k, v = attention.qkv(hidden_states).reshape(80, 3, attention.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q, k = rotary_utils.apply_rotary_pos_emb_native(q.contiguous(), k.contiguous(), angles.cos(), angles.sin())
        sdpa = vision_attention.VisionSdpaAttention(
            head_dim=q.shape[-1],
            num_heads=attention.num_heads,
            num_kv_heads=attention.num_heads,
            flatten_batch=True,
            softmax_scale=attention.scaling,
        )
        expected = attention.proj(sdpa(q, k, v, bsz=1, cu_seqlens=cu_seqlens).reshape(80, -1))
        patched = attention(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=(angles.cos(), angles.sin()))
    assert torch.equal(patched, expected)
