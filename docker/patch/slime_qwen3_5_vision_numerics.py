"""Compute the Qwen3.5 HF vision tower with the same kernels as SGLang's vision tower.

The trainer runs transformers' ``Qwen3_5VisionModel``. The rollout engine runs SGLang's
``Qwen3VLMoeVisionModel`` (``python/sglang/srt/models/qwen3_vl.py``). Both compute the same function, but
five ops use different kernels or a different rounding order. In bf16 the tower amplifies these one-ULP
differences into 5 to 10 percent relative L2 at its output (5.4% for Qwen3.5-0.8B, 10.4% for Qwen3.5-2B, 16
images). So the image features (and the log-probs of image samples) differ between training and rollout.
``match_sglang_vision_numerics`` replaces these five ops with SGLang's versions:

1. RoPE ``inv_freq`` in fp32. HF keeps it in a buffer that ``.to(bf16)`` casts to bf16. SGLang computes it in
   fp32 (``layers/rotary_embedding/base.py``, ``_compute_inv_freq``).
2. Position-embedding bilinear weights cast to bf16 before the product
   (``Qwen3VLMoeVisionModel.fast_pos_embed_interpolate_from_list``).
3. Attention with ``F.scaled_dot_product_attention`` and a ``[1, s, s]`` bool mask from ``cu_seqlens``
   (``layers/attention/vision.py``, ``VisionSdpaAttention``, ``flatten_batch=True``).
4. Patch embedding as unfold + ``F.linear`` (``layers/conv.py``, ``Conv3dLayer._forward_mulmat``).
5. RoPE apply with SGLang's ``torch.compile``-d ``apply_rotary_pos_emb_native``
   (``layers/rotary_embedding/utils.py``). The eager HF function rounds differently.

The tower is also called once per image, because the GEMM and attention kernels that run depend on the total
sequence length of the call.

The match is exact only under three conditions:

- The engine uses the SDPA vision attention backend. For an engine that Slime starts, pass
  ``--sglang-mm-attention-backend sdpa``. For an engine that Reef starts, set ``mm-attention-backend: sdpa``
  under ``inference.options``. On CUDA, SGLang uses another backend by default.
- The engine encodes each image in its own tower call. The regular engine path of the SGLang pin ``f7ea06e1``
  encodes the cache-miss images of one request (in the current prefill chunk) in one call. So this holds for at
  most one cache-miss image per request. The engine must encode images itself: with no
  ``encoder`` worker (EPD) and with ``mm-enable-dp-encoder`` off. The EPD encoder at the pin merges the images of
  up to ``SGLANG_ENCODER_MAX_BATCH_SIZE`` (default 8) requests in one call, and the DP encoder shards images
  across ranks.
- SGLang is the pin ``f7ea06e1``. Other revisions can encode or round differently.

Reef copies this file into the installed ``slime_plugins/models/`` at image build time
(``docker/Dockerfile.reef``). Remove it when Reef's Slime pin has the same change.
"""

from __future__ import annotations

import logging
import types
from collections.abc import Callable

import torch
from torch.nn import functional

logger = logging.getLogger(__name__)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_eager(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    # Same body as sglang.srt.layers.rotary_embedding.utils.apply_rotary_pos_emb_native at the pin f7ea06e1.
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(unsqueeze_dim).float()
    sin = sin.unsqueeze(unsqueeze_dim).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def get_local_rope_apply() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """A copy of SGLang's compiled RoPE apply, for environments without sglang."""
    return torch.compile(dynamic=True, backend="inductor")(apply_rotary_pos_emb_eager)


def get_rope_apply() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """Return SGLang's compiled ``apply_rotary_pos_emb_native``.

    The Reef image build checks that this import works. Without SGLang (for example outside the image), return the
    local compiled copy and log a warning.
    """
    try:
        from sglang.srt.layers.rotary_embedding.utils import apply_rotary_pos_emb_native

        return apply_rotary_pos_emb_native
    except ImportError as e:
        logger.warning(f"Cannot import SGLang's apply_rotary_pos_emb_native ({e!r}); using a local compiled copy.")
        return get_local_rope_apply()


def sglang_bilinear_weights(grid_thw: torch.Tensor, num_grid_per_side: int, spatial_merge_size: int) -> torch.Tensor:
    """SGLang's bilinear weights (bf16), in the token order of HF's ``get_vision_bilinear_indices_and_weights``.

    Returns a ``[4, n]`` tensor for the ``n`` patches of all images in ``grid_thw``.
    """
    device = grid_thw.device
    merge = spatial_merge_size
    out = []
    for t, h, w in grid_thw.tolist():
        h_idxs = torch.linspace(0, num_grid_per_side - 1, h, dtype=torch.float32, device=device)
        w_idxs = torch.linspace(0, num_grid_per_side - 1, w, dtype=torch.float32, device=device)
        dh = h_idxs - h_idxs.to(torch.long)
        dw = w_idxs - w_idxs.to(torch.long)
        dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
        # SGLang's formula and operation order, see fast_pos_embed_interpolate_from_list.
        w11 = dh_grid * dw_grid
        w10 = dh_grid - w11
        w01 = dw_grid - w11
        w00 = 1 - dh_grid - w01
        weights = torch.stack([w00, w01, w10, w11]).reshape(4, -1)
        # Row-major patch order -> merged-window order, repeated for each frame.
        h_idx = torch.arange(h, device=device).view(h // merge, merge)
        w_idx = torch.arange(w, device=device).view(w // merge, merge)
        reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).transpose(1, 2).flatten().repeat(t)
        out.append(weights[:, reorder])
    return torch.cat(out, dim=1).to(torch.bfloat16)


def block_diagonal_mask(cu_seqlens: torch.Tensor, device: torch.device) -> torch.Tensor:
    """``[1, s, s]`` bool mask that is True inside each sequence of ``cu_seqlens``."""
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
    segment = torch.repeat_interleave(torch.arange(lengths.numel(), device=device), lengths)
    return (segment[:, None] == segment[None, :]).unsqueeze(0)


def transformers_api_supported(version: str | None = None) -> bool:
    """The patch replaces methods with the signatures of transformers 5.12 and 5.13, the tested versions.

    transformers 5.15 renamed the position-embedding helper, and 5.17 changed the rotary embedding API. SGLang at
    the pin cannot import with transformers 5.14, so 5.14 was not checked against SGLang.
    """
    from packaging.version import Version

    if version is None:
        import transformers

        version = transformers.__version__
    return (5, 12) <= Version(version).release[:2] < (5, 14)


def match_sglang_vision_numerics(
    vision_model: torch.nn.Module, rope_apply: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None
) -> bool:
    """Patch a transformers ``Qwen3_5VisionModel`` in place. See the module docstring.

    Returns False and leaves the model unchanged for an unsupported transformers version. ``rope_apply`` is only for
    tests. Tests that do not need SGLang pass the local copy. Slime calls this function without it.
    """
    if not transformers_api_supported():
        import transformers

        logger.warning(
            f"transformers {transformers.__version__} is not supported; the Qwen3.5 vision tower is not "
            "matched to SGLang's numerics."
        )
        return False

    from transformers.modeling_outputs import BaseModelOutputWithPooling
    from transformers.vision_utils import get_vision_bilinear_indices_and_weights

    if rope_apply is None:
        rope_apply = get_rope_apply()

    def rotary_forward(self, position_ids):  # 1.
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=position_ids.device) / self.dim)
        )
        return (position_ids.unsqueeze(-1) * inv_freq).flatten(1)

    def patch_embed_forward(self, hidden_states):  # 4.
        weight = self.proj.weight
        x = hidden_states.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = x.to(dtype=weight.dtype)
        k1, k2, k3 = self.proj.kernel_size
        x = x.unfold(2, k1, k1).unfold(3, k2, k2).unfold(4, k3, k3)
        n, dp, hp, wp = x.shape[0], x.shape[2], x.shape[3], x.shape[4]
        x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(n, dp, hp, wp, -1)
        x = functional.linear(x, weight.reshape(self.proj.out_channels, -1), self.proj.bias)
        return x.permute(0, 4, 1, 2, 3).reshape(-1, self.embed_dim)

    def attention_forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):  # 3. and 5.
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        cos, sin = position_embeddings
        q, k = rope_apply(q.contiguous(), k.contiguous(), cos, sin)
        q, k, v = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))
        mask = block_diagonal_mask(cu_seqlens, q.device)
        out = functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=self.scaling, is_causal=False
        )
        out = out.transpose(1, 2).contiguous().reshape(seq_length, -1).contiguous()
        return self.proj(out)

    cls_forward = type(vision_model).forward

    def vision_forward(self, hidden_states, grid_thw, **kwargs):  # 2. and one call per image
        total_patch_count = int(grid_thw.prod(dim=1).sum())
        if total_patch_count != hidden_states.shape[0]:
            raise ValueError(f"pixel rows {hidden_states.shape[0]} != sum of grid t*h*w {total_patch_count}")
        pooled, last, offset = [], [], 0
        for grid in grid_thw:
            grid = grid.view(1, 3)
            patch_count = int(grid.prod())
            indices, _ = get_vision_bilinear_indices_and_weights(
                grid, num_grid_per_side=self.num_grid_per_side, spatial_merge_size=self.config.spatial_merge_size
            )
            weights = sglang_bilinear_weights(grid, self.num_grid_per_side, self.config.spatial_merge_size)
            output = cls_forward(
                self,
                hidden_states[offset : offset + patch_count],
                grid_thw=grid,
                bilinear_indices=indices.to(grid.device),
                bilinear_weights=weights,
                **kwargs,
            )
            pooled.append(output.pooler_output)
            last.append(output.last_hidden_state)
            offset += patch_count
        return BaseModelOutputWithPooling(last_hidden_state=torch.cat(last), pooler_output=torch.cat(pooled))

    vision_model.rotary_pos_emb.forward = types.MethodType(rotary_forward, vision_model.rotary_pos_emb)
    vision_model.patch_embed.forward = types.MethodType(patch_embed_forward, vision_model.patch_embed)
    for block in vision_model.blocks:
        block.attn.forward = types.MethodType(attention_forward, block.attn)
    vision_model.forward = types.MethodType(vision_forward, vision_model)
    return True
