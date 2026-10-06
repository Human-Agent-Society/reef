"""
Triton-optimized AlphaFold3-style Triangle Multiplicative Update (outgoing).

Forward-only implementation with two input paths based on channel dimension C:

For C=128 (5 of 7 test configs):
1. Single fused Triton kernel: input LayerNorm [B,N,N,C] FP32 → 5 FP16 linear
   projections (left_proj, right_proj, left_gate, right_gate, out_gate) using
   tensor-core tl.dot, then sigmoid gating + optional mask + transpose
   [B,N,N,H] → [B,H,N,N] for left/right, store raw out_gate.
2. Shape-specialized BMM contraction dispatch:
   - N % 128 == 0: cuBLAS (optimal alignment for all standard test cases)
   - All other N: single persistent Triton kernel with shape-agnostic tiling
     that adapts BLOCK_M/BLOCK_N/BLOCK_K to N's divisibility, using masking
     to handle boundary tiles without padding.
3. Fused output kernel: transpose [B*H,N,N]→[B,N,N,H] + LayerNorm over H +
   sigmoid gate + linear projection [H,C] → [B,N,N,C] float32 in one pass,
   with the contraction output cached in registers across C-blocks.

For C=384 (2 of 7 test configs):
1. Triton LayerNorm → FP16.
2. cuBLAS combined GEMM for all 5 projections.
3. Triton fused gate + mask + transpose (2 launches).
4. Shape-specialized BMM contraction (same dispatch as C=128).
5. Fused output kernel (same as C=128).
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ════════════════════════════════════════════════════════════════════════
# C=128 fused kernel: LayerNorm + 5 projections + gate + transpose
# ════════════════════════════════════════════════════════════════════════

@triton.jit
def fused_ln_proj_gate_transpose_kernel(
    x_ptr, norm_w_ptr, norm_b_ptr, proj_w_t_ptr,
    mask_ptr,
    left_out_ptr, right_out_ptr, gate_out_ptr,
    total_rows, N,
    C: tl.constexpr, H: tl.constexpr, FIVE_H: tl.constexpr,
    BLOCK_ROW: tl.constexpr, BLOCK_C: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """Fused: LayerNorm over C → 5 FP16 projections → sigmoid gate → mask →
    transpose [B,N,N,H] → [B,H,N,N] for left/right; store raw out_gate."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    row_mask = rows < total_rows
    offs_c = tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    offs_h = tl.arange(0, H)

    # ── Load input & LayerNorm over C ──
    x = tl.load(x_ptr + rows[:, None] * C + offs_c[None, :],
                mask=row_mask[:, None] & c_mask[None, :], other=0.0)
    mean = tl.sum(x, axis=1) / C
    xc = x - mean[:, None]
    xc = tl.where(c_mask[None, :], xc, 0.0)
    var = tl.sum(xc * xc, axis=1) / C
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    xn = xc * rstd[:, None]
    nw = tl.load(norm_w_ptr + offs_c, mask=c_mask, other=0.0)
    nb = tl.load(norm_b_ptr + offs_c, mask=c_mask, other=0.0)
    xn = xn * nw[None, :] + nb[None, :]
    x16 = xn.to(tl.float16)

    # ── Mask (fused into loads) ──
    if HAS_MASK:
        m = tl.load(mask_ptr + rows, mask=row_mask, other=0.0)

    # ── Transpose store indices: [B,N,N,H] → [B,H,N,N] ──
    b_idx = rows // (N * N)
    rib = rows % (N * N)
    base = b_idx * H * N * N + rib
    tptrs = base[None, :] + offs_h[:, None] * N * N

    # ── left_proj (weight rows 0:H) ──
    w = tl.load(proj_w_t_ptr + offs_c[:, None] * FIVE_H + offs_h[None, :],
                mask=c_mask[:, None], other=0.0)
    lval = tl.dot(x16, w)                                  # [BR, H] f32

    # ── left_gate (weight rows 2H:3H) ──
    oh = 2 * H + offs_h
    w = tl.load(proj_w_t_ptr + offs_c[:, None] * FIVE_H + oh[None, :],
                mask=c_mask[:, None], other=0.0)
    lgate = tl.dot(x16, w)
    lgsig = 1.0 / (1.0 + tl.exp(-lgate))
    lres = (lval * lgsig).to(tl.float16)
    if HAS_MASK:
        lres = lres * m[:, None].to(tl.float16)
    tl.store(left_out_ptr + tptrs, lres.trans(), mask=row_mask[None, :])

    # ── right_proj (weight rows H:2H) ──
    oh = H + offs_h
    w = tl.load(proj_w_t_ptr + offs_c[:, None] * FIVE_H + oh[None, :],
                mask=c_mask[:, None], other=0.0)
    rval = tl.dot(x16, w)

    # ── right_gate (weight rows 3H:4H) ──
    oh = 3 * H + offs_h
    w = tl.load(proj_w_t_ptr + offs_c[:, None] * FIVE_H + oh[None, :],
                mask=c_mask[:, None], other=0.0)
    rgate = tl.dot(x16, w)
    rgsig = 1.0 / (1.0 + tl.exp(-rgate))
    rres = (rval * rgsig).to(tl.float16)
    if HAS_MASK:
        rres = rres * m[:, None].to(tl.float16)
    tl.store(right_out_ptr + tptrs, rres.trans(), mask=row_mask[None, :])

    # ── out_gate (weight rows 4H:5H) — store raw (pre-sigmoid) ──
    oh = 4 * H + offs_h
    w = tl.load(proj_w_t_ptr + offs_c[:, None] * FIVE_H + oh[None, :],
                mask=c_mask[:, None], other=0.0)
    ogate = tl.dot(x16, w)
    tl.store(gate_out_ptr + rows[:, None] * H + offs_h[None, :],
             ogate.to(tl.float16), mask=row_mask[:, None])


# ════════════════════════════════════════════════════════════════════════
# Persistent Triton BMM contraction kernel
# Shape-agnostic: handles arbitrary N through masking, no padding required.
# ════════════════════════════════════════════════════════════════════════

@triton.jit
def persistent_bmm_kernel(
    a_ptr, b_ptr, c_ptr,
    BH, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Persistent batched matmul: C[bh, i, j] = Σ_k A[bh, i, k] * B[bh, j, k].

    Computes A @ B^T where A and B are [BH, N, N] contiguous.
    Shape-agnostic tiling: BLOCK_M, BLOCK_N, BLOCK_K are selected at launch
    time based on N's divisibility. Boundary tiles are handled via masking,
    eliminating the need for zero-padding on irregular N.

    Memory coalescing: A is loaded with K as the contiguous (last) dimension
    and B is loaded with K as the contiguous dimension, ensuring coalesced
    memory access patterns for both operands on H100.

    Persistent scheduling: each SM processes multiple tiles in a loop,
    amortizing launch overhead. GROUP_SIZE_M=4 provides L2 spatial locality.
    """
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(N, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_blocks_per_bh = num_pid_m * num_pid_n
    total_blocks = BH * num_blocks_per_bh

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for block_id in range(pid, total_blocks, NUM_SMS):
        pid_bh = block_id // num_blocks_per_bh
        pid_mn = block_id % num_blocks_per_bh

        group_id = pid_mn // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid_mn % group_size_m)
        pid_n = (pid_mn % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        m_mask = offs_m < N
        n_mask = offs_n < N

        # A[bh, i, k]: row stride = N, col stride = 1 (K is contiguous)
        a_ptrs = a_ptr + pid_bh * N * N + offs_m[:, None] * N + offs_k[None, :]
        # B[bh, j, k] loaded as [k, j]: K is contiguous, j stride = N
        b_ptrs = b_ptr + pid_bh * N * N + offs_k[:, None] + offs_n[None, :] * N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_iter in range(0, tl.cdiv(N, BLOCK_K)):
            k_offset = k_iter * BLOCK_K + offs_k
            k_mask = k_offset < N
            a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        c_ptrs = c_ptr + pid_bh * N * N + offs_m[:, None] * N + offs_n[None, :]
        tl.store(c_ptrs, acc.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])


# ════════════════════════════════════════════════════════════════════════
# C=384 path kernels (input side)
# ════════════════════════════════════════════════════════════════════════

@triton.jit
def layernorm_to_fp16_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    total_rows,
    C: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_ROW: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    row_mask = rows < total_rows
    offs_c = tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    x = tl.load(x_ptr + rows[:, None] * C + offs_c[None, :],
                mask=row_mask[:, None] & c_mask[None, :], other=0.0)
    mean = tl.sum(x, axis=1) / C
    xc = x - mean[:, None]
    xc = tl.where(c_mask[None, :], xc, 0.0)
    var = tl.sum(xc * xc, axis=1) / C
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    xn = xc * rstd[:, None]
    w = tl.load(weight_ptr + offs_c, mask=c_mask, other=0.0)
    b = tl.load(bias_ptr + offs_c, mask=c_mask, other=0.0)
    xn = xn * w[None, :] + b[None, :]
    tl.store(out_ptr + rows[:, None] * C + offs_c[None, :],
             xn.to(tl.float16), mask=row_mask[:, None] & c_mask[None, :])


@triton.jit
def fused_gate_transpose_kernel(
    val_ptr, gate_ptr, mask_ptr, out_ptr,
    total_rows, N,
    H: tl.constexpr, ROW_STRIDE: tl.constexpr,
    BLOCK_ROW: tl.constexpr, HAS_MASK: tl.constexpr,
):
    """Fused gate + mask + transpose."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    row_mask = rows < total_rows
    offs_h = tl.arange(0, H)
    val = tl.load(val_ptr + rows[:, None] * ROW_STRIDE + offs_h[None, :],
                  mask=row_mask[:, None], other=0.0)
    gate = tl.load(gate_ptr + rows[:, None] * ROW_STRIDE + offs_h[None, :],
                  mask=row_mask[:, None], other=0.0)
    gsig = 1.0 / (1.0 + tl.exp(-gate.to(tl.float32)))
    res = (val.to(tl.float32) * gsig).to(tl.float16)
    if HAS_MASK:
        m = tl.load(mask_ptr + rows, mask=row_mask, other=0.0)
        res = res * m[:, None].to(tl.float16)
    b_idx = rows // (N * N)
    rib = rows % (N * N)
    base = b_idx * H * N * N + rib
    tl.store(out_ptr + base[None, :] + offs_h[:, None] * N * N,
             res.trans(), mask=row_mask[None, :])


# ════════════════════════════════════════════════════════════════════════
# Unified fused output kernel: transpose + LayerNorm + gate + projection
# Caches the gated contraction output in registers across C-blocks.
# ════════════════════════════════════════════════════════════════════════

@triton.jit
def fused_output_kernel(
    bmm_out_ptr, out_gate_ptr,
    norm_w_ptr, norm_b_ptr,
    to_out_w_t_ptr, result_ptr,
    B, N,
    H: tl.constexpr, C: tl.constexpr, GATE_STRIDE: tl.constexpr,
    BLOCK_J: tl.constexpr, BLOCK_C: tl.constexpr,
    NUM_C_BLOCKS: tl.constexpr,
):
    """Fused output: transpose [B*H,N,N]→[B,N,N,H] + LayerNorm over H +
    sigmoid gate + linear projection [H,C] → [B,N,N,C] float32."""
    pid_bi = tl.program_id(0)
    pid_j = tl.program_id(1)
    b = pid_bi // N
    i = pid_bi % N
    j_start = pid_j * BLOCK_J
    offs_j = j_start + tl.arange(0, BLOCK_J)
    j_mask = offs_j < N
    offs_h = tl.arange(0, H)

    # Load bmm_out [H, BLOCK_J] from [B*H, N, N] and transpose → [BLOCK_J, H]
    bh_base = b * H * N * N + i * N
    bh = tl.load(bmm_out_ptr + bh_base + offs_h[:, None] * N * N + offs_j[None, :],
                 mask=j_mask[None, :], other=0.0).to(tl.float32)
    bh_t = bh.trans()  # [BLOCK_J, H]

    # LayerNorm over H
    mean = tl.sum(bh_t, axis=1) / H
    centered = bh_t - mean[:, None]
    var = tl.sum(centered * centered, axis=1) / H
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    xn = centered * rstd[:, None]
    nw = tl.load(norm_w_ptr + offs_h)
    nb = tl.load(norm_b_ptr + offs_h)
    xn = xn * nw[None, :] + nb[None, :]

    # Sigmoid gate
    gate_row = b * N * N + i * N + offs_j
    gate = tl.load(out_gate_ptr + gate_row[:, None] * GATE_STRIDE + offs_h[None, :],
                   mask=j_mask[:, None], other=0.0).to(tl.float32)
    gsig = 1.0 / (1.0 + tl.exp(-gate))
    gated = (xn * gsig).to(tl.float16)  # [BLOCK_J, H] — cached in registers

    # Loop over C-blocks, reusing cached gated values
    result_base = b * N * N * C + i * N * C
    for c_block in tl.static_range(NUM_C_BLOCKS):
        c_start = c_block * BLOCK_C
        offs_c = c_start + tl.arange(0, BLOCK_C)
        c_mask = offs_c < C

        w = tl.load(to_out_w_t_ptr + offs_h[:, None] * C + offs_c[None, :],
                    mask=c_mask[None, :], other=0.0)
        acc = tl.dot(gated, w)  # [BLOCK_J, BLOCK_C]

        tl.store(result_ptr + result_base + offs_j[:, None] * C + offs_c[None, :],
                 acc.to(tl.float32), mask=j_mask[:, None] & c_mask[None, :])


# ════════════════════════════════════════════════════════════════════════
# Shape-specialized BMM contraction dispatch
# ════════════════════════════════════════════════════════════════════════

def _bmm_contraction(left_bh, right_bh, B, H, N, device):
    """Dispatch BMM contraction with shape-specialized configs.

    out[bh, i, j] = Σ_k left[bh, i, k] * right[bh, j, k]
    Computed as left @ right^T via BMM.

    Dispatch logic:
    - N % 128 == 0: cuBLAS (optimal alignment for all standard test cases)
    - All other N: single persistent Triton kernel with shape-agnostic tiling.
      Tile sizes adapt to N's divisibility:
        N % 64 == 0  → 64×64×64 tiles, 4 warps
        N % 32 == 0  → 32×32×32 tiles, 4 warps
        otherwise    → 16×16×16 tiles, 2 warps
      Boundary tiles handled via masking — no zero-padding required.
    """
    BH = B * H
    left_view = left_bh.view(BH, N, N)
    right_view = right_bh.view(BH, N, N)

    if N % 128 == 0:
        # cuBLAS — optimal alignment for all standard test cases
        return torch.bmm(left_view, right_view.transpose(1, 2))

    # Persistent Triton BMM for non-128-divisible N — no padding needed
    if N % 64 == 0:
        BLOCK_M_P, BLOCK_N_P, BLOCK_K_P = 64, 64, 64
        NUM_WARPS_P = 4
    elif N % 32 == 0:
        BLOCK_M_P, BLOCK_N_P, BLOCK_K_P = 32, 32, 32
        NUM_WARPS_P = 4
    else:
        BLOCK_M_P, BLOCK_N_P, BLOCK_K_P = 16, 16, 16
        NUM_WARPS_P = 2

    GROUP_SIZE_M_P = 4
    NUM_STAGES_P = 3

    out_bh = torch.empty((BH, N, N), dtype=torch.float16, device=device)
    NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    grid_bmm = (NUM_SMS,)
    persistent_bmm_kernel[grid_bmm](
        left_view, right_view, out_bh,
        BH, N,
        BLOCK_M=BLOCK_M_P, BLOCK_N=BLOCK_N_P, BLOCK_K=BLOCK_K_P,
        GROUP_SIZE_M=GROUP_SIZE_M_P,
        NUM_SMS=NUM_SMS,
        num_warps=NUM_WARPS_P, num_stages=NUM_STAGES_P,
    )
    return out_bh


# ════════════════════════════════════════════════════════════════════════
# Entrypoint
# ════════════════════════════════════════════════════════════════════════

def custom_kernel(data):
    input_tensor, mask, weights, config = data
    B, N, _, C = input_tensor.shape
    H = config["hidden_dim"]

    norm_weight = weights["norm.weight"]
    norm_bias = weights["norm.bias"]
    to_out_norm_w = weights["to_out_norm.weight"]
    to_out_norm_b = weights["to_out_norm.bias"]
    to_out_w_t = weights["to_out.weight"].to(torch.float16).t().contiguous()  # [H, C]

    total_rows = B * N * N
    use_fused = (C == 128)

    if use_fused:
        # ── C=128 fused input path ──
        combined_w = torch.cat([
            weights["left_proj.weight"].to(torch.float16),
            weights["right_proj.weight"].to(torch.float16),
            weights["left_gate.weight"].to(torch.float16),
            weights["right_gate.weight"].to(torch.float16),
            weights["out_gate.weight"].to(torch.float16),
        ], dim=0)  # [5*H, C]
        combined_w_t = combined_w.t().contiguous()  # [C, 5*H]
        FIVE_H = 5 * H

        x_flat = input_tensor.reshape(total_rows, C)
        left_bh = torch.empty((B, H, N, N), dtype=torch.float16, device=input_tensor.device)
        right_bh = torch.empty((B, H, N, N), dtype=torch.float16, device=input_tensor.device)
        gate_buf = torch.empty((total_rows, H), dtype=torch.float16, device=input_tensor.device)

        BLOCK_ROW = 64
        grid = (triton.cdiv(total_rows, BLOCK_ROW),)

        fused_ln_proj_gate_transpose_kernel[grid](
            x_flat, norm_weight, norm_bias, combined_w_t,
            mask.reshape(-1).to(torch.float32) if mask is not None else None,
            left_bh, right_bh, gate_buf,
            total_rows, N,
            C=C, H=H, FIVE_H=FIVE_H,
            BLOCK_ROW=BLOCK_ROW, BLOCK_C=C,
            HAS_MASK=(mask is not None),
            num_warps=4, num_stages=3,
        )

        # ── Shape-specialized BMM contraction ──
        out_bh = _bmm_contraction(left_bh, right_bh, B, H, N, input_tensor.device)

        # ── Fused output: transpose + LayerNorm + gate + projection ──
        BLOCK_J_OUT = 64
        BLOCK_C_OUT = 128
        NUM_C_BLOCKS = triton.cdiv(C, BLOCK_C_OUT)
        grid_out = (B * N, triton.cdiv(N, BLOCK_J_OUT))
        result = torch.empty((B, N, N, C), dtype=torch.float32, device=input_tensor.device)
        fused_output_kernel[grid_out](
            out_bh, gate_buf,
            to_out_norm_w, to_out_norm_b,
            to_out_w_t, result,
            B, N,
            H=H, C=C, GATE_STRIDE=H,
            BLOCK_J=BLOCK_J_OUT, BLOCK_C=BLOCK_C_OUT,
            NUM_C_BLOCKS=NUM_C_BLOCKS,
            num_warps=4, num_stages=3,
        )

    else:
        # ── C=384 path ──
        combined_w = torch.cat([
            weights["left_proj.weight"].to(torch.float16),
            weights["right_proj.weight"].to(torch.float16),
            weights["left_gate.weight"].to(torch.float16),
            weights["right_gate.weight"].to(torch.float16),
            weights["out_gate.weight"].to(torch.float16),
        ], dim=0)  # [5*H, C]
        ROW_STRIDE = 5 * H

        # LayerNorm → FP16
        BLOCK_C_LN = 1
        while BLOCK_C_LN < C:
            BLOCK_C_LN *= 2
        BLOCK_ROW_LN = max(1, 8192 // BLOCK_C_LN)
        x_flat = input_tensor.reshape(total_rows, C)
        x16_flat = torch.empty((total_rows, C), dtype=torch.float16, device=input_tensor.device)
        grid_ln = (triton.cdiv(total_rows, BLOCK_ROW_LN),)
        layernorm_to_fp16_kernel[grid_ln](
            x_flat, norm_weight, norm_bias, x16_flat,
            total_rows,
            C=C, BLOCK_C=BLOCK_C_LN, BLOCK_ROW=BLOCK_ROW_LN,
            num_warps=4, num_stages=3,
        )

        # cuBLAS combined GEMM for all 5 projections
        proj_out = F.linear(x16_flat, combined_w)  # [total_rows, 5*H]

        left = proj_out[:, :H]
        right = proj_out[:, H:2*H]
        left_gate = proj_out[:, 2*H:3*H]
        right_gate = proj_out[:, 3*H:4*H]
        out_gate = proj_out[:, 4*H:5*H]

        # Gate + transpose (2 launches) with fused mask
        BLOCK_ROW_GT = 128
        grid_gt = (triton.cdiv(total_rows, BLOCK_ROW_GT),)
        left_bh = torch.empty((B, H, N, N), dtype=torch.float16, device=input_tensor.device)
        right_bh = torch.empty((B, H, N, N), dtype=torch.float16, device=input_tensor.device)

        if mask is not None:
            mask_flat = mask.reshape(-1).to(torch.float32)
            fused_gate_transpose_kernel[grid_gt](
                left, left_gate, mask_flat, left_bh,
                total_rows, N, H=H, ROW_STRIDE=ROW_STRIDE,
                BLOCK_ROW=BLOCK_ROW_GT, HAS_MASK=True, num_warps=8, num_stages=3)
            fused_gate_transpose_kernel[grid_gt](
                right, right_gate, mask_flat, right_bh,
                total_rows, N, H=H, ROW_STRIDE=ROW_STRIDE,
                BLOCK_ROW=BLOCK_ROW_GT, HAS_MASK=True, num_warps=8, num_stages=3)
        else:
            fused_gate_transpose_kernel[grid_gt](
                left, left_gate, None, left_bh,
                total_rows, N, H=H, ROW_STRIDE=ROW_STRIDE,
                BLOCK_ROW=BLOCK_ROW_GT, HAS_MASK=False, num_warps=8, num_stages=3)
            fused_gate_transpose_kernel[grid_gt](
                right, right_gate, None, right_bh,
                total_rows, N, H=H, ROW_STRIDE=ROW_STRIDE,
                BLOCK_ROW=BLOCK_ROW_GT, HAS_MASK=False, num_warps=8, num_stages=3)

        # ── Shape-specialized BMM contraction ──
        out_bh = _bmm_contraction(left_bh, right_bh, B, H, N, input_tensor.device)

        # ── Fused output: transpose + LayerNorm + gate + projection ──
        BLOCK_J_OUT = 64
        BLOCK_C_OUT = 128
        NUM_C_BLOCKS = triton.cdiv(C, BLOCK_C_OUT)
        grid_out = (B * N, triton.cdiv(N, BLOCK_J_OUT))
        result = torch.empty((B, N, N, C), dtype=torch.float32, device=input_tensor.device)
        fused_output_kernel[grid_out](
            out_bh, out_gate,
            to_out_norm_w, to_out_norm_b,
            to_out_w_t, result,
            B, N,
            H=H, C=C, GATE_STRIDE=ROW_STRIDE,
            BLOCK_J=BLOCK_J_OUT, BLOCK_C=BLOCK_C_OUT,
            NUM_C_BLOCKS=NUM_C_BLOCKS,
            num_warps=4, num_stages=3,
        )

    return result
