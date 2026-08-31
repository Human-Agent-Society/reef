diff --git a/dbscanlib/_kernels/__init__.py b/dbscanlib/_kernels/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/dbscanlib/_kernels/kernels/__init__.py b/dbscanlib/_kernels/kernels/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/dbscanlib/_kernels/kernels/distance/__init__.py b/dbscanlib/_kernels/kernels/distance/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/dbscanlib/_kernels/kernels/distance/triton/__init__.py b/dbscanlib/_kernels/kernels/distance/triton/__init__.py
new file mode 100644
index 0000000..63fceb5
--- /dev/null
+++ b/dbscanlib/_kernels/kernels/distance/triton/__init__.py
@@ -0,0 +1,11 @@
+"""distance triton backend (kNN squared-L2 gather only).
+
+Minimal re-export: only the tiled squared-L2 gather kernel that recovers
+true ``||x - c[idx]||^2`` per neighbour after the fused kNN pass. The
+other distance kernels from the upstream backend are not vendored here.
+"""
+from dbscanlib._kernels.kernels.distance.triton.knn_gather_l2sq import (
+    triton_knn_gather_sqdist,
+)
+
+__all__ = ["triton_knn_gather_sqdist"]
diff --git a/dbscanlib/_kernels/kernels/distance/triton/knn_gather_l2sq.py b/dbscanlib/_kernels/kernels/distance/triton/knn_gather_l2sq.py
new file mode 100644
index 0000000..e2e3fbe
--- /dev/null
+++ b/dbscanlib/_kernels/kernels/distance/triton/knn_gather_l2sq.py
@@ -0,0 +1,232 @@
+"""Tiled gather kernel: true squared L2 to selected kNN neighbours.
+
+After a fused kNN pass that ranks candidates with the shift-invariant
+score ``c_sq - 2 * <x, c>``, this kernel writes the **true**
+``||x[b, n] - c[b, idx[b, n, k]]||^2`` for every neighbour. Computing
+the difference directly (``(x - y)^2`` per d) avoids the cancellation
+inherent to the ``x^2 + y^2 - 2*x*y`` expansion, so the result is
+accurate even for fp32 inputs where the fused kNN kernel may have used
+TF32 in the cross-term.
+
+Complexity: ``O(B * N * K * D)`` -- cheap vs the ``O(B * N * M * D)``
+GEMM that found the candidates.
+
+Design
+------
+One program owns a ``(BN, K_BLOCK)`` output tile. The query row tile
+``x[b, n_block, :]`` is loaded once and reused across the ``K_BLOCK``
+neighbours, then streamed through ``D`` in ``BLOCK_D`` chunks. The
+corpus rows ``c[b, idx, :]`` are gather-loaded fresh per d-chunk into a
+3-D tile ``(BN, K_BLOCK, BLOCK_D)`` and subtracted directly from the
+broadcast x tile in fp32. The fp32 squared difference is summed along
+``D`` into the accumulator.
+
+Three regime presets pick ``BN`` and ``K_BLOCK`` so the c-tile fits in
+SMEM/registers and amortises one x-load over many neighbours:
+
+  * K <= 32:        ``K_BLOCK = next_pow2(K)``, ``BN = 16``.
+  * 32 < K <= 512:  ``K_BLOCK = 32``, ``BN = 8``; multiple K-tiles per row.
+  * K > 512:        ``K_BLOCK = 32``, ``BN = 4``.
+
+``BLOCK_D`` is chosen so the per-program SMEM footprint
+``BN * K_BLOCK * BLOCK_D * sizeof(dtype)`` stays under ~32 KB. The
+unmodified ``D`` (not padded) is passed as a constexpr so the Triton
+compiler unrolls the d-loop exactly.
+"""
+from __future__ import annotations
+
+import torch
+import triton
+import triton.language as tl
+
+
+def _next_pow2(n: int) -> int:
+    if n <= 1:
+        return 1
+    return 1 << (n - 1).bit_length()
+
+
+@triton.jit
+def _knn_gather_l2sq_kernel(
+    x_ptr, c_ptr, idx_ptr, out_ptr,
+    M,
+    stride_x_b, stride_x_n, stride_x_d,
+    stride_c_b, stride_c_m, stride_c_d,
+    stride_i_b, stride_i_n, stride_i_k,
+    stride_o_b, stride_o_n, stride_o_k,
+    N: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
+    BN: tl.constexpr, K_BLOCK: tl.constexpr,
+    BLOCK_D: tl.constexpr,
+    SINGLE_D_TILE: tl.constexpr,
+):
+    """Tiled (BN, K_BLOCK, BLOCK_D) gather + sq-distance accumulate.
+
+    Grid: ``(ceil(N / BN), ceil(K / K_BLOCK), B)``.
+
+    Each program owns one ``(BN, K_BLOCK)`` output tile. The fp32
+    accumulator stays in registers; the c-row tile ``(BN, K_BLOCK,
+    BLOCK_D)`` is reloaded per d-chunk to keep SMEM bounded.
+    """
+    pid_n = tl.program_id(0)
+    pid_k = tl.program_id(1)
+    pid_b = tl.program_id(2).to(tl.int64)
+
+    n_start = pid_n * BN
+    n_offs = (n_start + tl.arange(0, BN)).to(tl.int64)
+    n_mask = n_offs < N
+
+    k_start = pid_k * K_BLOCK
+    k_offs = (k_start + tl.arange(0, K_BLOCK)).to(tl.int64)
+    k_mask = k_offs < K
+
+    idx_tile = tl.load(
+        idx_ptr + pid_b * stride_i_b
+        + n_offs[:, None] * stride_i_n
+        + k_offs[None, :] * stride_i_k,
+        mask=n_mask[:, None] & k_mask[None, :], other=0,
+    ).to(tl.int64)
+    idx_tile = tl.maximum(idx_tile, 0)
+    idx_tile = tl.minimum(idx_tile, M - 1)
+
+    acc = tl.zeros((BN, K_BLOCK), dtype=tl.float32)
+
+    if SINGLE_D_TILE:
+        d_offs = tl.arange(0, BLOCK_D).to(tl.int64)
+        d_mask = d_offs < D
+        x_tile = tl.load(
+            x_ptr + pid_b * stride_x_b
+            + n_offs[:, None] * stride_x_n
+            + d_offs[None, :] * stride_x_d,
+            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
+        ).to(tl.float32)
+        c_tile = tl.load(
+            c_ptr + pid_b * stride_c_b
+            + idx_tile[:, :, None] * stride_c_m
+            + d_offs[None, None, :] * stride_c_d,
+            mask=(n_mask[:, None] & k_mask[None, :])[:, :, None]
+            & d_mask[None, None, :], other=0.0,
+        ).to(tl.float32)
+        diff = x_tile[:, None, :] - c_tile
+        acc = tl.sum(diff * diff, axis=2)
+    else:
+        for d_start in tl.range(0, D, BLOCK_D):
+            d_offs = (d_start + tl.arange(0, BLOCK_D)).to(tl.int64)
+            d_mask = d_offs < D
+            x_tile = tl.load(
+                x_ptr + pid_b * stride_x_b
+                + n_offs[:, None] * stride_x_n
+                + d_offs[None, :] * stride_x_d,
+                mask=n_mask[:, None] & d_mask[None, :], other=0.0,
+            ).to(tl.float32)
+            c_tile = tl.load(
+                c_ptr + pid_b * stride_c_b
+                + idx_tile[:, :, None] * stride_c_m
+                + d_offs[None, None, :] * stride_c_d,
+                mask=(n_mask[:, None] & k_mask[None, :])[:, :, None]
+                & d_mask[None, None, :], other=0.0,
+            ).to(tl.float32)
+            diff = x_tile[:, None, :] - c_tile
+            acc += tl.sum(diff * diff, axis=2)
+
+    tl.store(
+        out_ptr + pid_b * stride_o_b
+        + n_offs[:, None] * stride_o_n
+        + k_offs[None, :] * stride_o_k,
+        acc, mask=n_mask[:, None] & k_mask[None, :],
+    )
+
+
+def _pick_tile(K: int, D: int, dtype_bytes: int) -> tuple[int, int, int, bool]:
+    """Pick ``(BN, K_BLOCK, BLOCK_D, single_d_tile)`` for the kernel.
+
+    Regimes mirror the file docstring -- small ``K_BLOCK`` for large K,
+    larger ``BN`` for small K. ``BLOCK_D`` is sized so the 3-D c-tile
+    SMEM footprint stays around ~32 KB.
+    """
+    K_pad = _next_pow2(K)
+
+    if K_pad <= 32:
+        K_BLOCK = max(1, K_pad)
+        BN = 16
+    elif K_pad <= 512:
+        K_BLOCK = 32
+        BN = 8
+    else:
+        K_BLOCK = 32
+        BN = 4
+
+    target_bytes = 32 * 1024
+    per_d = BN * K_BLOCK * dtype_bytes
+    block_d_cap = max(16, target_bytes // max(1, per_d))
+    block_d_cap = min(block_d_cap, 256)
+
+    if D <= block_d_cap:
+        BLOCK_D = max(16, _next_pow2(D))
+        single_d_tile = True
+    else:
+        BLOCK_D = 64 if block_d_cap >= 64 else 32
+        single_d_tile = False
+
+    return BN, K_BLOCK, BLOCK_D, single_d_tile
+
+
+def triton_knn_gather_sqdist(
+    x: torch.Tensor,
+    c: torch.Tensor,
+    idx: torch.Tensor,
+    *,
+    out: torch.Tensor | None = None,
+) -> torch.Tensor:
+    """``out[b, n, k] = || x[b, n, :] - c[b, idx[b, n, k], :] ||^2`` (fp32).
+
+    Args:
+        x:    (B, N, D) query tensor, any float cuda dtype (bf16 / fp16 / fp32).
+        c:    (B, M, D) corpus, same dtype and same batch as ``x``.
+        idx:  (B, N, K) int32 / int64 column indices into ``c``.
+        out:  optional pre-allocated (B, N, K) fp32 output buffer.
+
+    Returns:
+        (B, N, K) fp32. The computation runs in fp32 throughout
+        (``diff = x.to(fp32) - c.to(fp32)``, ``sum(diff*diff)``) so the
+        result is bit-equivalent to the naive torch reference up to
+        accumulation order on the d-axis.
+    """
+    assert x.is_cuda and c.is_cuda and idx.is_cuda
+    assert x.dim() == 3 and c.dim() == 3 and idx.dim() == 3
+    B, N, D = x.shape
+    Bc, M, Dc = c.shape
+    Bi, Ni, K = idx.shape
+    assert B == Bc == Bi and N == Ni and D == Dc
+
+    if not x.is_contiguous():
+        x = x.contiguous()
+    if not c.is_contiguous():
+        c = c.contiguous()
+    if not idx.is_contiguous():
+        idx = idx.contiguous()
+
+    if out is None:
+        out = torch.empty((B, N, K), device=x.device, dtype=torch.float32)
+    else:
+        assert out.shape == (B, N, K) and out.dtype == torch.float32
+
+    dtype_bytes = x.element_size()
+    BN, K_BLOCK, BLOCK_D, single_d_tile = _pick_tile(K, D, dtype_bytes)
+
+    grid = (triton.cdiv(N, BN), triton.cdiv(K, K_BLOCK), B)
+    _knn_gather_l2sq_kernel[grid](
+        x, c, idx, out,
+        M,
+        x.stride(0), x.stride(1), x.stride(2),
+        c.stride(0), c.stride(1), c.stride(2),
+        idx.stride(0), idx.stride(1), idx.stride(2),
+        out.stride(0), out.stride(1), out.stride(2),
+        N=N, D=D, K=K,
+        BN=BN, K_BLOCK=K_BLOCK, BLOCK_D=BLOCK_D,
+        SINGLE_D_TILE=single_d_tile,
+        num_warps=4,
+    )
+    return out
+
+
+__all__ = ["triton_knn_gather_sqdist"]
diff --git a/dbscanlib/_kernels/kernels/flash_mst/__init__.py b/dbscanlib/_kernels/kernels/flash_mst/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/dbscanlib/_kernels/kernels/flash_mst/triton/__init__.py b/dbscanlib/_kernels/kernels/flash_mst/triton/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/dbscanlib/_kernels/kernels/flash_mst/triton/flash_mst.py b/dbscanlib/_kernels/kernels/flash_mst/triton/flash_mst.py
new file mode 100644
index 0000000..af8ace6
--- /dev/null
+++ b/dbscanlib/_kernels/kernels/flash_mst/triton/flash_mst.py
@@ -0,0 +1,611 @@
+"""flash-mst: GPU-resident dense Boruvka MST.
+
+Algorithm (per Boruvka iteration, all on GPU):
+  1. Per-component argmin (`_per_component_argmin_v2_kernel`): for each row v,
+     scan all N columns of MRD once; pack (weight_bits, dst_idx) as int64 with
+     +inf for same-component cells; reduce per-row to a single packed best
+     edge; atomic_min into OUT_PACKED[parent[v]] (the row's component bucket).
+     SAME kernel folds in OUT_SRC[parent[v]] = atomic_min(orig_i) so we get
+     the canonical source vertex per component without a separate kernel.
+  2. Concurrent union-find (`_concurrent_uf_kernel`): each component decodes
+     its packed-int64 best edge, finds roots with bounded path-walk, atomic-
+     CAS merges. Successful merges write an MST edge via atomic counter.
+  3. Pointer-jumping (`_pointer_jump_kernel`): parent[v] = parent[parent[v]]
+     in parallel for ~log iters until paths are flat. Triton kernel replaces
+     the `parent = parent[parent.to(int64)]` torch loop (saves N int64 casts).
+
+Persistent state (allocated ONCE at flash_mst entry, reused across rounds):
+  - parent[N] int32        — union-find array; doubles as the component label
+    for the next round's argmin (after pointer-jumping it is a valid label,
+    no torch.unique relabel needed).
+  - out_packed[N] int64    — per-CID atomic_min target, sized N (upper bound
+    on component count). Reset to +inf each round via Triton kernel.
+  - out_src[N]    int32    — per-CID canonical src; reset to MAX_INT32.
+  - mst_{src,dst,w}, edge_count — output.
+
+Why no argsort + no torch.unique relabel:
+  - The original code argsort-ed by component for cache locality; with
+    parent indexed directly the cache hit-rate is the same (parent[i] is
+    a single random read per row, then comp_j is a streaming scan which
+    L2-caches).
+  - `torch.unique` did a sort + inverse to remap parent[] to dense [0, n_C).
+    That step cost ~0.13 ms × log_2(N) iters of pointer-jumping. We skip it:
+    the argmin kernel uses `parent[i] != parent[j]` directly, which works
+    with sparse labels.
+
+Note: I tried a K-min Boruvka variant (build a per-row top-K cross-component
+edge table once, reuse for K mini-merges) — measured 1 outer round was enough
+to find 99.97% of MST edges at huge, but the K=4 table build is ~3-4× more
+compute-heavy per scan than the simple argmin (4× axis-min + masking + bitonic
+merge), and a 2nd cleanup round still costs a full N² scan, so it lost to the
+simple argmin. Reverted; orchestration cleanup + tile tuning gave the win.
+
+Packed int64 trick: for non-negative fp32 weights, the IEEE-754 bit pattern
+is monotonic (same ordering as float). Pack as
+    (weight_bits << 32) | dst_idx
+so atomic_min on int64 selects the smallest weight (with index as tiebreaker).
+"""
+
+import math
+import numpy as np
+import torch
+import triton
+import triton.language as tl
+
+
+# =============================================================================
+# Kernel 1: per-component min outgoing edge
+# Each program handles BLOCK_S sorted-contiguous points; per-row argmin over
+# the row of MRD (masked by component); atomic_min the result into the
+# corresponding component slot.
+# =============================================================================
+
+@triton.jit
+def _per_component_argmin_kernel(
+    MRD_ptr,                # (N, N) fp32
+    SORTED_IDX_ptr,         # (N,) int32
+    COMP_ptr,               # (N,) int32, indexed by orig_idx
+    OUT_PACKED_ptr,         # (n_components,) int64, atomic_min target
+    N,
+    BLOCK_S: tl.constexpr,
+    BLOCK_J: tl.constexpr,
+):
+    pid = tl.program_id(0)
+    s_offs = pid * BLOCK_S + tl.arange(0, BLOCK_S)
+    s_mask = s_offs < N
+
+    orig_i = tl.load(SORTED_IDX_ptr + s_offs, mask=s_mask, other=0)
+    comp_i = tl.load(COMP_ptr + orig_i, mask=s_mask, other=-1)
+
+    # IEEE-754 +inf bit pattern as int64 high word.
+    # 0x7F800000 << 32 = 9151314442816847872
+    best_packed = tl.full([BLOCK_S], 9151314442816847872, dtype=tl.int64)
+    SHIFT32 = tl.cast(4294967296, tl.int64)  # 2^32 as int64
+    n_i64 = tl.cast(N, tl.int64)
+
+    for j_start in range(0, N, BLOCK_J):
+        j_offs = j_start + tl.arange(0, BLOCK_J)
+        j_mask = j_offs < N
+        comp_j = tl.load(COMP_ptr + j_offs, mask=j_mask, other=-2)
+
+        row_offs = (orig_i.to(tl.int64)[:, None] * n_i64
+                    + j_offs.to(tl.int64)[None, :])
+        # Load and cast to fp32 (handles bf16 / fp16 / fp32 storage uniformly)
+        mrd_tile = tl.load(MRD_ptr + row_offs,
+                           mask=s_mask[:, None] & j_mask[None, :],
+                           other=float('inf')).to(tl.float32)
+
+        # Mask same-component
+        diff = (comp_j[None, :] != comp_i[:, None]) & j_mask[None, :] & s_mask[:, None]
+        mrd_tile = tl.where(diff, mrd_tile, float('inf'))
+
+        # Pack (weight_bits, j_idx) as int64. Use multiplication by 2^32 instead
+        # of `<< 32` to avoid int32 overflow warning.
+        wbits_u32 = mrd_tile.to(tl.uint32, bitcast=True)
+        wbits_i64 = wbits_u32.to(tl.int64)
+        j_i64 = j_offs.to(tl.int64)[None, :]
+        packed = wbits_i64 * SHIFT32 + j_i64
+
+        tile_best = tl.min(packed, axis=1)
+        best_packed = tl.minimum(best_packed, tile_best)
+
+    # Atomic_min into OUT_PACKED[comp_i] for each row in this block
+    out_addrs = OUT_PACKED_ptr + comp_i.to(tl.int64)
+    tl.atomic_min(out_addrs, best_packed, mask=s_mask)
+
+
+# =============================================================================
+# Kernel 2: write per-component canonical source (one orig_idx per component)
+# Run after argsort. For each component C in dense [0, n_components), picks
+# the FIRST orig_idx in sorted order (sorted_idx[comp_start[C]]).
+# =============================================================================
+
+@triton.jit
+def _set_per_component_src_kernel(
+    SORTED_IDX_ptr,         # (N,) int32
+    COMP_ptr,               # (N,) int32
+    OUT_SRC_ptr,            # (n_components,) int32
+    N,
+    BLOCK: tl.constexpr,
+):
+    pid = tl.program_id(0)
+    s_offs = pid * BLOCK + tl.arange(0, BLOCK)
+    s_mask = s_offs < N
+    orig = tl.load(SORTED_IDX_ptr + s_offs, mask=s_mask, other=0)
+    comp = tl.load(COMP_ptr + orig, mask=s_mask, other=0)
+    # The "first" element of each component is the one whose sorted position
+    # is smaller than all others with the same component. Use atomic_min on
+    # (sorted_pos, orig) pair — but for simplicity, just use atomic_min on
+    # orig (any vertex works as canonical).
+    target = OUT_SRC_ptr + comp.to(tl.int64)
+    # Use atomic_min so the smallest orig in each component wins
+    # (deterministic). orig values are int32 non-negative.
+    tl.atomic_min(target, orig, mask=s_mask)
+
+
+# =============================================================================
+# Kernel 3: concurrent union-find merge
+# For each component C, decode (weight, dst) from out_packed[C], set src from
+# out_src[C], find roots via bounded loop, atomic_cas to merge.
+# Successful merges write an MST edge via atomic counter.
+# =============================================================================
+
+@triton.jit
+def _concurrent_uf_kernel(
+    PARENT_ptr,             # (N,) int32, atomic
+    OUT_PACKED_ptr,         # (n_components,) int64
+    OUT_SRC_ptr,            # (n_components,) int32
+    MST_SRC_ptr,            # (N-1,) int32  — output edge sources
+    MST_DST_ptr,            # (N-1,) int32  — output edge targets
+    MST_W_ptr,              # (N-1,) fp32   — output edge weights
+    EDGE_COUNT_ptr,         # (1,) int32 atomic
+    n_components,
+    MAX_FIND: tl.constexpr,
+):
+    cid = tl.program_id(0)
+    if cid >= n_components:
+        return
+
+    packed = tl.load(OUT_PACKED_ptr + cid)
+    # Sentinel: high 32 bits = 0x7F800000 (+inf bit pattern) means "no edge found"
+    high32 = (packed >> 32).to(tl.int32)
+    if high32 >= 0x7F800000:
+        return
+
+    weight = high32.to(tl.float32, bitcast=True)
+    dst = (packed & 0xFFFFFFFF).to(tl.int32)
+    src = tl.load(OUT_SRC_ptr + cid)
+    if src < 0 or src == 2147483647:
+        return
+
+    # Single-pass: find roots with path-walking, then atomic_cas to merge.
+    # If CAS fails (concurrent thread merged hi first), we drop this edge —
+    # next Boruvka iter will re-propose. log(N) extra iters total (rare path).
+    rs = src
+    for _ in tl.static_range(MAX_FIND):
+        p = tl.load(PARENT_ptr + rs)
+        rs = tl.where(p == rs, rs, p)
+    rd = dst
+    for _ in tl.static_range(MAX_FIND):
+        p = tl.load(PARENT_ptr + rd)
+        rd = tl.where(p == rd, rd, p)
+
+    if rs == rd:
+        return
+
+    lo = tl.minimum(rs, rd)
+    hi = tl.maximum(rs, rd)
+    old = tl.atomic_cas(PARENT_ptr + hi, hi, lo)
+    if old == hi:
+        idx = tl.atomic_add(EDGE_COUNT_ptr, 1)
+        tl.store(MST_SRC_ptr + idx, src)
+        tl.store(MST_DST_ptr + idx, dst)
+        tl.store(MST_W_ptr + idx, weight)
+
+
+# =============================================================================
+# Generic edge-list union-find kernel (used by flash-dbscan CC, etc.).
+# Vectorised over BLOCK edges per program. Path-halving compresses the
+# tree during find via best-effort store-back. A merge counter lets the
+# caller exit as soon as no more edges can merge (a chain-shaped graph
+# with diameter ~ N can need 8+ passes).
+# =============================================================================
+
+@triton.jit
+def _union_edges_v2_kernel(
+    PARENT_ptr,         # (N,) int32 atomic
+    ROW_ptr,            # (E,) int32 — edge sources
+    COL_ptr,            # (E,) int32 — edge targets
+    MERGE_COUNTER_ptr,  # (1,) int32 — atomic counter; nonzero ⇒ keep iterating
+    n_edges,
+    BLOCK: tl.constexpr,
+    MAX_FIND: tl.constexpr,
+):
+    pid = tl.program_id(0)
+    e_offs = pid * BLOCK + tl.arange(0, BLOCK)
+    e_mask = e_offs < n_edges
+    u = tl.load(ROW_ptr + e_offs, mask=e_mask, other=0)
+    v = tl.load(COL_ptr + e_offs, mask=e_mask, other=0)
+
+    # find root of u with path-halving: each step does 2 hops
+    # (parent[ru] := parent[parent[ru]]) and writes back the compressed
+    # parent. Best-effort — concurrent writers to the same ru race, but
+    # since parent links are monotonically toward the root the result is
+    # always valid (no need for atomic_cas — a stale write only undoes
+    # one step of compression, never causes a cycle).
+    ru = u
+    for _ in tl.static_range(MAX_FIND):
+        p = tl.load(PARENT_ptr + ru, mask=e_mask, other=0)
+        gp = tl.load(PARENT_ptr + p, mask=e_mask, other=0)
+        # only compress if we actually moved up two levels
+        compress = (ru != p) & (p != gp) & e_mask
+        tl.store(PARENT_ptr + ru, gp, mask=compress)
+        ru = tl.where(p == ru, ru, gp)
+    rv = v
+    for _ in tl.static_range(MAX_FIND):
+        p = tl.load(PARENT_ptr + rv, mask=e_mask, other=0)
+        gp = tl.load(PARENT_ptr + p, mask=e_mask, other=0)
+        compress = (rv != p) & (p != gp) & e_mask
+        tl.store(PARENT_ptr + rv, gp, mask=compress)
+        rv = tl.where(p == rv, rv, gp)
+
+    diff = (ru != rv) & e_mask
+    lo = tl.minimum(ru, rv)
+    hi = tl.maximum(ru, rv)
+    # atomic_min instead of atomic_cas: parent[hi] := min(parent[hi], lo).
+    # Triton 3.6 doesn't support mask= on atomic_cas, but does on atomic_min.
+    # Semantics are actually cleaner for UF: parent[v] is monotonically
+    # non-increasing under atomic_min, so cycles are impossible by induction
+    # (we always have parent[v] ≤ v). "Made progress" = OLD > lo, i.e., we
+    # actually decreased parent[hi].
+    old = tl.atomic_min(PARENT_ptr + hi, lo, mask=diff)
+    progress = (old > lo) & diff
+    n_succ = tl.sum(progress.to(tl.int32))
+    # one atomic_add per CTA (not per edge) — negligible contention
+    if n_succ > 0:
+        tl.atomic_add(MERGE_COUNTER_ptr, n_succ)
+
+
+# =============================================================================
+# Triton compaction: replaces `torch.unique(parent, return_inverse=True)`.
+# After parent is fully flattened, parent[v] is either v (if v is a root) or
+# the root index. Then dense_id_at[v] = (number of roots in [0, v]) - 1 if v
+# is a root, otherwise undefined. We compute it via a torch cumsum on the
+# is_root bitmap (cheap, one launch) and then gather labels[v] = dense_id[parent[v]].
+# =============================================================================
+
+@triton.jit
+def _is_root_kernel(
+    PARENT_ptr,         # (N,) int32
+    IS_ROOT_ptr,        # (N,) int32 (0/1)
+    N,
+    BLOCK: tl.constexpr,
+):
+    pid = tl.program_id(0)
+    offs = pid * BLOCK + tl.arange(0, BLOCK)
+    mask = offs < N
+    p = tl.load(PARENT_ptr + offs, mask=mask, other=-1)
+    is_root = (p == offs.to(tl.int32)).to(tl.int32)
+    tl.store(IS_ROOT_ptr + offs, is_root, mask=mask)
+
+
+@triton.jit
+def _gather_labels_kernel(
+    PARENT_ptr,         # (N,) int32
+    DENSE_ID_ptr,       # (N,) int32 — dense id for each root, undefined for non-roots
+    LABELS_ptr,         # (N,) int32 — output
+    N,
+    BLOCK: tl.constexpr,
+):
+    pid = tl.program_id(0)
+    offs = pid * BLOCK + tl.arange(0, BLOCK)
+    mask = offs < N
+    p = tl.load(PARENT_ptr + offs, mask=mask, other=0)
+    label = tl.load(DENSE_ID_ptr + p.to(tl.int64), mask=mask, other=0)
+    tl.store(LABELS_ptr + offs, label, mask=mask)
+
+
+
+
+
+
+
+# =============================================================================
+# Kernel 4: pointer jumping path compression
+# parent[v] = parent[parent[v]] in parallel; iterate to convergence.
+# =============================================================================
+
+@triton.jit
+def _pointer_jump_kernel(
+    PARENT_ptr,
+    N,
+    BLOCK: tl.constexpr,
+):
+    pid = tl.program_id(0)
+    v_offs = pid * BLOCK + tl.arange(0, BLOCK)
+    v_mask = v_offs < N
+    p = tl.load(PARENT_ptr + v_offs, mask=v_mask, other=0)
+    # Cast p (int32) to int64 for pointer arithmetic safety on large N
+    pp = tl.load(PARENT_ptr + p.to(tl.int64), mask=v_mask, other=0)
+    tl.store(PARENT_ptr + v_offs, pp, mask=v_mask)
+
+
+
+def flash_cc_from_edges(rows: torch.Tensor, cols: torch.Tensor, N: int,
+                         max_find: int = 8, max_passes: int = 16):
+    """Connected components on the graph defined by edge list (rows, cols).
+
+    Iterative converging algorithm:
+      - Each pass: vectorized union with path-halving + atomic_cas hooks
+        (BLOCK=128 edges per program → 4 warps fully utilized vs the old
+         scalar 1-edge-per-warp launch).
+      - Then 6 calls to `_pointer_jump_kernel` to flatten the parent forest.
+      - Read the merge counter; exit as soon as a pass merges 0 edges.
+        For diameter-bound dense graphs this usually fires in 1-2 passes;
+        chain-shaped graphs (diameter ~N) converge in log_2(N) passes.
+
+    Args:
+        rows, cols: (E,) int32 — edge endpoints.
+        N: number of vertices.
+        max_find: bounded find depth per pass. With path-halving each step
+                  doubles the effective depth, so MAX_FIND=8 covers depth 256.
+        max_passes: hard ceiling (log2 of any conceivable graph diameter).
+
+    Returns:
+        labels: (N,) int32 — CC component id, dense in [0, n_components).
+    """
+    device = rows.device
+    parent = torch.arange(N, dtype=torch.int32, device=device)
+    E = rows.shape[0]
+
+    if E > 0:
+        BLOCK_EDGE = 128
+        BLOCK_JUMP = 1024
+        merge_counter = torch.zeros(1, dtype=torch.int32, device=device)
+        grid_edge = (triton.cdiv(E, BLOCK_EDGE),)
+        grid_jump = (triton.cdiv(N, BLOCK_JUMP),)
+
+        for _ in range(max_passes):
+            merge_counter.zero_()
+            _union_edges_v2_kernel[grid_edge](
+                parent, rows, cols, merge_counter, E,
+                BLOCK=BLOCK_EDGE, MAX_FIND=max_find, num_warps=4,
+            )
+            # Pointer-jump to flatten any remaining short chains. 6 iters
+            # cover depth ≤ 2^6 = 64 (post-compression chains are short).
+            for _ in range(6):
+                _pointer_jump_kernel[grid_jump](
+                    parent, N, BLOCK=BLOCK_JUMP, num_warps=4,
+                )
+            if merge_counter.item() == 0:
+                break
+
+    # Compact: roots have parent[v] == v; assign dense [0, n_cc) by index order.
+    is_root = torch.empty(N, dtype=torch.int32, device=device)
+    grid_compact = (triton.cdiv(N, 1024),)
+    _is_root_kernel[grid_compact](parent, is_root, N, BLOCK=1024, num_warps=4)
+    # cumsum gives 1-indexed dense IDs at root positions; subtract 1 → 0-indexed.
+    # At non-root positions the value is meaningless (we only gather via parent[v]).
+    dense_id = torch.cumsum(is_root, dim=0, dtype=torch.int32) - 1
+    labels = torch.empty(N, dtype=torch.int32, device=device)
+    _gather_labels_kernel[grid_compact](
+        parent, dense_id, labels, N, BLOCK=1024, num_warps=4,
+    )
+    return labels
+
+
+# =============================================================================
+# Kernel 5: per-component argmin with FOLDED canonical-src. Uses
+# ``orig_i = pid * BLOCK_S + lane`` directly (same cache hit-rate since
+# the column-side scan dominates the L2 traffic). The
+# ``OUT_SRC[component] = min(orig_idx)`` atomic_min is folded into the
+# same kernel, saving a separate launch per round.
+# =============================================================================
+
+@triton.jit
+def _per_component_argmin_v2_kernel(
+    MRD_ptr,                # (N, N) bf16 or fp32
+    PARENT_ptr,             # (N,) int32 — parent[i] is i's component label
+    OUT_PACKED_ptr,         # (N,) int64 — atomic_min target (sized to N)
+    OUT_SRC_ptr,            # (N,) int32 — atomic_min target for canonical src
+    N,
+    BLOCK_S: tl.constexpr,
+    BLOCK_J: tl.constexpr,
+):
+    pid = tl.program_id(0)
+    s_offs = pid * BLOCK_S + tl.arange(0, BLOCK_S)
+    s_mask = s_offs < N
+
+    orig_i = s_offs.to(tl.int64)
+    comp_i = tl.load(PARENT_ptr + orig_i, mask=s_mask, other=0)
+
+    # IEEE-754 +inf bit pattern as int64 high word: 0x7F800000 << 32
+    best_packed = tl.full([BLOCK_S], 9151314442816847872, dtype=tl.int64)
+    SHIFT32 = tl.cast(4294967296, tl.int64)
+    n_i64 = tl.cast(N, tl.int64)
+
+    for j_start in range(0, N, BLOCK_J):
+        j_offs = j_start + tl.arange(0, BLOCK_J)
+        j_mask = j_offs < N
+        comp_j = tl.load(PARENT_ptr + j_offs, mask=j_mask, other=0)
+
+        row_offs = orig_i[:, None] * n_i64 + j_offs.to(tl.int64)[None, :]
+        mrd_tile = tl.load(MRD_ptr + row_offs,
+                           mask=s_mask[:, None] & j_mask[None, :],
+                           other=float('inf')).to(tl.float32)
+
+        # Mask same-component
+        diff = (comp_j[None, :] != comp_i[:, None]) & j_mask[None, :] & s_mask[:, None]
+        mrd_tile = tl.where(diff, mrd_tile, float('inf'))
+
+        # Pack (weight_bits, j_idx) as int64 — the int64 axis-min uses a single
+        # comparator per element regardless of (weight, j) since the high bits
+        # dominate the ordering. Trying to track (best_w, best_j) as separate
+        # fp32+int (single tl.min + tl.argmin) was *slower* in benchmarking on
+        # H200 — the int64 reduction actually beats the fused fp32+argmin path.
+        wbits_u32 = mrd_tile.to(tl.uint32, bitcast=True)
+        wbits_i64 = wbits_u32.to(tl.int64)
+        j_i64 = j_offs.to(tl.int64)[None, :]
+        packed = wbits_i64 * SHIFT32 + j_i64
+
+        tile_best = tl.min(packed, axis=1)
+        best_packed = tl.minimum(best_packed, tile_best)
+
+    # Atomic_min into OUT_PACKED[comp_i] and OUT_SRC[comp_i] (folded)
+    out_addrs = OUT_PACKED_ptr + comp_i.to(tl.int64)
+    tl.atomic_min(out_addrs, best_packed, mask=s_mask)
+    src_addrs = OUT_SRC_ptr + comp_i.to(tl.int64)
+    tl.atomic_min(src_addrs, s_offs.to(tl.int32), mask=s_mask)
+
+
+# =============================================================================
+# Kernel 6: reset (N,)-sized int64 buffer to INF and int32 buffer to MAX_INT32
+# Faster than torch.full(...) by ~4× (~30us vs ~125us per pair at huge): single
+# kernel launch with one pass over both arrays vs two separate fill kernels +
+# Python tensor wrapping overhead.
+# =============================================================================
+
+@triton.jit
+def _reset_buffers_kernel(
+    OUT_PACKED_ptr,         # (N,) int64
+    OUT_SRC_ptr,            # (N,) int32
+    N,
+    BLOCK: tl.constexpr,
+):
+    pid = tl.program_id(0)
+    offs = pid * BLOCK + tl.arange(0, BLOCK)
+    mask = offs < N
+    INF_PACKED = tl.cast(9151314442816847872, tl.int64)  # 0x7F800000 << 32
+    MAX_I32 = tl.cast(2147483647, tl.int32)
+    tl.store(OUT_PACKED_ptr + offs, INF_PACKED, mask=mask)
+    tl.store(OUT_SRC_ptr + offs, MAX_I32, mask=mask)
+
+
+# =============================================================================
+# Python orchestration
+# =============================================================================
+
+def _pick_argmin_tile(N: int):
+    """Pick (BLOCK_S, BLOCK_J, num_warps, num_stages) for the argmin kernel.
+
+    Tuned on H200 (GPU 4) at bf16 MRD via `algorithms/hdbscan/_bench_mst.py`
+    (sub-process to avoid Triton autotune cache leakage between the per-tile
+    sweep and the E2E run).
+
+    Best per-N (of 4.8 TB/s peak BW):
+      N=20K:  BS=32, BJ=256, nw=4, ns=4 → 0.376 ms (44.4%)
+      N=80K:  BS=32, BJ=128, nw=4, ns=4 → 4.625 ms (57.7%)
+      N=200K: BS=32, BJ=128, nw=4, ns=3 → 28.08 ms (59.3%)
+    Larger BLOCK_J (256+) stalls at huge — the per-CTA register footprint of
+    the int64 packed tile (8 bytes × BLOCK_S × BLOCK_J) hurts occupancy and
+    spills into shared memory. Use BJ=128 for N >= 50K.
+    """
+    if N >= 50_000:
+        return 32, 128, 4, 3
+    return 32, 256, 4, 4
+
+
+def flash_mst(MRD: torch.Tensor) -> torch.Tensor:
+    """GPU-resident dense Boruvka MST on a symmetric (N, N) distance matrix.
+
+    Args:
+        MRD: (N, N) fp32 or bf16 — assumed symmetric, non-negative. Pass bf16
+             for 2× HBM speedup on the per-iter argmin scan; weight precision
+             reduces to ~7 bits but tie-breaking via packed (weight, dst) keeps
+             determinism.
+
+    Returns:
+        (N-1, 3) fp32 — MST edges as [src, dst, weight] sorted by weight.
+        src/dst are integer vertex IDs stored as fp32 (precise for N <= 2^24).
+    """
+    assert MRD.is_cuda and MRD.dtype in (torch.float32, torch.bfloat16) and MRD.ndim == 2
+    N = MRD.shape[0]
+    assert MRD.shape[1] == N
+    device = MRD.device
+
+    # Persistent state — allocated once, reused across all rounds.
+    parent = torch.arange(N, dtype=torch.int32, device=device)
+    out_packed = torch.empty(N, dtype=torch.int64, device=device)
+    out_src = torch.empty(N, dtype=torch.int32, device=device)
+
+    mst_src = torch.full((N - 1,), -1, dtype=torch.int32, device=device)
+    mst_dst = torch.full((N - 1,), -1, dtype=torch.int32, device=device)
+    mst_w = torch.zeros(N - 1, dtype=torch.float32, device=device)
+    edge_count = torch.zeros(1, dtype=torch.int32, device=device)
+
+    BLOCK_S, BLOCK_J, num_warps_argmin, num_stages_argmin = _pick_argmin_tile(N)
+    BLOCK_RESET = 1024
+    BLOCK_JUMP = 1024
+
+    # Standard Boruvka iter cap: log2(N) + small slack
+    max_iters = max(2, int(math.ceil(math.log2(max(N, 2)))) + 5)
+    target_edges = N - 1
+    prev_count = 0
+
+    for it in range(max_iters):
+        # Reset per-component buffers in one kernel (int64 + int32 fused)
+        grid_reset = (triton.cdiv(N, BLOCK_RESET),)
+        _reset_buffers_kernel[grid_reset](
+            out_packed, out_src, N,
+            BLOCK=BLOCK_RESET, num_warps=4,
+        )
+
+        # Per-component argmin: folds canonical-src into the kernel
+        grid_arg = (triton.cdiv(N, BLOCK_S),)
+        _per_component_argmin_v2_kernel[grid_arg](
+            MRD, parent, out_packed, out_src, N,
+            BLOCK_S=BLOCK_S, BLOCK_J=BLOCK_J,
+            num_warps=num_warps_argmin, num_stages=num_stages_argmin,
+        )
+
+        # Concurrent union-find: dispatch over [0, N). Non-root CIDs see
+        # OUT_PACKED[c] == INF and early-return — cheap (~2 instructions).
+        grid_uf = (N,)
+        _concurrent_uf_kernel[grid_uf](
+            parent, out_packed, out_src,
+            mst_src, mst_dst, mst_w, edge_count,
+            N,
+            MAX_FIND=8, num_warps=1,
+        )
+
+        # Pointer-jump until paths flat. Triton kernel (one launch ≈ 30 µs at
+        # huge) replaces the original log_2(N)-deep `parent[parent.to(int64)]`
+        # torch loop. 6 iters flatten depth ≤ 64 which covers all observed
+        # post-CAS chains; the next round's argmin handles any residual depth
+        # (a non-flat parent makes parent[i] != parent[j] occasionally true
+        # within a component → at worst one wasted edge proposal which the
+        # uf_merge rejects).
+        grid_jump = (triton.cdiv(N, BLOCK_JUMP),)
+        for _ in range(6):
+            _pointer_jump_kernel[grid_jump](
+                parent, N, BLOCK=BLOCK_JUMP, num_warps=4,
+            )
+
+        # Single host sync per iter to check completion. Cost: ~5 µs each;
+        # 8 iters at huge = 40 µs total (negligible vs ~30 ms argmin).
+        cur_count = int(edge_count.item())
+        if cur_count >= target_edges:
+            break
+        if cur_count == prev_count:
+            # No new edges this round — graph is disconnected.
+            break
+        prev_count = cur_count
+
+    n_edges = int(edge_count.item())
+    if n_edges != target_edges:
+        raise RuntimeError(
+            f"flash_mst: produced {n_edges} edges, expected {target_edges}. "
+            f"Graph may be disconnected, or merge kernel hit MAX_FIND."
+        )
+
+    # Sort by weight ascending for downstream HDBSCAN tree-building
+    sort_idx = torch.argsort(mst_w)
+    mst_src = mst_src[sort_idx]
+    mst_dst = mst_dst[sort_idx]
+    mst_w = mst_w[sort_idx]
+
+    # Pack as (N-1, 3) fp32 for return — int IDs preserved exactly for N<=2^24
+    out = torch.stack([mst_src.to(torch.float32),
+                        mst_dst.to(torch.float32),
+                        mst_w], dim=1)
+    return out
diff --git a/dbscanlib/_kernels/primitives/__init__.py b/dbscanlib/_kernels/primitives/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/dbscanlib/_kernels/primitives/dbscan/__init__.py b/dbscanlib/_kernels/primitives/dbscan/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/dbscanlib/_kernels/primitives/dbscan/triton/__init__.py b/dbscanlib/_kernels/primitives/dbscan/triton/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/dbscanlib/_kernels/primitives/dbscan/triton/dbscan.py b/dbscanlib/_kernels/primitives/dbscan/triton/dbscan.py
new file mode 100644
index 0000000..b9cc7b6
--- /dev/null
+++ b/dbscanlib/_kernels/primitives/dbscan/triton/dbscan.py
@@ -0,0 +1,346 @@
+"""flash-dbscan: GPU-resident DBSCAN.
+
+Two algorithmic paths, dispatched by D:
+
+  • D ≤ 4 (low-D): **GRID radius search**. Bin points into cells of side eps;
+    each point only needs to scan its 3^D=9 (D=2) or 27 (D=3) neighbor cells.
+    For typical D=2 spatial data with N ≈ M ≈ 100s/cell, this is ~250×
+    fewer distance evaluations than brute force.
+
+  • D ≥ 5 (high-D): **brute-force top-K** via flash_knn (bf16 GEMM + on-chip
+    top-K), then eps filter on the returned distances.
+
+Both paths feed the same downstream pipeline:
+  → core mask: deg ≥ min_samples
+  → connected components (atomic-CAS UF on GPU)
+  → border assignment: smallest CC label among in-eps core neighbors
+  → compact labels to dense [0, n_clusters)
+
+The grid path is exact for any D where the cell-side ≥ eps (so all in-eps
+neighbors lie in the 3^D neighborhood). For high-D the curse of
+dimensionality makes the per-cell cost grow as 3^D, hence the brute-force
+path for D ≥ 5.
+"""
+import math
+
+import torch
+import triton
+import triton.language as tl
+
+# High-D brute-force neighbour path: flashlib's flash_knn = a fused Triton kNN
+# (indices) + a tiled gather kernel (exact squared-L2 distances), both vendored
+# from the flashlib knn primitive under dbscanlib._kernels.
+from dbscanlib._kernels.primitives.knn.triton import flash_knn_triton
+from dbscanlib._kernels.kernels.distance.triton.knn_gather_l2sq import (
+    triton_knn_gather_sqdist,
+)
+
+
+def flash_knn(x, y, k, *, tol=None):
+    """Return ``(dist_sq, idx)`` top-k neighbours -- flashlib's flash_knn contract."""
+    idx = flash_knn_triton(x, y, k)
+    dist_sq = triton_knn_gather_sqdist(x, y, idx)
+    return dist_sq, idx
+from dbscanlib._kernels.kernels.flash_mst.triton.flash_mst import flash_cc_from_edges
+
+
+def _next_pow2(x):
+    return 1 << (int(x - 1).bit_length()) if x > 1 else 1
+
+
+# ---------------------------------------------------------------------------
+# GRID radius-search kernel (D=2)
+# ---------------------------------------------------------------------------
+# Each program handles BN query points. For each query, computes its grid
+# cell, iterates over the 3×3 neighbor cells, looks up each cell in a
+# hash table (linear probing) to find the (start, end) indices in
+# sorted_idx, and computes fp32 (xi - xj)² for each candidate; emits hits
+# (dist ≤ eps²) into the per-row neighbor buffer.
+#
+# Hash table layout: (HT_SIZE,) int64 packed as (cell_id << 32) | bin_idx.
+# Linear probing with a sentinel (-1) for empty slots.
+# ---------------------------------------------------------------------------
+
+@triton.jit
+def _grid_radius_2d_kernel(
+    X_ptr,                  # (N, 2) fp32
+    SORTED_PTR_ptr,         # (N,) int32 — point indices sorted by cell
+    CELL_START_ptr,         # (GW * GH,) int32 — dense 2D grid start
+    CELL_END_ptr,           # (GW * GH,) int32 — dense 2D grid end
+    DEG_ptr,                # (N,) int32
+    NBR_IDX_ptr,            # (N, K) int32
+    N: tl.constexpr, K: tl.constexpr,
+    GW: tl.constexpr, GH: tl.constexpr,   # grid dims
+    INV_EPS, EPS_SQ,
+    GRID_X_MIN, GRID_Y_MIN,
+    BN: tl.constexpr,
+):
+    """Grid radius search with dense 2D grid index.
+
+    Each query maps to its (cx, cy) cell; we scan the 3×3 neighborhood by
+    direct indexing into CELL_START/END arrays of size (GW × GH). For our
+    standardised D=2 inputs (~6σ in [-3, 3], eps=0.04 → grid ~150×150,
+    storage ~22500 int32 × 2 = 180 KB).
+    """
+    pid = tl.program_id(0)
+    n_offs = (pid * BN + tl.arange(0, BN)).to(tl.int64)
+    n_mask = n_offs < N
+
+    xi = tl.load(X_ptr + n_offs * 2, mask=n_mask, other=0.0)
+    yi = tl.load(X_ptr + n_offs * 2 + 1, mask=n_mask, other=0.0)
+
+    cxq = tl.floor((xi - GRID_X_MIN) * INV_EPS).to(tl.int32)
+    cyq = tl.floor((yi - GRID_Y_MIN) * INV_EPS).to(tl.int32)
+
+    cur_cnt = tl.zeros([BN], dtype=tl.int32)
+
+    # 3×3 cell neighborhood. Static-unrolled outer.
+    for nbi in tl.static_range(9):
+        dx = nbi // 3 - 1
+        dy = nbi % 3 - 1
+        cx = cxq + dx
+        cy = cyq + dy
+        # Bounds check (drop out-of-grid cells by mask).
+        in_grid = (cx >= 0) & (cx < GW) & (cy >= 0) & (cy < GH) & n_mask
+        cell_idx = cy * GW + cx
+        cell_idx_safe = tl.where(in_grid, cell_idx, 0)
+        s = tl.load(CELL_START_ptr + cell_idx_safe.to(tl.int64),
+                    mask=in_grid, other=0)
+        e = tl.load(CELL_END_ptr + cell_idx_safe.to(tl.int64),
+                    mask=in_grid, other=0)
+        range_len = tl.where(in_grid, e - s, 0)
+        max_iter = tl.max(range_len)
+
+        for j in tl.range(0, max_iter, 1):
+            in_range = (j < range_len) & in_grid
+            idx = s + j
+            pj = tl.load(SORTED_PTR_ptr + idx.to(tl.int64), mask=in_range, other=0)
+            xj = tl.load(X_ptr + pj.to(tl.int64) * 2, mask=in_range, other=0.0)
+            yj = tl.load(X_ptr + pj.to(tl.int64) * 2 + 1, mask=in_range, other=0.0)
+            dxv = xi - xj
+            dyv = yi - yj
+            dist = dxv * dxv + dyv * dyv
+            hit_eps = (dist <= EPS_SQ) & in_range
+            slot_addr = n_offs * K + cur_cnt.to(tl.int64)
+            emit = hit_eps & (cur_cnt < K)
+            tl.store(NBR_IDX_ptr + slot_addr, pj, mask=emit)
+            cur_cnt = cur_cnt + hit_eps.to(tl.int32)
+
+    cur_cnt = tl.minimum(cur_cnt, tl.full([BN], K, dtype=tl.int32))
+    tl.store(DEG_ptr + n_offs, cur_cnt, mask=n_mask)
+
+
+def _build_grid_index(X: torch.Tensor, eps: float):
+    """Build a dense 2D grid index over points X (N, 2) fp32.
+
+    Returns:
+        sorted_ptr:           (N,) int32 — point indices sorted by cell
+        cell_start, cell_end: (GW * GH,) int32 — start/end indices in sorted_ptr
+        gw, gh:               grid dims
+        inv_eps, x_min, y_min: kernel params
+    """
+    N = X.shape[0]
+    device = X.device
+    inv_eps = 1.0 / eps
+    x_min = float(X[:, 0].min().item())
+    y_min = float(X[:, 1].min().item())
+    x_max = float(X[:, 0].max().item())
+    y_max = float(X[:, 1].max().item())
+
+    GW = int((x_max - x_min) * inv_eps) + 2
+    GH = int((y_max - y_min) * inv_eps) + 2
+    n_cells = GW * GH
+
+    cx = ((X[:, 0] - x_min) * inv_eps).floor().to(torch.int32).clamp(0, GW - 1)
+    cy = ((X[:, 1] - y_min) * inv_eps).floor().to(torch.int32).clamp(0, GH - 1)
+    cell_id = cy.to(torch.int64) * GW + cx.to(torch.int64)
+
+    # Sort points by cell_id
+    sorted_cell, perm = torch.sort(cell_id, stable=True)
+    sorted_ptr = perm.to(torch.int32)
+
+    # Build dense (GW*GH,) start/end via bucket counts.
+    counts = torch.zeros(n_cells, dtype=torch.int32, device=device)
+    counts.scatter_add_(0, sorted_cell, torch.ones(N, dtype=torch.int32, device=device))
+    cell_end = torch.cumsum(counts, dim=0, dtype=torch.int32)
+    cell_start = cell_end - counts
+
+    return sorted_ptr, cell_start, cell_end, GW, GH, inv_eps, x_min, y_min
+
+
+def _flash_dbscan_grid(X: torch.Tensor, eps: float, min_samples: int,
+                       max_neighbors: int):
+    """D=2 grid-based path."""
+    N, D = X.shape
+    assert D == 2
+    device = X.device
+    K = max(min_samples, max_neighbors)
+    eps_sq = float(eps) ** 2
+
+    # Build grid index
+    (sorted_ptr, cell_start, cell_end, GW, GH,
+     inv_eps, x_min, y_min) = _build_grid_index(X, eps)
+
+    deg = torch.zeros(N, dtype=torch.int32, device=device)
+    nbr_idx = torch.full((N, K), -1, dtype=torch.int32, device=device)
+
+    BN = 32
+    grid = (triton.cdiv(N, BN),)
+    _grid_radius_2d_kernel[grid](
+        X.contiguous(), sorted_ptr, cell_start, cell_end,
+        deg, nbr_idx,
+        N=N, K=K, GW=GW, GH=GH,
+        INV_EPS=inv_eps, EPS_SQ=eps_sq,
+        GRID_X_MIN=x_min, GRID_Y_MIN=y_min,
+        BN=BN,
+        num_warps=4, num_stages=1,
+    )
+    return deg, nbr_idx, K
+
+
+def _flash_dbscan_brute(X: torch.Tensor, eps: float, min_samples: int,
+                         max_neighbors: int, *, tol=None):
+    """High-D brute-force path via flash_knn -- single bf16 kNN call.
+
+    Enumerates each point's top-``max_neighbors`` ε-candidates (in bf16
+    by default; pass ``tol=0`` to force fp32). Points with more than
+    ``max_neighbors`` true ε-neighbors will have their excess neighbours
+    silently truncated, but this does **not** change the DBSCAN
+    clustering:
+
+    * Core-mask correctness: a point is core iff its ε-degree ≥
+      ``min_samples``. As long as ``max_neighbors >= min_samples``
+      the bounded-K enumeration still observes ``min_samples``-many
+      ε-neighbours when they exist, so the core mask is exact.
+    * Cluster-membership correctness: two core points belong to the
+      same cluster iff they are reachable through a chain of
+      core-core ε-edges. Dense clusters expose Θ(max_neighbors) such
+      edges per core point, so any one of them is enough to merge
+      that core into the cluster's connected component. Edges
+      ranked beyond top-K are redundant for CC.
+
+    The historical K-grow loop ("double K until the K-th NN exceeds
+    ε") was reverted in 2026-05 -- the iterative re-launches grew K
+    geometrically in dense regimes (medium-D blob shapes hit K=1024
+    in 7 calls = 566 ms total) without changing ARI vs cuML beyond
+    the noise floor inherited from the bounded-K approximation.
+
+    Callers who actually need exhaustive enumeration (e.g. for
+    reach-distance diagnostics, not clustering) should pass a
+    larger ``max_neighbors`` once.
+    """
+    N, D = X.shape
+    device = X.device
+    K = max(min_samples, max_neighbors)
+    K = min(K, N)
+    eps_sq = float(eps) ** 2
+
+    # Default to fp32 KNN: at these shapes (N, M >= 16K) flash_knn
+    # routes to the "large_n" insert path (BN=128 single-pass per CTA),
+    # which is memory-bound and runs at the same speed in fp32 as in
+    # bf16 -- the .to(bf16) cast itself eats the would-be win. fp32
+    # also gives bit-exact ARI vs sklearn
+    # on the standard ε boundaries; bf16 quantisation flips a handful
+    # of borderline points (~3% on D=64 eps=8). Power users can opt
+    # into bf16 via ``tol=1e-3`` if they have looser ε.
+    if D < 16:
+        Xpad = torch.zeros(N, 16, device=device, dtype=X.dtype)
+        Xpad[:, :D] = X
+        X_kernel = Xpad.contiguous()
+    else:
+        X_kernel = X.contiguous()
+    knn_dist_sq, knn_idx = flash_knn(
+        X_kernel[None], X_kernel[None], k=K, tol=tol)
+    knn_dist_sq = knn_dist_sq[0]
+    knn_idx = knn_idx[0]
+
+    valid = knn_dist_sq <= eps_sq
+    deg = valid.sum(dim=1).to(torch.int32)
+    nbr_idx = torch.where(valid, knn_idx.to(torch.int32),
+                          torch.full_like(knn_idx, -1, dtype=torch.int32))
+    return deg, nbr_idx, K
+
+
+def flash_dbscan(X: torch.Tensor, eps: float, min_samples: int = 5,
+                 max_neighbors: int = 32, *, tol=None):
+    """End-to-end Triton DBSCAN.
+
+    Args:
+        X: (N, D) float32 CUDA tensor.
+        eps: distance threshold (Euclidean).
+        min_samples: minimum points in eps-neighborhood for a point to be core.
+        max_neighbors: per-point ε-candidate budget for the high-D brute
+            path. Default 32 -- core detection stays exact for any
+            ``max_neighbors >= min_samples``; the connected-component
+            stage only needs O(max_neighbors) redundant edges per
+            cluster, so cluster membership is robust to the bounded-K
+            approximation. Bump this if you need exhaustive ε-edge
+            enumeration (e.g. reach-distance diagnostics).
+        tol: residual tolerance forwarded to :func:`flash_knn` on the
+            high-D (D >= 3) brute-force path. ``None`` (default) keeps
+            fp32 storage -- the "large_n" single-pass insert path
+            (BN=128) flash_knn picks on these shapes is memory-bound,
+            so fp32 and bf16 run at the same speed; fp32 wins on ARI by
+            avoiding ε-boundary quantisation. Pass ``tol=1e-3`` to opt
+            into bf16/fp16 KNN storage (matches the historical HEAD
+            cast) when you want lower HBM pressure.
+
+    Returns:
+        labels: (N,) int32 -- cluster id (>= 0) or -1 for noise.
+    """
+    assert X.is_cuda
+    N, D = X.shape
+    device = X.device
+
+    # ── Stage 1: per-point in-eps neighbor list ─────────────────────────
+    # Path A: D ≤ 2 → grid (cell-side eps; 9-cell scan, exact in any dtype).
+    # Path B: D ≥ 3 → brute-force top-K via flash_knn (tol-routed dtype).
+    if D == 2:
+        # Grid kernel reads X via raw fp32 strides; cast internally.
+        # This is a kernel implementation detail, NOT a public dtype contract.
+        deg, nbr_idx, K = _flash_dbscan_grid(
+            X if X.dtype == torch.float32 else X.float(),
+            eps, min_samples, max_neighbors,
+        )
+    else:
+        deg, nbr_idx, K = _flash_dbscan_brute(
+            X, eps, min_samples, max_neighbors, tol=tol)
+
+    # ── Stage 2: core mask ──────────────────────────────────────────────
+    core_mask = deg >= min_samples
+
+    # ── Stage 3: edge construction (core-core only) ─────────────────────
+    K_eff = nbr_idx.shape[1]
+    nbr_idx_i64 = nbr_idx.to(torch.int64)
+    valid_slot = nbr_idx >= 0
+    core_per_row = core_mask[:, None].expand(-1, K_eff)
+    nbr_idx_safe = torch.where(valid_slot, nbr_idx_i64, torch.zeros_like(nbr_idx_i64))
+    core_per_col = core_mask[nbr_idx_safe] & valid_slot
+    edge_mask = valid_slot & core_per_row & core_per_col
+    rows = (torch.arange(N, device=device, dtype=torch.int32).view(-1, 1)
+            .expand(-1, K_eff).contiguous())[edge_mask].contiguous()
+    cols = nbr_idx[edge_mask].contiguous()
+    label_cc = flash_cc_from_edges(rows, cols, N)
+
+    # ── Stage 4: assign labels (core: CC label, else: -1 for now) ────────
+    INT_MAX = 2 ** 31 - 1
+    label = torch.where(core_mask, label_cc, torch.full_like(label_cc, -1))
+
+    # ── Stage 5: border assignment ───────────────────────────────────────
+    nbr_labels = label[nbr_idx_safe]
+    nbr_is_core = core_mask[nbr_idx_safe] & valid_slot
+    border_cand = torch.where(valid_slot & nbr_is_core, nbr_labels,
+                               torch.full_like(nbr_labels, INT_MAX))
+    min_core_label = border_cand.min(dim=1).values
+    is_border = (~core_mask) & (min_core_label != INT_MAX) & (min_core_label >= 0)
+    label = torch.where(is_border, min_core_label, label)
+
+    # ── Stage 6: compact ─────────────────────────────────────────────────
+    valid = label >= 0
+    if valid.any():
+        unique, inv = torch.unique(label[valid], return_inverse=True)
+        compact = torch.full_like(label, -1)
+        compact[valid] = inv.to(torch.int32)
+        label = compact
+
+    return label
diff --git a/dbscanlib/_kernels/primitives/knn/__init__.py b/dbscanlib/_kernels/primitives/knn/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/dbscanlib/_kernels/primitives/knn/triton/__init__.py b/dbscanlib/_kernels/primitives/knn/triton/__init__.py
new file mode 100644
index 0000000..f5834ec
--- /dev/null
+++ b/dbscanlib/_kernels/primitives/knn/triton/__init__.py
@@ -0,0 +1,41 @@
+"""knn triton backend.
+
+Re-exports the public Python wrappers from each component file.
+``@triton.jit`` kernels stay private to their file.
+
+Kernel layout:
+
+* :mod:`_common`   -- shared helpers: ``_next_pow2``, micro-bench,
+  IEEE-sortable u32 transform, ``_INF_PACKED`` sentinel.
+* :mod:`sortmerge` -- packed-uint64 sort-merge top-K (x²-free). Routed
+  to only for the small-Q + medium-K Pattern-A corner.
+* :mod:`insert`    -- iterative argmin-insert top-K (x²-free). The
+  general path -- BN in {8, 16, 32, 64, 128} covers everything from
+  small-Q search to large-Q build.
+* :mod:`_row_norm` -- ``_fast_row_sq`` / ``_get_or_compute_csq`` for
+  the CuteDSL FA3 wrapper (it needs ``c_sq`` as a kernel argument).
+* :mod:`dispatch`  -- host-side router. Single public entry
+  :func:`flash_knn_triton` runs through :func:`_heuristic_config`
+  unconditionally -- the per-CTA-count check inside the heuristic
+  picks single-pass vs M-split. Per-shape config (BN/BM/mode/mps/
+  ns_pipe) also falls out of :func:`_heuristic_config`.
+"""
+from dbscanlib._kernels.primitives.knn.triton._common import (
+    _next_pow2,
+    _bench_quick,
+)
+from dbscanlib._kernels.primitives.knn.triton._row_norm import (
+    _fast_row_sq,
+    _get_or_compute_csq,
+)
+from dbscanlib._kernels.primitives.knn.triton.dispatch import (
+    flash_knn_triton,
+    flash_knn_triton_small_n,
+    flash_knn_triton_large_n,
+)
+
+__all__ = [
+    "flash_knn_triton",
+    "flash_knn_triton_small_n",
+    "flash_knn_triton_large_n",
+]
diff --git a/dbscanlib/_kernels/primitives/knn/triton/_common.py b/dbscanlib/_kernels/primitives/knn/triton/_common.py
new file mode 100644
index 0000000..686e85b
--- /dev/null
+++ b/dbscanlib/_kernels/primitives/knn/triton/_common.py
@@ -0,0 +1,81 @@
+"""Shared utilities for flash_knn Triton dispatch + kernels.
+
+Lives in one place so :mod:`dispatch`, :mod:`sortmerge` and :mod:`insert`
+all import the same primitives:
+
+  * :func:`_next_pow2` -- D / K padding helper.
+  * :func:`_bench_quick` -- micro-bench used by the autotuner.
+  * :func:`_fp32_to_sortable_u32` / :func:`_sortable_u32_to_fp32` --
+    branch-free IEEE-sortable u32 transform. With the x²-free score
+    ``s = c² - 2⟨x, c⟩`` the quantity is signed, so the packed-uint64
+    sort-merge / final-sort needs an ascending-u32 = ascending-fp32
+    mapping. The transform is 3 ops, exact for non-NaN fp32.
+  * :data:`_INF_PACKED` -- (sortable +inf << 32) | -1 idx sentinel,
+    used as the initial fill of the sortmerge running top-K so the
+    early-exit ``chunk_best < sortable_inv(top)`` always fires until
+    real values arrive.
+"""
+from __future__ import annotations
+
+import torch
+import triton
+import triton.language as tl
+
+
+def _next_pow2(n: int) -> int:
+    """Smallest power-of-two >= ``max(1, n)``. Used for D / K padding."""
+    if n <= 1:
+        return 1
+    return 1 << (n - 1).bit_length()
+
+
+def _bench_quick(fn, *, warmup: int = 3, reps: int = 5) -> float:
+    """Median-ish wall time in ms; cheap enough to call once per autotune cfg."""
+    for _ in range(warmup):
+        fn()
+    torch.cuda.synchronize()
+    s = torch.cuda.Event(enable_timing=True)
+    e = torch.cuda.Event(enable_timing=True)
+    s.record()
+    for _ in range(reps):
+        fn()
+    e.record()
+    torch.cuda.synchronize()
+    return s.elapsed_time(e) / reps
+
+
+# Sortable upper-bits = sortable(+inf) = 0xFF800000; lower-bits = -1 (idx
+# sentinel) = 0xFFFFFFFF. ``_sortable_u32_to_fp32(0xFF800000) == +inf`` so
+# the sortmerge early-exit ``chunk_best < +inf`` always fires until the
+# running top-K starts to hold real values.
+#
+# Wrapped in ``tl.constexpr`` so ``@triton.jit`` kernels can reference it
+# directly as a module-level global (un-wrapped Python ints are not
+# accessible from inside a JIT'd kernel as of Triton 3.x).
+_INF_PACKED = tl.constexpr(0xFF800000_FFFFFFFF)
+
+
+@triton.jit
+def _fp32_to_sortable_u32(x):
+    """Map ascending fp32 -> ascending uint32 (branch-free, 3 ops).
+
+    Standard IEEE-754 trick: arithmetic-shift the int32 view by 31 (so
+    MSB=1 / negative produces 0xFFFFFFFF, MSB=0 / positive produces 0),
+    OR with 0x80000000 to flip the sign bit of positives, then XOR with
+    the original bits. NaNs map into 0xFF800001..0xFFFFFFFF (unused in
+    practice).
+    """
+    bits = x.to(tl.uint32, bitcast=True)
+    sign_mask = (bits.to(tl.int32, bitcast=True) >> 31).to(tl.uint32, bitcast=True)
+    flip = sign_mask | 0x80000000
+    return bits ^ flip
+
+
+@triton.jit
+def _sortable_u32_to_fp32(s):
+    """Inverse of :func:`_fp32_to_sortable_u32`."""
+    sign_mask = (s.to(tl.int32, bitcast=True) >> 31).to(tl.uint32, bitcast=True)
+    # In sortable space: MSB=1 -> was positive (flip back via 0x80000000);
+    #                    MSB=0 -> was negative (flip back via 0xFFFFFFFF).
+    flip = (sign_mask ^ 0xFFFFFFFF) | 0x80000000
+    return (s ^ flip).to(tl.float32, bitcast=True)
diff --git a/dbscanlib/_kernels/primitives/knn/triton/_row_norm.py b/dbscanlib/_kernels/primitives/knn/triton/_row_norm.py
new file mode 100644
index 0000000..39fd0c8
--- /dev/null
+++ b/dbscanlib/_kernels/primitives/knn/triton/_row_norm.py
@@ -0,0 +1,93 @@
+"""Fast row sum-of-squares (||x_i||^2) helper + per-tensor cache.
+
+Tiny utility, but shared by the FA3 fused KNN path which needs the
+pre-computed query/corpus norms to fold into the squared-L2 distance
+formula ``||x - c||^2 = ||x||^2 + ||c||^2 - 2 <x, c>``.
+
+This module deliberately stays narrow:
+  - ``_row_sq_kernel`` is a pure Triton row-norm kernel (no cross
+    matrix, no top-K) — it never materialises an N x M tensor.
+  - ``_fast_row_sq`` wraps the kernel.
+  - ``_get_or_compute_csq`` caches results per ``(data_ptr, shape,
+    dtype)`` so repeat queries on the same corpus skip the recompute.
+"""
+from __future__ import annotations
+
+import math
+
+import torch
+import triton
+import triton.language as tl
+
+from dbscanlib._kernels.primitives.knn.triton._common import _next_pow2
+
+
+@triton.jit
+def _row_sq_kernel(
+    x_ptr, out_ptr,
+    stride_x_b, stride_x_n, stride_x_d,
+    stride_o_b, stride_o_n,
+    B: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
+    BN: tl.constexpr, BD: tl.constexpr,
+):
+    """Row sum-of-squares for ``(B, N, D) -> (B, N)`` fp32.
+
+    Casts input to fp32 inside the kernel; reads input exactly once
+    and writes output exactly once. At BN=128 BD=128 nw=4 this hits
+    ~36% peak HBM BW on H200 (typical for tall-skinny reads).
+    """
+    pid_n = tl.program_id(0)
+    pid_b = tl.program_id(1).to(tl.int64)
+    n_offs = (pid_n * BN + tl.arange(0, BN)).to(tl.int64)
+    n_mask = n_offs < N
+    acc = tl.zeros([BN], dtype=tl.float32)
+    for d_start in range(0, D, BD):
+        d_offs = (d_start + tl.arange(0, BD)).to(tl.int64)
+        d_mask = d_offs < D
+        x = tl.load(
+            x_ptr + pid_b * stride_x_b
+            + n_offs[:, None] * stride_x_n
+            + d_offs[None, :] * stride_x_d,
+            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
+        )
+        x_f = x.to(tl.float32)
+        acc += tl.sum(x_f * x_f, axis=1)
+    tl.store(out_ptr + pid_b * stride_o_b + n_offs * stride_o_n,
+             acc, mask=n_mask)
+
+
+def _fast_row_sq(x: torch.Tensor) -> torch.Tensor:
+    """Return ``(B, N)`` fp32 row sum-of-squares via the Triton kernel."""
+    assert x.is_cuda and x.ndim == 3
+    B, N, D = x.shape
+    out = torch.empty(B, N, device=x.device, dtype=torch.float32)
+    BN = 128 if N >= 128 else _next_pow2(max(N, 16))
+    BD = 128 if D >= 128 else _next_pow2(max(D, 16))
+    grid = (math.ceil(N / BN), B)
+    _row_sq_kernel[grid](
+        x, out,
+        x.stride(0), x.stride(1), x.stride(2),
+        out.stride(0), out.stride(1),
+        B=B, N=N, D=D, BN=BN, BD=BD,
+        num_warps=4,
+    )
+    return out
+
+
+# Per-data-ptr cache so repeated queries on the same corpus don't pay
+# the row-norm kernel cost.
+_csq_cache: dict = {}
+
+
+def _get_or_compute_csq(c: torch.Tensor) -> torch.Tensor:
+    """Cached ``||c_i||^2`` (fp32) keyed by ``(data_ptr, shape, dtype)``."""
+    key = (c.data_ptr(), tuple(c.shape), c.dtype)
+    cached = _csq_cache.get(key)
+    if cached is not None and cached.device == c.device:
+        return cached
+    csq = _fast_row_sq(c)
+    _csq_cache[key] = csq
+    if len(_csq_cache) > 8:
+        for old_key in list(_csq_cache.keys())[:-8]:
+            del _csq_cache[old_key]
+    return csq
diff --git a/dbscanlib/_kernels/primitives/knn/triton/dispatch.py b/dbscanlib/_kernels/primitives/knn/triton/dispatch.py
new file mode 100644
index 0000000..1476b18
--- /dev/null
+++ b/dbscanlib/_kernels/primitives/knn/triton/dispatch.py
@@ -0,0 +1,1178 @@
+"""flash_knn -- host-side dispatcher: kernel/config selection + 2-stage launch.
+
+Backs the public :func:`dbscanlib._kernels.primitives.knn.flash_knn` entry point.
+
+Top-level wrappers
+------------------
+
+* :func:`flash_knn_triton` -- one-shot kNN, returns ``(B, N, k)`` int32
+  indices. The shape-only heuristic now handles both build (large-N,
+  single-pass per CTA) and search (small-N, M-split flash-decode)
+  without any host-side SM-saturation gate -- ``ctas_no_split`` is
+  inspected after BN is picked, so build shapes (``BN=128`` ->
+  ``ctas_no_split`` already saturates 132 SMs) and search shapes
+  (small BN -> M-split for tail-fill) both fall out of the same
+  decision tree.
+* :func:`flash_knn_triton_small_n` / :func:`flash_knn_triton_large_n`
+  -- explicit overrides for callers (e.g. K-means assign) that know
+  their shape category at construction time.
+* :func:`_heuristic_config`, :func:`_autotune` -- shape-only heuristic
+  (default, fast first call) + opt-in brute-force autotune (slower
+  first call, cached steady-state). Both share the same kernel grid;
+  ``force_path="large_n"`` collapses M-splits down to one per CTA.
+
+Two kernels live behind this dispatcher:
+
+* :mod:`insert`    -- iterative argmin-insert top-K. Picked by the
+  heuristic on virtually every shape.
+* :mod:`sortmerge` -- packed-uint64 sort-merge top-K with the IEEE-
+  sortable u32 transform. Routed to for the **Pattern-A small-Q**
+  corner (``B*N <= 8 or (B*N <= 16 and K in {32, 64})``) at medium-K
+  + ``M <= 200K`` where the wider sort per chunk beats insert's
+  argmin loop on this regime's autotune data. Otherwise insert wins;
+  sortmerge is kept in :func:`_gen_configs` so the offline tuner can
+  keep verifying that.
+
+Pipelining depth
+----------------
+Both kernels take a ``NUM_STAGES_PIPE`` constexpr for the M-loop's
+``tl.range`` pipelining factor. The dispatcher tunes it per shape from
+a 201-shape ns-sweep (upstream ``bench/sweep_ns_by_k.py``): K<=8 +
+small tile wants ``ns=3`` to hide HBM latency, K>=64 + big tile wants
+``ns=2`` (registers tight), wide-D wants ``ns=2`` uniformly, narrow
+tile + anything wants ``ns=1`` to avoid spills. See the comments
+inside :func:`_heuristic_config` for the empirical breakdown.
+
+Distance recovery
+-----------------
+The kernels emit **signed shifted scores** ``s = c_sq - 2*<x, c>``, not
+true squared L2. ``argmin-k`` over ``s`` matches ``argmin-k`` over true
+squared L2 (since ``-||x||^2`` is a per-row constant), but the returned
+score is not a valid distance. The :func:`flash_knn` public wrapper
+calls :func:`dbscanlib._kernels.kernels.distance.triton_knn_gather_sqdist` on the
+returned indices to write the true ``||x - c[idx]||^2`` per neighbour.
+"""
+from __future__ import annotations
+
+import math
+from typing import Optional
+
+import torch
+
+from dbscanlib._kernels.primitives.knn.triton._common import _next_pow2, _bench_quick
+from dbscanlib._kernels.primitives.knn.triton.sortmerge import _flash_knn_sortmerge_kernel
+from dbscanlib._kernels.primitives.knn.triton.insert import _flash_knn_insert_kernel
+
+
+# ── SRAM estimation ────────────────────────────────────────────────────
+
+
+def _estimate_sram(bn, bm, D, K, d_inner, num_stages, *,
+                   sortmerge=False, dtype_bytes=2):
+    """Per-CTA SRAM estimate (bytes) for the unified kernel — coarse.
+
+    Counts only what Triton **definitely** keeps in SMEM: the x tile
+    + c tile (per iteration of the inner D loop). Everything else
+    (cross fp32 accumulator, score tile, topK heap, sort scratch) is
+    register-resident at the tile sizes used by the dispatcher's
+    heuristic, and the NUM_STAGES_PIPE multi-buffering of the c tile
+    is sometimes elided by Triton's pipeliner. Including any of these
+    in the estimate produces 50-200 KB of over-count that **wrongly
+    shrinks** configs the kernel could actually launch — a bf16
+    regression demonstrated by the ``D=256`` cells in
+    ``benchmarks/results/micro_knn_dtype_smem_fit.md``.
+
+    This estimator is therefore **optimistic on purpose**. Its only
+    job is to bias the autotune candidate filter away from configs
+    that obviously can't fit (e.g. ``BN=128, BM=128, D=256, fp32``
+    needs >232 KB by raw arithmetic). The source of truth for "does
+    this fit?" is ``_run``'s runtime ``OutOfResources`` catch + shrink
+    loop. ``dtype_bytes`` is still threaded through so fp32 inputs
+    correctly double the x/c contribution.
+    """
+    x_sub = bn * d_inner * dtype_bytes
+    c_sub = bm * d_inner * dtype_bytes
+    c_sq = bm * 4
+    base = x_sub + c_sub + c_sq
+    if sortmerge:
+        chunk = bn * bm * 8
+        base += chunk
+    return base
+
+
+def _smem_limit(device) -> int:
+    """Per-block dynamic shared-memory budget for ``device``.
+
+    Mirrors :func:`dbscanlib._kernels.primitives.kmeans.triton.assign._smem_limit`
+    — Triton uses opt-in dynamic SMEM; prefer that attribute when
+    available, fall back to static limit, and finally to a conservative
+    48 KiB for old PyTorch builds.
+    """
+    props = torch.cuda.get_device_properties(device)
+    for attr in (
+        "shared_memory_per_block_optin",
+        "max_shared_memory_per_block_optin",
+        "shared_memory_per_block",
+        "max_shared_memory_per_block",
+    ):
+        v = getattr(props, attr, None)
+        if v:
+            return int(v)
+    return 48 * 1024
+
+
+def _fit_config_to_smem(cfg: dict, D: int, K: int,
+                        dtype_bytes: int, smem_limit: int) -> dict:
+    """Shrink ``(BN, BM, NUM_STAGES_PIPE)`` until the kernel fits SMEM.
+
+    Returns ``cfg`` unchanged when it already fits. Otherwise enumerates
+    power-of-two reductions on the three SMEM-bearing axes and picks the
+    one that maximises ``BN * BM * num_stages`` (work-per-program tile)
+    among those that fit, breaking ties towards the original aspect ratio.
+
+    Why only these three axes?
+      * ``D_INNER``: changing it flips between single-D-tile and D-split
+        paths, which would also change the kernel's perf profile. We
+        keep the heuristic's D-split decision and shrink the cheaper
+        axes first.
+      * ``TOPK_PAD``: determined by ``K``; cannot be reduced without
+        breaking correctness.
+      * ``kernel_mode``, ``num_warps``, ``M_PER_SPLIT``, ``NUM_SPLITS``:
+        these don't affect per-CTA SMEM. Stable.
+
+    Pattern (and rationale) mirrors :func:`dbscanlib._kernels.primitives.kmeans.\
+triton.assign._fit_config_to_smem`.
+    """
+    BN0 = int(cfg["BN"])
+    BM0 = int(cfg["BM"])
+    D_INNER = int(cfg["D_INNER"])
+    NS0 = int(cfg.get("NUM_STAGES_PIPE", 2))
+    sortmerge = cfg.get("kernel_mode") == "sortmerge"
+
+    cur = _estimate_sram(BN0, BM0, D, K, D_INNER, NS0,
+                         sortmerge=sortmerge, dtype_bytes=dtype_bytes)
+    if cur <= smem_limit:
+        return cfg
+
+    def _pow2_down_to(v, lo):
+        out = []
+        x = v
+        while x >= lo:
+            out.append(x)
+            x //= 2
+        return out
+
+    # BN must stay >= 8 (matches _gen_configs lowest BN rung);
+    # BM must stay >= 32 for the inner loop to keep WGMMA shape;
+    # NUM_STAGES_PIPE must stay >= 1.
+    bn_cands = _pow2_down_to(BN0, 8)
+    bm_cands = _pow2_down_to(BM0, 32)
+    ns_cands = list(range(NS0, 0, -1))
+
+    best = None
+    best_key = None
+    for bn in bn_cands:
+        for bm in bm_cands:
+            # sortmerge requires BM == TOPK_PAD; do not shrink BM there.
+            if sortmerge and bm != BM0:
+                continue
+            for ns in ns_cands:
+                if _estimate_sram(bn, bm, D, K, D_INNER, ns,
+                                  sortmerge=sortmerge,
+                                  dtype_bytes=dtype_bytes) > smem_limit:
+                    continue
+                aspect_penalty = abs((bn / max(bm, 1)) - (BN0 / max(BM0, 1)))
+                # Prefer: larger total tile work, closer aspect ratio,
+                # larger BN (more N-parallelism), larger NS (better pipelining).
+                key = (bn * bm * ns, -aspect_penalty, bn, ns)
+                if best_key is None or key > best_key:
+                    best_key = key
+                    best = (bn, bm, ns)
+
+    if best is None:
+        raise RuntimeError(
+            f"flash_knn: cannot fit kernel into shared memory "
+            f"(D={D}, K={K}, D_INNER={D_INNER}, dtype_bytes={dtype_bytes}, "
+            f"smem_limit={smem_limit}). Original config "
+            f"BN={BN0}, BM={BM0}, NS={NS0} needs {cur} bytes."
+        )
+
+    bn, bm, ns = best
+    out = dict(cfg)
+    out["BN"] = bn
+    out["BM"] = bm
+    out["NUM_STAGES_PIPE"] = ns
+    return out
+
+
+def _pipe_stages_for(K, d_inner, D):
+    """``NUM_STAGES_PIPE`` candidates for the autotune M-loop sweep.
+
+    The K-step inner loop adds register pressure that scales with K, so
+    deeper pipelining (which double-/triple-buffers the next C-tile) is
+    only profitable when K is small. Empirically:
+
+      * D-split path (``d_inner < D``) -- kernel uses ns=1 by construction.
+      * K >= 64 -- only try {1, 2}; ns >= 3 spills registers and tanks perf
+        (probe showed ns=3 going from 750 -> 1220 us at K=128).
+      * K <= 32 -- try {1, 2, 3} so big pipelines can hide HBM latency
+        for huge-M shapes.
+    """
+    if d_inner < D:
+        return [1]
+    if K >= 64:
+        return [1, 2]
+    return [1, 2, 3]
+
+
+def _gen_configs(D, K, *, dtype_bytes=2, smem_limit=None):
+    """Generate candidate kernel configs (sortmerge + insert), SRAM-validated.
+
+    Covers ``BN ∈ {8, 16, 32, 64, 128}`` × ``num_warps ∈ {2, 4}`` × the
+    BM / D_INNER / NUM_STAGES_PIPE combinations that fit per-CTA SMEM
+    on H200.
+
+    The ``BN = 8`` rung unlocks the small-N + medium-K Pattern-A regime
+    (``B*N <= 8`` + ``K ∈ {16, 32, 64, 128}``) where a small row tile +
+    sortmerge ``BM = max(32, K)`` wins over insert. ``tl.dot`` silently
+    pads the M dimension to 16 on Triton 3.x sm_90, so BN=8 still
+    compiles.
+
+    ``dtype_bytes`` and ``smem_limit`` parameterise the SMEM check so
+    fp32 inputs don't slip through with a config that OOR's at launch.
+    Defaults preserve the pre-fix behaviour (bf16 / 220 KB) for any
+    caller that didn't yet plumb dtype awareness through.
+    """
+    # 220 KB stays the default — slightly below the 227 KB H200 opt-in
+    # limit to leave room for static SMEM the compiler adds.
+    max_sram = 220_000 if smem_limit is None else smem_limit
+    d_pad = _next_pow2(D)
+    topk_pad_sm = max(32, _next_pow2(K))
+    topk_pad_ins = _next_pow2(K)
+    d_inner_cands = [d_pad] if D <= 256 else [128]
+
+    configs = []
+
+    # sort-merge: BM == TOPK_PAD; BN ∈ {8, 16, 32} (BN >= 64 never wins
+    # for sortmerge in autotune data).
+    bm_sm = topk_pad_sm
+    for bn in [8, 16, 32]:
+        for nw in [2, 4]:
+            for d_inner in d_inner_cands:
+                num_d_iters = math.ceil(D / d_inner)
+                ns = 2 if num_d_iters == 1 else 1
+                sram = _estimate_sram(
+                    bn, bm_sm, D, K, d_inner, ns,
+                    sortmerge=True, dtype_bytes=dtype_bytes,
+                )
+                if sram <= max_sram:
+                    for ns_pipe in _pipe_stages_for(K, d_inner, D):
+                        configs.append({
+                            "BN": bn, "BM": bm_sm, "D_INNER": d_inner,
+                            "num_warps": nw,
+                            "TOPK_PAD": topk_pad_sm,
+                            "kernel_mode": "sortmerge",
+                            "NUM_STAGES_PIPE": ns_pipe,
+                        })
+
+    # iterative insert: BM decoupled from K, BN ∈ {8, 16, 32, 64, 128}.
+    for bn in [8, 16, 32, 64, 128]:
+        for nw in [2, 4]:
+            for bm in [64, 128, 256]:
+                for d_inner in d_inner_cands:
+                    num_d_iters = math.ceil(D / d_inner)
+                    ns = 2 if num_d_iters == 1 else 1
+                    sram = _estimate_sram(
+                        bn, bm, D, K, d_inner, ns,
+                        sortmerge=False, dtype_bytes=dtype_bytes,
+                    )
+                    if sram <= max_sram:
+                        for ns_pipe in _pipe_stages_for(K, d_inner, D):
+                            configs.append({
+                                "BN": bn, "BM": bm, "D_INNER": d_inner,
+                                "num_warps": nw,
+                                "TOPK_PAD": topk_pad_ins,
+                                "kernel_mode": "insert",
+                                "NUM_STAGES_PIPE": ns_pipe,
+                            })
+
+    return configs
+
+
+def _gen_m_splits(M, BM, *, B=1, N=1, BN=16):
+    """M_PER_SPLIT candidates targeting {1, 2, 4, 8, 16} waves on 132 SMs.
+
+    The wave count is critical -- Stage-2 reduce scales linearly with
+    NUM_SPLITS, while too few splits leaves SMs idle. The autotune
+    sweep confirms that wave=2 is the winner on huge-M + medium-N
+    shapes (e.g. ``1×128×10M×64×10``), which the original {1, 4, 16}
+    grid missed.
+
+    Capped at 4096×BM tiles (``MAX_MPS_TILES``) to bound the static
+    M-loop trip count for fast Triton compilation. Also includes the
+    single-pass case when M itself fits the cap (subsumes the large-N
+    kernel as ``num_splits = 1``).
+    """
+    NUM_SMS = 132
+    MAX_MPS_TILES = 4096
+    num_n_tiles = max(1, (N + BN - 1) // BN)
+    max_mps = min(M, MAX_MPS_TILES * BM)
+    candidates = set()
+    for waves in [1, 2, 4, 8, 16]:
+        target_splits = max(2, (NUM_SMS * waves) // (num_n_tiles * B))
+        target_splits = min(target_splits, max(2, M // BM))
+        mps = (M + target_splits - 1) // target_splits
+        mps = ((mps + BM - 1) // BM) * BM
+        mps = max(mps, BM)
+        mps = min(mps, max_mps)
+        candidates.add(mps)
+    if M <= max_mps:
+        single = ((M + BM - 1) // BM) * BM
+        candidates.add(single)
+    return sorted(candidates)
+
+
+# ── shape-only heuristic ───────────────────────────────────────────────
+
+
+def _heuristic_config(B, N, M, D, K, *, force_path=None,
+                      dtype_bytes=2, smem_limit=None):
+    """Pick a kernel config from shape alone -- no autotune.
+
+    When ``dtype_bytes`` and ``smem_limit`` are supplied (the dispatcher
+    plumbs them from ``x.element_size()`` and the device's opt-in SMEM
+    cap), the picked config is post-processed by :func:`_fit_config_to_smem`
+    so that fp32 inputs at large D never produce a config that OOR's at
+    launch.
+
+    Derived from a 92-shape autotune sweep on H200/bf16. The data
+    showed clean decision boundaries on three axes:
+
+      * **BN by NB-bucket + M-bump**:
+        NB <= 8   -> BN=8;
+        NB <= 32  -> BN=16 (32 at M>=5M);
+        NB <= 256 -> BN=64 (128 at M>=5M + narrow-D + small-K);
+        NB <= 2K  -> BN=64 (128 at M>=5M);
+        NB >= 8K  -> BN=128 (64 for D>=256 + K>=32).
+
+        The M-bump rule fixes the ``(1, 128, 10M, 64, 10)`` regression
+        identified by klib -- at M=10M each extra N-tile costs
+        ~300us of c-replication HBM traffic.
+
+      * **BM by K**:
+        K<=4         -> 128/256 (256 only at very-small M + small NB);
+        K=5..16      -> 128 default (256 at huge-M);
+        K=32         -> 128 default (256 at NB<=8 + M<=200K);
+        K>=64        -> 64 default (sortmerge at NB<=8).
+
+      * **kernel_mode**:
+        ``sortmerge`` only when ``NB <= 8`` and ``K ∈ {32, 64, 128}``
+        and ``D <= 256`` and ``M <= 200K`` (Pattern-A1), plus a
+        secondary ``NB <= 16 + K ∈ {32, 64} + M <= 200K`` corner
+        (Pattern-A2). All other shapes use ``insert``.
+
+      * **num_warps**: 4 everywhere except tiny tiles (BN=8 + BM<=64 +
+        K<=4 + huge M, where nw=2 wins by reducing register pressure)
+        and the BN=16 + huge-M + small-K corner where nw=2 frees a
+        warp set for the topK epilogue.
+
+      * **target waves**: 1 for build (NB>=8K) and very-large K;
+        2 for medium/large M; 4 for small M + small K.
+
+      * **NUM_STAGES_PIPE**: 2 by default; 3 for BN>=64 + BM<=128 +
+        K<=8 (small tile + tiny K hides HBM latency) and for
+        BN ∈ [16, 32] + BM=256 + K>=64 (big tile keeps K loop fed);
+        1 for narrow-tile + any K (avoid register spills) and for the
+        D-split path; 2 for D_INNER>=256 and for BN>=128. Without
+        this rule, K>=64 + D=256 shapes hit catastrophic regressions
+        (e.g. ``(1,16,100K,256,128)`` 439us -> 2429us at ns=1, a 5.5x
+        slowdown); conversely K<=32 + huge-M shapes save 20-45% vs
+        ns=2 (e.g. ``(1,1,10M,128,4)`` 1426us -> 785us at ns=1).
+
+    Args:
+        B, N, M, D, K: shape parameters.
+        force_path: ``"large_n"`` for single-pass (one M-split per CTA),
+            ``None`` for the wave-targeted M-split.
+
+    Returns:
+        Config dict: ``BN, BM, D_INNER, TOPK_PAD, kernel_mode,
+        num_warps, M_PER_SPLIT, NUM_SPLITS, NUM_STAGES_PIPE``.
+    """
+    NUM_SMS = 132
+    NB = N * B
+    TOPK_PAD = _next_pow2(K)
+    D_INNER = _next_pow2(D) if D <= 256 else 128
+    is_large_n = (force_path == "large_n")
+    # MAX_MPS_TILES bounds the per-CTA M-loop iteration count. The
+    # original cap of 512 was too tight for huge M with BM=128 (cap
+    # = 65K), which forced fewer splits than autotune wanted. Bump
+    # to 4096 so we never bottleneck on this cap.
+    MAX_MPS_TILES = 4096
+
+    # ───────────────────────────────────────────────────────────────
+    # Step 1: kernel_mode, BN, BM, num_warps
+    # ───────────────────────────────────────────────────────────────
+    if NB <= 8:
+        # Tiny query: BN=8 default.
+        BN = 8
+        # Sortmerge fast-path: K ∈ {32, 64, 128}, M <= 200K. Now
+        # extended to ANY D (autotune ``(1,1,100K,1024,32)`` picks
+        # BN=8 BM=32 sort).
+        if (not is_large_n) and K in (32, 64, 128) and M <= 200_000:
+            BM = max(32, _next_pow2(K))
+            sram = _estimate_sram(8, BM, D, K, D_INNER,
+                                  num_stages=(2 if D_INNER >= D else 1),
+                                  sortmerge=True)
+            if sram <= 220_000:
+                kernel_mode = "sortmerge"
+                num_warps = 2 if BM >= 128 else 4
+            else:
+                kernel_mode = "insert"
+                BM = 256 if M <= 200_000 else 128
+                num_warps = 4
+        else:
+            kernel_mode = "insert"
+            if K <= 4:
+                # K=4 + D>=256 + huge M -> BM=64 nw=2 (autotune
+                # ``(1,1,1M,256,4)`` picks this). HBM-bound; BM=256
+                # adds wasted SMEM.
+                if D >= 256 and M >= 500_000:
+                    BM = 64
+                else:
+                    BM = 256
+            elif K <= 16:
+                # D>=256 + K=16 -> BM=128 (autotune (1,1,100K,256,16))
+                if D >= 256 and K >= 16:
+                    BM = 128
+                else:
+                    BM = 256 if (M >= 500_000 or K >= 16) else 128
+            elif K == 32:
+                # autotune (1,1,1M,256,32) prefers BM=128 at D>=256.
+                # At narrower D, BM=256 wins.
+                if D >= 256 and M >= 500_000:
+                    BM = 128
+                else:
+                    BM = 256
+            elif K <= 64:
+                BM = 64
+            else:  # K=128
+                BM = 256
+            # nw=2 for small/medium-D BM=64 cases
+            if BM == 64 and K <= 4 and D >= 512 and M >= 500_000:
+                # D >= 512 hits the D-split path (D_INNER capped at 128);
+                # nw=2 was calibrated for this regime at (1, 1, 1M, 256, 4)
+                # but later re-benching at D=256 showed nw=4 ns=1 wins by
+                # 1.48-1.81x there (single-D-iter, multibuffered C). Keep
+                # nw=2 only for the D-split path; D=256 falls to the
+                # default nw=4 below and gets the ns=1 override in Step 3.
+                num_warps = 2
+            elif BM >= 128 and K <= 4 and D <= 64 and M >= 5_000_000:
+                # NB<=8 + K<=4 + narrow-D + huge-M: nw=2 wins by 20-30 %.
+                # 4 warps split the SM register file too thinly for the
+                # K=4 argmin-insert epilogue at BM>=128, so the compiler
+                # spills; halving the warp count doubles regs/warp and
+                # lets ILP recover. Verified at D=64 across M ∈ [5M, 60M]
+                # (D=128 strictly prefers nw=4 at this BM; do not bump
+                # the threshold without re-benching).
+                num_warps = 2
+            else:
+                num_warps = 4
+
+    elif NB <= 32:
+        kernel_mode = "insert"
+        # Sortmerge also wins at NB <= 16 + K ∈ {32, 64} (autotune
+        # ``(1,16,100K,256,32)`` picks BN=8 BM=32 sortmerge).
+        if (not is_large_n) and NB <= 16 and K in (32, 64) and M <= 200_000:
+            BN = 8
+            BM = max(32, _next_pow2(K))
+            sram = _estimate_sram(8, BM, D, K, D_INNER,
+                                  num_stages=(2 if D_INNER >= D else 1),
+                                  sortmerge=True)
+            if sram <= 220_000:
+                kernel_mode = "sortmerge"
+                num_warps = 4
+            else:
+                BN, BM = 16, 256
+        if kernel_mode == "insert":
+            if M >= 5_000_000 and N >= 32:
+                BN = 32
+            else:
+                BN = max(8, min(16, _next_pow2(N)))
+            if K <= 4:
+                BM = 128
+            elif K <= 16:
+                BM = 256 if M <= 2_000_000 else 128
+            else:  # K >= 32
+                BM = 256
+            if K <= 4 and M >= 500_000:
+                num_warps = 2
+            elif K <= 16 and M >= 5_000_000 and BN <= 16:
+                num_warps = 2
+            elif K <= 16 and NB >= 17 and M >= 5_000_000:
+                # autotune (1,32,10M,64,8) BN=32 nw=2 wins
+                num_warps = 2
+            else:
+                num_warps = 4
+
+    elif NB <= 256:
+        # Medium queries: BN must scale with M to control
+        # c-replication, but ONLY when K is very small (K<=4) or
+        # very large (K>=32). At K ∈ {5..16} the autotune consistently
+        # picks BN=64 even at M=10M, because K=8..16 hits a sweet
+        # spot where the topk-insert + Stage-2 cost of 1-wave
+        # BN=128 outweighs the c-HBM savings.
+        kernel_mode = "insert"
+        big_NB = (NB >= 192)
+
+        # K=128 special-case: medium NB + K=128, autotune picks
+        # BN=8 BM=256 insert (NOT sortmerge!). The K-step argmin loop
+        # at BN=64 K=128 is too expensive -- better to chop N into
+        # many small tiles.
+        if K >= 128 and M <= 200_000:
+            BN = 8
+            BM = 256
+        elif M >= 5_000_000 and D <= 64 and (K <= 4 or K >= 32):
+            BN = 128
+        elif big_NB and M >= 5_000_000:
+            BN = 128
+        elif big_NB and D >= 256 and M <= 200_000 and K <= 4:
+            BN = 128
+        elif D >= 256 and K <= 16 and (NB >= 64 and M >= 500_000):
+            # D=256 + K<=16 + (medium NB + medium M): BN=128 wins.
+            # autotune (1,128,1M,256,8) picks BN=128 BM=64.
+            BN = 128
+        else:
+            BN = 64
+        BN = min(BN, max(8, _next_pow2(N)))
+        # BM
+        if K >= 128:
+            BM = 256  # already covered above, keep consistent here
+        elif K <= 4 and D <= 128 and M <= 200_000:
+            BM = 256
+        elif D >= 256 and K <= 16:
+            BM = 64
+        elif K <= 32:
+            BM = 64 if (BN == 64 and K == 32 and D >= 256) else 128
+        elif K <= 64:
+            BM = 128
+        else:
+            BM = 128
+        num_warps = 4
+
+    elif NB <= 2048:
+        kernel_mode = "insert"
+        if M >= 5_000_000:
+            BN = 128
+        elif M >= 500_000 and D >= 256 and K <= 16:
+            BN = 128
+        else:
+            BN = 64
+        BN = min(BN, max(8, _next_pow2(N)))
+        if K <= 4 and M >= 500_000:
+            BM = 64
+        elif K <= 16 and M >= 500_000 and D >= 128:
+            BM = 64
+        elif K >= 64:
+            BM = 64
+        else:
+            BM = 128
+        num_warps = 4
+
+    else:  # NB >= 8K (build / large-eval regime)
+        kernel_mode = "insert"
+        # autotune patterns:
+        #   NB=50K-100K D<=128 K<=8  -> BN=128 (halves HBM)
+        #   NB=10K     K>=8           -> BN=64  (more topk work per row)
+        #   NB>=10K    D>=256 K<=16   -> BN=128
+        #   NB>=10K    D>=256 K>=32   -> BN=64
+        if D >= 256 and K <= 16:
+            BN = 128
+        elif D >= 256 and K >= 32:
+            BN = 64
+        elif K <= 4:
+            BN = 128
+        elif K <= 8 and NB >= 30_000:
+            # Larger NB -> BN=128 saves enough HBM to overcome topk cost
+            BN = 128
+        else:
+            BN = 64
+        if K <= 4 and D >= 256:
+            BM = 64  # SMEM pressure at D=256 + BM=128
+        elif K <= 4:
+            BM = 128 if (D <= 64 or BN == 128) else 64
+        elif K <= 8 and BN == 128:
+            BM = 128
+        elif K >= 64 and BN == 64 and D <= 128 and NB >= 30_000:
+            # klib carve-out: NB in [30K, ~200K] + BN=64 + K>=64 +
+            # D<=128 wins big at BM=128 splits=1 vs BM=64 splits=2.
+            # Empirically at (1, 48K, 48K, 64, 64) the upstream rule
+            # (BM=64, target_splits=2) runs 17.95 ms, while BM=128
+            # splits=1 ns=2 runs 11.71 ms (-35%); at (1, 64K, 64K, 64,
+            # 64) 29.20 ms -> 18.91 ms (-35%). Bounded on K>=64 (K=32
+            # autotune at this NB shows BM=64 splits=1 wins, ~1 ms over
+            # BM=128 splits=1) and on NB>=30K so the upstream rule
+            # still applies to smaller-NB shapes where its M-split
+            # target was directly autotune-derived.
+            BM = 128
+        else:
+            BM = 64
+        num_warps = 4
+
+    # ───────────────────────────────────────────────────────────────
+    # Step 2: M_PER_SPLIT (target waves)
+    # ───────────────────────────────────────────────────────────────
+    num_n_tiles = max(1, math.ceil(N / BN))
+    ctas_no_split = num_n_tiles * B
+
+    # Workaround for an insert-kernel correctness issue with the
+    # D-split path: when ``M_PER_SPLIT == BM`` (single M-loop
+    # iteration) and the d-loop has >=4 chunks, the result is wrong.
+    # Bumping the minimum mps to 2*BM keeps the M-loop at >=2 iters.
+    needs_min_2_iters = (D_INNER < D and (D + D_INNER - 1) // D_INNER >= 4)
+    min_mps = (2 * BM) if needs_min_2_iters else BM
+
+    if is_large_n:
+        M_PER_SPLIT = ((M + BM - 1) // BM) * BM
+    elif ctas_no_split >= NUM_SMS * 8:
+        # Massively oversaturated (e.g. N=100K, BN=64, ctas=1563): one
+        # split fully utilises SMs.
+        M_PER_SPLIT = ((M + BM - 1) // BM) * BM
+    elif K >= 32 and BN == 64 and D <= 128 and NB >= 30_000 and not is_large_n:
+        # klib carve-out (pairs with the BM=128 K>=64 rule above
+        # plus K=32 BM=64 standalone): the K>=32 build regime at
+        # NB>=30K wants splits=1 -- 2 splits doubles the K-step
+        # argmin work per row and the Stage-2 reduce, neither of
+        # which the autotune at NB=10K K=8 accounted for. Empirically
+        # saves ~6 ms on (1, 48K, 48K, 64, 64) and ~10 ms on
+        # (1, 64K, 64K, 64, 64) vs target_splits=2, and ~1 ms on
+        # (1, 56K, 56K, 64, 32) vs target_splits=2.
+        M_PER_SPLIT = ((M + BM - 1) // BM) * BM
+    else:
+        if ctas_no_split >= NUM_SMS * 4:
+            # ctas >= 528: 2 splits for tail-fill at very large builds
+            # (autotune (1,100K,100K,64,4) BN=128 ctas=782 -> 2 splits)
+            target_splits = 2
+        elif ctas_no_split >= NUM_SMS * 2:
+            # ctas in [264, 528): single split already 2-4 waves; adding
+            # splits hurts more than it helps. autotune
+            # (1,50K,50K,64,8) BN=128 ctas=391 -> 1 split.
+            target_splits = 1
+        elif ctas_no_split >= NUM_SMS:
+            # ctas in [132, 264): 3 splits for better tail. autotune
+            # (1,10K,10K,64,16) BN=64 ctas=157 -> 3 splits.
+            target_splits = 3
+        else:
+            # Under-saturated: target_waves by K (autotune-derived).
+            # Use FLOOR div: snaps to integer wave count, which matches
+            # autotune choices on shapes like NB=1024 K=32 ctas=16
+            # (16 splits = 2 waves, 17 splits = 2.06 waves but bad
+            # tail).
+            if kernel_mode == "sortmerge":
+                target_waves = 2
+            elif K >= 128 and NB >= 64:
+                # K=128 + medium NB: 2w wins (autotune
+                # (1,128,100K,128,128) picks 16 splits at ctas=16).
+                target_waves = 2
+            elif K >= 128:
+                target_waves = 1
+            elif K >= 64 and ctas_no_split <= 2:
+                # K=64 + tiny ctas: 1 wave wins (autotune
+                # (1,128,100K,128,64) picks 66 splits = 1w).
+                target_waves = 1
+            elif K >= 64:
+                target_waves = 2
+            elif K <= 4 and ctas_no_split == 1 and (NB <= 32 or M <= 1_000_000):
+                # K=4 single-CTA-stack: 4 waves at small NB or small M.
+                # autotune NB=1 K=4 -> 521 splits; (1,128,10M,64,4)
+                # however picks 264 splits (2w) -- too much M per CTA
+                # for more splits to help.
+                target_waves = 4
+            elif K <= 8 and ctas_no_split >= 8 and ctas_no_split <= 32:
+                target_waves = 4 if M >= 500_000 else 2
+            elif K <= 8 and ctas_no_split == 1 and M >= 5_000_000 and BN <= 32:
+                target_waves = 4
+            elif B >= 2 and N <= 8 and K <= 8:
+                # Batched B>=2 + single-query + K<=8: 4w wins
+                target_waves = 4
+            else:
+                target_waves = 2
+            target_splits = max(1, NUM_SMS * target_waves // ctas_no_split)
+            target_splits = min(target_splits, max(1, math.ceil(M / BM)))
+        mps_raw = math.ceil(M / target_splits)
+        mps = math.ceil(mps_raw / BM) * BM
+        mps = max(min_mps, min(mps, ((M + BM - 1) // BM) * BM))
+        mps = min(mps, MAX_MPS_TILES * BM)
+        M_PER_SPLIT = mps
+
+    NUM_SPLITS = math.ceil(M / M_PER_SPLIT)
+
+    # ───────────────────────────────────────────────────────────────
+    # Step 3: NUM_STAGES_PIPE (M-loop pipeline depth)
+    # ───────────────────────────────────────────────────────────────
+    # Derived from a 201-shape × 4-ns sweep on H200 (upstream
+    # ``bench/sweep_ns_by_k.py``), cross-checked against direct probes
+    # on N>=64 build shapes. Three competing effects:
+    #
+    #   * Each extra pipeline stage adds another buffered C-tile,
+    #     costing ``BM * D_INNER * 2`` bytes of SMEM/registers. Wide
+    #     D-tiles amortise this well; narrow ones don't.
+    #   * The K-step argmin inner loop holds live state proportional
+    #     to ``BN * TOPK_PAD * 8`` bytes. For narrow rows (BN<=16) +
+    #     large K, registers get tight and deeper pipelines spill.
+    #   * Medium-N tiles (BN>=64) have abundant compute per tile to
+    #     amortise prefetch, but get little benefit at very large K
+    #     because the K loop dominates anyway.
+    #
+    # Empirical winners (mode_ns from the 201-shape sweep):
+    #
+    #   D_INNER >= 256 (wide D)               -> ns=2  (uniform across K)
+    #   BN >= 128 (huge build N)              -> ns=2  (registers tight)
+    #   BN >= 64 + BM <= 128 + K <= 8         -> ns=3  (small tile + tiny K)
+    #   BN >= 64                              -> ns=2  (BM=256 SMEM-bound)
+    #   BN ∈ [16,32] + BM=256 + K >= 64       -> ns=3  (big-tile large-K)
+    #   otherwise (narrow tile + any K)       -> ns=1  (avoid spills)
+    #
+    # Without this rule, K>=64 + D=256 shapes hit catastrophic
+    # regressions (e.g. (1,16,100K,256,128) 439us -> 2429us at ns=1,
+    # a 5.5x slowdown). Conversely K<=32 + huge-M shapes save 20-45%
+    # vs ns=2 (e.g. (1,1,10M,128,4) 1426us -> 785us at ns=1).
+    if D_INNER < D:
+        NUM_STAGES_PIPE = 1  # D-split path is hard-coded to ns=1
+    elif (BN == 8 and BM == 64 and D_INNER == 256
+          and K <= 4 and M >= 500_000):
+        # NB<=8 + K<=4 + D=256 + BM=64 + huge-M: ns=1 wins by 1.48-1.81x
+        # over the default ns=2. Direct verification at K ∈ {1, 2, 4} ×
+        # M ∈ {1M, 5M, 10M, 30M}: (nw=4, ns=1) lifts %peak HBM from
+        # 39-48 % (ns=2) to 57-85 % (ns=1). The ns=2 default below was
+        # calibrated for wider-N tiles; at BN=8 the deeper pipeline
+        # spends too many registers on prefetch and the K=4 epilogue
+        # spills. Pairs with the nw=4 default the rule above falls into
+        # at D=256.
+        NUM_STAGES_PIPE = 1
+    elif D_INNER >= 256:
+        NUM_STAGES_PIPE = 2
+    elif BN >= 128:
+        NUM_STAGES_PIPE = 2
+    elif BN >= 64 and BM <= 128 and K <= 8:
+        NUM_STAGES_PIPE = 3
+    elif BN >= 64:
+        NUM_STAGES_PIPE = 2
+    elif BN >= 16 and BM >= 256 and K >= 64:
+        NUM_STAGES_PIPE = 3
+    else:
+        NUM_STAGES_PIPE = 1
+
+    cfg = {
+        "BN": BN, "BM": BM, "D_INNER": D_INNER,
+        "TOPK_PAD": (
+            max(32, _next_pow2(K)) if kernel_mode == "sortmerge"
+            else _next_pow2(K)
+        ),
+        "kernel_mode": kernel_mode, "num_warps": num_warps,
+        "M_PER_SPLIT": M_PER_SPLIT,
+        "NUM_SPLITS": NUM_SPLITS,
+        "NUM_STAGES_PIPE": NUM_STAGES_PIPE,
+    }
+
+    # Dtype-aware SMEM shrink. The shape-only heuristic above was tuned
+    # on bf16 and may pick (BN=128, BM=128, D_INNER=128) for shapes that
+    # cleanly fit ~210 KB at 2 bytes/elt but blow past the 227 KB H200
+    # opt-in cap at 4 bytes/elt. The fitter shrinks BN/BM/NUM_STAGES_PIPE
+    # by powers of 2 until the kernel fits — leaving D_INNER and
+    # kernel_mode (which encode different perf regimes) alone.
+    if smem_limit is not None:
+        cfg = _fit_config_to_smem(cfg, D, K, dtype_bytes, smem_limit)
+
+    return cfg
+
+
+# ── autotune (opt-in) ──────────────────────────────────────────────────
+
+
+_autotune_cache: dict = {}
+
+
+def _autotune(x, c, k, *, force_path=None):
+    """Brute-force autotune over the candidate config grid.
+
+    Cached per ``(B, N, M, D, k, dtype, force_path)`` shape; first call
+    costs ~30-60 s of compile time, subsequent calls hit the cache in
+    sub-ms.
+
+    The candidate grid is dtype-aware: ``_gen_configs`` filters out
+    configs whose SMEM exceeds the device opt-in limit at this dtype,
+    so fp32 inputs at large D never have an unrunnable config in the
+    search space.
+    """
+    B, N, D = x.shape
+    M = c.shape[1]
+    dtype_bytes = x.element_size()
+    key = (B, N, M, D, k, x.dtype, force_path)
+
+    if key in _autotune_cache:
+        return _autotune_cache[key]
+
+    smem_limit = _smem_limit(x.device)
+    configs = _gen_configs(D, k,
+                           dtype_bytes=dtype_bytes,
+                           smem_limit=min(smem_limit, 220_000))
+    if not configs:
+        raise RuntimeError(
+            f"No valid flash_knn config for D={D}, K={k}, "
+            f"dtype={x.dtype} (element_size={dtype_bytes})"
+        )
+
+    # Drop configs with BN > next_pow2(N) -- they would just pad N to BN
+    # and run the same kernel slower.
+    max_bn = max(16, _next_pow2(N))
+    configs = [cfg for cfg in configs if cfg["BN"] <= max_bn]
+
+    best_time = float('inf')
+    best_cfg = None
+    max_partial_bytes = 4 * 1024**3
+
+    for cfg in configs:
+        bn = cfg["BN"]
+        bm = cfg["BM"]
+        d_inner = cfg["D_INNER"]
+        topk_pad = cfg["TOPK_PAD"]
+        kernel_mode = cfg["kernel_mode"]
+        nw = cfg["num_warps"]
+        ns_pipe = cfg.get("NUM_STAGES_PIPE", 2)
+
+        if force_path == "large_n":
+            mps_list = [((M + bm - 1) // bm) * bm]
+        else:
+            mps_list = _gen_m_splits(M, bm, B=B, N=N, BN=bn)
+
+        for mps in mps_list:
+            num_splits = math.ceil(M / mps)
+            partial_bytes = B * num_splits * N * k * 8
+            if partial_bytes > max_partial_bytes:
+                continue
+
+            try:
+                pv = torch.empty((B, N, num_splits, k),
+                                 device=x.device, dtype=torch.float32)
+                pi = torch.empty((B, N, num_splits, k),
+                                 device=x.device, dtype=torch.int32)
+                grid = (num_splits, math.ceil(N / bn), B)
+                pv_s0, pv_s1, pv_s2, pv_s3 = pv.stride()
+                pi_s0, pi_s1, pi_s2, pi_s3 = pi.stride()
+
+                if kernel_mode == "sortmerge":
+                    def run(grid=grid, bn=bn, bm=bm, d_inner=d_inner,
+                            topk_pad=topk_pad, mps=mps, nw=nw,
+                            num_splits=num_splits, pv=pv, pi=pi,
+                            ns_pipe=ns_pipe,
+                            pv_s0=pv_s0, pv_s1=pv_s1, pv_s2=pv_s2, pv_s3=pv_s3,
+                            pi_s0=pi_s0, pi_s1=pi_s1, pi_s2=pi_s2, pi_s3=pi_s3):
+                        _flash_knn_sortmerge_kernel[grid](
+                            x, c, pv, pi,
+                            x.stride(0), x.stride(1), x.stride(2),
+                            c.stride(0), c.stride(1), c.stride(2),
+                            pv_s0, pv_s2, pv_s1, pv_s3,
+                            pi_s0, pi_s2, pi_s1, pi_s3,
+                            N=N, M=M, D=D, K=k, M_PER_SPLIT=mps,
+                            BN=bn, BM=bm, D_INNER=d_inner,
+                            TOPK_PAD=topk_pad,
+                            NUM_STAGES_PIPE=ns_pipe,
+                            num_warps=nw,
+                        )
+                        if num_splits > 1:
+                            pv_flat = pv.view(B, N, -1)
+                            pv_flat.topk(k, dim=-1, largest=False)
+                else:
+                    max_steps = min(k, bm)
+                    def run(grid=grid, bn=bn, bm=bm, d_inner=d_inner,
+                            topk_pad=topk_pad, mps=mps, nw=nw,
+                            num_splits=num_splits, pv=pv, pi=pi,
+                            max_steps=max_steps, ns_pipe=ns_pipe,
+                            pv_s0=pv_s0, pv_s1=pv_s1, pv_s2=pv_s2, pv_s3=pv_s3,
+                            pi_s0=pi_s0, pi_s1=pi_s1, pi_s2=pi_s2, pi_s3=pi_s3):
+                        _flash_knn_insert_kernel[grid](
+                            x, c, pv, pi,
+                            x.stride(0), x.stride(1), x.stride(2),
+                            c.stride(0), c.stride(1), c.stride(2),
+                            pv_s0, pv_s2, pv_s1, pv_s3,
+                            pi_s0, pi_s2, pi_s1, pi_s3,
+                            N=N, M=M, D=D, K=k, M_PER_SPLIT=mps,
+                            BN=bn, BM=bm, D_INNER=d_inner,
+                            TOPK_PAD=topk_pad, MAX_STEPS=max_steps,
+                            NUM_STAGES_PIPE=ns_pipe,
+                            num_warps=nw,
+                        )
+                        if num_splits > 1:
+                            pv_flat = pv.view(B, N, -1)
+                            pv_flat.topk(k, dim=-1, largest=False)
+
+                t = _bench_quick(run)
+                if t < best_time:
+                    best_time = t
+                    best_cfg = {**cfg, "M_PER_SPLIT": mps,
+                                "NUM_SPLITS": num_splits,
+                                "NUM_STAGES_PIPE": ns_pipe}
+
+                del pv, pi
+            except Exception:
+                continue
+
+    if best_cfg is None:
+        raise RuntimeError(
+            f"All flash_knn configs failed for B={B}, N={N}, M={M}, D={D}, K={k}")
+
+    _autotune_cache[key] = best_cfg
+    return best_cfg
+
+
+# ── runner ─────────────────────────────────────────────────────────────
+
+
+_OOR_FALLBACK_CACHE: dict = {}
+
+
+def _try_launch(x, c, cfg, B, N, M, D, k):
+    """Single launch attempt; returns idxs or raises ``triton.compiler.errors.OutOfResources``.
+
+    Factored out of ``_run`` so the OOR-shrink retry loop can call it
+    repeatedly with a shrunk cfg.
+    """
+    bn = cfg["BN"]
+    bm = cfg["BM"]
+    d_inner = cfg["D_INNER"]
+    topk_pad = cfg["TOPK_PAD"]
+    mps = cfg["M_PER_SPLIT"]
+    num_splits = cfg["NUM_SPLITS"]
+    kernel_mode = cfg["kernel_mode"]
+    nw = cfg["num_warps"]
+    num_stages_pipe = cfg.get("NUM_STAGES_PIPE", 2)
+
+    partial_vals = torch.empty((B, N, num_splits, k),
+                               device=x.device, dtype=torch.float32)
+    partial_idxs = torch.empty((B, N, num_splits, k),
+                               device=x.device, dtype=torch.int32)
+    grid = (num_splits, math.ceil(N / bn), B)
+
+    pv_s0, pv_s1, pv_s2, pv_s3 = partial_vals.stride()
+    pi_s0, pi_s1, pi_s2, pi_s3 = partial_idxs.stride()
+
+    if kernel_mode == "sortmerge":
+        _flash_knn_sortmerge_kernel[grid](
+            x, c, partial_vals, partial_idxs,
+            x.stride(0), x.stride(1), x.stride(2),
+            c.stride(0), c.stride(1), c.stride(2),
+            pv_s0, pv_s2, pv_s1, pv_s3,
+            pi_s0, pi_s2, pi_s1, pi_s3,
+            N=N, M=M, D=D, K=k, M_PER_SPLIT=mps,
+            BN=bn, BM=bm, D_INNER=d_inner,
+            TOPK_PAD=topk_pad,
+            NUM_STAGES_PIPE=num_stages_pipe,
+            num_warps=nw,
+        )
+    else:
+        max_steps = min(k, bm)
+        _flash_knn_insert_kernel[grid](
+            x, c, partial_vals, partial_idxs,
+            x.stride(0), x.stride(1), x.stride(2),
+            c.stride(0), c.stride(1), c.stride(2),
+            pv_s0, pv_s2, pv_s1, pv_s3,
+            pi_s0, pi_s2, pi_s1, pi_s3,
+            N=N, M=M, D=D, K=k, M_PER_SPLIT=mps,
+            BN=bn, BM=bm, D_INNER=d_inner,
+            TOPK_PAD=topk_pad, MAX_STEPS=max_steps,
+            NUM_STAGES_PIPE=num_stages_pipe,
+            num_warps=nw,
+        )
+
+    if num_splits == 1:
+        return partial_idxs[:, :, 0, :].contiguous()
+
+    pv = partial_vals.view(B, N, -1)
+    pi = partial_idxs.view(B, N, -1)
+    _, sel = pv.topk(k, dim=-1, largest=False, sorted=True)
+    out_idxs = pi.gather(-1, sel.to(torch.int64)).to(torch.int32)
+    return out_idxs
+
+
+def _shrink_cfg_on_oor(cfg: dict) -> Optional[dict]:
+    """Halve the cheapest SMEM-bearing axis. ``None`` when nothing left.
+
+    Order:
+      1. NUM_STAGES_PIPE 3 → 2 → 1 (drops the c-tile pipeline buffer
+         — usually free or nearly-free at this depth).
+      2. Tend toward a near-square tile by **halving the larger of
+         (BN, BM)** first. WGMMA shape efficiency drops rapidly once
+         BM<32 on Hopper (MMA atom is 64x16x16), so we prefer to
+         shrink BN when BN > BM. For BN=128 BM=64 (typical build
+         regime) this gives BN=64 BM=64 in one step, which empirically
+         matches the kmeans-style ``_fit_config_to_smem`` choice and
+         beats BN=128 BM=32 by ~20 % at D=1024 fp32.
+      3. Floors: BN ≥ 8, BM ≥ 32 (both required by the WGMMA atom).
+
+    sortmerge requires ``BM == TOPK_PAD``; skip BM shrinks and only
+    halve BN.
+
+    Also recomputes ``M_PER_SPLIT`` so it remains a multiple of the
+    new BM (the kernel's inner ``tl.range`` walk requires this).
+    """
+    ns = int(cfg.get("NUM_STAGES_PIPE", 2))
+    bm = int(cfg["BM"])
+    bn = int(cfg["BN"])
+    sortmerge = cfg.get("kernel_mode") == "sortmerge"
+
+    out = dict(cfg)
+    if ns > 1:
+        out["NUM_STAGES_PIPE"] = ns - 1
+        return out
+    if sortmerge:
+        if bn > 8:
+            out["BN"] = bn // 2
+            return out
+        return None
+    # Non-sortmerge: prefer shrinking the larger axis.
+    if bn > bm and bn > 8:
+        out["BN"] = bn // 2
+        return out
+    if bm > 32:
+        out["BM"] = bm // 2
+        new_bm = out["BM"]
+        out["M_PER_SPLIT"] = ((int(out["M_PER_SPLIT"]) + new_bm - 1)
+                              // new_bm) * new_bm
+        return out
+    if bn > 8:
+        out["BN"] = bn // 2
+        return out
+    return None
+
+
+def _run(x: torch.Tensor, c: torch.Tensor, k: int, *,
+         force_path: Optional[str] = None,
+         autotune: bool = False) -> torch.Tensor:
+    """Shared dispatcher -- launch stage 1, optional stage 2 reduce, return idxs.
+
+    Returns indices only ``(B, N, k) int32``. The wrapper at
+    :func:`dbscanlib._kernels.primitives.knn.flash_knn` calls the gather kernel to
+    produce true squared distances per neighbour.
+
+    On ``OutOfResources`` at launch, we shrink one SMEM-bearing axis
+    (NS → BM → BN) and retry, caching the surviving config per
+    ``(D, K, dtype, force_path)`` so subsequent calls with the same
+    shape skip the failed compiles. This is the source of truth for
+    "does this fit?"; ``_estimate_sram`` only provides a coarse hint
+    for the autotune candidate filter.
+    """
+    assert x.is_cuda and c.is_cuda
+    assert x.dtype == c.dtype
+    B, N, D = x.shape
+    M = c.shape[1]
+    assert c.shape == (B, M, D)
+    assert 1 <= k <= M
+
+    if autotune:
+        cfg = _autotune(x, c, k, force_path=force_path)
+    else:
+        cfg = _heuristic_config(
+            B, N, M, D, k,
+            force_path=force_path,
+            dtype_bytes=x.element_size(),
+            smem_limit=min(_smem_limit(x.device), 220_000),
+        )
+
+    # Surviving-config cache keyed on the parts the dispatcher can't
+    # vary (D, K, dtype, force_path). If a previous call already
+    # shrunk past an OOR for this shape, start with that smaller cfg.
+    fb_key = (D, k, x.dtype, force_path,
+              cfg["kernel_mode"], cfg.get("D_INNER"))
+    if fb_key in _OOR_FALLBACK_CACHE:
+        cached = _OOR_FALLBACK_CACHE[fb_key]
+        # Use the cached cfg only if it's smaller than the current one
+        # on at least one SMEM-bearing axis (otherwise the heuristic
+        # already picked something safe for this shape).
+        sm_cur = (cfg["BN"], cfg["BM"], cfg.get("NUM_STAGES_PIPE", 2))
+        sm_cached = (cached["BN"], cached["BM"],
+                     cached.get("NUM_STAGES_PIPE", 2))
+        if sm_cached < sm_cur:
+            cfg = {**cfg, **{k_: cached[k_] for k_ in
+                              ("BN", "BM", "NUM_STAGES_PIPE")}}
+            # Re-align mps to new BM
+            new_bm = cfg["BM"]
+            cfg["M_PER_SPLIT"] = ((cfg["M_PER_SPLIT"] + new_bm - 1)
+                                  // new_bm) * new_bm
+            cfg["NUM_SPLITS"] = math.ceil(M / cfg["M_PER_SPLIT"])
+
+    # Import lazily so we don't pay it on every dispatch.
+    try:
+        from triton.runtime.errors import OutOfResources
+    except ImportError:
+        try:
+            from triton.compiler.errors import OutOfResources
+        except ImportError:
+            OutOfResources = RuntimeError  # safest fallback
+
+    last_err = None
+    initial_cfg = cfg
+    shrunk = False
+    for _attempt in range(8):
+        try:
+            result = _try_launch(x, c, cfg, B, N, M, D, k)
+            # Cache the surviving cfg so subsequent calls with the
+            # same (D, K, dtype, ...) shape skip the failed compile.
+            if shrunk:
+                _OOR_FALLBACK_CACHE[fb_key] = {
+                    "BN": cfg["BN"], "BM": cfg["BM"],
+                    "NUM_STAGES_PIPE": cfg.get("NUM_STAGES_PIPE", 2),
+                }
+            return result
+        except OutOfResources as e:
+            last_err = e
+            new_cfg = _shrink_cfg_on_oor(cfg)
+            if new_cfg is None:
+                break
+            cfg = new_cfg
+            shrunk = True
+            continue
+
+    raise last_err if last_err is not None else RuntimeError(
+        "flash_knn: exhausted OOR shrink attempts")
+
+
+# ── public API ─────────────────────────────────────────────────────────
+
+
+def flash_knn_triton_small_n(x: torch.Tensor, c: torch.Tensor, k: int,
+                              *, autotune: bool = False) -> torch.Tensor:
+    """Force the M-split (search) path -- ``(B, N, k) int32`` indices."""
+    return _run(x, c, k, force_path=None, autotune=autotune)
+
+
+def flash_knn_triton_large_n(x: torch.Tensor, c: torch.Tensor, k: int,
+                              *, autotune: bool = False) -> torch.Tensor:
+    """Force the single-pass (build) path -- ``(B, N, k) int32`` indices."""
+    return _run(x, c, k, force_path="large_n", autotune=autotune)
+
+
+def flash_knn_triton(x: torch.Tensor, c: torch.Tensor, k: int,
+                     *, autotune: bool = False) -> torch.Tensor:
+    """Universal Triton dispatch -- the heuristic picks BN/BM/mode/
+    mps/ns_pipe based on shape, including whether to single-pass or
+    M-split. The per-CTA-count check inside :func:`_heuristic_config`
+    handles both build and eval shapes without a host-side forced path.
+
+    Neither path materialises an N×M cross or distance matrix to HBM,
+    and neither computes or loads ``x_sq`` -- the two hard contracts
+    klib's KNN imposes on every Triton entry point here.
+
+    Args:
+        x: ``(B, N, D)`` bf16 / fp16 / fp32 query tensor.
+        c: ``(B, M, D)`` corpus, same dtype.
+        k: number of neighbours.
+        autotune: ``False`` (default) uses the shape-only heuristic --
+            first call pays a single Triton compile (~0.5 s). ``True``
+            runs the full brute-force sweep + caches per shape (~30-90 s
+            first call).
+
+    Returns:
+        idxs: ``(B, N, k)`` int32 -- nearest-neighbour indices, sorted
+        by ascending true squared L2 distance (ties broken by index).
+    """
+    return _run(x, c, k, force_path=None, autotune=autotune)
diff --git a/dbscanlib/_kernels/primitives/knn/triton/insert.py b/dbscanlib/_kernels/primitives/knn/triton/insert.py
new file mode 100644
index 0000000..7e72d76
--- /dev/null
+++ b/dbscanlib/_kernels/primitives/knn/triton/insert.py
@@ -0,0 +1,204 @@
+"""flash_knn — Triton iterative-insert top-K kernel (x²-free, signed score).
+
+Same x²-free signed score as ``sortmerge.py`` — see that module's docstring
+for the rationale. The insert kernel differs in two ways:
+
+  * ``BM`` is decoupled from ``TOPK_PAD`` (sortmerge requires ``BM == TOPK_PAD``).
+    The autotune sweeps ``BM ∈ {64, 128, 256}`` independently.
+  * Top-K is maintained as an unsorted ``(BN, TOPK_PAD)`` register array;
+    each chunk does at most ``MAX_STEPS = min(K, BM)`` argmin->insert
+    passes. fp32 ``<`` works natively on signed scores, so only the
+    trailing output sort uses the sortable-u32 transform.
+
+Empirically the insert kernel wins on virtually every shape past the
+trivial ``K == 1`` case; sortmerge is kept for completeness (autotune
+still considers it, and it's the natural fit when ``K`` matches a
+hardware-sortable tile size).
+
+Public name: ``_flash_knn_insert_kernel`` — called from
+``klib/primitives/knn/triton/dispatch.py``.
+"""
+import triton
+import triton.language as tl
+
+from dbscanlib._kernels.primitives.knn.triton._common import (
+    _fp32_to_sortable_u32,
+    _sortable_u32_to_fp32,
+)
+
+
+@triton.jit
+def _flash_knn_insert_kernel(
+    x_ptr, c_ptr,
+    partial_val_ptr, partial_idx_ptr,
+    stride_x_b, stride_x_n, stride_x_d,
+    stride_c_b, stride_c_m, stride_c_d,
+    stride_pv_b, stride_pv_s, stride_pv_n, stride_pv_k,
+    stride_pi_b, stride_pi_s, stride_pi_n, stride_pi_k,
+    N: tl.constexpr, M: tl.constexpr,
+    D: tl.constexpr, K: tl.constexpr,
+    M_PER_SPLIT: tl.constexpr,
+    BN: tl.constexpr, BM: tl.constexpr,
+    D_INNER: tl.constexpr,
+    TOPK_PAD: tl.constexpr,
+    MAX_STEPS: tl.constexpr,
+    NUM_STAGES_PIPE: tl.constexpr = 2,
+):
+    """Iterative-insert top-K on signed shifted distance.
+
+    Grid: ``(num_m_splits, ceil(N/BN), B)``.
+    Two compile-time paths gated on ``D_INNER >= D`` (persistent x +
+    pipelined M-loop vs D-split with sequential M-loop). The M-loop
+    pipelining depth is exposed as ``NUM_STAGES_PIPE`` so the
+    dispatcher can tune it per shape (see :mod:`dispatch` for the
+    heuristic rules; defaults to 2 if not provided).
+    """
+    pid_s = tl.program_id(0)
+    pid_n = tl.program_id(1)
+    pid_b = tl.program_id(2)
+    pid_b = pid_b.to(tl.int64)
+
+    n_start = pid_n * BN
+    n_offs = (n_start + tl.arange(0, BN)).to(tl.int64)
+    n_mask = n_offs < N
+
+    topk_vals = tl.full([BN, TOPK_PAD], float('inf'), dtype=tl.float32)
+    topk_idxs = tl.full([BN, TOPK_PAD], -1, dtype=tl.int32)
+    topk_max = tl.full([BN], float('inf'), dtype=tl.float32)
+    k_range = tl.arange(0, TOPK_PAD)
+    bm_range = tl.arange(0, BM)
+    m_base = pid_s.to(tl.int64) * M_PER_SPLIT
+
+    if D_INNER >= D:
+        d_offs = tl.arange(0, D_INNER).to(tl.int64)
+        d_mask = d_offs < D
+        x_tile = tl.load(
+            x_ptr + pid_b * stride_x_b
+            + n_offs[:, None] * stride_x_n
+            + d_offs[None, :] * stride_x_d,
+            mask=n_mask[:, None] & d_mask[None, :], other=0.0)
+
+        for m_local in tl.range(0, M_PER_SPLIT, BM, num_stages=NUM_STAGES_PIPE):
+            m_start = m_base + m_local
+            m_offs = m_start + bm_range.to(tl.int64)
+            m_mask = m_offs < M
+
+            c_tile = tl.load(
+                c_ptr + pid_b * stride_c_b
+                + m_offs[:, None] * stride_c_m
+                + d_offs[None, :] * stride_c_d,
+                mask=m_mask[:, None] & d_mask[None, :], other=0.0)
+
+            c_f = c_tile.to(tl.float32)
+            c_sq_tile = tl.sum(c_f * c_f, axis=1)
+            cross = tl.dot(x_tile, tl.trans(c_tile)).to(tl.float32)
+            score = c_sq_tile[None, :] - 2.0 * cross
+            score = tl.where(m_mask[None, :], score, float('inf'))
+
+            chunk_best = tl.min(score)
+            threshold_worst = tl.max(topk_max)
+            if chunk_best < threshold_worst:
+                _active = tl.full([1], 1, dtype=tl.int32)
+                for _step in range(MAX_STEPS):
+                    if tl.max(_active) > 0:
+                        row_min = tl.min(score, axis=1)
+                        row_argmin = tl.argmin(score, axis=1)
+                        do_insert = row_min < topk_max
+                        n_inserts = tl.sum(do_insert.to(tl.int32))
+                        if n_inserts > 0:
+                            topk_argmax = tl.argmax(topk_vals, axis=1)
+                            replace_mask = (k_range[None, :] == topk_argmax[:, None])
+                            insert_mask = do_insert[:, None] & replace_mask
+                            topk_vals = tl.where(insert_mask, row_min[:, None], topk_vals)
+                            topk_idxs = tl.where(
+                                insert_mask,
+                                (m_start + row_argmin.to(tl.int64))[:, None].to(tl.int32),
+                                topk_idxs,
+                            )
+                            topk_max = tl.max(topk_vals, axis=1)
+                            used_mask = (bm_range[None, :] == row_argmin[:, None])
+                            score = tl.where(used_mask & do_insert[:, None], float('inf'), score)
+                        _active = tl.where(
+                            n_inserts > 0,
+                            tl.full([1], 1, dtype=tl.int32),
+                            tl.full([1], 0, dtype=tl.int32),
+                        )
+
+    else:
+        for m_local in tl.range(0, M_PER_SPLIT, BM, num_stages=1):
+            m_start = m_base + m_local
+            m_offs = m_start + bm_range.to(tl.int64)
+            m_mask = m_offs < M
+
+            cross = tl.zeros([BN, BM], dtype=tl.float32)
+            c_sq_tile = tl.zeros([BM], dtype=tl.float32)
+            for d_start in range(0, D, D_INNER):
+                d_offs = (d_start + tl.arange(0, D_INNER)).to(tl.int64)
+                d_mask = d_offs < D
+
+                x_sub = tl.load(
+                    x_ptr + pid_b * stride_x_b
+                    + n_offs[:, None] * stride_x_n
+                    + d_offs[None, :] * stride_x_d,
+                    mask=n_mask[:, None] & d_mask[None, :], other=0.0)
+
+                c_sub = tl.load(
+                    c_ptr + pid_b * stride_c_b
+                    + m_offs[:, None] * stride_c_m
+                    + d_offs[None, :] * stride_c_d,
+                    mask=m_mask[:, None] & d_mask[None, :], other=0.0)
+
+                cross += tl.dot(x_sub, tl.trans(c_sub)).to(tl.float32)
+                c_f = c_sub.to(tl.float32)
+                c_sq_tile += tl.sum(c_f * c_f, axis=1)
+
+            score = c_sq_tile[None, :] - 2.0 * cross
+            score = tl.where(m_mask[None, :], score, float('inf'))
+
+            chunk_best = tl.min(score)
+            threshold_worst = tl.max(topk_max)
+            if chunk_best < threshold_worst:
+                _active = tl.full([1], 1, dtype=tl.int32)
+                for _step in range(MAX_STEPS):
+                    if tl.max(_active) > 0:
+                        row_min = tl.min(score, axis=1)
+                        row_argmin = tl.argmin(score, axis=1)
+                        do_insert = row_min < topk_max
+                        n_inserts = tl.sum(do_insert.to(tl.int32))
+                        if n_inserts > 0:
+                            topk_argmax = tl.argmax(topk_vals, axis=1)
+                            replace_mask = (k_range[None, :] == topk_argmax[:, None])
+                            insert_mask = do_insert[:, None] & replace_mask
+                            topk_vals = tl.where(insert_mask, row_min[:, None], topk_vals)
+                            topk_idxs = tl.where(
+                                insert_mask,
+                                (m_start + row_argmin.to(tl.int64))[:, None].to(tl.int32),
+                                topk_idxs,
+                            )
+                            topk_max = tl.max(topk_vals, axis=1)
+                            used_mask = (bm_range[None, :] == row_argmin[:, None])
+                            score = tl.where(used_mask & do_insert[:, None], float('inf'), score)
+                        _active = tl.where(
+                            n_inserts > 0,
+                            tl.full([1], 1, dtype=tl.int32),
+                            tl.full([1], 0, dtype=tl.int32),
+                        )
+
+    val_sortable = _fp32_to_sortable_u32(topk_vals)
+    val_bits = val_sortable.to(tl.uint64)
+    idx_bits = topk_idxs.to(tl.uint32).to(tl.uint64)
+    packed = (val_bits << 32) | idx_bits
+    packed = tl.sort(packed, dim=1)
+    topk_score = _sortable_u32_to_fp32((packed >> 32).to(tl.uint32))
+    topk_idx = (packed & 0xFFFFFFFF).to(tl.int32)
+
+    k_offs = tl.arange(0, TOPK_PAD).to(tl.int64)
+    k_mask = k_offs < K
+    write_mask = n_mask[:, None] & k_mask[None, :]
+    pid_s_i64 = pid_s.to(tl.int64)
+    tl.store(partial_val_ptr + pid_b * stride_pv_b + pid_s_i64 * stride_pv_s
+             + n_offs[:, None] * stride_pv_n + k_offs[None, :] * stride_pv_k,
+             topk_score, mask=write_mask)
+    tl.store(partial_idx_ptr + pid_b * stride_pi_b + pid_s_i64 * stride_pi_s
+             + n_offs[:, None] * stride_pi_n + k_offs[None, :] * stride_pi_k,
+             topk_idx, mask=write_mask)
diff --git a/dbscanlib/_kernels/primitives/knn/triton/sortmerge.py b/dbscanlib/_kernels/primitives/knn/triton/sortmerge.py
new file mode 100644
index 0000000..0d4a0e4
--- /dev/null
+++ b/dbscanlib/_kernels/primitives/knn/triton/sortmerge.py
@@ -0,0 +1,181 @@
+"""flash_knn — Triton sort-merge top-K kernel (x²-free, signed score).
+
+The kernel computes ``argmin-K`` over the *signed* shifted distance
+
+    s(n, m) = ||c[m]||² − 2·⟨x[n], c[m]⟩      (== ||x[n] − c[m]||² − ||x[n]||²)
+
+The ``||x||²`` term is constant per row and so does not affect the top-K
+selection over ``m``. Dropping it eliminates the ``x_sq`` HBM tensor, its
+precompute pass, its per-tile load, one fp32 ADD per accumulator element,
+and the ``dist >= 0`` underflow clamp. The trade-off is that ``s`` is no
+longer non-negative; the packed-uint64 sort-merge top-K therefore uses
+an IEEE-sortable u32 transform (positives flip just the sign bit, negatives
+flip every bit) so ascending uint32 order matches ascending fp32 order on
+signed inputs.
+
+Public name: ``_flash_knn_sortmerge_kernel`` — called from
+``klib/primitives/knn/triton/dispatch.py``. Stage 1 writes signed
+fp32 scores + int32 idxs to a partial buffer; Stage 2 either returns
+``partial[:, 0]`` directly (single split) or runs ``torch.topk(...,
+largest=False)`` to reduce across splits.
+
+Same kernel covers both ``small-N M-split flash-decode`` (multiple
+splits per query block) and ``large-N single-pass`` (M_PER_SPLIT == M)
+— the host-side dispatcher just chooses the grid + M_PER_SPLIT.
+"""
+import triton
+import triton.language as tl
+
+from dbscanlib._kernels.primitives.knn.triton._common import (
+    _INF_PACKED,
+    _fp32_to_sortable_u32,
+    _sortable_u32_to_fp32,
+)
+
+
+@triton.jit
+def _flash_knn_sortmerge_kernel(
+    x_ptr, c_ptr,
+    partial_val_ptr, partial_idx_ptr,
+    stride_x_b, stride_x_n, stride_x_d,
+    stride_c_b, stride_c_m, stride_c_d,
+    stride_pv_b, stride_pv_s, stride_pv_n, stride_pv_k,
+    stride_pi_b, stride_pi_s, stride_pi_n, stride_pi_k,
+    N: tl.constexpr, M: tl.constexpr,
+    D: tl.constexpr, K: tl.constexpr,
+    M_PER_SPLIT: tl.constexpr,
+    BN: tl.constexpr, BM: tl.constexpr,
+    D_INNER: tl.constexpr,
+    TOPK_PAD: tl.constexpr,
+    NUM_STAGES_PIPE: tl.constexpr = 2,
+):
+    """Packed-uint64 sort-merge top-K with IEEE-sortable u32 score bits.
+
+    Grid: ``(num_m_splits, ceil(N/BN), B)``. ``BM == TOPK_PAD``.
+
+    Two compile-time paths gated on ``D_INNER >= D``:
+      * persistent x tile + pipelined M-loop (single tile covers all of D)
+      * D-split with sequential M-loop (D-chunked GEMM for wide D)
+
+    ``NUM_STAGES_PIPE`` controls the M-loop ``tl.range`` pipelining
+    depth (host-side dispatch tunes it per shape; defaults to 2 if not
+    provided -- matches the pre-port hard-coded value).
+    """
+    pid_s = tl.program_id(0)
+    pid_n = tl.program_id(1)
+    pid_b = tl.program_id(2)
+    pid_b = pid_b.to(tl.int64)
+
+    n_start = pid_n * BN
+    n_offs = (n_start + tl.arange(0, BN)).to(tl.int64)
+    n_mask = n_offs < N
+
+    topk_packed = tl.full([BN, TOPK_PAD], _INF_PACKED, dtype=tl.uint64)
+    bm_range = tl.arange(0, BM)
+    m_base = pid_s.to(tl.int64) * M_PER_SPLIT
+
+    if D_INNER >= D:
+        d_offs = tl.arange(0, D_INNER).to(tl.int64)
+        d_mask = d_offs < D
+        x_tile = tl.load(
+            x_ptr + pid_b * stride_x_b
+            + n_offs[:, None] * stride_x_n
+            + d_offs[None, :] * stride_x_d,
+            mask=n_mask[:, None] & d_mask[None, :], other=0.0)
+
+        for m_local in tl.range(0, M_PER_SPLIT, BM, num_stages=NUM_STAGES_PIPE):
+            m_start = m_base + m_local
+            m_offs = m_start + bm_range.to(tl.int64)
+            m_mask = m_offs < M
+
+            c_tile = tl.load(
+                c_ptr + pid_b * stride_c_b
+                + m_offs[:, None] * stride_c_m
+                + d_offs[None, :] * stride_c_d,
+                mask=m_mask[:, None] & d_mask[None, :], other=0.0)
+
+            c_f = c_tile.to(tl.float32)
+            c_sq_tile = tl.sum(c_f * c_f, axis=1)
+            cross = tl.dot(x_tile, tl.trans(c_tile)).to(tl.float32)
+            score = c_sq_tile[None, :] - 2.0 * cross
+            score = tl.where(m_mask[None, :], score, float('inf'))
+
+            chunk_best = tl.min(score)
+            topk_worst_packed = tl.max(topk_packed)
+            topk_worst_val = _sortable_u32_to_fp32(
+                (topk_worst_packed >> 32).to(tl.uint32))
+            if chunk_best < topk_worst_val:
+                score_sortable = _fp32_to_sortable_u32(score)
+                idx_vals = (m_start + bm_range.to(tl.int64)).to(tl.int32)
+                idx_bits = idx_vals.to(tl.uint32).to(tl.uint64)
+                chunk_packed = (score_sortable.to(tl.uint64) << 32) | idx_bits[None, :]
+
+                chunk_packed = chunk_packed ^ tl.full([1], 0xFFFFFFFFFFFFFFFF, dtype=tl.uint64)
+                chunk_packed = tl.sort(chunk_packed, dim=1)
+                chunk_packed = chunk_packed ^ tl.full([1], 0xFFFFFFFFFFFFFFFF, dtype=tl.uint64)
+
+                merged = tl.minimum(topk_packed, chunk_packed)
+                topk_packed = tl.sort(merged, dim=1)
+
+    else:
+        for m_local in tl.range(0, M_PER_SPLIT, BM, num_stages=1):
+            m_start = m_base + m_local
+            m_offs = m_start + bm_range.to(tl.int64)
+            m_mask = m_offs < M
+
+            cross = tl.zeros([BN, BM], dtype=tl.float32)
+            c_sq_tile = tl.zeros([BM], dtype=tl.float32)
+            for d_start in range(0, D, D_INNER):
+                d_offs = (d_start + tl.arange(0, D_INNER)).to(tl.int64)
+                d_mask = d_offs < D
+
+                x_sub = tl.load(
+                    x_ptr + pid_b * stride_x_b
+                    + n_offs[:, None] * stride_x_n
+                    + d_offs[None, :] * stride_x_d,
+                    mask=n_mask[:, None] & d_mask[None, :], other=0.0)
+
+                c_sub = tl.load(
+                    c_ptr + pid_b * stride_c_b
+                    + m_offs[:, None] * stride_c_m
+                    + d_offs[None, :] * stride_c_d,
+                    mask=m_mask[:, None] & d_mask[None, :], other=0.0)
+
+                cross += tl.dot(x_sub, tl.trans(c_sub)).to(tl.float32)
+                c_f = c_sub.to(tl.float32)
+                c_sq_tile += tl.sum(c_f * c_f, axis=1)
+
+            score = c_sq_tile[None, :] - 2.0 * cross
+            score = tl.where(m_mask[None, :], score, float('inf'))
+
+            chunk_best = tl.min(score)
+            topk_worst_packed = tl.max(topk_packed)
+            topk_worst_val = _sortable_u32_to_fp32(
+                (topk_worst_packed >> 32).to(tl.uint32))
+            if chunk_best < topk_worst_val:
+                score_sortable = _fp32_to_sortable_u32(score)
+                idx_vals = (m_start + bm_range.to(tl.int64)).to(tl.int32)
+                idx_bits = idx_vals.to(tl.uint32).to(tl.uint64)
+                chunk_packed = (score_sortable.to(tl.uint64) << 32) | idx_bits[None, :]
+
+                chunk_packed = chunk_packed ^ tl.full([1], 0xFFFFFFFFFFFFFFFF, dtype=tl.uint64)
+                chunk_packed = tl.sort(chunk_packed, dim=1)
+                chunk_packed = chunk_packed ^ tl.full([1], 0xFFFFFFFFFFFFFFFF, dtype=tl.uint64)
+
+                merged = tl.minimum(topk_packed, chunk_packed)
+                topk_packed = tl.sort(merged, dim=1)
+
+    topk_score_sortable = (topk_packed >> 32).to(tl.uint32)
+    topk_score = _sortable_u32_to_fp32(topk_score_sortable)
+    topk_idx = (topk_packed & 0xFFFFFFFF).to(tl.int32)
+
+    k_offs = tl.arange(0, TOPK_PAD).to(tl.int64)
+    k_mask = k_offs < K
+    write_mask = n_mask[:, None] & k_mask[None, :]
+    pid_s_i64 = pid_s.to(tl.int64)
+    tl.store(partial_val_ptr + pid_b * stride_pv_b + pid_s_i64 * stride_pv_s
+             + n_offs[:, None] * stride_pv_n + k_offs[None, :] * stride_pv_k,
+             topk_score, mask=write_mask)
+    tl.store(partial_idx_ptr + pid_b * stride_pi_b + pid_s_i64 * stride_pi_s
+             + n_offs[:, None] * stride_pi_n + k_offs[None, :] * stride_pi_k,
+             topk_idx, mask=write_mask)
diff --git a/dbscanlib/dbscan.py b/dbscanlib/dbscan.py
index ab7fa8b..53710d0 100644
--- a/dbscanlib/dbscan.py
+++ b/dbscanlib/dbscan.py
@@ -1,70 +1,17 @@
-"""Naive DBSCAN baseline (correct, deterministic, GPU/CPU torch).
+"""Euclidean DBSCAN -- vendored best-in-repo Triton grid path (D=2).
 
-Euclidean DBSCAN. Chunked O(N^2) neighbour scan + GPU-friendly label
-propagation for the connected components of the core graph. Border rule matches
-flashlib: a non-core point joins the SMALLEST cluster label among its in-eps
-core neighbours; otherwise it is noise (-1).
+Delegates to the fused Triton grid radius-search + connected-components kernels
+vendored under ``dbscanlib._kernels``. Public contract is unchanged:
+
+    dbscan(x, eps, min_samples) -> labels  ((N,) int64; cluster id >=0, -1 noise)
 """
 from __future__ import annotations
 
 import torch
 
+from dbscanlib._kernels.primitives.dbscan.triton.dbscan import flash_dbscan
 
-def dbscan(x, eps, min_samples, max_neighbors=32):
-    del max_neighbors  # naive path scans all neighbours; knob only for the fast kernel
-    N = x.shape[0]
-    dev = x.device
-    eps2 = float(eps) * float(eps)
-    xn = (x * x).sum(1)                          # (N,)
-    CH = 4096
-    BIG = N  # sentinel label (> any real label)
-
-    def d2_chunk(lo, hi):
-        xb = x[lo:hi]
-        return (xb * xb).sum(1, keepdim=True) - 2.0 * (xb @ x.t()) + xn[None, :]
-
-    # 1) degree (in-eps count, includes self) -> core mask
-    deg = torch.zeros(N, device=dev, dtype=torch.int64)
-    for lo in range(0, N, CH):
-        hi = min(lo + CH, N)
-        deg[lo:hi] = (d2_chunk(lo, hi) <= eps2).sum(1)
-    core = deg >= int(min_samples)              # (N,) bool
 
-    # 2) connected components of the core-core eps graph via label propagation
-    labels = torch.arange(N, device=dev, dtype=torch.int64)
-    labels[~core] = BIG
-    for _ in range(1000):
-        new = labels.clone()
-        for lo in range(0, N, CH):
-            hi = min(lo + CH, N)
-            if not bool(core[lo:hi].any()):
-                continue
-            adj = (d2_chunk(lo, hi) <= eps2) & core[None, :]     # (chunk, N)
-            lab = torch.where(adj, labels[None, :], torch.full((1, N), BIG, device=dev, dtype=torch.int64))
-            mn = lab.min(1).values                                # (chunk,)
-            rows = slice(lo, hi)
-            upd = core[rows] & (mn < new[rows])
-            new[rows] = torch.where(upd, mn, new[rows])
-        if torch.equal(new, labels):
-            break
-        labels = new
-
-    # 3) border assignment: non-core within eps of a core -> smallest core label
-    for lo in range(0, N, CH):
-        hi = min(lo + CH, N)
-        nb = ~core[lo:hi]
-        if not bool(nb.any()):
-            continue
-        adj = (d2_chunk(lo, hi) <= eps2) & core[None, :]
-        lab = torch.where(adj, labels[None, :], torch.full((1, N), BIG, device=dev, dtype=torch.int64))
-        mn = lab.min(1).values
-        rows = slice(lo, hi)
-        take = nb & (mn < BIG)
-        labels[rows] = torch.where(take, mn, labels[rows])
-
-    # 4) compact to dense [0, n_clusters); noise (BIG) -> -1
-    out = torch.full((N,), -1, device=dev, dtype=torch.int64)
-    uniq = torch.unique(labels[labels < BIG])
-    for new_id, old in enumerate(uniq.tolist()):
-        out[labels == old] = new_id
-    return out
+def dbscan(x, eps, min_samples, max_neighbors=32):
+    labels = flash_dbscan(x, float(eps), int(min_samples), int(max_neighbors))
+    return labels.to(torch.int64)
