diff --git a/kmeanslib/_kernels/__init__.py b/kmeanslib/_kernels/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/kmeanslib/_kernels/primitives/__init__.py b/kmeanslib/_kernels/primitives/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/kmeanslib/_kernels/primitives/kmeans/__init__.py b/kmeanslib/_kernels/primitives/kmeans/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/kmeanslib/_kernels/primitives/kmeans/triton/__init__.py b/kmeanslib/_kernels/primitives/kmeans/triton/__init__.py
new file mode 100644
index 0000000..043b4b7
--- /dev/null
+++ b/kmeanslib/_kernels/primitives/kmeans/triton/__init__.py
@@ -0,0 +1,43 @@
+"""kmeans triton backend.
+
+Re-exports the public Python wrappers from each component file.
+``@triton.jit`` kernels stay private to their file.
+"""
+from kmeanslib._kernels.primitives.kmeans.triton.assign import (
+    _ceil_div,
+    euclid_assign_triton,
+    cosine_assign_triton,
+)
+from kmeanslib._kernels.primitives.kmeans.triton.kmeans import (
+    COMPILE_FLAG,
+    batch_kmeans_Euclid,
+    batch_kmeans_Cosine,
+    batch_kmeans_Dot,
+)
+from kmeanslib._kernels.primitives.kmeans.triton.update import (
+    triton_centroid_update_cosine,
+    torch_loop_centroid_update_cosine,
+    triton_centroid_update_euclid,
+    triton_centroid_update_sorted_cosine,
+    triton_centroid_update_sorted_euclid,
+    triton_centroid_finalize,
+    triton_lloyd_centroid_step_euclid,
+    main,
+)
+
+__all__ = [
+    "euclid_assign_triton",
+    "cosine_assign_triton",
+    "COMPILE_FLAG",
+    "batch_kmeans_Euclid",
+    "batch_kmeans_Cosine",
+    "batch_kmeans_Dot",
+    "triton_centroid_update_cosine",
+    "torch_loop_centroid_update_cosine",
+    "triton_centroid_update_euclid",
+    "triton_centroid_update_sorted_cosine",
+    "triton_centroid_update_sorted_euclid",
+    "triton_centroid_finalize",
+    "triton_lloyd_centroid_step_euclid",
+    "main",
+]
diff --git a/kmeanslib/_kernels/primitives/kmeans/triton/assign.py b/kmeanslib/_kernels/primitives/kmeans/triton/assign.py
new file mode 100644
index 0000000..5014a67
--- /dev/null
+++ b/kmeanslib/_kernels/primitives/kmeans/triton/assign.py
@@ -0,0 +1,1476 @@
+from typing import Optional
+import torch
+import triton
+import triton.language as tl
+
+# ===============================================================
+# Triton kernel: compute nearest-centroid IDs (Euclidean distance).
+#
+# x²-free score: the kernel ranks ``s(n, k) = c_sq[k] − 2·<x[n], c[k]>``
+# which is ``||x − c||² − ||x||²`` and so produces the same argmin
+# without loading any ``x_sq`` tensor. See assign_euclid in
+# klib/primitives/kmeans/torch_fallback.py for the matching CPU
+# reference and the gather kernel in klib/kernels/distance for
+# recovering the true distances post-hoc.
+#
+# Inputs:
+#   x           : (B, N, D)  float16 / float32
+#   centroids   : (B, K, D)  same dtype as x
+# Output:
+#   cluster_ids : (B, N)     int32   -- nearest centroid index per point
+# ===============================================================
+
+
+def _ceil_div(a: int, b: int) -> int:
+    return (a + b - 1) // b
+
+
+def _next_power_of_2(n: int) -> int:
+    p = 1
+    while p < n:
+        p <<= 1
+    return p
+
+
+# -----------------------------------------------------------------------------
+# Auto-tuning setup – explore various tile sizes / warp counts
+# -----------------------------------------------------------------------------
+
+_TUNE_CONFIGS = [
+    triton.Config({"BLOCK_N": BN, "BLOCK_K": BK}, num_stages=num_stages, num_warps=wp)
+    for BN in [32, 64, 128]
+    for BK in [32, 64, 128]
+    for wp in [4, 8]
+    for num_stages in [1, 2, 4]
+]
+
+
+def _cfg_keep(conf):
+    """Basic heuristic to prune unbalanced configs."""
+    BN = conf.kwargs["BLOCK_N"]
+    BK = conf.kwargs["BLOCK_K"]
+    # Avoid tiny tiles on many warps
+    if BN * BK < 32 * 32 and conf.num_warps > 4:
+        return False
+    return True
+
+_TUNE_CONFIGS = list(filter(_cfg_keep, _TUNE_CONFIGS))
+
+
+# Tuning grid for the split-D kernel. Adds a third tile dim ``BLOCK_D``
+# that controls how much of the feature dimension is materialised inside
+# each program at a time. Pruned to keep peak SMEM and register pressure
+# under control on conservative GPUs (GB10).
+_TUNE_CONFIGS_SPLIT_D = [
+    triton.Config({"BLOCK_N": BN, "BLOCK_K": BK, "BLOCK_D": BD},
+                  num_stages=num_stages, num_warps=wp)
+    for BN in [32, 64, 128]
+    for BK in [32, 64, 128]
+    for BD in [32, 64, 128]
+    for wp in [4, 8]
+    for num_stages in [1, 2, 4]
+]
+
+
+def _cfg_keep_split_d(conf):
+    BN = conf.kwargs["BLOCK_N"]
+    BK = conf.kwargs["BLOCK_K"]
+    BD = conf.kwargs["BLOCK_D"]
+    # Tiny tiles do not need many warps.
+    if BN * BK < 32 * 32 and conf.num_warps > 4:
+        return False
+    # Cap (BN, BK) tile size for register/SMEM safety. The (BN, BK) fp32
+    # cross accumulator is the same size as the small-D kernel, so keeping
+    # BN*BK <= 128*128 prevents new spill regressions.
+    if BN * BK > 128 * 128:
+        return False
+    # Prune the largest combined work tiles to keep tuning wall-clock down
+    # without losing useful configs.
+    if BN * BK * BD > 128 * 128 * 128:
+        return False
+    return True
+
+
+_TUNE_CONFIGS_SPLIT_D = list(filter(_cfg_keep_split_d, _TUNE_CONFIGS_SPLIT_D))
+
+_HALF_DTYPES = (torch.float16, torch.bfloat16)
+
+
+def _dtype_bytes(dtype) -> int:
+    """Element size in bytes for a torch / numpy-ish dtype.
+
+    Falls back to 2 (fp16) when the dtype is unknown to keep prior behaviour
+    (the heuristic was originally tuned with fp16 in mind).
+    """
+    if dtype is None:
+        return 2
+    if isinstance(dtype, torch.dtype):
+        return torch.tensor([], dtype=dtype).element_size()
+    # Allow callers to pass a raw byte size.
+    if isinstance(dtype, int):
+        return dtype
+    return 2
+
+
+def _is_half_dtype(dtype) -> bool:
+    """True for fp16/bf16 (the original tuning regime).
+
+    For these dtypes we skip the SMEM-fitting fallback entirely so heuristic
+    selection on already-validated GPUs (H100/H200/A100) is byte-for-byte
+    identical to the previous behaviour.
+    """
+    if dtype is None:
+        return True
+    if isinstance(dtype, torch.dtype):
+        return dtype in _HALF_DTYPES
+    return False
+
+
+def _smem_bytes(D: int, BN: int, BK: int, num_stages: int, dtype_bytes: int) -> int:
+    """Approximate dynamic shared-memory usage of `_euclid_assign_kernel`.
+
+    The kernel materialises:
+    - one ``x_tile`` of shape (BN, D) outside the K loop, and
+    - ``num_stages`` copies of ``c_tile`` of shape (D, BK) for the software
+      pipelined K loop.
+
+    Other buffers (c_sq, masks, accumulators) are negligible compared
+    to these and are ignored. The x²-free kernel does not load ``x_sq``.
+    """
+    return D * dtype_bytes * (BN + num_stages * BK)
+
+
+def _smem_limit(device) -> int:
+    """Per-block dynamic shared-memory budget for ``device``.
+
+    Triton uses opt-in dynamic shared memory; prefer that attribute when
+    available, fall back to the static limit, and finally to a conservative
+    48 KiB for very old PyTorch builds.
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
+def _fit_config_to_smem(
+    cfg: dict,
+    D: int,
+    dtype_bytes: int,
+    smem_limit: int,
+) -> dict:
+    """Return a config that fits ``smem_limit`` and is closest to ``cfg``.
+
+    The original config is returned unchanged whenever it already fits. If
+    not, we enumerate all power-of-two ``(BLOCK_N, BLOCK_K, num_stages)``
+    that are no larger than the original and pick the one that maximises
+    work-per-program tile (``BLOCK_N * BLOCK_K * num_stages``), breaking
+    ties towards the original aspect ratio. This avoids the pitfall of a
+    pure greedy halving (e.g. shrinking BLOCK_K all the way to 16 when only
+    a single halving was needed).
+
+    Raises ``RuntimeError`` if even ``(BN=16, BK=16, S=1)`` does not fit –
+    this only happens for absurdly large D combined with fp32 on tiny-SMEM
+    GPUs.
+    """
+    BN0 = int(cfg["BLOCK_N"])
+    BK0 = int(cfg["BLOCK_K"])
+    W0 = int(cfg["num_warps"])
+    S0 = int(cfg["num_stages"])
+
+    if _smem_bytes(D, BN0, BK0, S0, dtype_bytes) <= smem_limit:
+        return {"BLOCK_N": BN0, "BLOCK_K": BK0, "num_warps": W0, "num_stages": S0}
+
+    def _pow2_down_to_16(v):
+        out = []
+        x = v
+        while x >= 16:
+            out.append(x)
+            x //= 2
+        return out
+
+    best = None
+    best_key = None
+    for BN in _pow2_down_to_16(BN0):
+        for BK in _pow2_down_to_16(BK0):
+            for S in range(S0, 0, -1):
+                if _smem_bytes(D, BN, BK, S, dtype_bytes) > smem_limit:
+                    continue
+                # Prefer larger total tile work, then closer aspect ratio
+                # to the original, then larger BLOCK_N (more parallelism
+                # along N), then larger num_stages (better pipelining).
+                aspect_penalty = abs(
+                    (BN / max(BK, 1)) - (BN0 / max(BK0, 1))
+                )
+                key = (BN * BK * S, -aspect_penalty, BN, S)
+                if best_key is None or key > best_key:
+                    best_key = key
+                    best = (BN, BK, S)
+
+    if best is None:
+        raise RuntimeError(
+            f"euclid_assign_triton: cannot fit kernel into shared memory "
+            f"(D={D}, dtype_bytes={dtype_bytes}, smem_limit={smem_limit}). "
+            f"Even BLOCK_N=16, BLOCK_K=16, num_stages=1 needs "
+            f"{_smem_bytes(D, 16, 16, 1, dtype_bytes)} bytes."
+        )
+
+    BN, BK, S = best
+    W = W0
+    # Tiny tiles do not benefit from many warps and may even fail to compile
+    # for some Triton versions; cap to 4.
+    if BN * BK <= 32 * 32 and W > 4:
+        W = 4
+
+    return {"BLOCK_N": BN, "BLOCK_K": BK, "num_warps": W, "num_stages": S}
+
+
+def _smem_bytes_split_d(BD: int, BN: int, BK: int, num_stages: int, dtype_bytes: int) -> int:
+    """SMEM estimate for ``_euclid_assign_kernel_split_d`` per program.
+
+    The split-D kernel materialises:
+    - one ``x_chunk`` of shape (BN, BD) per D-tile, and
+    - ``num_stages`` copies of ``c_chunk`` of shape (BD, BK) for the software
+      pipelined inner D loop.
+    """
+    return BD * dtype_bytes * (BN + num_stages * BK)
+
+
+def _smallD_kernel_fits_smem(D: int, dtype_bytes: int, smem_limit: int) -> bool:
+    """Return True if even the tiniest small-D kernel config fits SMEM.
+
+    Used by the wrapper to fall back to split-D when the small-D kernel can
+    not run at all (e.g., GB10 + fp32 + D=448).
+    """
+    return _smem_bytes(D, 16, 16, 1, dtype_bytes) <= smem_limit
+
+
+def _fit_config_to_smem_split_d(
+    cfg: dict,
+    D: int,
+    dtype_bytes: int,
+    smem_limit: int,
+) -> dict:
+    """Shrink a split-D config until it fits ``smem_limit``.
+
+    Mirrors ``_fit_config_to_smem`` with the additional ``BLOCK_D`` axis.
+    Returns the largest-work config that fits, breaking ties towards the
+    original aspect ratio. ``BLOCK_D`` is also clamped to D when D < BD0
+    (no point materialising more dims than exist).
+    """
+    BN0 = int(cfg["BLOCK_N"])
+    BK0 = int(cfg["BLOCK_K"])
+    W0 = int(cfg["num_warps"])
+    S0 = int(cfg["num_stages"])
+    BD0 = int(cfg["BLOCK_D"])
+    # No point letting BD exceed next_pow2(D) (the loop would still run
+    # once). We round D up to the next power-of-2 because BLOCK_D feeds
+    # into ``tl.arange(0, BLOCK_D)`` which Triton requires to be pow2.
+    D_ceil = _next_power_of_2(max(D, 16))
+    BD0 = min(BD0, D_ceil)
+
+    if _smem_bytes_split_d(BD0, BN0, BK0, S0, dtype_bytes) <= smem_limit:
+        return {
+            "BLOCK_N": BN0, "BLOCK_K": BK0, "BLOCK_D": BD0,
+            "num_warps": W0, "num_stages": S0,
+        }
+
+    def _pow2_down_to_16(v):
+        out = []
+        x = v
+        while x >= 16:
+            out.append(x)
+            x //= 2
+        return out
+
+    best = None
+    best_key = None
+    for BN in _pow2_down_to_16(BN0):
+        for BK in _pow2_down_to_16(BK0):
+            for BD in _pow2_down_to_16(BD0):
+                for S in range(S0, 0, -1):
+                    if _smem_bytes_split_d(BD, BN, BK, S, dtype_bytes) > smem_limit:
+                        continue
+                    aspect_penalty = abs(
+                        (BN / max(BK, 1)) - (BN0 / max(BK0, 1))
+                    )
+                    key = (BN * BK * BD * S, -aspect_penalty, BN, BD, S)
+                    if best_key is None or key > best_key:
+                        best_key = key
+                        best = (BN, BK, BD, S)
+
+    if best is None:
+        raise RuntimeError(
+            f"euclid_assign_triton (split-D): cannot fit kernel into shared "
+            f"memory (D={D}, dtype_bytes={dtype_bytes}, smem_limit={smem_limit})."
+        )
+
+    BN, BK, BD, S = best
+    W = W0
+    if BN * BK <= 32 * 32 and W > 4:
+        W = 4
+    return {
+        "BLOCK_N": BN, "BLOCK_K": BK, "BLOCK_D": BD,
+        "num_warps": W, "num_stages": S,
+    }
+
+
+# -----------------------------------------------------------------------------
+# Per-arch small-D heuristic functions. These bodies are the original
+# hand-tuned tables, moved verbatim here so the top-level dispatcher stays
+# small and so the split-D path can live alongside without touching them.
+# Any change here must be guarded by examples/regression_fp16_smalld.py.
+# -----------------------------------------------------------------------------
+
+
+def _heuristic_euclid_config_h200_smallD(N: int, K: int, D: int, dtype) -> dict:
+    """H200 small-D heuristic for the x²-free Euclidean kernel.
+
+    Derived from a fresh grid sweep (``scripts/tune_euclid_h200.py`` in
+    flash-kmeans) after dropping the ``||x||²`` load and the underflow
+    clamp from ``_euclid_assign_kernel``: N ∈ {65536, 1048576},
+    K ∈ {256, 4096, 65536, 200000}, D ∈ {64, 128, 256, 512}, B=1,
+    fp16 + fp32.
+
+    fp32:
+        - D=64 K<=256                  : BN=128 BK=64  W=4 S=1
+        - D=64 256<K<=65K              : BN=128 BK=128 W=4 S=1
+        - D=64 K>65K                   : BN=128 BK=64  W=4 S=1
+        - D=128 K<=4K                  : BN=128 BK=64  W=4 S=1
+        - D=128 K>4K                   : BN=128 BK=64  W=8 S=1
+        - D=256                        : BN=128 BK=64  W=8 S=1
+        - D=512 (and 320/384/448)      : BN=64  BK=32  W=4 S=1
+          (fp32 + wide-D only fits H200's 226 KiB SMEM with the smallest tile)
+
+    fp16 / bf16:
+        - D=64 K<=256                  : BN=128 BK=128 W=4 S=2
+        - D=64 256<K<200K              : BN=64  BK=128 W=4 S=4
+          (deeper pipeline + 2x D-amortisation)
+        - D=64 K>=200K                 : BN=128 BK=64  W=4 S=4
+        - D=128 K<=256                 : BN=128 BK=64  W=4 S=1
+        - D=128 256<K<=65K             : BN=128 BK=128 W=8 S=2
+        - D=128 K>65K                  : BN=128 BK=64  W=4 S=1
+        - D=256 K<=4K                  : BN=128 BK=64  W=4 S=1
+        - D=256 K>4K                   : BN=128 BK=64  W=8 S=1
+        - D>=320 (i.e. D=512)          : BN=128 BK=64  W=8 S=1
+
+    Tiny N (<65536) shrinks BN to 64 to avoid wasted work -- kept as a
+    guard since the new sweep only covered N>=65536.
+    """
+    half = _is_half_dtype(dtype)
+
+    if not half:
+        if D <= 64:
+            if K <= 256:
+                return {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1}
+            if K <= 65536:
+                return {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 4, "num_stages": 1}
+            return {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1}
+        if D <= 128:
+            if K <= 4096:
+                return {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1}
+            return {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 8, "num_stages": 1}
+        if D <= 256:
+            return {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 8, "num_stages": 1}
+        cfg = {"BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1}
+        if N < 65536:
+            cfg = dict(cfg)
+            cfg["BLOCK_N"] = 32
+        return cfg
+
+    if D <= 64:
+        if K <= 256:
+            cfg = {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 4, "num_stages": 2}
+        elif K < 200_000:
+            cfg = {"BLOCK_N": 64, "BLOCK_K": 128, "num_warps": 4, "num_stages": 4}
+        else:
+            cfg = {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 4}
+    elif D <= 128:
+        if K <= 256:
+            cfg = {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1}
+        elif K <= 65536:
+            cfg = {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 8, "num_stages": 2}
+        else:
+            cfg = {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1}
+    elif D <= 256:
+        if K <= 4096:
+            cfg = {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1}
+        else:
+            cfg = {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 8, "num_stages": 1}
+    else:
+        cfg = {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 8, "num_stages": 1}
+
+    if N < 65536:
+        cfg = dict(cfg)
+        cfg["BLOCK_N"] = 64
+
+    return cfg
+
+
+def _heuristic_euclid_config_h100_smallD(N: int, K: int, D: int, dtype) -> dict:
+    # H100 tuned heuristic (more conservative on D=64 mid-K vs H200).
+    block_n = 128
+    block_k = 64
+    num_warps = 4
+    num_stages = 1
+
+    if D >= 512:
+        block_n = 128
+        block_k = 64
+        num_warps = 8
+        num_stages = 1
+    elif D >= 256:
+        block_n = 128
+        block_k = 64
+        if K <= 1024:
+            num_warps = 8
+            num_stages = 1
+        elif K <= 16384:
+            num_warps = 4
+            num_stages = 1
+        else:
+            num_warps = 8
+            num_stages = 1
+    else:
+        # D <= 128
+        if D <= 64:
+            if K <= 1024:
+                block_k = 64
+                num_warps = 4
+                num_stages = 2
+            elif K <= 16384:
+                block_k = 64
+                num_warps = 4
+                num_stages = 2
+            elif K <= 65536:
+                block_k = 128
+                num_warps = 4
+                num_stages = 4
+            else:
+                block_k = 64
+                num_warps = 4
+                num_stages = 4
+        else:
+            # D == 128
+            if K <= 1024:
+                block_k = 64
+                num_warps = 4
+                num_stages = 1
+            elif K <= 65536:
+                block_k = 128
+                num_warps = 8
+                num_stages = 2
+            else:
+                block_k = 64
+                num_warps = 4
+                num_stages = 4
+
+    if N < 65536:
+        block_n = 64
+
+    return {
+        "BLOCK_N": block_n,
+        "BLOCK_K": block_k,
+        "num_warps": num_warps,
+        "num_stages": num_stages,
+    }
+
+
+def _heuristic_euclid_config_a100_smallD(N: int, K: int, D: int, dtype) -> dict:
+    # Robust default on A100 across tuned grid.
+    block_n = 128
+    block_k = 32
+    num_warps = 4
+    num_stages = 2
+
+    if D == 128:
+        # Small-N cases tend to prefer a larger K tile.
+        if N <= 65536:
+            block_k = 64
+    elif D == 256:
+        # D=256 benefits from deeper pipeline at larger K.
+        if K >= 65536:
+            block_k = 32
+            num_stages = 4
+        elif K >= 1024 and N <= 262144:
+            block_k = 64
+            num_stages = 4
+
+    return {
+        "BLOCK_N": block_n,
+        "BLOCK_K": block_k,
+        "num_warps": num_warps,
+        "num_stages": num_stages,
+    }
+
+
+def _heuristic_euclid_config_gb10_smallD(N: int, K: int, D: int, dtype) -> dict:
+    # GB10 (Grace Blackwell, ~80 SMs, ~99 KiB SMEM/SM) tuned heuristic.
+    # Derived from a grid sweep over N in {65536, 262144, 1048576},
+    # K in {256..200000}, D in {64,128,256,512}, B in {1, 32}, fp16.
+    if D >= 512:
+        if K <= 256:
+            return {"BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1}
+        return {"BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 8, "num_stages": 1}
+
+    if D >= 256:
+        if K <= 256:
+            return {"BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1}
+        return {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 2}
+
+    if D >= 128:
+        if K <= 256:
+            return {"BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1}
+        if K <= 1024:
+            if N <= 65536:
+                return {"BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1}
+            return {"BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2}
+        if K <= 65536:
+            return {"BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1}
+        return {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1}
+
+    # D <= 64
+    if K <= 256 and N <= 65536:
+        return {"BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1}
+    return {"BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1}
+
+
+def _heuristic_euclid_config_fallback_smallD(N: int, K: int, D: int, dtype) -> dict:
+    # Conservative default for unknown architectures (prioritize avoiding OOR).
+    return {
+        "BLOCK_N": 64,
+        "BLOCK_K": 32,
+        "num_warps": 4,
+        "num_stages": 1,
+    }
+
+
+_KNOWN_ARCHS = ("H200", "H100", "A100", "GB10")
+
+
+def _is_known_arch(gpu_name: str) -> bool:
+    return any(tag in gpu_name for tag in _KNOWN_ARCHS)
+
+
+def _arch_smallD_picker(gpu_name: str):
+    if "H200" in gpu_name:
+        return _heuristic_euclid_config_h200_smallD
+    if "H100" in gpu_name:
+        return _heuristic_euclid_config_h100_smallD
+    if "A100" in gpu_name:
+        return _heuristic_euclid_config_a100_smallD
+    if "GB10" in gpu_name:
+        return _heuristic_euclid_config_gb10_smallD
+    return _heuristic_euclid_config_fallback_smallD
+
+
+def _heuristic_euclid_config(
+    N: int,
+    K: int,
+    D: int,
+    *,
+    device: Optional[torch.device] = None,
+    dtype=None,
+):
+    """Architecture-aware heuristic config selection without autotune.
+
+    Per-GPU sub-functions own the actual lookup tables. This function only
+    routes to the right one and (for non-half dtypes) post-processes the
+    config through ``_fit_config_to_smem`` so fp32 / large-D inputs do not
+    OOR on small-SMEM GPUs.
+
+    For fp16/bf16 the picked config is returned **byte-for-byte** as the
+    original tables produced (``examples/regression_fp16_smalld.py``
+    enforces this).
+    """
+    if device is None:
+        device = torch.device("cuda")
+    gpu_name = torch.cuda.get_device_properties(device).name.upper()
+
+    cfg = _arch_smallD_picker(gpu_name)(N, K, D, dtype)
+
+    if _is_half_dtype(dtype):
+        return cfg
+
+    dtype_bytes = _dtype_bytes(dtype)
+    smem_limit = _smem_limit(device)
+    return _fit_config_to_smem(cfg, D, dtype_bytes, smem_limit)
+
+
+# -----------------------------------------------------------------------------
+# Split-D heuristic. Per-arch tables stay conservative for now and rely on
+# ``_fit_config_to_smem_split_d`` for SMEM safety. H200 and H100 have freshly
+# tuned tables; A100/GB10 still use a single default until tuning data lands.
+# -----------------------------------------------------------------------------
+
+
+def _heuristic_euclid_config_h200_largeD(N: int, K: int, D: int, dtype) -> dict:
+    """H200 split-D heuristic.
+
+    Derived from a focused grid sweep over D ∈ {1024, 2048, 4096},
+    K ∈ {256..65536}, N ∈ {65536, 262144, 1048576}, B=1, fp16+fp32.
+    Patterns:
+      - fp16/bf16 (2 bytes): wide tile (BN=128, BK=128) with BD=64 is the
+        clear winner across D=1024 and D=4096 (3/3 N votes per K bucket).
+        D=2048 with K ≥ 4096 prefers the deeper-D tile (BN=64, BK=128, BD=128)
+        because the centroid stream dominates and a fatter D chunk amortises
+        loads better.
+      - fp32 (4 bytes): same shape but BD shrinks to 32 to keep SMEM under
+        budget. D=1024 with K ≥ 4096 splits BN→64 and grows BD→64 (more
+        D-axis amortisation when the centroid set is large).
+    """
+    half = _is_half_dtype(dtype)
+
+    if half:
+        if D >= 4096:
+            return {"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_D": 64,
+                    "num_warps": 8, "num_stages": 4}
+        if D >= 2048:
+            if K <= 1024:
+                return {"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_D": 64,
+                        "num_warps": 8, "num_stages": 4}
+            return {"BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_D": 128,
+                    "num_warps": 4, "num_stages": 4}
+        # D ≈ 1024 (also covers D in (512, 1024) when small-D kernel doesn't fit)
+        return {"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_D": 64,
+                "num_warps": 8, "num_stages": 4}
+
+    # fp32 / wider
+    if D >= 2048:
+        return {"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_D": 32,
+                "num_warps": 8, "num_stages": 4}
+    # D ≈ 1024 fp32
+    if K <= 1024:
+        return {"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_D": 32,
+                "num_warps": 8, "num_stages": 4}
+    return {"BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_D": 64,
+            "num_warps": 4, "num_stages": 4}
+
+
+def _heuristic_euclid_config_h100_largeD(N: int, K: int, D: int, dtype) -> dict:
+    """H100 split-D heuristic.
+
+    Derived from a focused grid sweep over D ∈ {1024, 2048, 4096},
+    K ∈ {256, 1024, 4096, 16384, 65536}, N ∈ {65536, 262144, 1048576},
+    B=1, fp16+fp32. Two cells (K=65536 D=4096 N=1048576 fp16/fp32) were
+    skipped due to multi-hour cost; their config is extrapolated from the
+    matching K=65536 K=16384/N=1048576 winners (same dominant shape).
+
+    Patterns:
+      - fp16/bf16: BN=128 BK=128 BD=64 W=8 S=4 dominates D=1024 (K≥1024)
+        and D=4096. D=2048 with K ∈ [1024, 16384] prefers BN=64 BK=128
+        BD=128 W=4 S=4 — a deeper D tile amortises centroid loads when
+        the centroid set is moderate; the wider N tile only pays off
+        once K=65536 makes per-program K-streaming dominate. K=256
+        uniformly prefers the smaller N tile (BN=64, W=4) — too few K
+        tiles to justify the wider N-axis program.
+      - fp32: BN=128 BK=128 BD=32 W=8 S=4 dominates D ≥ 2048. D=1024
+        prefers BN=64 BK=128 BD=64 W=4 S=4 across all K — distinct from
+        H200 which keeps BD=32 at small K, because H100's narrower L2
+        benefits from the wider D chunk amortising the centroid stream.
+    """
+    half = _is_half_dtype(dtype)
+
+    if half:
+        # K=256 has too few centroid tiles to justify a wide N program.
+        if K <= 256:
+            return {"BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_D": 64,
+                    "num_warps": 4, "num_stages": 4}
+        if D >= 4096:
+            return {"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_D": 64,
+                    "num_warps": 8, "num_stages": 4}
+        if D >= 2048:
+            # K ∈ [1024, 16384]: deeper D tile (BD=128) amortises centroid
+            # loads better. K=65536: the long K stream dominates and the
+            # wider N tile (BN=128, BD=64) wins.
+            if K <= 16384:
+                return {"BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_D": 128,
+                        "num_warps": 4, "num_stages": 4}
+            return {"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_D": 64,
+                    "num_warps": 8, "num_stages": 4}
+        # D ≈ 1024 (also covers D ∈ (512, 1024) when small-D doesn't fit).
+        return {"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_D": 64,
+                "num_warps": 8, "num_stages": 4}
+
+    # fp32 / wider
+    if D >= 2048:
+        return {"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_D": 32,
+                "num_warps": 8, "num_stages": 4}
+    # D ≈ 1024 fp32 — wider BD=64 wins consistently across K on H100.
+    return {"BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_D": 64,
+            "num_warps": 4, "num_stages": 4}
+
+
+def _heuristic_euclid_config_a100_largeD(N: int, K: int, D: int, dtype) -> dict:
+    return {
+        "BLOCK_N": 64,
+        "BLOCK_K": 32,
+        "BLOCK_D": 32,
+        "num_warps": 4,
+        "num_stages": 2,
+    }
+
+
+def _heuristic_euclid_config_gb10_largeD(N: int, K: int, D: int, dtype) -> dict:
+    return {
+        "BLOCK_N": 64,
+        "BLOCK_K": 32,
+        "BLOCK_D": 32,
+        "num_warps": 4,
+        "num_stages": 1,
+    }
+
+
+def _heuristic_euclid_config_fallback_largeD(N: int, K: int, D: int, dtype) -> dict:
+    return {
+        "BLOCK_N": 32,
+        "BLOCK_K": 32,
+        "BLOCK_D": 32,
+        "num_warps": 4,
+        "num_stages": 1,
+    }
+
+
+def _arch_largeD_picker(gpu_name: str):
+    if "H200" in gpu_name:
+        return _heuristic_euclid_config_h200_largeD
+    if "H100" in gpu_name:
+        return _heuristic_euclid_config_h100_largeD
+    if "A100" in gpu_name:
+        return _heuristic_euclid_config_a100_largeD
+    if "GB10" in gpu_name:
+        return _heuristic_euclid_config_gb10_largeD
+    return _heuristic_euclid_config_fallback_largeD
+
+
+def _heuristic_euclid_config_split_d(
+    N: int,
+    K: int,
+    D: int,
+    *,
+    device: Optional[torch.device] = None,
+    dtype=None,
+):
+    """Heuristic config picker for the split-D Euclid assign kernel.
+
+    Always post-processes through ``_fit_config_to_smem_split_d`` so the
+    selected config is guaranteed to fit SMEM regardless of dtype/D.
+    """
+    if device is None:
+        device = torch.device("cuda")
+    gpu_name = torch.cuda.get_device_properties(device).name.upper()
+    cfg = _arch_largeD_picker(gpu_name)(N, K, D, dtype)
+    dtype_bytes = _dtype_bytes(dtype)
+    smem_limit = _smem_limit(device)
+    return _fit_config_to_smem_split_d(cfg, D, dtype_bytes, smem_limit)
+
+
+# Hard threshold: D > this triggers split-D dispatch even when SMEM would
+# nominally fit, so the fp16/bf16/fp32 + D ≤ 512 regime stays on the
+# original kernel path (matches the regime the existing tables were tuned
+# in). Larger D goes through the split-D path.
+_SMALL_D_MAX = 512
+
+
+def _is_power_of_2(n: int) -> bool:
+    return n > 0 and (n & (n - 1)) == 0
+
+
+def _need_split_d(D: int, dtype, device) -> bool:
+    """Decide whether to dispatch to the split-D kernel.
+
+    Split-D is needed when:
+      1. D exceeds the small-D regime (the original kernel can't tile D).
+      2. D is **not a power of 2** — the small-D kernel uses
+         ``tl.arange(0, D)`` which Triton requires to be a power of 2.
+         The split-D kernel uses ``tl.arange(0, BLOCK_D)`` (always a
+         power of 2) with a ``d_mask = d_offsets < D`` guard, so any D
+         value works.
+      3. The small-D kernel cannot fit even at minimum tile (BN=16, BK=16,
+         S=1) — SMEM safety net for awkward dtype/D/GPU triples.
+      4. The GPU is unknown to the heuristic. The small-D path relies on
+         per-arch tuning tables; on unfamiliar architectures we have no
+         data to trust those configs and the SMEM probe may also be
+         unreliable. Split-D with the conservative fallback (small BN/BK/BD,
+         num_stages=1) is the safer choice — ``_fit_config_to_smem_split_d``
+         then guarantees the launch fits regardless of how small the SMEM
+         budget actually turns out to be.
+    """
+    if D > _SMALL_D_MAX:
+        return True
+    if not _is_power_of_2(D):
+        return True
+    # tl.dot requires the contraction axis (D) to be >= 16.
+    if D < 16:
+        return True
+    if device is None:
+        device = torch.device("cuda")
+    gpu_name = torch.cuda.get_device_properties(device).name.upper()
+    if not _is_known_arch(gpu_name):
+        return True
+    dtype_bytes = _dtype_bytes(dtype)
+    smem_limit = _smem_limit(device)
+    return not _smallD_kernel_fits_smem(D, dtype_bytes, smem_limit)
+
+
+@triton.jit
+def _euclid_assign_kernel(
+    x_ptr,                 # *f16 / *f32 [B, N, D]
+    c_ptr,                 # *f16 / *f32 [B, K, D]
+    c_sq_ptr,              # *f32         [B, K]
+    out_ptr,               # *i32         [B, N]
+    B: tl.constexpr,
+    N: tl.constexpr,
+    K: tl.constexpr,
+    D: tl.constexpr,
+    stride_x_b: tl.constexpr,
+    stride_x_n: tl.constexpr,
+    stride_x_d: tl.constexpr,
+    stride_c_b: tl.constexpr,
+    stride_c_k: tl.constexpr,
+    stride_c_d: tl.constexpr,
+    stride_csq_b: tl.constexpr,
+    stride_csq_k: tl.constexpr,
+    stride_out_b: tl.constexpr,
+    stride_out_n: tl.constexpr,
+    BLOCK_N: tl.constexpr,
+    BLOCK_K: tl.constexpr,
+):
+    """Each program handles a tile of BLOCK_N points for a given batch element.
+
+    The kernel iterates over the centroid dimension K in chunks of BLOCK_K and
+    maintains the running minimum distance as well as the corresponding index
+    for every point in the tile.
+    """
+    pid_n = tl.program_id(0)          # tile index along N dimension
+    pid_b = tl.program_id(1)          # batch index
+    pid_b = pid_b.to(tl.int64)
+
+    n_start = pid_n * BLOCK_N
+    n_offsets = n_start + tl.arange(0, BLOCK_N)
+    n_offsets = n_offsets.to(tl.int64)
+    n_mask = n_offsets < N
+
+    # ------------------------------------------------------------------
+    # Load x tile  (BLOCK_N, D)
+    # ------------------------------------------------------------------
+    offs_d = tl.arange(0, D).to(tl.int64)
+    # Compute pointer for x block: base + b*stride_x_b + n*stride_x_n + d*stride_x_d
+    x_ptrs = (
+        x_ptr
+        + pid_b * stride_x_b
+        + n_offsets[:, None] * stride_x_n
+        + offs_d[None, :] * stride_x_d
+    )
+    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
+    x_tile = x_tile  # compute in f32
+
+    # Init best distance / index
+    best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)  # large number
+    best_idx = tl.zeros((BLOCK_N,), tl.int32)
+
+    # ------------------------------------------------------------------
+    # Iterate over centroids in chunks of BLOCK_K
+    # ------------------------------------------------------------------
+    for k_start in range(0, K, BLOCK_K):
+        k_offsets = k_start + tl.arange(0, BLOCK_K)
+        k_offsets = k_offsets.to(tl.int64)
+        k_mask = k_offsets < K
+
+        # Load centroid tile  (D, BLOCK_K)
+        c_ptrs = (
+            c_ptr
+            + pid_b * stride_c_b
+            + k_offsets[None, :] * stride_c_k
+            + offs_d[:, None] * stride_c_d
+        )
+        c_tile = tl.load(c_ptrs, mask=k_mask[None, :], other=0.0)
+        c_tile = c_tile
+
+        # load c_sq for the tile  (BLOCK_K,)
+        csq_ptrs = c_sq_ptr + pid_b * stride_csq_b + k_offsets * stride_csq_k
+        cent_sq = tl.load(csq_ptrs, mask=k_mask, other=0.0).to(tl.float32)
+
+        # # Compute centroid squared norms (BLOCK_K,)
+        # cent_sq = tl.sum(c_tile * c_tile, axis=0).to(tl.float32)
+
+        # Compute cross term (BLOCK_N, BLOCK_K) = x_tile @ c_tile
+        cross = tl.dot(x_tile, c_tile).to(tl.float32)
+
+        # Squared-distance rank score (same argmin as ||x-c||^2); ||x||^2 omitted.
+        dist = cent_sq[None, :] - 2.0 * cross
+
+        # Mask out invalid centroid columns before reduction
+        dist = tl.where(k_mask[None, :], dist, 3.4e38)
+
+        curr_min = tl.min(dist, axis=1)
+        curr_idx = tl.argmin(dist, axis=1)
+
+        update = curr_min < best_dist
+        best_dist = tl.where(update, curr_min, best_dist)
+        best_idx = tl.where(update, k_start + curr_idx, best_idx)
+
+    # ------------------------------------------------------------------
+    # Write results
+    # ------------------------------------------------------------------
+    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
+    tl.store(out_ptrs, best_idx, mask=n_mask)
+
+_euclid_assign_kernel_autotuned = triton.autotune(_TUNE_CONFIGS, key=["N", "K"])(_euclid_assign_kernel)
+
+
+# ===============================================================
+# Split-D Euclid assign kernel.
+#
+# This kernel mirrors ``_euclid_assign_kernel`` but tiles the feature
+# dimension D into chunks of size ``BLOCK_D`` so the per-program SMEM
+# footprint is bounded by BLOCK_D rather than D. The K-streaming property
+# (no full distance matrix materialised) is preserved: outer loop is K,
+# inner D loop accumulates ``cross (BN, BK)`` in registers across D chunks,
+# distance and best-index are computed at the end of each K iteration.
+# ===============================================================
+@triton.jit
+def _euclid_assign_kernel_split_d(
+    x_ptr,                 # *f16 / *f32 [B, N, D]
+    c_ptr,                 # *f16 / *f32 [B, K, D]
+    c_sq_ptr,              # *f32         [B, K]
+    out_ptr,               # *i32         [B, N]
+    B: tl.constexpr,
+    N: tl.constexpr,
+    K: tl.constexpr,
+    D: tl.constexpr,
+    stride_x_b: tl.constexpr,
+    stride_x_n: tl.constexpr,
+    stride_x_d: tl.constexpr,
+    stride_c_b: tl.constexpr,
+    stride_c_k: tl.constexpr,
+    stride_c_d: tl.constexpr,
+    stride_csq_b: tl.constexpr,
+    stride_csq_k: tl.constexpr,
+    stride_out_b: tl.constexpr,
+    stride_out_n: tl.constexpr,
+    BLOCK_N: tl.constexpr,
+    BLOCK_K: tl.constexpr,
+    BLOCK_D: tl.constexpr,
+):
+    pid_n = tl.program_id(0)
+    pid_b = tl.program_id(1)
+    pid_b = pid_b.to(tl.int64)
+
+    n_start = pid_n * BLOCK_N
+    n_offsets = n_start + tl.arange(0, BLOCK_N)
+    n_offsets = n_offsets.to(tl.int64)
+    n_mask = n_offsets < N
+
+    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
+
+    best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)
+    best_idx = tl.zeros((BLOCK_N,), tl.int32)
+
+    for k_start in range(0, K, BLOCK_K):
+        k_offsets = k_start + tl.arange(0, BLOCK_K)
+        k_offsets = k_offsets.to(tl.int64)
+        k_mask = k_offsets < K
+
+        csq_ptrs = c_sq_ptr + pid_b * stride_csq_b + k_offsets * stride_csq_k
+        cent_sq = tl.load(csq_ptrs, mask=k_mask, other=0.0).to(tl.float32)
+
+        # cross accumulator lives in registers across the D loop.
+        cross = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
+
+        for d_start in range(0, D, BLOCK_D):
+            d_offsets = d_start + offs_d
+            d_mask = d_offsets < D
+
+            x_ptrs = (
+                x_ptr
+                + pid_b * stride_x_b
+                + n_offsets[:, None] * stride_x_n
+                + d_offsets[None, :] * stride_x_d
+            )
+            x_chunk = tl.load(
+                x_ptrs,
+                mask=n_mask[:, None] & d_mask[None, :],
+                other=0.0,
+            )
+
+            c_ptrs = (
+                c_ptr
+                + pid_b * stride_c_b
+                + k_offsets[None, :] * stride_c_k
+                + d_offsets[:, None] * stride_c_d
+            )
+            c_chunk = tl.load(
+                c_ptrs,
+                mask=k_mask[None, :] & d_mask[:, None],
+                other=0.0,
+            )
+
+            cross += tl.dot(x_chunk, c_chunk).to(tl.float32)
+
+        dist = cent_sq[None, :] - 2.0 * cross
+        dist = tl.where(k_mask[None, :], dist, 3.4e38)
+
+        curr_min = tl.min(dist, axis=1)
+        curr_idx = tl.argmin(dist, axis=1)
+
+        update = curr_min < best_dist
+        best_dist = tl.where(update, curr_min, best_dist)
+        best_idx = tl.where(update, k_start + curr_idx, best_idx)
+
+    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
+    tl.store(out_ptrs, best_idx, mask=n_mask)
+
+
+_euclid_assign_kernel_split_d_autotuned = triton.autotune(
+    _TUNE_CONFIGS_SPLIT_D, key=["N", "K", "D"]
+)(_euclid_assign_kernel_split_d)
+
+
+@triton.jit
+def _cosine_assign_kernel(
+    x_ptr,                 # *f16 / *f32 [B, N, D]
+    c_ptr,                 # *f16 / *f32 [B, K, D]
+    out_ptr,               # *i32         [B, N]
+    B: tl.constexpr,
+    N: tl.constexpr,
+    K: tl.constexpr,
+    D: tl.constexpr,
+    stride_x_b: tl.constexpr,
+    stride_x_n: tl.constexpr,
+    stride_x_d: tl.constexpr,
+    stride_c_b: tl.constexpr,
+    stride_c_k: tl.constexpr,
+    stride_c_d: tl.constexpr,
+    stride_out_b: tl.constexpr,
+    stride_out_n: tl.constexpr,
+    BLOCK_N: tl.constexpr,
+    BLOCK_K: tl.constexpr,
+):
+    """Each program handles a tile of BLOCK_N points for a given batch element.
+
+    The kernel iterates over the centroid dimension K in chunks of BLOCK_K and
+    maintains the running minimum distance as well as the corresponding index
+    for every point in the tile.
+    """
+    pid_n = tl.program_id(0)          # tile index along N dimension
+    pid_b = tl.program_id(1)          # batch index
+    pid_b = pid_b.to(tl.int64)
+
+    n_start = pid_n * BLOCK_N
+    n_offsets = n_start + tl.arange(0, BLOCK_N)
+    n_offsets = n_offsets.to(tl.int64)
+    n_mask = n_offsets < N
+
+    # ------------------------------------------------------------------
+    # Load x tile  (BLOCK_N, D)
+    # ------------------------------------------------------------------
+    offs_d = tl.arange(0, D).to(tl.int64)
+    # Compute pointer for x block: base + b*stride_x_b + n*stride_x_n + d*stride_x_d
+    x_ptrs = (
+        x_ptr
+        + pid_b * stride_x_b
+        + n_offsets[:, None] * stride_x_n
+        + offs_d[None, :] * stride_x_d
+    )
+    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
+    x_tile = x_tile  # compute in f32
+
+    # Init best distance / index
+    best_dist = tl.full((BLOCK_N,), -3.4e38, tl.float32)  # less is worse 
+    best_idx = tl.zeros((BLOCK_N,), tl.int32)
+
+    # ------------------------------------------------------------------
+    # Iterate over centroids in chunks of BLOCK_K
+    # ------------------------------------------------------------------
+    for k_start in range(0, K, BLOCK_K):
+        k_offsets = k_start + tl.arange(0, BLOCK_K)
+        k_offsets = k_offsets.to(tl.int64)
+        k_mask = k_offsets < K
+
+        # Load centroid tile  (D, BLOCK_K)
+        c_ptrs = (
+            c_ptr
+            + pid_b * stride_c_b
+            + k_offsets[None, :] * stride_c_k
+            + offs_d[:, None] * stride_c_d
+        )
+        c_tile = tl.load(c_ptrs, mask=k_mask[None, :], other=0.0)
+        c_tile = c_tile
+
+        # Compute cosine distance (BLOCK_N, BLOCK_K) = x_tile @ c_tile
+        cross = tl.dot(x_tile, c_tile).to(tl.float32)
+
+        # Mask out invalid centroid columns before reduction.
+        # Use a sentinel below any real cosine/dot score so masked tail lanes
+        # never win the argmax. (0.0 is a real cosine value: any input row whose
+        # cosine to all real centroids is < 0 would otherwise route to a masked
+        # lane, returning an out-of-range cluster id.)
+        dist = tl.where(k_mask[None, :], cross, -torch.finfo(torch.float32).max)
+
+        curr_max = tl.max(dist, axis=1)
+        curr_idx = tl.argmax(dist, axis=1)
+
+        update = curr_max > best_dist
+        best_dist = tl.where(update, curr_max, best_dist)
+        best_idx = tl.where(update, k_start + curr_idx, best_idx)
+
+    # ------------------------------------------------------------------
+    # Write results
+    # ------------------------------------------------------------------
+    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
+    tl.store(out_ptrs, best_idx, mask=n_mask)
+
+_cosine_assign_kernel_autotuned = triton.autotune(_TUNE_CONFIGS, key=["N", "K"])(_cosine_assign_kernel)
+
+
+# ===============================================================
+# Split-D Cosine assign kernel. Same loop structure as Euclid split-D
+# but tracks running argmax over the dot-product (cosine score with
+# normalized inputs).
+# ===============================================================
+@triton.jit
+def _cosine_assign_kernel_split_d(
+    x_ptr,
+    c_ptr,
+    out_ptr,
+    B: tl.constexpr,
+    N: tl.constexpr,
+    K: tl.constexpr,
+    D: tl.constexpr,
+    stride_x_b: tl.constexpr,
+    stride_x_n: tl.constexpr,
+    stride_x_d: tl.constexpr,
+    stride_c_b: tl.constexpr,
+    stride_c_k: tl.constexpr,
+    stride_c_d: tl.constexpr,
+    stride_out_b: tl.constexpr,
+    stride_out_n: tl.constexpr,
+    BLOCK_N: tl.constexpr,
+    BLOCK_K: tl.constexpr,
+    BLOCK_D: tl.constexpr,
+):
+    pid_n = tl.program_id(0)
+    pid_b = tl.program_id(1)
+    pid_b = pid_b.to(tl.int64)
+
+    n_start = pid_n * BLOCK_N
+    n_offsets = n_start + tl.arange(0, BLOCK_N)
+    n_offsets = n_offsets.to(tl.int64)
+    n_mask = n_offsets < N
+
+    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
+
+    best_dist = tl.full((BLOCK_N,), -3.4e38, tl.float32)
+    best_idx = tl.zeros((BLOCK_N,), tl.int32)
+
+    for k_start in range(0, K, BLOCK_K):
+        k_offsets = k_start + tl.arange(0, BLOCK_K)
+        k_offsets = k_offsets.to(tl.int64)
+        k_mask = k_offsets < K
+
+        cross = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
+
+        for d_start in range(0, D, BLOCK_D):
+            d_offsets = d_start + offs_d
+            d_mask = d_offsets < D
+
+            x_ptrs = (
+                x_ptr
+                + pid_b * stride_x_b
+                + n_offsets[:, None] * stride_x_n
+                + d_offsets[None, :] * stride_x_d
+            )
+            x_chunk = tl.load(
+                x_ptrs,
+                mask=n_mask[:, None] & d_mask[None, :],
+                other=0.0,
+            )
+
+            c_ptrs = (
+                c_ptr
+                + pid_b * stride_c_b
+                + k_offsets[None, :] * stride_c_k
+                + d_offsets[:, None] * stride_c_d
+            )
+            c_chunk = tl.load(
+                c_ptrs,
+                mask=k_mask[None, :] & d_mask[:, None],
+                other=0.0,
+            )
+
+            cross += tl.dot(x_chunk, c_chunk).to(tl.float32)
+
+        # Mask invalid centroid columns to a sentinel below any real score.
+        dist = tl.where(k_mask[None, :], cross, -torch.finfo(torch.float32).max)
+
+        curr_max = tl.max(dist, axis=1)
+        curr_idx = tl.argmax(dist, axis=1)
+
+        update = curr_max > best_dist
+        best_dist = tl.where(update, curr_max, best_dist)
+        best_idx = tl.where(update, k_start + curr_idx, best_idx)
+
+    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
+    tl.store(out_ptrs, best_idx, mask=n_mask)
+
+
+_cosine_assign_kernel_split_d_autotuned = triton.autotune(
+    _TUNE_CONFIGS_SPLIT_D, key=["N", "K", "D"]
+)(_cosine_assign_kernel_split_d)
+
+
+# ---------------------------------------------------------------
+# Python wrapper
+# ---------------------------------------------------------------
+
+def euclid_assign_triton(
+    x: torch.Tensor,
+    centroids: torch.Tensor,
+    out: torch.Tensor = None,
+    c_sq: torch.Tensor = None,
+    *,
+    BLOCK_N: int = 128,
+    BLOCK_K: int = 128,
+    num_warps: Optional[int] = None,
+    num_stages: Optional[int] = None,
+    config: Optional[dict] = None,
+    use_heuristic: bool = True,
+) -> torch.Tensor:
+    """Return nearest-centroid indices using Triton kernel.
+
+    Args:
+        x         : (B, N, D) float16 / float32 (on CUDA)
+        centroids : (B, K, D) same dtype/device as x
+        out       : (B, N)    int32   – (option) pre-allocated output tensor (on CUDA)
+        c_sq      : (B, K)    float32 – (option) ||centroids||^2 per centroid (on CUDA)
+
+    Returns:
+        cluster_ids (B, N) int32 (callers can cast to int64 if desired)
+    Extra:
+        config        : {"BLOCK_N","BLOCK_K","num_warps","num_stages"} to force a config
+        use_heuristic : use a fixed heuristic config instead of autotune
+    """
+    assert x.is_cuda and centroids.is_cuda, "All tensors must be on CUDA"
+    # assert x.dtype in (torch.float16, torch.float32), "x must be fp16/fp32"
+    assert centroids.dtype == x.dtype, "centroids dtype mismatch"
+
+    B, N, D = x.shape
+    K = centroids.shape[1]
+    assert centroids.shape == (B, K, D), "centroids shape mismatch"
+
+    # x = x.contiguous()
+    # centroids = centroids.contiguous()
+
+    if out is None:
+        out = torch.empty((B, N), device=x.device, dtype=torch.int32)
+    if c_sq is None:
+        c_sq = (centroids.to(torch.float32) ** 2).sum(-1)
+
+    # Strides (in elements)
+    stride_x_b, stride_x_n, stride_x_d = x.stride()
+    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
+    stride_csq_b, stride_csq_k = c_sq.stride()
+    stride_out_b, stride_out_n = out.stride()
+
+    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)
+
+    use_split_d = _need_split_d(D, x.dtype, x.device)
+
+    selected_config = None
+    if config is not None:
+        selected_config = config
+    elif num_warps is not None or num_stages is not None:
+        if num_warps is None or num_stages is None:
+            raise ValueError("num_warps and num_stages must be set together")
+        selected_config = {
+            "BLOCK_N": BLOCK_N,
+            "BLOCK_K": BLOCK_K,
+            "num_warps": num_warps,
+            "num_stages": num_stages,
+        }
+        if use_split_d and "BLOCK_D" not in selected_config:
+            # Caller supplied a small-D-shaped config but D demands split-D.
+            # Use a sane default for BLOCK_D and let SMEM fitter shrink.
+            selected_config = dict(selected_config)
+            selected_config["BLOCK_D"] = 64
+            selected_config = _fit_config_to_smem_split_d(
+                selected_config,
+                D,
+                _dtype_bytes(x.dtype),
+                _smem_limit(x.device),
+            )
+    elif use_heuristic:
+        if use_split_d:
+            selected_config = _heuristic_euclid_config_split_d(
+                N, K, D, device=x.device, dtype=x.dtype
+            )
+        else:
+            selected_config = _heuristic_euclid_config(
+                N, K, D, device=x.device, dtype=x.dtype
+            )
+
+    if use_split_d:
+        if selected_config is None:
+            _euclid_assign_kernel_split_d_autotuned[grid](
+                x, centroids, c_sq, out,
+                B, N, K, D,
+                stride_x_b, stride_x_n, stride_x_d,
+                stride_c_b, stride_c_k, stride_c_d,
+                stride_csq_b, stride_csq_k,
+                stride_out_b, stride_out_n,
+            )
+        else:
+            _euclid_assign_kernel_split_d[grid](
+                x, centroids, c_sq, out,
+                B, N, K, D,
+                stride_x_b, stride_x_n, stride_x_d,
+                stride_c_b, stride_c_k, stride_c_d,
+                stride_csq_b, stride_csq_k,
+                stride_out_b, stride_out_n,
+                BLOCK_N=selected_config["BLOCK_N"],
+                BLOCK_K=selected_config["BLOCK_K"],
+                BLOCK_D=selected_config["BLOCK_D"],
+                num_warps=selected_config["num_warps"],
+                num_stages=selected_config["num_stages"],
+            )
+        return out
+
+    if selected_config is not None:
+        _euclid_assign_kernel[grid](
+            x,
+            centroids,
+            c_sq,
+            out,
+            B,
+            N,
+            K,
+            D,
+            stride_x_b,
+            stride_x_n,
+            stride_x_d,
+            stride_c_b,
+            stride_c_k,
+            stride_c_d,
+            stride_csq_b,
+            stride_csq_k,
+            stride_out_b,
+            stride_out_n,
+            BLOCK_N=selected_config["BLOCK_N"],
+            BLOCK_K=selected_config["BLOCK_K"],
+            num_warps=selected_config["num_warps"],
+            num_stages=selected_config["num_stages"],
+        )
+    else:
+        _euclid_assign_kernel_autotuned[grid](
+            x,
+            centroids,
+            c_sq,
+            out,
+            B,
+            N,
+            K,
+            D,
+            stride_x_b,
+            stride_x_n,
+            stride_x_d,
+            stride_c_b,
+            stride_c_k,
+            stride_c_d,
+            stride_csq_b,
+            stride_csq_k,
+            stride_out_b,
+            stride_out_n,
+        )
+    return out
+
+
+def cosine_assign_triton(x: torch.Tensor, centroids: torch.Tensor, out: torch.Tensor = None,
+                         *, BLOCK_N: int = 128, BLOCK_K: int = 128) -> torch.Tensor:
+    """Return nearest(cosine similarity)-centroid indices using Triton kernel.
+
+    Args:
+        x         : (B, N, D) float16 / float32 (on CUDA)
+        centroids : (B, K, D) same dtype/device as x
+
+    Returns:
+        cluster_ids (B, N) int32 (callers can cast to int64 if desired)
+    """
+    assert x.is_cuda and centroids.is_cuda, "All tensors must be on CUDA"
+    # assert x.dtype in (torch.float16, torch.float32), "x must be fp16/fp32"
+    assert centroids.dtype == x.dtype, "centroids dtype mismatch"
+
+    B, N, D = x.shape
+    K = centroids.shape[1]
+    assert centroids.shape == (B, K, D), "centroids shape mismatch"
+
+    # x = x.contiguous()
+    # centroids = centroids.contiguous()
+
+    if out is None:
+        out = torch.empty((B, N), device=x.device, dtype=torch.int32)
+
+    # Strides (in elements)
+    stride_x_b, stride_x_n, stride_x_d = x.stride()
+    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
+    stride_out_b, stride_out_n = out.stride()
+
+    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)
+
+    if _need_split_d(D, x.dtype, x.device):
+        _cosine_assign_kernel_split_d_autotuned[grid](
+            x,
+            centroids,
+            out,
+            B,
+            N,
+            K,
+            D,
+            stride_x_b,
+            stride_x_n,
+            stride_x_d,
+            stride_c_b,
+            stride_c_k,
+            stride_c_d,
+            stride_out_b,
+            stride_out_n,
+        )
+        return out
+
+    # Small-D path. The existing autotune sweep includes configs that may
+    # not fit SMEM (e.g. fp16 D=512 BN=64 BK=64 S=4 → 320 KiB > H200 limit;
+    # fp32 D=512 BN=64 BK=32 S=4 → 393 KiB). The cosine kernel has the
+    # same tile shapes as euclid, so reuse the SMEM-aware euclid heuristic
+    # for config selection regardless of dtype. This keeps fp16/bf16 small-D
+    # behaviour byte-identical to the euclid path's heuristic (verified by
+    # examples/regression_fp16_smalld.py).
+    cfg = _heuristic_euclid_config(N, K, D, device=x.device, dtype=x.dtype)
+    _cosine_assign_kernel[grid](
+        x,
+        centroids,
+        out,
+        B,
+        N,
+        K,
+        D,
+        stride_x_b,
+        stride_x_n,
+        stride_x_d,
+        stride_c_b,
+        stride_c_k,
+        stride_c_d,
+        stride_out_b,
+        stride_out_n,
+        BLOCK_N=cfg["BLOCK_N"],
+        BLOCK_K=cfg["BLOCK_K"],
+        num_warps=cfg["num_warps"],
+        num_stages=cfg["num_stages"],
+    )
+    return out
diff --git a/kmeanslib/_kernels/primitives/kmeans/triton/kmeans.py b/kmeanslib/_kernels/primitives/kmeans/triton/kmeans.py
new file mode 100644
index 0000000..db5fb79
--- /dev/null
+++ b/kmeanslib/_kernels/primitives/kmeans/triton/kmeans.py
@@ -0,0 +1,234 @@
+import torch
+import torch.nn.functional as F
+from torch.cuda import nvtx
+from kmeanslib._kernels.primitives.kmeans.triton.assign import euclid_assign_triton, cosine_assign_triton
+from kmeanslib._kernels.primitives.kmeans.triton.update import (
+    triton_centroid_update_cosine,
+    triton_centroid_update_euclid,
+    triton_centroid_update_sorted_euclid,
+    triton_centroid_update_sorted_cosine,
+    triton_lloyd_centroid_step_euclid,
+)
+try:
+    from tqdm import trange
+except Exception:
+    trange = range
+
+# -------------------- Compiled single-iteration kernels --------------------
+
+# 1. Euclidean
+def _euclid_iter(x, centroids, use_heuristic=True):
+
+    cluster_ids = euclid_assign_triton(x, centroids, use_heuristic=use_heuristic)
+    centroids_new = triton_centroid_update_sorted_euclid(x, cluster_ids, centroids)
+
+    shift = (centroids_new - centroids).norm(dim=-1).max()
+    return centroids_new, shift, cluster_ids
+
+# 2. Cosine
+def _cosine_iter(x_norm, centroids):
+    # cos_sim = torch.einsum('bnd,bkd->bnk', x_norm, centroids)
+    # cluster_ids = cos_sim.argmax(dim=-1)
+    cluster_ids = cosine_assign_triton(x_norm, centroids)
+    centroids_new = triton_centroid_update_sorted_cosine(x_norm, cluster_ids, centroids)
+    # centroids_new = centroids_new.clone()
+    shift = (centroids_new - centroids).norm(dim=-1).max()
+    return centroids_new, shift, cluster_ids
+
+# 3. Dot-product
+def _dot_iter(x, centroids):
+    # sim = torch.einsum('bnd,bkd->bnk', x, centroids)
+    # cluster_ids = sim.argmax(dim=-1)
+    cluster_ids = cosine_assign_triton(x, centroids)
+    centroids_new = triton_centroid_update_sorted_cosine(x, cluster_ids, centroids)
+    # centroids_new = centroids_new.clone()
+    shift = (centroids_new - centroids).norm(dim=-1).max()
+    return centroids_new, shift, cluster_ids
+
+COMPILE_FLAG = False
+
+try:
+    if COMPILE_FLAG:
+        _euclid_iter_compiled = torch.compile(_euclid_iter, dynamic=True, mode="reduce-overhead")
+        _cosine_iter_compiled = torch.compile(_cosine_iter, dynamic=True, mode="reduce-overhead")
+        _dot_iter_compiled    = torch.compile(_dot_iter,    dynamic=True, mode="reduce-overhead")
+    else:
+        _euclid_iter_compiled = _euclid_iter
+        _cosine_iter_compiled = _cosine_iter
+        _dot_iter_compiled    = _dot_iter
+except Exception:  # pragma: no cover
+    _euclid_iter_compiled = _euclid_iter
+    _cosine_iter_compiled = _cosine_iter
+    _dot_iter_compiled    = _dot_iter
+
+def batch_kmeans_Euclid(
+    x,
+    n_clusters,
+    max_iters=100,
+    tol=0.0,
+    init_centroids=None,
+    verbose=False,
+    *,
+    use_heuristic=True,
+    fused=True,
+):
+    """
+    Batched KMeans clustering in PyTorch using Euclidean distance.
+
+    Args:
+        x: Tensor of shape (B, N, D), batch_size B, N points per batch, D dims.
+        n_clusters: Number of clusters.
+        max_iters: Max number of iterations.
+        tol: Relative tolerance for center movement.
+        verbose: Print loss for each iter.
+        use_heuristic: Use heuristic Triton config (skip autotune).
+        fused: If True (default), use the fused Lloyd path with preallocated
+               sums/cnts/new/shift buffers and ping-pong centroid swap (no
+               .clone() per iter). Falls back to per-iter alloc when False.
+    Returns:
+        cluster_ids: (B, N) LongTensor, cluster assignment for each point.
+        centroids: (B, n_clusters, D) final cluster centers.
+    """
+    B, N, D = x.shape
+    K = n_clusters
+
+    if init_centroids is None:
+        # Randomly select initial centers from x
+        indices = torch.randint(0, N, (B, K), device=x.device)
+        centroids = torch.gather(
+            x,
+            dim=1,
+            index=indices[..., None].expand(-1, -1, D)
+        )  # (B, K, D)
+    else:
+        centroids = init_centroids
+
+    centroids = centroids.view(B, K, D).contiguous()
+
+    if not fused:
+        # ----- per-iter alloc + .clone() path -----
+        for it in range(max_iters):
+            centroids_new, center_shift, cluster_ids = _euclid_iter_compiled(
+                x, centroids, use_heuristic
+            )
+            if verbose:
+                print(f"Iter {it}, center shift: {center_shift.item():.6f}")
+            if center_shift < tol:
+                break
+            centroids = centroids_new.clone()
+        return cluster_ids, centroids, it + 1
+
+    # ----- fused path: preallocated buffers + ping-pong centroid swap -----
+    # Two centroid buffers swapped each iter so we never .clone().
+    cent_a = centroids
+    cent_b = torch.empty_like(centroids)
+
+    sums_buf = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
+    cnts_buf = torch.zeros((B, K), device=x.device, dtype=torch.int32)
+    shift_buf = torch.empty((B, K), device=x.device, dtype=torch.float32)
+
+    cur, nxt = cent_a, cent_b
+    cluster_ids = None
+    it = 0
+    for it in range(max_iters):
+        cluster_ids = euclid_assign_triton(x, cur, use_heuristic=use_heuristic)
+        # writes new centroids into `nxt`, returns scalar GPU tensor for shift
+        new_cent, _, max_shift = triton_lloyd_centroid_step_euclid(
+            x, cluster_ids, cur,
+            sums_buf=sums_buf,
+            cnts_buf=cnts_buf,
+            new_buf=nxt,
+            shift_buf=shift_buf,
+        )
+        if verbose:
+            print(f"Iter {it}, center shift: {max_shift.item():.6f}")
+        # swap before convergence check so `cur` always points to the latest
+        cur, nxt = nxt, cur
+        # Convergence check: `max_shift` is a 0-D GPU tensor, so
+        # `if max_shift < tol` triggers `tensor.__bool__()` which forces
+        # a per-iter cuda sync (~2.4 ms/iter on H200, drains the kernel
+        # pipeline). The short-circuit on `tol > 0.0` keeps the default
+        # `tol=0.0` path sync-free; users opting into early-exit accept
+        # the sync as part of that contract.
+        if tol > 0.0 and max_shift < tol:
+            break
+
+    return cluster_ids, cur, it + 1
+
+
+def batch_kmeans_Cosine(x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False):
+    """
+    Batched KMeans clustering in PyTorch using Cosine similarity.
+
+    Args:
+        x: Tensor of shape (B, N, D), batch_size B, N points per batch, D dims.
+        n_clusters: Number of clusters.
+        max_iters: Max number of iterations.
+        tol: Relative tolerance for center movement.
+        verbose: Print loss for each iter.
+    Returns:
+        cluster_ids: (B, N) LongTensor, cluster assignment for each point.
+        centroids: (B, n_clusters, D) final cluster centers.
+    """
+    B, N, D = x.shape
+
+    # Normalize input vectors for cosine similarity
+    x_norm = F.normalize(x, p=2, dim=-1)  # (B, N, D)
+
+    if init_centroids is None:
+        # Randomly select initial centers from x_norm
+        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
+        centroids = torch.gather(
+            x_norm,
+            dim=1,
+            index=indices[..., None].expand(-1, -1, D)
+        ) # (B, n_clusters, D)
+    else:
+        centroids = init_centroids
+
+    centroids = centroids.view(B, n_clusters, D)
+    centroids = F.normalize(centroids, p=2, dim=-1)  # Ensure centroids are normalized
+
+    for it in range(max_iters):
+        # ---- compiled single iteration ----
+        centroids_new, center_shift, cluster_ids = _cosine_iter_compiled(x_norm, centroids)
+
+        # 4. Check for convergence
+        if verbose:
+            print(f"Iter {it}, center shift: {center_shift.item():.6f}")
+        if center_shift < tol:
+            break
+        centroids = centroids_new.clone()
+
+    return cluster_ids, centroids, it + 1
+
+
+def batch_kmeans_Dot(x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False):
+    """
+    Batched KMeans clustering in PyTorch using raw dot-product as similarity.
+
+    """
+    B, N, D = x.shape
+
+    if init_centroids is None:
+        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
+        centroids = torch.gather(
+            x,
+            dim=1,
+            index=indices[..., None].expand(-1, -1, D)
+        )
+    else:
+        centroids = init_centroids
+
+    centroids = centroids.view(B, n_clusters, D)
+
+    for it in range(max_iters):
+        centroids_new, center_shift, cluster_ids = _dot_iter_compiled(x, centroids)
+
+        if verbose:
+            print(f"Iter {it} (dot), center shift: {center_shift.item():.6f}")
+        if center_shift < tol:
+            break
+        centroids = centroids_new.clone()
+
+    return cluster_ids, centroids, it + 1
diff --git a/kmeanslib/_kernels/primitives/kmeans/triton/update.py b/kmeanslib/_kernels/primitives/kmeans/triton/update.py
new file mode 100644
index 0000000..88ff697
--- /dev/null
+++ b/kmeanslib/_kernels/primitives/kmeans/triton/update.py
@@ -0,0 +1,640 @@
+import torch
+import torch.nn.functional as F
+import triton
+import triton.language as tl
+try:
+    from tqdm import trange
+except Exception:
+    trange = range
+
+
+def _ceil_div(a: int, b: int) -> int:
+    return (a + b - 1) // b
+
+
+@triton.jit
+def _centroid_update_kernel(
+    x_ptr,                # *f16 / *f32 [B, N, D]
+    cluster_ptr,          # *i32        [B, N]
+    sum_ptr,              # *f32        [B, K, D]
+    count_ptr,            # *i32        [B, K]
+    # --- strides (elements) ---
+    stride_x_b, stride_x_n, stride_x_d,
+    stride_sum_b, stride_sum_k, stride_sum_d,
+    stride_count_b, stride_count_k,
+    B: tl.constexpr,
+    N: tl.constexpr,
+    D: tl.constexpr,
+    K: tl.constexpr,
+    BLOCK_D: tl.constexpr,   # number of dims processed per program
+):
+    """Each program processes 1 token across BLOCK_D dims using atomics with general strides."""
+    pid = tl.program_id(axis=0)
+    token_idx = pid  # range: [0, B*N)
+
+    # Derive (b, n)
+    b = (token_idx // N).to(tl.int64)
+    n = (token_idx % N).to(tl.int64)
+
+    # pointer to this token's feature vector
+    x_offset = b * stride_x_b + n * stride_x_n
+    x_tok_ptr = x_ptr + x_offset
+
+    cluster_idx = tl.load(cluster_ptr + b * N + n)
+    cluster_idx = tl.where(cluster_idx < K, cluster_idx, 0)
+    cluster_idx = cluster_idx.to(tl.int64)
+
+    # base ptr for centroid accum array
+    centroid_base = b * stride_sum_b + cluster_idx * stride_sum_k
+
+    offs = tl.arange(0, BLOCK_D).to(tl.int64)
+    for d_start in range(0, D, BLOCK_D):
+        mask = offs + d_start < D
+        feats = tl.load(x_tok_ptr + (d_start + offs) * stride_x_d, mask=mask, other=0.0)
+        feats = feats.to(tl.float32)
+        dest_ptr = sum_ptr + centroid_base + (d_start + offs) * stride_sum_d
+        tl.atomic_add(dest_ptr, feats, mask=mask)
+
+    tl.atomic_add(count_ptr + b * stride_count_b + cluster_idx * stride_count_k, 1)
+
+
+def triton_centroid_update_cosine(x_norm: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor):
+    """Compute centroids using custom Triton kernel.
+
+    Args:
+        x_norm (Tensor): (B, N, D) normalized input vectors (float16/float32)
+        cluster_ids (LongTensor): (B, N) cluster assignment per point
+        old_centroids (Tensor): (B, K, D) previous centroids (same dtype as x_norm)
+
+    Returns:
+        Tensor: (B, K, D) updated and L2-normalized centroids (dtype == x_norm.dtype)
+    """
+    assert x_norm.is_cuda and cluster_ids.is_cuda, "Input tensors must be on CUDA device"
+    B, N, D = x_norm.shape
+    K = old_centroids.shape[1]
+    assert cluster_ids.shape == (B, N)
+
+    # Allocate accumulation buffers
+    centroid_sums = torch.zeros((B, K, D), device=x_norm.device, dtype=torch.float32)
+    centroid_counts = torch.zeros((B, K), device=x_norm.device, dtype=torch.int32)
+
+    # Launch Triton kernel – one program per token
+    total_tokens = B * N
+    BLOCK_D = 128  # tuneable
+
+    grid = (total_tokens,)
+    _centroid_update_kernel[grid](
+        x_norm,
+        cluster_ids.to(torch.int32),
+        centroid_sums,
+        centroid_counts,
+        x_norm.stride(0), x_norm.stride(1), x_norm.stride(2),
+        centroid_sums.stride(0), centroid_sums.stride(1), centroid_sums.stride(2),
+        centroid_counts.stride(0), centroid_counts.stride(1),
+        B, N, D, K,
+        BLOCK_D=BLOCK_D,
+    )
+
+    # Compute means; keep old centroid if empty cluster
+    counts_f = centroid_counts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
+    centroids = centroid_sums / counts_f
+
+    # For clusters with zero count, revert to old centroids
+    zero_mask = (centroid_counts == 0).unsqueeze(-1)
+    centroids = torch.where(zero_mask, old_centroids.to(torch.float32), centroids)
+
+    centroids = centroids.to(x_norm.dtype)
+    centroids = F.normalize(centroids, p=2, dim=-1)
+    return centroids
+
+
+def torch_loop_centroid_update_cosine(x_norm: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor):
+    """Reference Python implementation (double for-loop)"""
+    B, N, D = x_norm.shape
+    K = old_centroids.shape[1]
+    new_centroids = torch.zeros_like(old_centroids)
+    for b in range(B):
+        for k in range(K):
+            mask = cluster_ids[b] == k
+            if mask.any():
+                new_centroids[b, k] = F.normalize(x_norm[b][mask].mean(dim=0, dtype=x_norm.dtype), p=2, dim=0)
+            else:
+                new_centroids[b, k] = old_centroids[b, k]
+    return new_centroids
+
+
+def triton_centroid_update_euclid(x: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor):
+    """Compute centroids for Euclidean KMeans using Triton.
+
+    Args:
+        x (Tensor): (B, N, D) input vectors (float16/float32)
+        cluster_ids (LongTensor): (B, N) cluster assignment per point
+        old_centroids (Tensor): (B, K, D) previous centroids (same dtype as x)
+
+    Returns:
+        Tensor: (B, K, D) updated centroids (dtype == x.dtype)
+    """
+    assert x.is_cuda and cluster_ids.is_cuda, "Input tensors must be on CUDA device"
+    B, N, D = x.shape
+    K = old_centroids.shape[1]
+    assert cluster_ids.shape == (B, N)
+
+    # Allocate accumulation buffers
+    centroid_sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
+    centroid_counts = torch.zeros((B, K), device=x.device, dtype=torch.int32)
+
+    total_tokens = B * N
+    BLOCK_D = 128  # tuneable
+    grid = (total_tokens,)
+
+    _centroid_update_kernel[grid](
+        x,
+        cluster_ids.to(torch.int32),
+        centroid_sums,
+        centroid_counts,
+        x.stride(0), x.stride(1), x.stride(2),
+        centroid_sums.stride(0), centroid_sums.stride(1), centroid_sums.stride(2),
+        centroid_counts.stride(0), centroid_counts.stride(1),
+        B, N, D, K,
+        BLOCK_D=BLOCK_D,
+    )
+
+    # Compute means; keep old centroid if empty cluster
+    counts_f = centroid_counts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
+    centroids = centroid_sums / counts_f
+
+    # For clusters with zero count, revert to old centroids
+    zero_mask = (centroid_counts == 0).unsqueeze(-1)
+    centroids = torch.where(zero_mask, old_centroids.to(torch.float32), centroids)
+
+    return centroids.to(x.dtype)
+
+
+# ------------------------------ NEW: chunk-wise centroid update (sorted ids) ------------------------------
+
+def _next_power_of_2(n: int) -> int:
+    p = 1
+    while p < n:
+        p <<= 1
+    return p
+
+
+@triton.jit
+def _centroid_update_chunk_kernel(
+    x_ptr,                # *f16 / *f32 [B, N, D] – ORIGINAL ORDER
+    sorted_idx_ptr,       # *i32        [B, N]    – indices after sort
+    sorted_cluster_ptr,   # *i32        [B, N]    – cluster ids in sorted order
+    sum_ptr,              # *f32        [B, K, D]
+    count_ptr,            # *i32        [B, K]
+    # strides
+    stride_x_b, stride_x_n, stride_x_d,
+    stride_idx_b, stride_idx_n, stride_cluster_b, stride_cluster_n,
+    stride_sum_b, stride_sum_k, stride_sum_d,
+    stride_count_b, stride_count_k,
+    B: tl.constexpr,
+    N: tl.constexpr,
+    D: tl.constexpr,
+    K: tl.constexpr,
+    BLOCK_N: tl.constexpr,
+    BLOCK_D: tl.constexpr,
+):
+    """Each program processes **BLOCK_N consecutive, already-sorted tokens**.
+
+    Because the tokens are sorted by cluster id, identical ids appear in
+    contiguous runs.  We therefore accumulate a local sum/count for the
+    current run and perform **a single atomic update per run**, instead of
+    per-token.
+
+    ``BLOCK_D`` tiles the feature dimension so that non-power-of-2 D
+    values work (Triton requires ``tl.arange`` ranges to be power-of-2).
+    When ``BLOCK_D >= D`` the D-loop executes once — no perf regression
+    for already-power-of-2 inputs.
+    """
+    pid_chunk = tl.program_id(axis=0)
+    pid_b     = tl.program_id(axis=1)
+
+    b = pid_b.to(tl.int64)
+    chunk_start = (pid_chunk * BLOCK_N).to(tl.int64)
+
+    if chunk_start >= N:
+        return
+
+    idx_batch_base     = sorted_idx_ptr + b * stride_idx_b
+    cid_batch_base     = sorted_cluster_ptr + b * stride_cluster_b
+    x_batch_base       = x_ptr + b * stride_x_b
+
+    offs_token = tl.arange(0, BLOCK_N).to(tl.int64)
+    offs_dim   = tl.arange(0, BLOCK_D).to(tl.int64)
+
+    token_idx  = chunk_start + offs_token
+    valid_tok  = token_idx < N
+    first_token_idx = chunk_start
+    last_token_idx = tl.minimum(chunk_start + BLOCK_N, N) - 1
+
+    first_id = tl.load(cid_batch_base + first_token_idx)
+    last_id = tl.load(cid_batch_base + last_token_idx)
+    all_ids = tl.load(cid_batch_base + token_idx * stride_cluster_n, mask=valid_tok, other=-1)
+
+    all_tokens_idxs = tl.load(idx_batch_base + token_idx * stride_idx_n, mask=valid_tok, other=-1)
+    all_tokens_idxs = all_tokens_idxs.to(tl.int64)
+
+    for cid in range(first_id, last_id + 1):
+        cluster_mask = all_ids == cid
+        cluster_size = tl.sum(cluster_mask.to(tl.int32))
+        if cluster_size != 0:
+            tl.atomic_add(count_ptr + b*stride_count_b + cid*stride_count_k, cluster_size)
+            for d_start in range(0, D, BLOCK_D):
+                d_offsets = d_start + offs_dim
+                d_mask = d_offsets < D
+                row_ptrs = (x_batch_base
+                            + all_tokens_idxs[:, None] * stride_x_n
+                            + d_offsets[None, :] * stride_x_d)
+                cluster_feats = tl.load(
+                    row_ptrs,
+                    mask=cluster_mask[:, None] & d_mask[None, :],
+                    other=0.0,
+                )
+                cluster_feats = cluster_feats.to(tl.float32)
+                sum_feats = tl.sum(cluster_feats, axis=0)
+                dest_ptr = (sum_ptr + b * stride_sum_b
+                            + cid * stride_sum_k
+                            + d_offsets * stride_sum_d)
+                tl.atomic_add(dest_ptr, sum_feats, mask=d_mask)
+
+
+# ---------------------------------------------------------------------------------------------
+
+def triton_centroid_update_sorted_cosine(x_norm: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor,
+                                         *, BLOCK_N: int = 256):
+    """Fast centroid update assuming **cluster_ids are sorted along N**.
+
+    This helper will sort the assignments (together with `x_norm`) and launch the
+    chunk kernel above.  Compared to the naive per-token kernel it performs *one
+    atomic add per run of identical ids* instead of per token, providing large
+    speed-ups when clusters are reasonably sized.
+    """
+    assert x_norm.is_cuda and cluster_ids.is_cuda, "Inputs must be on CUDA"
+    B, N, D = x_norm.shape
+    K = old_centroids.shape[1]
+    assert cluster_ids.shape == (B, N)
+
+    # -------- sort per-batch --------
+    sorted_cluster_ids, sorted_idx = torch.sort(cluster_ids, dim=-1)
+    sorted_idx_int = sorted_idx.to(torch.int32)
+
+    # accumulation buffers
+    centroid_sums = torch.zeros((B, K, D), device=x_norm.device, dtype=torch.float32)
+    centroid_cnts = torch.zeros((B, K),    device=x_norm.device, dtype=torch.int32)
+
+    BLOCK_D = _next_power_of_2(D)
+    grid = (triton.cdiv(N, BLOCK_N), B)
+    _centroid_update_chunk_kernel[grid](
+        x_norm,
+        sorted_idx_int,
+        sorted_cluster_ids.to(torch.int32),
+        centroid_sums,
+        centroid_cnts,
+        x_norm.stride(0), x_norm.stride(1), x_norm.stride(2),
+        sorted_idx_int.stride(0), sorted_idx_int.stride(1),
+        sorted_cluster_ids.stride(0), sorted_cluster_ids.stride(1),
+        centroid_sums.stride(0), centroid_sums.stride(1), centroid_sums.stride(2),
+        centroid_cnts.stride(0), centroid_cnts.stride(1),
+        B, N, D, K,
+        BLOCK_N=BLOCK_N,
+        BLOCK_D=BLOCK_D,
+    )
+
+    # finalise – convert to means, handle empty clusters, renormalise
+    counts_f = centroid_cnts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
+    centroids = centroid_sums / counts_f
+    empty_mask = (centroid_cnts == 0).unsqueeze(-1)
+    centroids = torch.where(empty_mask, old_centroids.to(torch.float32), centroids)
+    centroids = centroids.to(x_norm.dtype)
+    centroids = F.normalize(centroids, p=2, dim=-1)
+    return centroids
+
+def triton_centroid_update_sorted_euclid(x: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor,
+                                         *, BLOCK_N: int = 256, centroid_sums: torch.Tensor = None, centroid_cnts: torch.Tensor = None, calculate_new: bool = True):
+    """Fast centroid update for *Euclidean* KMeans assuming cluster IDs are pre-sorted.
+
+    Parameters
+    ----------
+    x : Tensor [B, N, D]
+        Input feature vectors (no normalization assumed).
+    cluster_ids : LongTensor [B, N]
+        Cluster assignment for each point.
+    old_centroids : Tensor [B, K, D]
+        Previous centroids (used to fill empty clusters).
+    BLOCK_N : int, optional
+        Tokens per Triton program (affects occupancy/perf).
+    centroid_sums : Tensor [B, K, D], optional
+        Pre-allocated accumulation buffer for sums.  If None, a new buffer is created.
+    centroid_cnts : Tensor [B, K], optional
+        Pre-allocated accumulation buffer for counts.  If None, a new buffer is created.
+    calculate_new : bool, default=True
+        If True, compute and return the new centroids.  If False, only update the
+        accumulation buffers.
+
+    Returns
+    _________
+        centroids_new : Tensor [B, K, D] or None
+            Updated centroids if `calculate_new` is True; otherwise None.
+    """
+    assert x.is_cuda and cluster_ids.is_cuda, "Inputs must be on CUDA device"
+    B, N, D = x.shape
+    K = old_centroids.shape[1]
+
+    # Batch-wise sort of cluster assignments
+    sorted_cluster_ids, sorted_idx = torch.sort(cluster_ids, dim=-1)
+    sorted_idx_int = sorted_idx.to(torch.int32)
+
+    if centroid_sums is None:
+        centroid_sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
+    else:
+        assert centroid_sums.shape == (B, K, D)
+    
+    if centroid_cnts is None:
+        centroid_cnts = torch.zeros((B, K),    device=x.device, dtype=torch.int32)
+    else:
+        assert centroid_cnts.shape == (B, K)
+
+    BLOCK_D = _next_power_of_2(D)
+    grid = (triton.cdiv(N, BLOCK_N), B)
+    _centroid_update_chunk_kernel[grid](
+        x,                       # original features
+        sorted_idx_int,          # gather indices
+        sorted_cluster_ids.to(torch.int32),
+        centroid_sums,
+        centroid_cnts,
+        x.stride(0), x.stride(1), x.stride(2),
+        sorted_idx_int.stride(0), sorted_idx_int.stride(1),
+        sorted_cluster_ids.stride(0), sorted_cluster_ids.stride(1),
+        centroid_sums.stride(0), centroid_sums.stride(1), centroid_sums.stride(2),
+        centroid_cnts.stride(0), centroid_cnts.stride(1),
+        B, N, D, K,
+        BLOCK_N=BLOCK_N,
+        BLOCK_D=BLOCK_D,
+    )
+
+    if calculate_new:
+        # Convert sums to means; replace empty clusters with old centroids
+        counts_f = centroid_cnts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
+        centroids = centroid_sums / counts_f
+        empty_mask = (centroid_cnts == 0).unsqueeze(-1)
+        centroids = torch.where(empty_mask, old_centroids.to(torch.float32), centroids)
+        return centroids.to(x.dtype)
+    else:
+        return None
+# ------------------------------ END new implementation ------------------------------
+
+
+# =============================================================================
+# Fused centroid finalize + per-iter Lloyd helper
+# =============================================================================
+#
+# `_centroid_finalize_kernel` collapses 6 host-side ops into one Triton kernel:
+#     1. cnts.float()             count cast f32
+#     2. clamp(cnts, min=1)       safe-divide guard
+#     3. sums / cnts              per-element divide
+#     4. empty-mask where         fall back to old centroid where cnts==0
+#     5. cast back to original dtype
+#     6. (new - old).norm(dim=-1) per-cluster shift (host then takes max)
+#
+# Outputs (B, K) per-cluster shift so the host max reduction is over K, not B*K*D.
+#
+# `triton_lloyd_centroid_step_euclid` glues sort + chunk-update + finalize using
+# preallocated sums / cnts / new / shift buffers reused across Lloyd iterations,
+# so per-iter allocation cost is zero.
+
+@triton.jit
+def _centroid_finalize_kernel(
+    sums_ptr,           # *f32  [B, K, D]
+    cnts_ptr,           # *i32  [B, K]
+    old_ptr,            # *T    [B, K, D]   (T = output dtype)
+    new_ptr,            # *T    [B, K, D]   (output)
+    shift_ptr,          # *f32  [B, K]      (output: per-cluster ‖new − old‖₂)
+    stride_sums_b, stride_sums_k, stride_sums_d,
+    stride_cnts_b, stride_cnts_k,
+    stride_old_b, stride_old_k, stride_old_d,
+    stride_new_b, stride_new_k, stride_new_d,
+    stride_shift_b, stride_shift_k,
+    K: tl.constexpr,
+    D: tl.constexpr,
+    BLOCK_D: tl.constexpr,
+):
+    pid_k = tl.program_id(0)
+    pid_b = tl.program_id(1)
+
+    cnt = tl.load(cnts_ptr + pid_b * stride_cnts_b + pid_k * stride_cnts_k).to(tl.float32)
+    inv = 1.0 / tl.maximum(cnt, 1.0)
+    is_empty = cnt == 0.0
+
+    sq_acc = tl.zeros([], dtype=tl.float32)
+    offs_d = tl.arange(0, BLOCK_D)
+    n_blocks = (D + BLOCK_D - 1) // BLOCK_D
+
+    for blk in range(n_blocks):
+        d_idx = blk * BLOCK_D + offs_d
+        d_mask = d_idx < D
+
+        sum_off = pid_b * stride_sums_b + pid_k * stride_sums_k + d_idx * stride_sums_d
+        old_off = pid_b * stride_old_b + pid_k * stride_old_k + d_idx * stride_old_d
+        new_off = pid_b * stride_new_b + pid_k * stride_new_k + d_idx * stride_new_d
+
+        s = tl.load(sums_ptr + sum_off, mask=d_mask, other=0.0).to(tl.float32)
+        old_v = tl.load(old_ptr + old_off, mask=d_mask, other=0.0).to(tl.float32)
+
+        # divide; if empty, fall back to old centroid (shift contribution = 0)
+        new_v = tl.where(is_empty, old_v, s * inv)
+
+        # accumulate squared shift
+        diff = new_v - old_v
+        sq_acc += tl.sum(tl.where(d_mask, diff * diff, 0.0))
+
+        tl.store(new_ptr + new_off, new_v, mask=d_mask)
+
+    shift = tl.sqrt(sq_acc)
+    tl.store(shift_ptr + pid_b * stride_shift_b + pid_k * stride_shift_k, shift)
+
+
+def triton_centroid_finalize(
+    sums: torch.Tensor,        # (B, K, D) fp32
+    cnts: torch.Tensor,        # (B, K) int32
+    old_centroids: torch.Tensor,  # (B, K, D) original dtype
+    *,
+    out: torch.Tensor = None,
+    shift: torch.Tensor = None,
+    BLOCK_D: int = 128,
+):
+    """Fused finalize: sums/cnts → new centroids + per-cluster shifts.
+
+    Replaces the host pipeline:
+        cnts_f = cnts.float().unsqueeze(-1).clamp(min=1)
+        new = sums / cnts_f
+        new = where(cnts==0, old, new)
+        new = new.to(old.dtype)
+        shift = (new - old).norm(dim=-1)        # (B, K)
+
+    Returns (new_centroids, shift) where shift has shape (B, K) — caller takes
+    `.max()` over the K axis for the convergence criterion.
+    """
+    assert sums.is_cuda and cnts.is_cuda and old_centroids.is_cuda
+    B, K, D = sums.shape
+    assert cnts.shape == (B, K)
+    assert old_centroids.shape == (B, K, D)
+
+    if out is None:
+        out = torch.empty_like(old_centroids)
+    if shift is None:
+        shift = torch.empty((B, K), device=sums.device, dtype=torch.float32)
+
+    grid = (K, B)
+    _centroid_finalize_kernel[grid](
+        sums, cnts, old_centroids, out, shift,
+        sums.stride(0), sums.stride(1), sums.stride(2),
+        cnts.stride(0), cnts.stride(1),
+        old_centroids.stride(0), old_centroids.stride(1), old_centroids.stride(2),
+        out.stride(0), out.stride(1), out.stride(2),
+        shift.stride(0), shift.stride(1),
+        K=K, D=D, BLOCK_D=BLOCK_D,
+    )
+    return out, shift
+
+
+def triton_lloyd_centroid_step_euclid(
+    x: torch.Tensor,
+    cluster_ids: torch.Tensor,
+    old_centroids: torch.Tensor,
+    *,
+    BLOCK_N: int = 256,
+    sums_buf: torch.Tensor = None,
+    cnts_buf: torch.Tensor = None,
+    new_buf: torch.Tensor = None,
+    shift_buf: torch.Tensor = None,
+):
+    """Single Lloyd centroid step: sort → chunk-update → fused finalize.
+
+    All accumulator + output buffers can be preallocated and reused across
+    iterations for zero per-iter allocation cost.
+
+    Returns
+    -------
+    new_centroids : (B, K, D), original dtype  (== `new_buf` if provided)
+    cluster_ids   : (B, N) int (echoed back; unchanged)
+    max_shift     : scalar fp32 GPU tensor — `(new - old).norm(-1).max()`
+                    Caller can `.item()` to get host-side scalar.
+    """
+    B, N, D = x.shape
+    K = old_centroids.shape[1]
+
+    if sums_buf is None:
+        sums_buf = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
+    else:
+        sums_buf.zero_()
+    if cnts_buf is None:
+        cnts_buf = torch.zeros((B, K), device=x.device, dtype=torch.int32)
+    else:
+        cnts_buf.zero_()
+
+    # Reuse the existing sorted-update kernel to fill sums_buf / cnts_buf (no
+    # final divide — we delegate that to the fused finalize below).
+    triton_centroid_update_sorted_euclid(
+        x, cluster_ids, old_centroids,
+        BLOCK_N=BLOCK_N,
+        centroid_sums=sums_buf,
+        centroid_cnts=cnts_buf,
+        calculate_new=False,
+    )
+
+    new_centroids, shift_per_k = triton_centroid_finalize(
+        sums_buf, cnts_buf, old_centroids,
+        out=new_buf,
+        shift=shift_buf,
+    )
+    # max over K → (B,), then max over B → scalar (matches existing
+    # `(cnew - cold).norm(dim=-1).max()` semantics)
+    max_shift = shift_per_k.amax(dim=-1).amax()
+    return new_centroids, cluster_ids, max_shift
+
+
+def main():
+    torch.manual_seed(0)
+
+    B, N, D = 32, 74256, 128  # modest sizes for quick correctness test
+    K = 1000
+    dtype = torch.float16
+
+    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
+    x_norm = F.normalize(x, p=2, dim=-1)
+
+    cluster_ids = torch.randint(0, K, (B, N), device="cuda", dtype=torch.int32)
+
+    # Random old centroids for handling empty clusters
+    old_centroids = F.normalize(torch.randn(B, K, D, device="cuda", dtype=dtype), p=2, dim=-1)
+
+    # ---------------- Correctness check (compile Triton kernel) ----------------
+    ref_centroids = torch_loop_centroid_update_cosine(x_norm, cluster_ids, old_centroids)
+    tri_centroids = triton_centroid_update_cosine(x_norm, cluster_ids, old_centroids)  # this call triggers compilation
+    tri_sorted_centroids = triton_centroid_update_sorted_cosine(x_norm, cluster_ids, old_centroids)
+
+    # Validate correctness (includes first-run compile)
+    if torch.allclose(ref_centroids, tri_centroids, atol=1e-3, rtol=1e-3):
+        print("Centroid update: PASS ✅")
+    else:
+        max_diff = (ref_centroids - tri_centroids).abs().max().item()
+        print(f"Centroid update: FAIL ❌ | max diff = {max_diff}")
+
+    # Validate new sorted kernel
+    if torch.allclose(ref_centroids, tri_sorted_centroids, atol=1e-3, rtol=1e-3):
+        print("Sorted centroid update: PASS ✅")
+    else:
+        max_diff = (ref_centroids - tri_sorted_centroids).abs().max().item()
+        print(f"Sorted centroid update: FAIL ❌ | max diff = {max_diff}")
+
+
+    # show some examples
+    print(f"ref_centroids[0,0:5,0:5]: {ref_centroids[0,0:5,0:5]}")
+    print(f"tri_centroids[0,0:5,0:5]: {tri_centroids[0,0:5,0:5]}")
+    print(f"tri_sorted_centroids[0,0:5,0:5]: {tri_sorted_centroids[0,0:5,0:5]}")
+
+    # ---------------- Efficiency benchmark (exclude compile) ----------------
+    repeats = 20
+
+    # Torch loop timing
+    torch.cuda.synchronize()
+    start = torch.cuda.Event(enable_timing=True)
+    end = torch.cuda.Event(enable_timing=True)
+    start.record()
+    for _ in trange(repeats):
+        torch_loop_centroid_update_cosine(x_norm, cluster_ids, old_centroids)
+    end.record(); torch.cuda.synchronize()
+    torch_time = start.elapsed_time(end) / repeats  # average per run (ms)
+
+    # Triton timing (already compiled)
+    torch.cuda.synchronize()
+    start = torch.cuda.Event(enable_timing=True)
+    end = torch.cuda.Event(enable_timing=True)
+    start.record()
+    for _ in trange(repeats):
+        triton_centroid_update_cosine(x_norm, cluster_ids, old_centroids)
+    end.record(); torch.cuda.synchronize()
+    triton_time = start.elapsed_time(end) / repeats  # average per run (ms)
+
+    # Sorted Triton timing (already compiled)
+    torch.cuda.synchronize()
+    start = torch.cuda.Event(enable_timing=True)
+    end = torch.cuda.Event(enable_timing=True)
+    start.record()
+    for _ in trange(repeats):
+        triton_centroid_update_sorted_cosine(x_norm, cluster_ids, old_centroids)
+    end.record(); torch.cuda.synchronize()
+    triton_sorted_time = start.elapsed_time(end) / repeats  # average per run (ms)
+
+    print(f"\n=== Efficiency (average over {repeats} runs, exclude compile) ===")
+    print(f"Torch loop   : {torch_time:.2f} ms")
+    print(f"Triton kernel: {triton_time:.2f} ms (speed-up x{torch_time / triton_time:.2f})")
+    print(f"Triton sorted: {triton_sorted_time:.2f} ms (speed-up x{torch_time / triton_sorted_time:.2f})")
+
+
+if __name__ == "__main__":
+    main()
diff --git a/kmeanslib/kmeans.py b/kmeanslib/kmeans.py
index 2eadfdb..9823476 100644
--- a/kmeanslib/kmeans.py
+++ b/kmeanslib/kmeans.py
@@ -1,80 +1,31 @@
-"""One Lloyd K-Means step -- the function you optimise.
+"""Euclidean (squared-L2) one Lloyd step -- vendored best-in-repo Triton path.
 
-The judge owns the iteration loop, the data, the initial centroids, and the
-iteration count; you provide a single Lloyd step and it is called repeatedly.
-This shipped step is intentionally naive (materialise a (chunk, K) distance
-matrix with a bf16 matmul + argmin, then a PyTorch scatter update).
-
-Contract (do NOT change):
+Delegates to the Triton assign + sorted centroid-update kernels vendored under
+``kmeanslib._kernels`` (the fastest single-iteration Euclidean K-Means path in
+the source repo). Public contract is unchanged:
 
     step(x, centroids) -> (labels, new_centroids)
 
-    x            : (N, D) bfloat16 CUDA tensor of points.
-    centroids    : (K, D) bfloat16 tensor -- the current centroids.
-    labels       : (N,) int64 -- nearest-centroid assignment of every point to
-                   `centroids` (a full assignment of all N points).
-    new_centroids: (K, D) -- centroids recomputed as the mean of each cluster
-                   (empty clusters keep their previous centroid).
-
-This is exactly one Lloyd iteration: assign to `centroids`, then update. You may
-add modules/kernels under ``kmeanslib`` and rewrite the body of ``step``,
-``_assign`` and ``_update`` freely -- including **fusing the assign + update into
-a single kernel** -- as long as ``step(x, centroids) -> (labels, new_centroids)``
-is preserved. You cannot change how many times it runs, the data, or the initial
-centroids; the judge owns the loop and calls ``step`` a fixed number of times.
+This is exactly one Lloyd iteration -- assign every point to its nearest
+``centroids`` (Triton distance kernel), then recompute the centroids as the mean
+of each cluster (Triton sorted scatter-update). The judge owns the loop.
 """
 from __future__ import annotations
 
 import torch
 
+from kmeanslib._kernels.primitives.kmeans.triton.assign import euclid_assign_triton
+from kmeanslib._kernels.primitives.kmeans.triton.update import (
+    triton_centroid_update_sorted_euclid,
+)
 
-def _assign(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
-    """Nearest-centroid assignment by squared-L2 distance.
-
-    Naive: for each chunk of points, materialise the (chunk, K) distance matrix
-    with a bf16 matmul, then argmin. ``argmin ||x-c||^2 == argmin (||c||^2 - 2
-    x.c^T)`` since ``||x||^2`` is constant per row. bf16 in, fp32 accumulation.
-    """
-    c = centroids.to(x.dtype)
-    c_sq = (c.float() * c.float()).sum(1)                    # (K,) fp32
-    labels = torch.empty(x.shape[0], device=x.device, dtype=torch.long)
-    for lo in range(0, x.shape[0], 16384):
-        xb = x[lo:lo + 16384]
-        dist = c_sq[None, :] - 2.0 * (xb @ c.t()).float()   # (chunk, K) fp32
-        labels[lo:lo + 16384] = torch.argmin(dist, dim=1)
-    return labels
-
-
-def _update(
-    x: torch.Tensor,
-    labels: torch.Tensor,
-    n_clusters: int,
-    old_centroids: torch.Tensor,
-) -> torch.Tensor:
-    """Recompute each centroid as the mean of its assigned points.
-
-    Empty clusters keep their previous centroid.
-    """
-    N, D = x.shape
-    sums = torch.zeros((n_clusters, D), device=x.device, dtype=torch.float32)
-    counts = torch.zeros((n_clusters,), device=x.device, dtype=torch.float32)
-    sums.index_add_(0, labels, x.float())
-    counts.index_add_(0, labels, torch.ones(N, device=x.device, dtype=torch.float32))
-    empty = counts == 0
-    counts = counts.clamp_min(1.0)
-    new = sums / counts[:, None]
-    if empty.any():
-        new[empty] = old_centroids[empty].float()
-    return new.to(x.dtype)
-
-
-def step(x: torch.Tensor, centroids: torch.Tensor):
-    """One Lloyd iteration: assign to `centroids`, then recompute them.
 
-    Returns ``(labels, new_centroids)``. See the module docstring for the contract.
-    """
+def step(x, centroids):
     if x.ndim != 2:
-        raise ValueError(f"x must be 2-D (N, D); got shape {tuple(x.shape)}")
-    labels = _assign(x, centroids)
-    new_centroids = _update(x, labels, centroids.shape[0], centroids)
-    return labels, new_centroids
+        raise ValueError(f"x must be 2-D (N, D); got {tuple(x.shape)}")
+    # The vendored kernels are batched (B, N, D); run a single instance as B=1.
+    xb = x.unsqueeze(0).contiguous()
+    cb = centroids.to(x.dtype).unsqueeze(0).contiguous()
+    labels = euclid_assign_triton(xb, cb, use_heuristic=True)       # (1, N)
+    new_c = triton_centroid_update_sorted_euclid(xb, labels, cb)    # (1, K, D)
+    return labels.squeeze(0).to(torch.long), new_c.squeeze(0).to(x.dtype)
