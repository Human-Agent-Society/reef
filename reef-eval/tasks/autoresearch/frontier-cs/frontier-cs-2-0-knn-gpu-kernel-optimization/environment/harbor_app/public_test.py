"""Public GPU self-test -- IDENTICAL to the final graded evaluation.

This runs the EXACT same workloads, thresholds, timing, and seeds that the hidden
judge uses to grade your submission (all read from ``/app/task_config.json``,
the same config the judge reads), on a Modal GPU, and reports the same
per-workload pass/fail + speedup + geomean + predicted score (0-100) you would
receive on submission. There is no longer a separate, smaller "public" workload
set -- what you see here is what you get graded on.

Requires MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in the environment.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, "/opt")  # flash_gpu.py is baked at /opt

APP_DIR = os.environ.get("APP_DIR", "/app")
TASK_CONFIG_PATH = os.environ.get("TASK_CONFIG_PATH", "/app/task_config.json")


def _load_eval() -> dict:
    """The judge grades from task_config.json's `evaluation` block; read the same."""
    try:
        doc = json.loads(Path(TASK_CONFIG_PATH).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"could not read {TASK_CONFIG_PATH}: {exc}")
        return {}
    return doc.get("evaluation", doc)


EV = _load_eval()


def g(key, default):
    v = EV.get(key, default)
    return default if v is None else v


PRIMITIVE = str(g("primitive", ""))
PKG = str(g("pkg", ""))
REF_MODULE = str(g("ref_module", ""))
SPEEDUP_TARGET = float(g("speedup_target", 8.0))
BASELINE_DIR = str(g("baseline_source", "/opt/flash_ref"))


def _workloads() -> list:
    """Exactly the judge's final-role workload set: ALL workloads, same seed
    derivation ``base_seed + 1000*(i+1)`` -- identical data to the graded run."""
    wls = [dict(w) for w in g("workloads", [])]
    base = int(g("base_seed", 20260701))
    for i, w in enumerate(wls):
        w.setdefault("seed", base + 1000 * (i + 1))
    return wls


def _cfg() -> dict:
    """Byte-for-byte the judge's _build_cfg()."""
    return {
        "primitive": PRIMITIVE,
        "pkg": PKG,
        "ref_module": REF_MODULE,
        "gpu": str(g("gpu", "H100")),
        "cuda_image": str(g("cuda_image", "nvidia/cuda:12.4.1-devel-ubuntu22.04")),
        "pip": list(g("pip", ["torch", "numpy"])),
        "app_name": str(g("app_name", "flash-kernel-public")),
        "modal_timeout_seconds": int(g("modal_timeout_seconds", 1800)),
        "warmup": int(g("warmup_iters", 3)),
        "iters": int(g("timed_iters", 7)),
        "inertia_tolerance": float(g("inertia_tolerance", 0.02)),
        "recall_threshold": float(g("recall_threshold", 0.99)),
        "captured_tolerance": float(g("captured_tolerance", 0.02)),
        "ortho_tolerance": float(g("ortho_tolerance", 0.02)),
        "ari_threshold": float(g("ari_threshold", 0.99)),
    }


def geometric_mean(values: list) -> float:
    if not values:
        return 0.0
    return math.exp(sum(math.log(max(v, 1e-9)) for v in values) / len(values))


def score_from_speedup(gm: float) -> float:
    if gm <= 0:
        return 0.0
    raw = 100.0 * math.log(gm) / math.log(max(SPEEDUP_TARGET, 1.0000001))
    return max(0.0, min(100.0, raw))


def _read(root: str, sub: str = "") -> dict:
    base = Path(root)
    scan = base / sub if sub else base
    return {str(p.relative_to(base)): p.read_text(encoding="utf-8", errors="replace")
            for p in scan.rglob("*.py")}


def main() -> int:
    import flash_gpu  # baked at /opt/flash_gpu.py
    if not flash_gpu.modal_available():
        print("Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET to run the GPU self-test.")
        return 1
    workloads = _workloads()
    cfg = _cfg()
    if not workloads:
        print("no workloads found in task_config.json; cannot run.")
        return 1
    payload = {
        "baseline_files": _read(BASELINE_DIR),
        "patched_files": _read(APP_DIR, PKG),
        "workloads": workloads,
        "cfg": cfg,
    }
    try:
        result = flash_gpu.run_remote(payload)
    except Exception as exc:  # noqa: BLE001
        print(f"GPU run failed: {exc}")
        return 1
    if not result.get("ok"):
        print(f"worker error: {result.get('error')}")
        return 1

    if PRIMITIVE in ("knn", "ivfpq"):
        mlabel, gate = "recall@k", f">= {cfg['recall_threshold']}"
    elif PRIMITIVE == "dbscan":
        mlabel, gate = "ARI", f">= {cfg['ari_threshold']}"
    elif PRIMITIVE == "kmeans":
        mlabel, gate = "inertia", f"<= (1+{cfg['inertia_tolerance']}) x ref (lower is better)"
    else:
        mlabel, gate = "quality", ""

    print("=== FINAL-EQUIVALENT self-test: identical workloads / thresholds / seeds / "
          "timing to the graded judge ===")
    print(f"correctness gate: {mlabel} {gate}   |   speedup_target = {SPEEDUP_TARGET:g}")
    print(f"{'workload':9s} {'status':22s} {'speedup':>9s}   {mlabel}: agent / ref")
    rows = result.get("rows", [])
    speedups = []
    any_fail = False
    for row in rows:
        av, rv = row.get("agent_val"), row.get("ref_val")
        q = (f"{av:.4f} / {rv:.4f}"
             if isinstance(av, (int, float)) and isinstance(rv, (int, float)) else "n/a")
        if row.get("ok"):
            sp = f"{row['speedup']:.2f}x"
            speedups.append(max(float(row["speedup"]), 0.01))  # same clamp as the judge
            st = "OK"
        else:
            sp = "-"
            st = "FAIL:" + str(row.get("reason", ""))
            any_fail = True
        print(f"{row['id']:9s} {st:22s} {sp:>9s}   {q}")

    # Scoring is byte-for-byte the judge's full_evaluation: ANY gate failure -> 0.
    print()
    if any_fail or len(speedups) != len(rows) or not speedups:
        print("RESULT: at least one workload FAILED the correctness gate -> a submission now "
              "would be INVALID and score 0/100. Fix the failing workload(s) before submitting.")
        return 0
    gm = geometric_mean(speedups)
    score = score_from_speedup(gm)
    print(f"geomean speedup = {gm:.3f}x  over the naive baseline")
    print(f"PREDICTED FINAL SCORE = {score:.2f} / 100   (100 == {SPEEDUP_TARGET:g}x geomean)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
