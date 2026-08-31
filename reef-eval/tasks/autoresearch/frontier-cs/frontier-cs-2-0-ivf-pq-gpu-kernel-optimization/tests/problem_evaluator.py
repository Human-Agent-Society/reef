"""Generic evaluator for the flashlib GPU kernel-optimization task family.

Identical across all four tasks (kmeans / knn / pca / truncated_svd); every
primitive-specific value comes from the ``evaluation`` block of the task's
config (delivered to the judge as ``/judge/task_config.json``), which is not
present in the agent workspace.

Flow: statically validate the patch (only ``<pkg>/**`` may change; no external
optimized libs / env / process / network access) -> apply it to the pristine
package baked at ``clean_source`` -> hand the frozen baseline + patched package
sources to a Modal GPU worker (``flash_gpu.run_remote``) that times both on
fresh per-iteration data and verifies clustering/retrieval/subspace quality on
every iteration -> gate on the worker's per-workload verdict -> score by the
geometric-mean speedup.

Without Modal credentials or the baked judge sources (e.g. repo CI on a box with
no GPU/Modal), the evaluator still validates the patch policy and returns a
smoke pass, so ``python3 evaluator.py reference.patch`` works anywhere.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TASK_CONFIG_PATH = Path("/judge/task_config.json")


def _load_task_config() -> dict[str, Any]:
    try:
        payload = json.loads(TASK_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


TASK_CONFIG = _load_task_config()
EVAL = TASK_CONFIG.get("evaluation", {}) if isinstance(TASK_CONFIG.get("evaluation"), dict) else {}


def _get(name: str, default):
    return EVAL.get(name, default)


def _get_int(name: str, default: int) -> int:
    try:
        return int(EVAL.get(name, default))
    except Exception:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(EVAL.get(name, default))
    except Exception:
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw = EVAL.get(name, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


# --- Per-task identity: the ONLY lines that differ across the four tasks. ---
# Kept as constants (not config) so the patch policy is enforceable locally / in
# CI without /judge/task_config.json. Everything else is config-driven and only
# needed on the GPU path.
PRIMITIVE = "ivfpq"
PKG = "ivfpqlib"
REF_MODULE = "refivfpq"

MAX_PATCH_BYTES = _get_int("max_patch_bytes", 2_000_000)
MAX_CHANGED_FILES = _get_int("max_changed_files", 80)
CLEAN_SOURCE = Path(f"/opt/{PKG}-clean")
BASELINE_SOURCE = Path(f"/opt/{PRIMITIVE}_ref")
SPEEDUP_TARGET = _get_float("speedup_target", 8.0)

ALLOWED_PATTERNS = (f"{PKG}/**",)
DENIED_PATTERNS = (
    "evaluator.py",
    "flash_gpu.py",
    "reference.patch",
    f"{REF_MODULE}.py",
    f"**/{REF_MODULE}.py",
    "**/conftest.py",
    "**/test_*.py",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
)
# Defense in depth (the load-bearing anti-tamper defense is structural, inside
# the GPU worker: it captures its perf_counter / synchronize references before
# importing the submission, regenerates data every iteration, and re-verifies
# quality each iteration). These patterns are matched surgically so that
# legitimate vendored/optimized kernel source is not rejected:
#   * external optimized libraries, matched as IMPORT statements (not bare words,
#     so a comment mentioning a name is fine); none are installed on the GPU
#     image anyway;
#   * process / network / measurement-tamper patterns that never appear in a
#     real GPU kernel.
FORBIDDEN_IMPORT_RE = re.compile(
    r"(?:^|\n)\s*(?:import|from)\s+(flashlib|cuml|cudf|cupy|faiss|sklearn|scikit|cutlass|cuvs)\b"
)
FORBIDDEN_PATTERN_RE = re.compile(
    r"\bsubprocess\b|\bsocket\b|\burllib\b|\brequests\b|os\.system|"
    r"torch\.cuda\.synchronize\s*=|time\.perf_counter\s*=|setattr\(\s*torch\.cuda"
)


@dataclass(frozen=True)
class PatchFile:
    old_path: str
    new_path: str
    added_lines: tuple[str, ...]

    @property
    def path(self) -> str:
        return self.new_path if self.new_path != "/dev/null" else self.old_path


def _match(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def _invalid(message: str, metrics: dict[str, Any] | None = None):
    payload = metrics or {}
    payload.setdefault("valid_patch", 0)
    return 0.0, 0.0, message, payload


def _parse_patch(text: str) -> list[PatchFile]:
    files: list[PatchFile] = []
    old = new = ""
    added: list[str] = []
    in_file = False
    for line in text.splitlines():
        if line.startswith("diff --git "):
            if in_file:
                files.append(PatchFile(old, new, tuple(added)))
            in_file, old, new, added = True, "", "", []
            continue
        if not in_file:
            continue
        if line.startswith("--- "):
            old = line[4:].strip()
            old = old[2:] if old.startswith("a/") else old
        elif line.startswith("+++ "):
            new = line[4:].strip()
            new = new[2:] if new.startswith("b/") else new
        elif line.startswith("+") and not line.startswith("+++ "):
            added.append(line[1:])
    if in_file:
        files.append(PatchFile(old, new, tuple(added)))
    return files


def _validate_path(path: str) -> tuple[bool, str]:
    if not path or path == "/dev/null":
        return True, ""
    if path.startswith("/") or ".." in Path(path).parts:
        return False, f"unsafe patch path: {path}"
    if _match(path, DENIED_PATTERNS):
        return False, f"changed file is outside task boundary: {path}"
    if not _match(path, ALLOWED_PATTERNS):
        return False, f"changed file is not allowlisted (only {PKG}/** may change): {path}"
    if not path.endswith(".py"):
        return False, f"only Python files may change: {path}"
    return True, ""


def validate_patch(patch_path: Path) -> tuple[bool, str, dict[str, Any]]:
    if not patch_path.exists():
        return False, "solution patch does not exist", {}
    size = patch_path.stat().st_size
    if size > MAX_PATCH_BYTES:
        return False, f"patch is too large ({size} bytes > {MAX_PATCH_BYTES})", {}
    text = patch_path.read_text(encoding="utf-8", errors="replace")
    files = _parse_patch(text)
    metrics: dict[str, Any] = {
        "patch_bytes": size,
        "patch_sha256": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest(),
        "changed_files": len(files),
    }
    if len(files) > MAX_CHANGED_FILES:
        return False, f"too many changed files ({len(files)} > {MAX_CHANGED_FILES})", metrics
    for pf in files:
        path = pf.path
        if pf.new_path == "/dev/null":
            return False, f"deleting files is outside task boundary: {pf.old_path}", metrics
        if pf.old_path != "/dev/null" and pf.old_path != pf.new_path:
            ok, err = _validate_path(pf.old_path)
            if not ok:
                return False, f"rename source is outside task boundary: {err}", metrics
        ok, err = _validate_path(path)
        if not ok:
            return False, err, metrics
        added = "\n".join(pf.added_lines)
        m = FORBIDDEN_IMPORT_RE.search(added)
        if m:
            return False, f"{path}: forbidden import of an external optimized library ({m.group(1)})", metrics
        m = FORBIDDEN_PATTERN_RE.search(added)
        if m:
            return False, f"{path}: forbidden pattern in added code ({m.group(0).strip()[:40]})", metrics
    metrics["valid_patch"] = 1
    return True, "patch accepted by static policy", metrics


def sanitize(text: str) -> str:
    text = re.sub(r"/tmp/[A-Za-z0-9_./-]+", "<tmp>", text or "")
    text = re.sub(r"/opt/[A-Za-z0-9_./-]+", "<judge>", text)
    text = re.sub(r"\bN=\d+|\bseed[=:]?\s*\d+", "<hidden>", text, flags=re.IGNORECASE)
    return text[-600:]


def geometric_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.exp(sum(math.log(max(v, 1e-9)) for v in values) / len(values))


def score_from_speedup(gm: float) -> float:
    if gm <= 0:
        return 0.0
    raw = 100.0 * math.log(gm) / math.log(max(SPEEDUP_TARGET, 1.0000001))
    return max(0.0, min(100.0, raw))


def is_final_role() -> bool:
    return os.environ.get("FRONTIER_SUBMISSION_ROLE", "agent") == "final"


def _workloads(final_role: bool) -> list[dict[str, Any]]:
    workloads = list(_get("workloads", []) or [])
    if not final_role:
        n = _get_int("agent_workload_count", 3)
        workloads = workloads[:n]
    base = _get_int("base_seed", 20260701)
    for i, w in enumerate(workloads):
        w.setdefault("seed", base + 1000 * (i + 1))
    return workloads


def _build_cfg() -> dict[str, Any]:
    return {
        "primitive": PRIMITIVE,
        "pkg": PKG,
        "ref_module": REF_MODULE,
        "gpu": str(_get("gpu", "H100")),
        "cuda_image": str(_get("cuda_image", "nvidia/cuda:12.4.1-devel-ubuntu22.04")),
        "pip": list(_get("pip", ["torch==2.5.1", "triton==3.1.0", "numpy"])),
        "app_name": str(_get("app_name", "flash-kernel-eval")),
        "dataset_volume": _get("dataset_volume", None),   # Modal Volume with real SIFT/GIST datasets, mounted at /data
        "modal_timeout_seconds": _get_int("modal_timeout_seconds", 1800),
        "warmup": _get_int("warmup_iters", 3),
        "iters": _get_int("timed_iters", 7),
        "inertia_tolerance": _get_float("inertia_tolerance", 0.02),
        "recall_threshold": _get_float("recall_threshold", 0.99),
        "captured_tolerance": _get_float("captured_tolerance", 0.02),
        "ortho_tolerance": _get_float("ortho_tolerance", 0.02),
        "ari_threshold": _get_float("ari_threshold", 0.99),
    }


def _read_dir(root: Path, rel_to: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in root.rglob("*.py"):
        out[str(p.relative_to(rel_to))] = p.read_text(encoding="utf-8", errors="replace")
    return out


def full_evaluation(patch_path: Path, metrics: dict[str, Any]):
    final_role = is_final_role()
    metrics["submission_role"] = "final" if final_role else "agent"
    workloads = _workloads(final_role)

    # Import the Modal harness baked into the judge image; missing harness or
    # missing credentials/sources -> policy smoke pass.
    sys.path.insert(0, "/opt")
    try:
        import flash_gpu  # type: ignore
    except Exception:
        metrics["full_benchmark"] = 0
        return 1.0, 1.0, "patch policy smoke passed; GPU harness not available in this environment", metrics
    if not flash_gpu.modal_available() or not CLEAN_SOURCE.exists() or not BASELINE_SOURCE.exists():
        metrics["full_benchmark"] = 0
        return 1.0, 1.0, "patch policy smoke passed; Modal/judge sources not configured in this environment", metrics

    with tempfile.TemporaryDirectory(prefix="flash_kernel_opt_") as tmp:
        tmp_root = Path(tmp)
        patched = tmp_root / "patched"
        shutil.copytree(CLEAN_SOURCE, patched)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_root),
               "LC_ALL": "C", "LANG": "C"}
        if int(metrics.get("changed_files", 0)) > 0:
            try:
                subprocess.run(["git", "apply", "--check", str(patch_path)], cwd=patched, env=env,
                               check=True, capture_output=True, text=True, timeout=60)
                subprocess.run(["git", "apply", str(patch_path)], cwd=patched, env=env,
                               check=True, capture_output=True, text=True, timeout=60)
            except subprocess.CalledProcessError as exc:
                metrics["stderr_tail"] = sanitize(exc.stderr or "")
                return _invalid("patch does not apply to the clean package tree", metrics)
            metrics["applied_patch"] = 1
        else:
            metrics["used_empty_patch"] = 1

        payload = {
            "baseline_files": _read_dir(BASELINE_SOURCE, BASELINE_SOURCE),
            "patched_files": _read_dir(patched, patched),
            "workloads": workloads,
            "cfg": _build_cfg(),
        }
        try:
            result = flash_gpu.run_remote(payload)
        except Exception as exc:
            metrics["error_detail"] = sanitize(str(exc))
            return _invalid("GPU evaluation failed", metrics)

    if not result.get("ok"):
        metrics["worker_error"] = sanitize(str(result.get("error", "unknown")))
        return _invalid("GPU benchmark worker failed", metrics)

    speedups: list[float] = []
    per_workload: dict[str, Any] = {}
    for row in result.get("rows", []):
        if not row.get("ok"):
            metrics["failed_workload"] = {"reason": row.get("reason", "error")}
            return _invalid("submission failed the quality gate or crashed on a hidden workload", metrics)
        speedups.append(max(float(row["speedup"]), 0.01))
        per_workload[row["id"]] = {"speedup": row["speedup"]}
    if not speedups:
        return _invalid("no workloads were evaluated", metrics)

    gm = geometric_mean(speedups)
    bounded = score_from_speedup(gm)
    metrics.update({"full_benchmark": 1, "workload_count": len(speedups), "geomean_speedup": gm})
    if _get_bool("expose_per_workload_metrics", False):
        metrics["per_workload"] = per_workload
    return bounded, bounded, f"{PRIMITIVE} geomean speedup {gm:.3f}x over the naive baseline", metrics


def evaluate(solution_path: str) -> tuple[float, float, str, dict[str, Any]]:
    patch_path = Path(solution_path)
    ok, message, metrics = validate_patch(patch_path)
    if not ok:
        return _invalid(message, metrics)
    try:
        return full_evaluation(patch_path, metrics)
    except Exception as exc:  # noqa: BLE001
        metrics["error_type"] = type(exc).__name__
        metrics["error_detail"] = sanitize(str(exc))
        return _invalid("evaluation failed", metrics)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: evaluator.py SOLUTION_PATCH", file=sys.stderr)
        return 2
    score, unbounded, message, metrics = evaluate(argv[1])
    print(json.dumps({"score": score, "score_unbounded": unbounded,
                      "message": message, "metrics": metrics}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
