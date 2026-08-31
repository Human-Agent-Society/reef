"""Evaluator for the direct BBOPlace ISPD2005 Frontier-CS 2.0 problem."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import pickle
import pwd
import shutil
import subprocess
import sys
import tempfile
import traceback
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


DATASET = "ispd2005"
BENCHMARKS = ("adaptec1",)
QUICK_BENCHMARKS = (BENCHMARKS[0],)
MAX_CANDIDATES = 1
TIMEOUT_SECONDS = int(os.environ.get("BBOPLACE_SOLUTION_TIMEOUT_SECONDS", "10800"))
BBOPLACE_ROOT = Path(os.environ.get("BBOPLACE_ROOT", "/opt/bboplace-bench"))
INF_HPWL_THRESHOLD = 1e15

# BBOPlace-Bench report, Table III, MGO + Vanilla-EA MP-HPWL.
# Reported unit is x10^5; constants below are raw HPWL values relaxed by 1.2x.
BASELINE_HPWL = {
    "adaptec1": 6.96e5,
    "adaptec2": 73.752e5,
    "adaptec3": 67.356e5,
    "adaptec4": 68.148e5,
    "bigblue1": 2.76e5,
    "bigblue3": 62.88e5,
}

_EVALUATORS: dict[str, Any] = {}


def _protect_evaluator_source() -> None:
    try:
        evaluator_path = Path(__file__).resolve()
        if str(evaluator_path).startswith(("/judge/", "/tests/")) and os.geteuid() == 0:
            evaluator_path.chmod(0o600)
    except Exception:
        pass


_protect_evaluator_source()


def _solution_preexec():
    if os.name != "posix":
        return None
    try:
        if os.geteuid() != 0:
            return None
        nobody = pwd.getpwnam("nobody")
    except Exception:
        return None

    def demote() -> None:
        os.setgid(nobody.pw_gid)
        os.setuid(nobody.pw_uid)

    return demote


def _ensure_runtime_paths() -> None:
    if not BBOPLACE_ROOT.exists():
        raise RuntimeError(
            f"BBOPlace runtime not found at {BBOPLACE_ROOT}; the judge image must include it"
        )
    for rel in ("src", "config", f"benchmarks/{DATASET}"):
        path = BBOPLACE_ROOT / rel
        if not path.exists():
            raise RuntimeError(f"BBOPlace judge image is missing {path}")
    for path in (
        BBOPLACE_ROOT,
        BBOPLACE_ROOT / "src",
        BBOPLACE_ROOT / "benchmarks",
        BBOPLACE_ROOT / "thirdparty",
        BBOPLACE_ROOT / "thirdparty" / "dreamplace",
    ):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    os.environ["PYTHONPATH"] = ":".join(sys.path)


def _install_runtime_shims() -> None:
    if "ray" not in sys.modules:
        ray = ModuleType("ray")

        class RemoteFunction:
            def __init__(self, fn):
                self.fn = fn

            def remote(self, *args, **kwargs):
                return self.fn(*args, **kwargs)

        def remote(*args, **kwargs):
            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return RemoteFunction(args[0])

            def decorator(fn):
                return RemoteFunction(fn)

            return decorator

        ray.remote = remote  # type: ignore[attr-defined]
        ray.get = lambda value: value  # type: ignore[attr-defined]
        ray.init = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        sys.modules["ray"] = ray

    if "matplotlib" not in sys.modules:
        matplotlib = ModuleType("matplotlib")
        pyplot = ModuleType("matplotlib.pyplot")
        patches = ModuleType("matplotlib.patches")
        pyplot.figure = lambda *args, **kwargs: SimpleNamespace(  # type: ignore[attr-defined]
            add_subplot=lambda *a, **k: SimpleNamespace(
                axes=SimpleNamespace(
                    xaxis=SimpleNamespace(set_visible=lambda *_: None),
                    yaxis=SimpleNamespace(set_visible=lambda *_: None),
                ),
                add_patch=lambda *_args, **_kwargs: None,
            ),
            savefig=lambda *_args, **_kwargs: None,
        )
        pyplot.close = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        patches.Rectangle = lambda *args, **kwargs: object()  # type: ignore[attr-defined]
        sys.modules["matplotlib"] = matplotlib
        sys.modules["matplotlib.pyplot"] = pyplot
        sys.modules["matplotlib.patches"] = patches


def _load_bbo_evaluator_class():
    _ensure_runtime_paths()
    _install_runtime_shims()
    import yaml  # type: ignore
    from config.benchmark import (  # type: ignore
        BENCHMARK_DIR,
        ROOT_DIR,
        benchmark_dict,
        benchmark_n_macro_dict,
        benchmark_type_dict,
    )
    from src.placedb import PlaceDB  # type: ignore
    placer_package = ModuleType("src.placer")
    placer_package.__path__ = [str(Path(ROOT_DIR) / "src" / "placer")]  # type: ignore[attr-defined]
    sys.modules.setdefault("src.placer", placer_package)
    from src.placer.mgo_placer import MaskGuidedOptimizationPlacer  # type: ignore

    class Evaluator:
        def __init__(self, args: Namespace):
            config_path = Path(ROOT_DIR) / "config"
            file_config_dict: dict[str, Any] = {}
            with (config_path / "default.yaml").open("r", encoding="utf-8") as f:
                file_config_dict.update(yaml.load(f, Loader=yaml.FullLoader) or {})
            with (config_path / "placer" / "mgo.yaml").open("r", encoding="utf-8") as f:
                file_config_dict.update(yaml.load(f, Loader=yaml.FullLoader) or {})

            benchmark_base = None
            for candidate_base, names in benchmark_dict.items():
                if args.benchmark in names:
                    benchmark_base = candidate_base
                    break
            if benchmark_base is None:
                raise RuntimeError(f"benchmark is not registered: {args.benchmark}")

            file_config_dict.update(
                {
                    "ROOT_DIR": ROOT_DIR,
                    "SOURCE_DIR": str(Path(ROOT_DIR) / "src"),
                    "THIRDPARTY_DIR": str(Path(ROOT_DIR) / "thirdparty"),
                    "placer": "mgo",
                    "benchmark": args.benchmark,
                    "benchmark_base": benchmark_base,
                    "benchmark_path": str(Path(BENCHMARK_DIR) / benchmark_base / args.benchmark),
                    "benchmark_type": benchmark_type_dict[benchmark_base],
                    "n_macro": benchmark_n_macro_dict[benchmark_base],
                    "eval_gp_hpwl": False,
                    "n_cpu_max": 1,
                    "result_path": str(
                        Path(tempfile.gettempdir())
                        / "frontier_bboplace_results"
                        / DATASET
                        / args.benchmark
                    ),
                    "unique_token": "frontier_cs_2_0",
                }
            )
            args.__dict__.update({k: v for k, v in file_config_dict.items() if k not in args.__dict__})
            Path(args.result_path).mkdir(parents=True, exist_ok=True)
            self.args = args
            self.placedb = PlaceDB(args=args)
            self.placer = MaskGuidedOptimizationPlacer(args=args, placedb=self.placedb)

        @property
        def n_dim(self):
            return self.placer.placedb.node_cnt * 2

    return Evaluator


def _make_args(benchmark: str) -> Namespace:
    return Namespace(
        placer="mgo",
        benchmark=benchmark,
        eval_gp_hpwl=False,
        seed=1,
        use_wandb=False,
        error_redirect=False,
        n_cpu_max=1,
        gpu=0,
    )


def _ensure_evaluator(benchmark: str) -> Any:
    if benchmark in _EVALUATORS:
        return _EVALUATORS[benchmark]
    if benchmark not in BENCHMARKS:
        raise RuntimeError(f"unknown benchmark: {benchmark}")
    Evaluator = _load_bbo_evaluator_class()
    _EVALUATORS[benchmark] = Evaluator(_make_args(benchmark))
    return _EVALUATORS[benchmark]


def _benchmark_info(benchmark: str, evaluator: Any) -> dict[str, Any]:
    placedb = evaluator.placer.placedb
    return {
        "dataset": DATASET,
        "benchmark": benchmark,
        "placer": "mgo",
        "metric": "mp_hpwl",
        "objective": "minimize",
        "dim": int(evaluator.n_dim),
        "node_cnt": int(placedb.node_cnt),
        "net_cnt": int(getattr(placedb, "net_cnt", len(getattr(placedb, "net_info", {})))),
        "canvas_width": float(placedb.canvas_width),
        "canvas_height": float(placedb.canvas_height),
        "n_grid_x": int(evaluator.args.n_grid_x),
        "n_grid_y": int(evaluator.args.n_grid_y),
        "bounds_kind": "mgo_repeated_grid",
        "submission_format": "direct_json",
        "max_candidates_per_submission": MAX_CANDIDATES,
        "baseline_hpwl": float(BASELINE_HPWL[benchmark]),
        "baseline_source": "BBOPlace-Bench Table III, MGO + Vanilla-EA MP-HPWL, unit x10^5, relaxed by 1.2x",
    }


def prepare() -> dict[str, Any]:
    _ensure_runtime_paths()
    return {
        "dataset": DATASET,
        "benchmarks": list(BENCHMARKS),
        "quick_feedback_benchmarks": list(QUICK_BENCHMARKS),
        "bboplace_root": str(BBOPLACE_ROOT),
        "max_candidates_per_submission": MAX_CANDIDATES,
        "load_mode": "lazy_per_benchmark",
    }


def _selected_benchmarks() -> tuple[tuple[str, ...], str]:
    return BENCHMARKS, "single_instance"


def _run_solution(solution_path: str, info: dict[str, Any]) -> Any:
    payload = json.loads(Path(solution_path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "fill" in payload:
            return [float(payload["fill"])] * int(info["dim"])
        if "placement" in payload:
            return payload["placement"]
        if "candidate" in payload:
            return payload["candidate"]
        if "candidates" in payload:
            return payload["candidates"]
        if "x" in payload and "y" in payload:
            x = payload["x"]
            y = payload["y"]
            if not isinstance(x, list) or not isinstance(y, list):
                raise ValueError("x and y must be lists")
            return [*x, *y]
        raise ValueError("JSON object must contain placement, candidate, candidates, or x/y")
    return payload


def _normalize_candidates(raw: Any, *, dim: int, node_cnt: int, n_grid_x: int, n_grid_y: int):
    import numpy as np

    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 1:
        if arr.size != dim:
            raise ValueError(f"expected one candidate of length {dim}, got length {arr.size}")
        arr = arr.reshape(1, dim)
    elif arr.ndim == 2:
        if arr.shape[1] != dim:
            raise ValueError(f"expected candidates with dimension {dim}, got {arr.shape[1]}")
    else:
        raise ValueError("candidates must be a 1D vector or a 2D list/array")

    if arr.shape[0] < 1:
        raise ValueError("at least one candidate is required")
    if arr.shape[0] > MAX_CANDIDATES:
        raise ValueError(
            f"too many candidates: got {arr.shape[0]}, maximum is {MAX_CANDIDATES}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("all candidate coordinates must be finite")
    x = arr[:, :node_cnt]
    y = arr[:, node_cnt:]
    if np.any(x < 0.0) or np.any(x > float(n_grid_x)):
        raise ValueError(f"x-grid coordinates must be in [0, {n_grid_x}]")
    if np.any(y < 0.0) or np.any(y > float(n_grid_y)):
        raise ValueError(f"y-grid coordinates must be in [0, {n_grid_y}]")
    return arr


def _evaluate_candidates(evaluator: Any, candidates: Any) -> tuple[float, float, int, int]:
    import numpy as np

    hpwl_values, overlap_values, _macro_pos = evaluator.placer.evaluate(candidates)
    hpwl_arr = np.asarray(hpwl_values, dtype=float).reshape(-1)
    overlap_arr = np.asarray(overlap_values, dtype=float).reshape(-1)
    if hpwl_arr.size != candidates.shape[0]:
        raise RuntimeError("BBOPlace returned an unexpected number of HPWL values")
    if not np.all(np.isfinite(hpwl_arr)):
        raise RuntimeError("BBOPlace returned a non-finite HPWL")
    best_index = int(np.argmin(hpwl_arr))
    if float(hpwl_arr[best_index]) >= INF_HPWL_THRESHOLD:
        raise ValueError("BBOPlace could not legalize any submitted candidate")
    overlap = float(overlap_arr[best_index]) if overlap_arr.size > best_index else math.nan
    return float(hpwl_arr[best_index]), overlap, best_index, int(candidates.shape[0])


def evaluate(solution_path: str) -> tuple[float, float, str, dict[str, Any]]:
    try:
        per_benchmark: list[dict[str, Any]] = []
        bounded_scores: list[float] = []
        raw_scores: list[float] = []

        selected_benchmarks, evaluation_scope = _selected_benchmarks()
        for benchmark in selected_benchmarks:
            evaluator = _ensure_evaluator(benchmark)
            info = _benchmark_info(benchmark, evaluator)
            raw_candidates = _run_solution(solution_path, info)
            candidates = _normalize_candidates(
                raw_candidates,
                dim=info["dim"],
                node_cnt=info["node_cnt"],
                n_grid_x=info["n_grid_x"],
                n_grid_y=info["n_grid_y"],
            )
            candidate_hpwl, overlap_rate, candidate_index, n_candidates = _evaluate_candidates(
                evaluator, candidates
            )
            baseline_hpwl = BASELINE_HPWL[benchmark]
            raw_score = 100.0 * (baseline_hpwl - candidate_hpwl) / baseline_hpwl
            bounded_score = max(0.0, raw_score)
            bounded_scores.append(bounded_score)
            raw_scores.append(raw_score)
            per_benchmark.append(
                {
                    "benchmark": benchmark,
                    "candidate_hpwl": candidate_hpwl,
                    "baseline_hpwl": baseline_hpwl,
                    "raw_score": raw_score,
                    "score": bounded_score,
                    "overlap_rate": overlap_rate,
                    "candidate_index": candidate_index,
                    "n_candidates": n_candidates,
                }
            )

        score = sum(bounded_scores) / len(bounded_scores)
        score_unbounded = sum(raw_scores) / len(raw_scores)
        message = (
            f"dataset={DATASET}; scope={evaluation_scope}; benchmarks={len(selected_benchmarks)}; "
            f"mean_score={score:.6f}; mean_score_unbounded={score_unbounded:.6f}; "
            "metric=MP-HPWL; baseline=1.2x relaxed MGO paper constants"
        )
        metrics = {
            "dataset": DATASET,
            "evaluation_scope": evaluation_scope,
            "direct_benchmark": BENCHMARKS[0],
            "benchmark_count": len(selected_benchmarks),
            "full_suite_benchmark_count": len(BENCHMARKS),
            "score_formula": "max(0, 100 * (baseline_hpwl - candidate_hpwl) / baseline_hpwl)",
            "mean_candidate_hpwl": sum(item["candidate_hpwl"] for item in per_benchmark)
            / len(per_benchmark),
            "per_benchmark": per_benchmark,
        }
        return score, score_unbounded, message, metrics
    except subprocess.TimeoutExpired:
        return 0.0, 0.0, f"timed out after {TIMEOUT_SECONDS}s", {}
    except Exception as exc:
        return 0.0, 0.0, f"evaluation failed: {exc}", {"traceback": traceback.format_exc()}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: evaluator.py /path/to/solution.json", file=sys.stderr)
        return 1
    score, score_unbounded, message, metrics = evaluate(argv[1])
    print(message, file=sys.stderr)
    if metrics:
        print(json.dumps(metrics, indent=2), file=sys.stderr)
    print(f"{score:.12f} {score_unbounded:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
