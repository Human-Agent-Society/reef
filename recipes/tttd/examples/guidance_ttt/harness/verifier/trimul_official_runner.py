from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml


def build_case_file(cases: list[dict[str, Any]]) -> str:
    """Serialize cases exactly as libkernelbot's build_test_string does."""
    lines = ["; ".join(f"{key}: {value}" for key, value in case.items()) for case in cases]
    return "\n".join(lines) + "\n"


def _parse_popcorn_output(raw: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in raw.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip():
            parsed[key.strip()] = value.strip()
    return parsed


def _run_mode(
    workspace: Path,
    *,
    mode: str,
    cases: list[dict[str, Any]],
    timeout_s: int,
    subprocess_env: dict[str, str] | None,
) -> dict[str, Any]:
    cases_path = workspace / f"{mode}_cases.txt"
    cases_path.write_text(build_case_file(cases), encoding="utf-8")
    pipe_read, pipe_write = os.pipe()
    env = dict(subprocess_env or os.environ)
    env["POPCORN_FD"] = str(pipe_write)
    started_at = time.monotonic()
    completed: subprocess.CompletedProcess[str] | None = None
    timed_out = False
    timeout_message = ""
    try:
        completed = subprocess.run(
            [sys.executable, "eval.py", mode, str(cases_path)],
            cwd=workspace,
            env=env,
            pass_fds=[pipe_write],
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout_s)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        timeout_message = f"official {mode} evaluation timed out after {timeout_s}s"
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
    else:
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    finally:
        os.close(pipe_write)
    try:
        popcorn_output = os.fdopen(pipe_read, "r").read()
    finally:
        elapsed_s = time.monotonic() - started_at
    result = _parse_popcorn_output(popcorn_output)
    return {
        "mode": mode,
        "passed": not timed_out and result.get("check") == "pass",
        "returncode": None if completed is None else completed.returncode,
        "timed_out": timed_out,
        "error": timeout_message,
        "elapsed_s": elapsed_s,
        "result": result,
        "stdout": stdout[-8000:],
        "stderr": stderr[-8000:],
    }


def _benchmark_records(result: dict[str, str]) -> list[dict[str, Any]]:
    try:
        count = int(result["benchmark-count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("official leaderboard output has no valid benchmark-count") from exc
    records: list[dict[str, Any]] = []
    for index in range(count):
        prefix = f"benchmark.{index}"
        try:
            mean_ns = float(result[f"{prefix}.mean"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"official leaderboard output has no valid {prefix}.mean") from exc
        if not math.isfinite(mean_ns) or mean_ns <= 0:
            raise ValueError(f"official leaderboard returned invalid {prefix}.mean={mean_ns!r}")
        records.append(
            {
                "index": index,
                "spec": result.get(f"{prefix}.spec", ""),
                "runs": int(float(result.get(f"{prefix}.runs", "0"))),
                "mean_ns": mean_ns,
                "mean_us": mean_ns / 1000.0,
                "std_ns": float(result.get(f"{prefix}.std", "nan")),
                "best_ns": float(result.get(f"{prefix}.best", "nan")),
                "worst_ns": float(result.get(f"{prefix}.worst", "nan")),
            }
        )
    return records


def geometric_mean_runtime_us(records: list[dict[str, Any]]) -> float:
    if not records:
        raise ValueError("cannot score an empty benchmark set")
    log_sum = sum(math.log(float(record["mean_us"])) for record in records)
    return math.exp(log_sum / len(records))


def run_official_trimul_evaluation(
    code: str,
    *,
    evaluator_dir: str | Path,
    timeout_s: int = 1100,
    subprocess_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run the vendored TTT-Discover correctness and H100 leaderboard evaluator."""
    source_dir = Path(evaluator_dir)
    task_path = source_dir / "task.yml"
    if not task_path.exists():
        raise FileNotFoundError(f"TriMul evaluator task.yml not found under {source_dir}")
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    tests = list(task.get("tests") or [])
    benchmarks = list(task.get("benchmarks") or [])
    if len(tests) != 18 or len(benchmarks) != 7:
        raise ValueError(
            f"TriMul evaluator contract changed: expected 18 tests and 7 benchmarks, "
            f"got {len(tests)} and {len(benchmarks)}"
        )

    started_at = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="guidance-ttt-trimul-eval-") as tmp_dir:
        workspace = Path(tmp_dir) / "trimul"
        shutil.copytree(source_dir, workspace)
        (workspace / "submission.py").write_text(code, encoding="utf-8")
        test_run = _run_mode(
            workspace,
            mode="test",
            cases=tests,
            timeout_s=timeout_s,
            subprocess_env=subprocess_env,
        )
        leaderboard_run: dict[str, Any] | None = None
        records: list[dict[str, Any]] = []
        score_us: float | None = None
        error = ""
        if test_run["passed"]:
            leaderboard_run = _run_mode(
                workspace,
                mode="leaderboard",
                cases=benchmarks,
                timeout_s=timeout_s,
                subprocess_env=subprocess_env,
            )
            if leaderboard_run["passed"]:
                try:
                    records = _benchmark_records(leaderboard_run["result"])
                    score_us = geometric_mean_runtime_us(records)
                except ValueError as exc:
                    error = str(exc)
            else:
                error = str(leaderboard_run.get("error") or "leaderboard correctness/timing failed")
        else:
            error = str(test_run.get("error") or "public correctness tests failed")

    all_correct = bool(test_run["passed"] and leaderboard_run and leaderboard_run["passed"] and score_us)
    return {
        "report": {
            "all_correct": all_correct,
            "score_us": score_us,
            "ranking_by": "geom",
            "test_count": len(tests),
            "benchmark_count": len(benchmarks),
            "benchmarks": records,
            "test": test_run,
            "leaderboard": leaderboard_run,
            "error": error,
        },
        "provider": "official_trimul_evaluator",
        "elapsed_s": time.monotonic() - started_at,
    }
