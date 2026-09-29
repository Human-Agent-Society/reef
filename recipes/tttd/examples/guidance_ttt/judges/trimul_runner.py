"""Run the pinned TriMul evaluator and preserve its structured timing results."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal, TypedDict

import yaml


class TriMulCase(TypedDict):
    """Input fields from the pinned evaluator's task.yml, with upstream names."""

    seqlen: int
    bs: int
    dim: int
    hiddendim: int
    seed: int
    nomask: bool
    distribution: str


class EvaluationModeResult(TypedDict):
    mode: Literal["test", "leaderboard"]
    passed: bool
    returncode: int | None
    timed_out: bool
    error: str
    elapsed_s: float
    result: dict[str, str]
    stdout: str
    stderr: str


class BenchmarkRecord(TypedDict):
    index: int
    spec: str
    runs: int
    mean_ns: float
    mean_us: float
    std_ns: float
    best_ns: float
    worst_ns: float


class TriMulReport(TypedDict):
    all_correct: bool
    score_us: float | None
    ranking_by: Literal["geom"]
    test_count: int
    benchmark_count: int
    benchmarks: list[BenchmarkRecord]
    test: EvaluationModeResult
    leaderboard: EvaluationModeResult | None
    error: str


class TriMulEvaluationResult(TypedDict):
    report: TriMulReport
    provider: Literal["official_trimul_evaluator"]
    elapsed_s: float


def build_case_file(cases: list[TriMulCase]) -> str:
    """Serialize cases exactly as libkernelbot's build_test_string does."""
    lines = ["; ".join(f"{key}: {value}" for key, value in case.items()) for case in cases]
    return "\n".join(lines) + "\n"


def parse_popcorn_output(raw: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in raw.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip():
            parsed[key.strip()] = value.strip()
    return parsed


def run_mode(
    workspace: Path,
    *,
    mode: Literal["test", "leaderboard"],
    cases: list[TriMulCase],
    timeout_s: int,
    subprocess_env: dict[str, str] | None,
) -> EvaluationModeResult:
    cases_path = workspace / f"{mode}_cases.txt"
    cases_path.write_text(build_case_file(cases), encoding="utf-8")
    with tempfile.TemporaryFile(mode="w+") as output:
        pipe_write = output.fileno()
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
        output.seek(0)
        popcorn_output = output.read()
    elapsed_s = time.monotonic() - started_at
    result = parse_popcorn_output(popcorn_output)
    return {
        "mode": mode,
        "passed": not timed_out
        and completed is not None
        and completed.returncode == 0
        and result.get("check") == "pass",
        "returncode": None if completed is None else completed.returncode,
        "timed_out": timed_out,
        "error": timeout_message,
        "elapsed_s": elapsed_s,
        "result": result,
        "stdout": stdout[-8000:],
        "stderr": stderr[-8000:],
    }


def benchmark_records(result: dict[str, str]) -> list[BenchmarkRecord]:
    try:
        count = int(result["benchmark-count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("official leaderboard output has no valid benchmark-count") from exc
    records: list[BenchmarkRecord] = []
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


def geometric_mean_runtime_us(records: list[BenchmarkRecord]) -> float:
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
) -> TriMulEvaluationResult:
    """Run the vendored TTT-Discover correctness and H100 leaderboard evaluator."""
    source_dir = Path(evaluator_dir)
    task_path = source_dir / "task.yml"
    if not task_path.exists():
        raise FileNotFoundError(f"TriMul evaluator task.yml not found under {source_dir}")
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    tests: list[TriMulCase] = list(task.get("tests") or [])
    benchmarks: list[TriMulCase] = list(task.get("benchmarks") or [])
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
        test_run = run_mode(
            workspace,
            mode="test",
            cases=tests,
            timeout_s=timeout_s,
            subprocess_env=subprocess_env,
        )
        leaderboard_run: EvaluationModeResult | None = None
        records: list[BenchmarkRecord] = []
        score_us: float | None = None
        error = ""
        if test_run["passed"]:
            leaderboard_run = run_mode(
                workspace,
                mode="leaderboard",
                cases=benchmarks,
                timeout_s=max(1, int(timeout_s - (time.monotonic() - started_at))),
                subprocess_env=subprocess_env,
            )
            if leaderboard_run["passed"]:
                try:
                    records = benchmark_records(leaderboard_run["result"])
                    score_us = geometric_mean_runtime_us(records)
                except ValueError as exc:
                    error = str(exc)
            else:
                error = leaderboard_run["error"] or "leaderboard correctness/timing failed"
        else:
            error = test_run["error"] or "public correctness tests failed"

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
