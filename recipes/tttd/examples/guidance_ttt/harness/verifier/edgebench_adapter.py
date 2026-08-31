from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_VLIW_JUDGE_IMAGE = "seededge/edgebench.judge.vliw_kernel_optimization:5cdef0021634"
DEFAULT_VLIW_WORKSPACE = "/home/workspace/sebench_performance_takehome"
VLIW_RESCALE_BASELINE = 4475.526541978607
VLIW_RESCALE_EXPERT = 1000.0
VLIW_DEFAULT_REWARD_SCALE = 1_000_000.0


class EdgeBenchEnvironmentError(RuntimeError):
    """Raised when the configured EdgeBench judge cannot evaluate a submission."""


@dataclass(frozen=True)
class EdgeBenchResult:
    valid: bool
    cycles: float | None
    normalized_score: float
    message: str
    artifacts: dict[str, Any]


def rescale_vliw_cycles(cycles: float | None) -> float:
    """Apply EdgeBench's pinned log_min mapping and clip the result to [0, 100]."""
    if cycles is None or not math.isfinite(float(cycles)) or float(cycles) <= 0:
        return 0.0
    numerator = math.log(VLIW_RESCALE_BASELINE / float(cycles))
    denominator = math.log(VLIW_RESCALE_BASELINE / VLIW_RESCALE_EXPERT)
    return max(0.0, min(100.0, 100.0 * numerator / denominator))


def vliw_training_reward(
    cycles: float | None,
    *,
    mode: str = "inverse_cycles",
    scale: float = VLIW_DEFAULT_REWARD_SCALE,
) -> float:
    """Return a maximize-oriented RL reward while preserving cycles as the raw metric."""
    if cycles is None or not math.isfinite(float(cycles)) or float(cycles) <= 0:
        return 0.0
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "inverse_cycles":
        if not math.isfinite(float(scale)) or float(scale) <= 0:
            raise ValueError(f"reward_scale must be finite and positive, got {scale!r}")
        return float(scale) / float(cycles)
    if normalized_mode in {"official", "official_log_min"}:
        return rescale_vliw_cycles(cycles)
    raise ValueError(
        f"Unknown VLIW reward_mode {mode!r}; expected 'inverse_cycles' or 'official_log_min'"
    )


def evaluate_vliw_solution(
    code: str,
    *,
    timeout_s: int,
    config: dict[str, Any] | None = None,
) -> EdgeBenchResult:
    judge_config = dict(config or {})
    provider = str(judge_config.get("provider", "docker")).strip().lower()
    if provider in {"docker", "local_docker"}:
        payload = _evaluate_with_docker(code, timeout_s=timeout_s, config=judge_config)
    elif provider == "http":
        payload = _evaluate_with_http(code, timeout_s=timeout_s, config=judge_config)
    else:
        raise EdgeBenchEnvironmentError(f"Unknown EdgeBench verifier provider: {provider!r}")
    return _result_from_payload(payload)


def _evaluate_with_docker(code: str, *, timeout_s: int, config: dict[str, Any]) -> dict[str, Any]:
    docker_binary = str(config.get("docker_binary", "docker"))
    judge_image = str(config.get("judge_image", DEFAULT_VLIW_JUDGE_IMAGE))
    workspace = str(config.get("workspace", DEFAULT_VLIW_WORKSPACE)).rstrip("/")
    platform = str(config.get("platform", "linux/amd64"))
    cpus = str(config.get("cpus", 4))
    memory = str(config.get("memory", "8g"))
    runner_timeout = max(1, min(int(config.get("runner_timeout_s", 600)), int(timeout_s)))

    with tempfile.TemporaryDirectory(prefix="guidance-ttt-vliw-") as tmp_dir:
        submission_dir = Path(tmp_dir)
        solution_path = submission_dir / "solution.py"
        report_path = submission_dir / "report.json"
        solution_path.write_text(code)
        solution_path.chmod(0o644)
        command = [
            docker_binary,
            "run",
            "--rm",
            "--platform",
            platform,
            "--network",
            "none",
            "--cpus",
            cpus,
            "--memory",
            memory,
            "-v",
            f"{solution_path}:{workspace}/solution.py:ro",
            "-v",
            f"{submission_dir}:/submission-output",
            "--workdir",
            workspace,
            "--entrypoint",
            "python",
            judge_image,
            "runner.py",
            "--solution",
            "solution.py",
            "--cases",
            "test_cases/hidden_cases.json",
            "--output",
            "/submission-output/report.json",
        ]
        started_at = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=runner_timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise EdgeBenchEnvironmentError(f"Docker executable not found: {docker_binary}") from exc
        except subprocess.TimeoutExpired as exc:
            raise EdgeBenchEnvironmentError(
                f"EdgeBench Docker evaluation timed out after {runner_timeout}s"
            ) from exc
        elapsed_s = time.monotonic() - started_at
        if not report_path.exists():
            output = ((completed.stdout or "") + (completed.stderr or ""))[-4000:]
            raise EdgeBenchEnvironmentError(
                "EdgeBench judge did not produce report.json; "
                f"exit_code={completed.returncode}; output={output}"
            )
        try:
            report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise EdgeBenchEnvironmentError(f"Cannot parse EdgeBench report.json: {exc}") from exc
        return {
            "report": report,
            "runner_returncode": completed.returncode,
            "stdout": (completed.stdout or "")[-4000:],
            "stderr": (completed.stderr or "")[-4000:],
            "elapsed_s": elapsed_s,
            "provider": "docker",
            "judge_image": judge_image,
        }


def _evaluate_with_http(code: str, *, timeout_s: int, config: dict[str, Any]) -> dict[str, Any]:
    endpoint_env = str(config.get("endpoint_env", "VLIW_JUDGE_URL"))
    endpoint = str(config.get("endpoint") or os.environ.get(endpoint_env, "")).rstrip("/")
    if not endpoint:
        raise EdgeBenchEnvironmentError(
            f"EdgeBench HTTP verifier requires endpoint or environment variable {endpoint_env}"
        )
    api_key_env = str(config.get("api_key_env", "VLIW_JUDGE_TOKEN"))
    api_key = str(config.get("api_key") or os.environ.get(api_key_env, ""))
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(
            {
                "solution": code,
                "runner_timeout_s": int(config.get("runner_timeout_s", 600)),
            }
        ).encode(),
        headers=headers,
        method="POST",
    )
    request_timeout = max(1.0, min(float(config.get("request_timeout_s", timeout_s)), float(timeout_s)))
    max_retries = max(0, int(config.get("max_retries", 2)))
    retry_backoff_s = max(0.0, float(config.get("retry_backoff_s", 1.0)))
    started_at = time.monotonic()
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                payload = json.loads(response.read().decode())
            if not isinstance(payload, dict):
                raise EdgeBenchEnvironmentError(
                    f"EdgeBench HTTP verifier returned {type(payload).__name__}, expected an object"
                )
            payload.setdefault("provider", "http")
            payload.setdefault("elapsed_s", time.monotonic() - started_at)
            payload["http_attempts"] = attempt + 1
            return payload
        except urllib.error.HTTPError as exc:
            retryable = exc.code in {408, 429, 500, 502, 503, 504}
            if not retryable or attempt >= max_retries:
                body = exc.read().decode(errors="replace")[:2000]
                raise EdgeBenchEnvironmentError(
                    f"EdgeBench HTTP verifier failed with HTTP {exc.code}: {body}"
                ) from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            if attempt >= max_retries:
                raise EdgeBenchEnvironmentError(
                    f"EdgeBench HTTP verifier failed after {attempt + 1} attempts: {exc}"
                ) from exc
        time.sleep(retry_backoff_s * (2**attempt))
    raise AssertionError("unreachable")


def _result_from_payload(payload: dict[str, Any]) -> EdgeBenchResult:
    report = payload.get("report") if isinstance(payload.get("report"), dict) else payload
    cycles_value = report.get("score_cycles")
    valid = bool(report.get("all_correct") and cycles_value is not None)
    cycles = float(cycles_value) if valid else None
    if cycles is not None and (not math.isfinite(cycles) or cycles <= 0):
        valid = False
        cycles = None
    normalized_score = rescale_vliw_cycles(cycles)
    error = str(report.get("error") or "").strip()
    message = (
        f"cycles={cycles:g}; lower is better; EdgeBench score={normalized_score:.6f}"
        if valid and cycles is not None
        else f"invalid submission; no cycle score{': ' + error if error else ''}"
    )
    artifacts = {
        "provider": payload.get("provider"),
        "elapsed_s": payload.get("elapsed_s"),
        "runner_returncode": payload.get("runner_returncode"),
        "judge_image": payload.get("judge_image"),
        "best_cycles": report.get("best_cycles"),
        "passed_thresholds": report.get("passed_thresholds") or [],
        "results": report.get("results") or [],
        "stdout": payload.get("stdout", ""),
        "stderr": payload.get("stderr", ""),
        "http_attempts": payload.get("http_attempts"),
        "normalized_score": normalized_score,
    }
    return EdgeBenchResult(
        valid=valid,
        cycles=cycles,
        normalized_score=normalized_score,
        message=message,
        artifacts=artifacts,
    )
