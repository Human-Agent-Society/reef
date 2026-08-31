from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class TriMulEnvironmentError(RuntimeError):
    """Raised when the configured official H100 evaluator is unavailable."""


@dataclass(frozen=True)
class TriMulResult:
    valid: bool
    score_us: float | None
    message: str
    artifacts: dict[str, Any]


def evaluate_trimul_solution(
    code: str,
    *,
    timeout_s: int,
    config: dict[str, Any] | None = None,
) -> TriMulResult:
    judge_config = dict(config or {})
    provider = str(judge_config.get("provider", "http")).strip().lower()
    if provider != "http":
        raise TriMulEnvironmentError("TriMul evaluation requires an HTTP H100 verifier service")
    payload = _evaluate_with_http(code, timeout_s=timeout_s, config=judge_config)
    return _result_from_payload(payload)


def _evaluate_with_http(code: str, *, timeout_s: int, config: dict[str, Any]) -> dict[str, Any]:
    endpoint_env = str(config.get("endpoint_env", "TRIMUL_JUDGE_URL"))
    endpoint = str(config.get("endpoint") or os.environ.get(endpoint_env, "")).rstrip("/")
    if not endpoint:
        raise TriMulEnvironmentError(
            f"TriMul HTTP verifier requires endpoint or environment variable {endpoint_env}"
        )
    api_key_env = str(config.get("api_key_env", "TRIMUL_JUDGE_TOKEN"))
    api_key = str(config.get("api_key") or os.environ.get(api_key_env, ""))
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(
            {
                "solution": code,
                "runner_timeout_s": int(config.get("runner_timeout_s", 1100)),
            }
        ).encode(),
        headers=headers,
        method="POST",
    )
    request_timeout = max(1.0, min(float(config.get("request_timeout_s", timeout_s)), float(timeout_s)))
    max_retries = max(0, int(config.get("max_retries", 2)))
    retry_backoff_s = max(0.0, float(config.get("retry_backoff_s", 2.0)))
    started_at = time.monotonic()
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                payload = json.loads(response.read().decode())
            if not isinstance(payload, dict):
                raise TriMulEnvironmentError(
                    f"TriMul verifier returned {type(payload).__name__}, expected an object"
                )
            payload.setdefault("provider", "http")
            payload.setdefault("elapsed_s", time.monotonic() - started_at)
            payload["http_attempts"] = attempt + 1
            return payload
        except urllib.error.HTTPError as exc:
            retryable = exc.code in {408, 429, 500, 502, 503, 504}
            if not retryable or attempt >= max_retries:
                body = exc.read().decode(errors="replace")[:2000]
                raise TriMulEnvironmentError(
                    f"TriMul verifier failed with HTTP {exc.code}: {body}"
                ) from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            if attempt >= max_retries:
                raise TriMulEnvironmentError(
                    f"TriMul verifier failed after {attempt + 1} attempts: {exc}"
                ) from exc
        time.sleep(retry_backoff_s * (2**attempt))
    raise AssertionError("unreachable")


def _result_from_payload(payload: dict[str, Any]) -> TriMulResult:
    report = payload.get("report") if isinstance(payload.get("report"), dict) else payload
    raw_score = report.get("score_us")
    valid = bool(report.get("all_correct") and raw_score is not None)
    score_us = float(raw_score) if valid else None
    if score_us is not None and (not math.isfinite(score_us) or score_us <= 0):
        valid = False
        score_us = None
    error = str(report.get("error") or "").strip()
    message = (
        f"official H100 geometric-mean runtime={score_us:.6f} us; lower is better"
        if valid and score_us is not None
        else f"invalid TriMul submission; no runtime score{': ' + error if error else ''}"
    )
    artifacts = {
        "provider": payload.get("provider"),
        "elapsed_s": payload.get("elapsed_s"),
        "http_attempts": payload.get("http_attempts"),
        "ranking_by": report.get("ranking_by"),
        "evaluation_gpu": payload.get("evaluation_gpu"),
        "runtime_versions": payload.get("runtime_versions") or {},
        "test_count": report.get("test_count"),
        "benchmark_count": report.get("benchmark_count"),
        "benchmarks": report.get("benchmarks") or [],
        "test": report.get("test"),
        "leaderboard": report.get("leaderboard"),
    }
    return TriMulResult(valid=valid, score_us=score_us, message=message, artifacts=artifacts)
