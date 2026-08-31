#!/usr/bin/env python3
"""Iterative submission helper for Frontier-CS Harbor tasks.

The agent runs this whenever it wants to test the solution against the
graded judge. Each call:

1. Writes a `started` record to /logs/agent/submissions.jsonl (so we know the
   call existed even if everything downstream fails).
2. POSTs /app/solution.cpp (or the first arg) to the black-box judge sidecar.
3. Writes a `done`/`error` record with the same submission_uuid, then prints
   a concise score + feedback summary so the agent sees it on stdout.

Exit code semantics:
- 0: judge returned a result (score may be low — that's still success).
- 2: solution file missing / empty.
- 3: judge request failed.
- 5: unexpected runtime error.

The post-hoc analysis joins submissions.jsonl with the agent's trajectory
file by ordinal to attach tokens_so_far. See ../../../../../analysis/.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

JUDGE_URL = os.environ.get("JUDGE_URL", "http://judge:8082").rstrip("/")
PROBLEM_ID = os.environ.get("PROBLEM_ID", "")
SUBMISSIONS_LOG = Path("/logs/agent/submissions.jsonl")
JUDGE_TIMEOUT_SECONDS = float(os.environ.get("SUBMIT_MAX_POLL_TIME", "600"))


def now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def log_record(record: dict) -> None:
    SUBMISSIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SUBMISSIONS_LOG.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def wait_for_judge() -> None:
    deadline = time.time() + JUDGE_TIMEOUT_SECONDS
    last_error: Exception | None = None
    printed_waiting = False
    while time.time() < deadline:
        try:
            response = requests.get(f"{JUDGE_URL}/health", timeout=5)
            if response.status_code == 200:
                return
            last_error = RuntimeError(response.text)
        except Exception as exc:
            last_error = exc
        if not printed_waiting:
            print("[submit] waiting for judge service", flush=True)
            printed_waiting = True
        time.sleep(1)
    raise RuntimeError(f"judge service is not ready at {JUDGE_URL}: {last_error}")


def evaluate_with_judge(submission_uuid: str, code: str) -> dict:
    wait_for_judge()
    response = requests.post(
        f"{JUDGE_URL}/evaluate",
        json={
            "submission_uuid": submission_uuid,
            "submission_role": "agent",
            "problem_id": PROBLEM_ID,
            "code": code,
        },
        timeout=JUDGE_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("status") not in ("done", "error"):
        raise RuntimeError(str(result.get("message") or result))
    return result


def main() -> int:
    solution_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/app/solution.cpp")
    sub_uuid = str(uuid.uuid4())
    code_chars = 0

    log_record(
        {
            "submission_uuid": sub_uuid,
            "ts": now_iso(),
            "status": "started",
            "problem_id": PROBLEM_ID,
            "solution_path": str(solution_path),
            "code_chars": code_chars,
        }
    )

    if not solution_path.exists():
        msg = f"Solution file {solution_path} does not exist"
        print(f"[submit] ERROR: {msg}", file=sys.stderr)
        log_record(
            {
                "submission_uuid": sub_uuid,
                "ts": now_iso(),
                "status": "error",
                "error": msg,
            }
        )
        return 2

    code = solution_path.read_text()
    code_chars = len(code)
    if not code.strip():
        msg = f"Solution file {solution_path} is empty"
        print(f"[submit] ERROR: {msg}", file=sys.stderr)
        log_record(
            {
                "submission_uuid": sub_uuid,
                "ts": now_iso(),
                "status": "error",
                "error": msg,
                "code_chars": code_chars,
            }
        )
        return 2

    try:
        start = time.time()
        result = evaluate_with_judge(sub_uuid, code)
        elapsed_seconds = time.time() - start
    except Exception as exc:
        msg = f"Judge request failed: {exc}"
        print(f"[submit] ERROR: {msg}", file=sys.stderr)
        log_record(
            {
                "submission_uuid": sub_uuid,
                "ts": now_iso(),
                "status": "error",
                "error": msg,
            }
        )
        return 3

    score_raw = result.get("score") or 0.0
    try:
        score = float(score_raw) / 100.0
    except (TypeError, ValueError):
        score = 0.0
    status = result.get("status", "unknown")
    detail = result.get("message") or ""
    score_unbounded = result.get("score_unbounded", score_raw)
    metrics = result.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}

    log_record(
        {
            "submission_uuid": sub_uuid,
            "ts": now_iso(),
            "status": status,
            "score": score,
            "score_raw": score_raw,
            "score_unbounded": score_unbounded,
            "detail": detail,
            "elapsed_seconds": elapsed_seconds,
            "code_chars": code_chars,
            "metrics": metrics,
        }
    )

    print(f"[submit] uuid={sub_uuid}")
    print(
        f"[submit] status={status} score={score:.4f} "
        f"(raw={score_raw}/100) code_chars={code_chars}"
    )
    if score_unbounded != score_raw:
        print(f"[submit] unbounded={score_unbounded}")
    if detail:
        snippet = detail if len(detail) <= 500 else detail[:500] + "..."
        print(f"[submit] detail: {snippet}")
    if metrics:
        metric_text = " ".join(
            f"{key}={value}" for key, value in sorted(metrics.items())
        )
        print(f"[submit] metrics: {metric_text}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"[submit] FATAL: {exc!r}", file=sys.stderr)
        sys.exit(5)
