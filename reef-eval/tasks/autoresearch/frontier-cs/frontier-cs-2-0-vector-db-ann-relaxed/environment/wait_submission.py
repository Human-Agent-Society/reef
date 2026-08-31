#!/usr/bin/env python3
"""Wait for one Frontier-CS 2.0 async submission."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

JUDGE_URL = os.environ.get("JUDGE_URL", "http://judge:8082").rstrip("/")
SUBMISSIONS_LOG = Path("/logs/agent/submissions.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_uuid")
    parser.add_argument("--timeout", type=float, default=0.0, help="seconds; 0 means wait forever")
    parser.add_argument("--json", action="store_true", help="print raw JSON result")
    parser.add_argument("--log-agent", action="store_true", help="mirror status changes to /logs/agent/submissions.jsonl")
    parser.add_argument("--quiet", action="store_true", help="suppress progress text")
    return parser.parse_args()


def now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def log_agent_record(record: dict) -> None:
    payload = dict(record)
    payload.setdefault("ts", now_iso())
    SUBMISSIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SUBMISSIONS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def fetch(submission_uuid: str) -> dict:
    response = requests.get(f"{JUDGE_URL}/submission/{submission_uuid}", timeout=10)
    response.raise_for_status()
    payload = response.json()
    submission = payload.get("submission")
    if not isinstance(submission, dict):
        raise RuntimeError(str(payload))
    return submission


def print_result(record: dict, *, raw_json: bool) -> None:
    if raw_json:
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return
    status = record.get("status")
    print(f"[wait] uuid={record.get('submission_uuid')} status={status}")
    if status == "done":
        score = float(record.get("score", 0.0))
        reward = score / 100.0
        print(f"[wait] score={score:.4f}/100 reward={reward:.4f}")
        if record.get("score_unbounded") != score:
            print(f"[wait] unbounded={record.get('score_unbounded')}")
        message = str(record.get("message", ""))
        if message:
            print(f"[wait] detail: {message[:800]}")
        metrics = record.get("metrics", {})
        if isinstance(metrics, dict) and metrics:
            metric_text = " ".join(f"{key}={value}" for key, value in sorted(metrics.items()))
            print(f"[wait] metrics: {metric_text}")


def main() -> int:
    args = parse_args()
    deadline = time.time() + args.timeout if args.timeout > 0 else None
    last_status: str | None = None
    while True:
        record = fetch(args.submission_uuid)
        status = str(record.get("status", "unknown"))
        if status != last_status:
            if args.log_agent:
                log_agent_record(record)
            if not args.json and not args.quiet:
                print(f"[wait] status={status}", flush=True)
            last_status = status
        if status in {"done", "error", "cancelled"}:
            if not args.quiet:
                print_result(record, raw_json=args.json)
            return 0 if status == "done" else 1
        if deadline is not None and time.time() >= deadline:
            if not args.quiet:
                print_result(record, raw_json=args.json)
            return 124
        time.sleep(1)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
