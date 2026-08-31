#!/usr/bin/env python3
"""List Frontier-CS 2.0 async submissions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import requests

JUDGE_URL = os.environ.get("JUDGE_URL", "http://judge:8082").rstrip("/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print raw JSON")
    return parser.parse_args()


def score_text(record: dict) -> str:
    if record.get("status") != "done":
        return "-"
    try:
        return f"{float(record.get('score', 0.0)):.2f}"
    except (TypeError, ValueError):
        return "-"


def main() -> int:
    args = parse_args()
    response = requests.get(f"{JUDGE_URL}/submissions", timeout=10)
    response.raise_for_status()
    payload = response.json()
    submissions = payload.get("submissions", [])
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if not submissions:
        print("No submissions yet.")
        return 0
    print("uuid                                  status     score    code_chars  submitted_at")
    for record in submissions:
        if not isinstance(record, dict):
            continue
        uuid = str(record.get("submission_uuid", ""))[:36]
        status = str(record.get("status", "unknown"))[:10]
        code_chars = record.get("code_chars")
        code_text = str(code_chars) if code_chars is not None else "-"
        timestamp = record.get("ts") or datetime.now(timezone.utc).isoformat()
        print(f"{uuid:36}  {status:10} {score_text(record):>7} {code_text:>11}  {timestamp}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
