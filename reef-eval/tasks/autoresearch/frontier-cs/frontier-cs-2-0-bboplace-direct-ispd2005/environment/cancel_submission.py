#!/usr/bin/env python3
"""Cancel a queued or running Frontier-CS 2.0 async submission."""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests

JUDGE_URL = os.environ.get("JUDGE_URL", "http://judge:8082").rstrip("/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_uuid")
    parser.add_argument("--json", action="store_true", help="print raw JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    response = requests.post(
        f"{JUDGE_URL}/submission/{args.submission_uuid}/cancel",
        json={},
        timeout=10,
    )
    try:
        payload = response.json()
    except Exception:
        payload = {"status": "error", "error": response.text}
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif response.status_code == 200:
        print(f"[cancel] uuid={args.submission_uuid} status=cancelled")
    else:
        print(f"[cancel] failed: {payload}", file=sys.stderr)
    return 0 if response.status_code == 200 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
