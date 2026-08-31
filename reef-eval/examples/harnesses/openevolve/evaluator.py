"""OpenEvolve evaluator backed exclusively by reef-eval's judge."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _candidate(program_path: str) -> dict:
    completed = subprocess.run(
        [sys.executable, program_path],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-1_000:] or "candidate exited non-zero")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("candidate printed no JSON solution")
    value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise TypeError("candidate JSON must be an object")
    return value


def _submit(solution: dict) -> dict:
    request = urllib.request.Request(
        os.environ["JUDGE_URL"].rstrip("/") + "/submit",
        data=json.dumps(solution).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        if error.code == 429:
            return {
                "score": 0.0,
                "remaining": 0,
                "error": "submission budget exhausted",
            }
        raise RuntimeError(f"judge returned HTTP {error.code}: {body}") from error


def evaluate(program_path: str) -> dict[str, float]:
    """Run one candidate and return the judge score OpenEvolve optimizes."""
    try:
        result = _submit(_candidate(program_path))
        return {"score": float(result["score"])}
    except Exception as error:
        print(f"candidate evaluation failed: {error}", file=sys.stderr)
        return {"score": 0.0}


if __name__ == "__main__":
    print(evaluate(str(Path(sys.argv[1]))))
