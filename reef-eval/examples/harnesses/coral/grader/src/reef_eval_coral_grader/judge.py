"""Small stdlib client for reef-eval's judge protocol."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path


class SubmissionBudgetExhausted(RuntimeError):
    pass


def submit_file(solution_file: Path, judge_url: str | None = None) -> dict:
    solution = json.loads(solution_file.read_text())
    if not isinstance(solution, dict):
        raise TypeError("solution.json must contain a JSON object")
    url = (judge_url or os.environ["JUDGE_URL"]).rstrip("/") + "/submit"
    request = urllib.request.Request(
        url,
        data=json.dumps(solution).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        if error.code == 429:
            raise SubmissionBudgetExhausted(
                "reef-eval submission budget exhausted"
            ) from error
        raise RuntimeError(
            f"reef-eval judge returned HTTP {error.code}: {body}"
        ) from error
