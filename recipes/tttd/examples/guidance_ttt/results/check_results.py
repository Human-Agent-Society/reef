#!/usr/bin/env python3
"""Validate the compact Guidance-TTT result records."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent


def validate(results_dir: Path = RESULTS_DIR) -> int:
    payload = json.loads((results_dir / "runs.json").read_text())
    runs = {run["task"]: run for run in payload["runs"]}
    if set(runs) != {"polyomino", "trimul"}:
        raise AssertionError("expected exactly the Polyomino and TriMul runs")

    repeat = runs["trimul"]["fixed_candidate_reevaluation"]
    scores = repeat["scores_microseconds"]
    if not math.isclose(statistics.fmean(scores), repeat["mean_microseconds"]):
        raise AssertionError("TriMul repeat mean does not match")
    if not math.isclose(statistics.stdev(scores), repeat["sample_std_microseconds"]):
        raise AssertionError("TriMul repeat standard deviation does not match")
    if not repeat["all_correct"]:
        raise AssertionError("TriMul fixed-candidate repeats were not all correct")

    return len(runs)


if __name__ == "__main__":
    run_count = validate()
    print(f"validated {run_count} runs")
