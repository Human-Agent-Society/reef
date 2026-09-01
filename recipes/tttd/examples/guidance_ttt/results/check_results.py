#!/usr/bin/env python3
"""Validate the compact Guidance-TTT result records."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent


def validate(results_dir: Path = RESULTS_DIR) -> tuple[int, int]:
    payload = json.loads((results_dir / "runs.json").read_text())
    runs = {run["task"]: run for run in payload["runs"]}
    if set(runs) != {"polyomino", "trimul"}:
        raise AssertionError("expected exactly the Polyomino and TriMul runs")

    with (results_dir / "trajectory.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    for task, run in runs.items():
        task_rows = [row for row in rows if row["task"] == task]
        updates = run["search"]["updates"]
        if len(task_rows) != updates + 1:
            raise AssertionError(f"{task}: expected a baseline and {updates} updates")
        if [int(row["update"]) for row in task_rows] != list(range(updates + 1)):
            raise AssertionError(f"{task}: update numbers are not contiguous")

        values = [float(row["cumulative_best"]) for row in task_rows]
        if not math.isclose(values[0], run["search"]["seed_score"]):
            raise AssertionError(f"{task}: baseline does not match runs.json")
        if not math.isclose(values[-1], run["search"]["best_score"]):
            raise AssertionError(f"{task}: final trajectory value does not match runs.json")

        direction = run["search"]["score_direction"]
        pairs = zip(values, values[1:])
        if direction == "max" and any(current > following for current, following in pairs):
            raise AssertionError(f"{task}: cumulative best score decreased")
        if direction == "min" and any(current < following for current, following in pairs):
            raise AssertionError(f"{task}: cumulative best latency increased")

        sampled = sum(int(row["sampled_rollouts"]) for row in task_rows[1:])
        valid = sum(int(row["valid_rollouts"]) for row in task_rows[1:])
        if sampled != run["search"]["sampled_rollouts"]:
            raise AssertionError(f"{task}: sampled-rollout total does not match")
        if valid != run["search"]["valid_rollouts"]:
            raise AssertionError(f"{task}: valid-rollout total does not match")

    repeat = runs["trimul"]["fixed_candidate_reevaluation"]
    scores = repeat["scores_microseconds"]
    if not math.isclose(statistics.fmean(scores), repeat["mean_microseconds"]):
        raise AssertionError("TriMul repeat mean does not match")
    if not math.isclose(statistics.stdev(scores), repeat["sample_std_microseconds"]):
        raise AssertionError("TriMul repeat standard deviation does not match")
    if not repeat["all_correct"]:
        raise AssertionError("TriMul fixed-candidate repeats were not all correct")

    return len(runs), len(rows)


if __name__ == "__main__":
    run_count, row_count = validate()
    print(f"validated {run_count} runs and {row_count} trajectory rows")
