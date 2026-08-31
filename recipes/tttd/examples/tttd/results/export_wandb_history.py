#!/usr/bin/env python3
"""Export the numeric Reef training history from resumed W&B run files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


FIELDS = [
    "task",
    "reef/step",
    "reef/run_step",
    "reef/outcome",
    "rollout/rewards",
    "rollout/kl",
    "rollout/response_lengths",
    "rollout/truncated",
    "train/importance_ratio",
    "train/grad_norm",
    "train/ppo_kl",
    "perf/step_time",
    "perf/train_time",
    "perf/train_wait_time",
    "perf/wait_time_ratio",
    "perf/actor_train_tok_per_s",
]


def parse_value(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def history_rows(path: Path):
    store = DataStore()
    store.open_for_scan(str(path))
    while data := store.scan_data():
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if not record.HasField("history"):
            continue
        row = {}
        for item in record.history.item:
            key = item.key or "/".join(item.nested_key)
            if key:
                row[key] = parse_value(item.value_json)
        if "reef/step" in row:
            yield row


def export_task(task: str, root: Path):
    by_step = {}
    run_files = sorted(root.glob("run-*/run-*.wandb"))
    if not run_files:
        raise FileNotFoundError(f"No W&B run files found below {root}")
    for run_file in run_files:
        for row in history_rows(run_file):
            step = int(row["reef/step"])
            current = by_step.setdefault(step, {})
            current.update({key: value for key, value in row.items() if value is not None})
    for step in sorted(by_step):
        yield {"task": task, **by_step[step]}


def parse_task(value: str):
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", type=parse_task, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int)
    args = parser.parse_args()

    rows = []
    for task, root in args.task:
        task_rows = list(export_task(task, root))
        if args.expected_steps is not None and len(task_rows) != args.expected_steps:
            raise RuntimeError(f"{task} has {len(task_rows)} committed W&B rows; expected {args.expected_steps}")
        rows.extend(task_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
