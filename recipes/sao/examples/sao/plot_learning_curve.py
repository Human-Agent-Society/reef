#!/usr/bin/env python3
"""Plot the SAO example's learning curve directly from the retained records.

Reads ``rollouts.jsonl`` — one JSON object per scored rollout, written by
``harness/agent.py`` as the rollout is graded — and renders three things
side by side so a reviewer can audit the figure without trusting a summary:

1. Raw per-rollout outcome (0 or 1) as a scatter, coloured by task.
2. The running mean of the outcome sequence.
3. A 95% bootstrap confidence band on the mean at each rollout index.

Also prints per-arm summary and the release-id sequence, and marks each new
serving version with a vertical line so it is obvious *which* rollouts trained
against the same weights and which trained against fresh ones.

Usage:
    python plot_learning_curve.py <records.jsonl> [--output plot.png]

The records module ships alongside the example, so a reproducer never has to
guess how a figure was produced.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_records(path: Path) -> list[dict]:
    """Read one record per line, oldest first, dropping empties and unscored lines."""
    records: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        entry = json.loads(line)
        if entry.get("score") is None:
            continue
        records.append(entry)
    records.sort(key=lambda item: item.get("recorded_at", 0.0))
    return records


def _short_gold(gold: str) -> str:
    """Compact task label derived from the gold answer."""
    if not gold:
        return "?"
    return gold.strip("$").strip("\\").replace("frac", "").replace("{", "").replace("}", "")[:14]


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Per-prefix bootstrap CI on the cumulative mean; returns (low, high)."""
    rng = np.random.default_rng(0)
    n = len(values)
    if n == 0:
        return np.array([]), np.array([])
    boot = np.empty((n_boot, n))
    for boot_index in range(n_boot):
        resampled = rng.choice(values, size=n, replace=True)
        boot[boot_index] = np.cumsum(resampled) / np.arange(1, n + 1)
    low = np.quantile(boot, alpha / 2, axis=0)
    high = np.quantile(boot, 1 - alpha / 2, axis=0)
    return low, high


def plot(records: list[dict], output: Path, title: str) -> None:
    if not records:
        raise SystemExit("no records to plot")
    scores = np.array([float(r["score"]) for r in records])
    # Fall back to gold answer when the context did not carry a task name.
    tasks = [r.get("task_name", "?") if r.get("task_name") not in (None, "unknown", "?") else _short_gold(r.get("gold", "?")) for r in records]
    serving = [r.get("serving_release_id") for r in records]

    n = len(scores)
    indices = np.arange(1, n + 1)
    cum = np.cumsum(scores) / indices
    low, high = bootstrap_ci(scores)

    fig, ax = plt.subplots(figsize=(10, 5))
    task_names = sorted(set(tasks))
    palette = plt.get_cmap("tab10")
    for task_index, task_name in enumerate(task_names):
        mask = np.array([task == task_name for task in tasks])
        colour = palette(task_index)
        ax.scatter(indices[mask], scores[mask], marker="|", s=140, color=colour, label=f"{task_name} raw")
    ax.plot(indices, cum, color="black", linewidth=2, label="running mean")
    ax.fill_between(indices, low, high, color="black", alpha=0.15, label="95% bootstrap CI")

    # Mark each new serving release with a vertical line so the reader can see
    # exactly which rollouts trained after a weight update.
    seen_releases: set[str] = set()
    for index, release_id in enumerate(serving, start=1):
        if release_id and release_id not in seen_releases:
            seen_releases.add(release_id)
            ax.axvline(index, color="gray", linewidth=0.5, alpha=0.4)

    ax.set_ylim(-0.08, 1.08)
    ax.set_xlim(0.5, n + 0.5)
    ax.set_xlabel("Scored rollouts (chronological)")
    ax.set_ylabel("Reward (0 or 1)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def summarize(records: list[dict]) -> None:
    per_task: dict[str, list[float]] = defaultdict(list)
    per_version: dict[str, list[float]] = defaultdict(list)
    for record in records:
        per_task[record.get("task_name", "?")].append(float(record["score"]))
        per_version[str(record.get("serving_release_id"))].append(float(record["score"]))
    total = np.array([float(record["score"]) for record in records])
    print(f"records: {len(records)}  mean_reward: {total.mean():.4f}")
    for task_name, values in sorted(per_task.items()):
        array = np.array(values)
        print(f"  {task_name}: n={len(array)} mean={array.mean():.4f}")
    print("serving versions encountered:", len(per_version))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--output", type=Path, default=Path("learning_curve.png"))
    parser.add_argument("--title", default="SAO learning curve (raw rollouts with 95% bootstrap CI)")
    args = parser.parse_args()
    records = load_records(args.records)
    summarize(records)
    plot(records, args.output, args.title)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
