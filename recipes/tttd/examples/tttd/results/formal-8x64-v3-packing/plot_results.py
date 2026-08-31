#!/usr/bin/env python3
"""Generate the figures stored with the formal 8x64 packing results."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


ROOT = Path(__file__).resolve().parent
COLORS = {"packing26": "#2563a6", "packing32": "#d97706"}
LABELS = {"packing26": "Packing 26", "packing32": "Packing 32"}
TOLERANCE = 1e-12


def read_numeric_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {key: value if key in {"task", "reef/outcome"} else float(value) for key, value in row.items() if value != ""}
        for row in rows
    ]


def read_circles(task: str):
    rows = read_numeric_csv(ROOT / task / "circles.csv")
    expected = 26 if task == "packing26" else 32
    if len(rows) != expected:
        raise RuntimeError(f"{task} contains {len(rows)} circles; expected {expected}")
    for row in rows:
        radius = row["radius"]
        if radius < 0 or not all(math.isfinite(row[key]) for key in ("x", "y", "radius")):
            raise RuntimeError(f"{task} contains a negative or non-finite circle")
        margin = min(
            row["x"] - radius,
            row["y"] - radius,
            1 - row["x"] - radius,
            1 - row["y"] - radius,
        )
        if margin < -TOLERANCE:
            raise RuntimeError(f"{task} circle {int(row['circle'])} crosses the boundary")
    for first, circle in enumerate(rows):
        for second in range(first + 1, len(rows)):
            other = rows[second]
            distance = math.hypot(circle["x"] - other["x"], circle["y"] - other["y"])
            if distance < circle["radius"] + other["radius"] - TOLERANCE:
                raise RuntimeError(f"{task} circles {first} and {second} overlap")
    return rows


def certified_score(task: str):
    with (ROOT / task / "step-050.json").open(encoding="utf-8") as handle:
        return float(json.load(handle)["primary_metric_value"])


def plot_packings():
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.6), constrained_layout=True)
    for index, task in enumerate(("packing26", "packing32")):
        axis = axes[index]
        circles = read_circles(task)
        radii = [row["radius"] for row in circles]
        minimum, maximum = min(radii), max(radii)
        scale = maximum - minimum or 1
        for row in circles:
            shade = 0.18 + 0.64 * (row["radius"] - minimum) / scale
            axis.add_patch(
                Circle(
                    (row["x"], row["y"]),
                    row["radius"],
                    facecolor=plt.cm.Blues(shade),
                    edgecolor="#16324f",
                    linewidth=0.7,
                )
            )
        axis.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor="#202020", linewidth=1.2))
        axis.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), aspect="equal")
        axis.set_xticks([0, 0.5, 1])
        axis.set_yticks([0, 0.5, 1])
        axis.set_title(f"{LABELS[task]}\ncertified sum = {certified_score(task):.12f}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    figure.suptitle("Verified circle-packing configurations", fontsize=15)
    figure.savefig(ROOT / "packing_configurations.png", dpi=180)
    plt.close(figure)


def task_history(rows, task):
    selected = sorted((row for row in rows if row["task"] == task), key=lambda row: row["reef/step"])
    steps = [int(row["reef/step"]) for row in selected]
    if steps != list(range(1, 51)):
        raise RuntimeError(f"{task} W&B history does not contain committed steps 1 through 50")
    if any(row.get("reef/outcome") != "committed" for row in selected):
        raise RuntimeError(f"{task} W&B history contains a non-committed result")
    return selected


def plot_wandb_metrics():
    rows = read_numeric_csv(ROOT / "wandb_history.csv")
    histories = {task: task_history(rows, task) for task in ("packing26", "packing32")}
    panels = [
        ("rollout/rewards", "Mean rollout reward", lambda value: value),
        ("rollout/response_lengths", "Mean response length (tokens)", lambda value: value),
        ("train/ppo_kl", "Sampled-policy KL", lambda value: value),
        ("perf/step_time", "Step time (minutes)", lambda value: value / 60),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for index, (metric, title, transform) in enumerate(panels):
        axis = axes.flat[index]
        for task, history in histories.items():
            axis.plot(
                [row["reef/step"] for row in history],
                [transform(row[metric]) for row in history],
                color=COLORS[task],
                linewidth=1.5,
                marker="o",
                markersize=2.5,
                label=LABELS[task],
            )
        axis.set_title(title)
        axis.set_xlabel("Committed training step")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("REEF-TTT training metrics exported from W&B", fontsize=15)
    figure.savefig(ROOT / "wandb_training_metrics.png", dpi=180)
    plt.close(figure)


def main():
    plot_packings()
    plot_wandb_metrics()


if __name__ == "__main__":
    main()
