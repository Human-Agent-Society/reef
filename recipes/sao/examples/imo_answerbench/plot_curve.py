"""Held-out accuracy against optimizer step, the paper's Figure 3 shape.

    python plot_curve.py out.png --evals work/evals --arms sao grpo --sets aime2025 hmmt_feb2025

Reads ``eval-<arm>-<step>-<set>.jsonl`` files written by evaluate.py (one per
kept checkpoint) plus ``eval-base-<set>.jsonl`` for the untrained model, and
draws one panel per benchmark: accuracy with a 95% Wilson interval at each
evaluated step, one line per arm, the base rate as a dotted line at step 0.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def accuracy(path: str) -> tuple[float, int, float, float]:
    with open(path) as handle:
        rows = [json.loads(line) for line in handle if '"score"' in line]
    k = sum(r["score"] for r in rows)
    lo, hi = wilson(k, len(rows))
    return (k / max(1, len(rows)), len(rows), lo, hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--evals", required=True)
    parser.add_argument("--arms", nargs="+", default=["sao", "grpo"])
    parser.add_argument("--sets", nargs="+", default=["aime2025", "hmmt_feb2025"])
    parser.add_argument("--labels", nargs="*", default=None, help="display names for --arms")
    parser.add_argument(
        "--title", default="SAO vs GRPO(+DIS), Qwen3-30B-A3B-Thinking-2507, batch 128, held-out accuracy"
    )
    args = parser.parse_args()
    labels = dict(zip(args.arms, args.labels or args.arms, strict=False))
    colors = {"sao": "tab:red", "grpo": "tab:blue"}

    fig, axes = plt.subplots(1, len(args.sets), figsize=(6.5 * len(args.sets), 4.8), squeeze=False)
    for ax, bench in zip(axes[0], args.sets, strict=False):
        base = os.path.join(args.evals, f"eval-base-{bench}.jsonl")
        if os.path.exists(base):
            acc, n, lo, hi = accuracy(base)
            ax.axhline(acc, color="k", ls=":", lw=1.5, label=f"base {acc:.3f} (n={n})")
            ax.axhspan(lo, hi, color="k", alpha=0.06)
        for arm in args.arms:
            points = []
            for path in glob.glob(os.path.join(args.evals, f"eval-{arm}-*-{bench}.jsonl")):
                match = re.search(rf"eval-{arm}-(\d+)-{bench}\.jsonl$", path)
                if match:
                    points.append((int(match.group(1)), *accuracy(path)))
            points.sort()
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            err = [[p[1] - p[3] for p in points], [p[4] - p[1] for p in points]]
            ax.errorbar(
                xs,
                ys,
                yerr=err,
                marker="o",
                capsize=3,
                lw=2,
                color=colors.get(arm),
                label=f"{labels[arm]} (n={points[0][2]} per point)",
            )
        ax.set_title(bench)
        ax.set_xlabel("optimizer step (128 rollouts each)")
        ax.set_ylabel("accuracy, 95% interval")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(args.title)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
