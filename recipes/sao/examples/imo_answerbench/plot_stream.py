"""Plot a streaming SAO run: training reward per optimizer step, then held-out accuracy.

    python plot_stream.py out.png --records work/records/stream-*.jsonl \
        --eval base=work/eval-base.jsonl sao=work/eval-sao-final.jsonl --batch 8

Left: the mean reward of the rollouts that fed each optimizer step (batches of
``--batch`` in scoring order) with a moving average over four steps; a dotted
line marks the pool's untrained rate if ``--pool-base`` is given. Right: the
held-out accuracy of each evaluated arm with a 95% Wilson interval. Training
reward on the training pool and held-out accuracy are different quantities;
the plot keeps them in separate panels on purpose.
"""

from __future__ import annotations

import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (center - half, center + half)


def load(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--records", nargs="+", required=True, help="stream.py record files, one arm per file")
    parser.add_argument("--eval", nargs="*", default=[], help="label=path pairs of evaluate.py outputs")
    parser.add_argument("--batch", type=int, default=8, help="rollouts per optimizer step")
    parser.add_argument("--pool-base", type=float, default=None, help="untrained pass rate on the training pool")
    parser.add_argument("--title", default="SAO, one rollout per prompt, batches of many prompts")
    args = parser.parse_args()

    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})
    for path in args.records:
        rows = sorted((r for r in load(path) if "score" in r), key=lambda r: r["recorded_at"])
        steps = [rows[i : i + args.batch] for i in range(0, len(rows) - len(rows) % args.batch, args.batch)]
        means = [sum(r["score"] for r in step) / len(step) for step in steps]
        xs = list(range(1, len(means) + 1))
        label = path.rsplit("/", 1)[-1].replace(".jsonl", "")
        left.scatter(xs, means, s=16, alpha=0.5)
        if len(means) >= 4:
            avg = [sum(means[i - 3 : i + 1]) / 4 for i in range(3, len(means))]
            left.plot(
                xs[3:],
                avg,
                lw=2,
                label=f"{label}: {int(sum(r['score'] for r in rows))}/{len(rows)} rollouts (4-step avg)",
            )
        else:
            left.plot([], [], lw=2, label=f"{label}: {int(sum(r['score'] for r in rows))}/{len(rows)} rollouts")
    if args.pool_base is not None:
        left.axhline(
            args.pool_base, color="k", ls=":", lw=1.5, label=f"untrained rate on the pool {args.pool_base:.3f}"
        )
    left.set_xlabel(f"optimizer step ({args.batch} rollouts each, in scoring order)")
    left.set_ylabel("mean reward of the step's rollouts")
    left.set_ylim(-0.03, 1.03)
    left.grid(alpha=0.3)
    left.legend(fontsize=8, loc="best")
    left.set_title("training reward on the training pool")

    labels, values, lows, highs, counts = [], [], [], [], []
    for pair in args.eval:
        label, path = pair.split("=", 1)
        rows = [r for r in load(path) if "score" in r]
        total = sum(r["score"] for r in rows)
        lo, hi = wilson(total, len(rows))
        labels.append(label)
        values.append(total / max(1, len(rows)))
        lows.append(values[-1] - lo)
        highs.append(hi - values[-1])
        counts.append(f"{int(total)}/{len(rows)}")
    if labels:
        xs = range(len(labels))
        right.bar(
            xs,
            values,
            yerr=[lows, highs],
            capsize=4,
            color=["k" if lab == "base" else "tab:red" for lab in labels],
            alpha=0.85,
        )
        for x, c in zip(xs, counts, strict=False):
            right.text(x, 0.02, c, ha="center", color="w", fontsize=9, rotation=90)
        right.set_xticks(list(xs))
        right.set_xticklabels(labels)
    right.set_ylim(0, 1.0)
    right.set_title("held-out problems, fresh samples, 95% interval")
    right.grid(axis="y", alpha=0.3)
    fig.suptitle(args.title)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
