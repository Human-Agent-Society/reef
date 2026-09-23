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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation_stats import load_jsonl, wilson_interval


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
        rows = sorted((row for row in load_jsonl(path) if "score" in row), key=lambda row: row["recorded_at"])
        steps = [
            rows[start : start + args.batch] for start in range(0, len(rows) - len(rows) % args.batch, args.batch)
        ]
        means = [sum(row["score"] for row in step) / len(step) for step in steps]
        step_numbers = list(range(1, len(means) + 1))
        label = path.rsplit("/", 1)[-1].replace(".jsonl", "")
        correct = int(sum(row["score"] for row in rows))
        left.scatter(step_numbers, means, s=16, alpha=0.5)
        if len(means) >= 4:
            moving_average = [sum(means[index - 3 : index + 1]) / 4 for index in range(3, len(means))]
            left.plot(
                step_numbers[3:], moving_average, lw=2, label=f"{label}: {correct}/{len(rows)} rollouts (4-step avg)"
            )
        else:
            left.plot([], [], lw=2, label=f"{label}: {correct}/{len(rows)} rollouts")
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
        rows = [row for row in load_jsonl(path) if "score" in row]
        total = sum(row["score"] for row in rows)
        low, high = wilson_interval(total, len(rows))
        labels.append(label)
        values.append(total / max(1, len(rows)))
        lows.append(values[-1] - low)
        highs.append(high - values[-1])
        counts.append(f"{int(total)}/{len(rows)}")
    if labels:
        positions = range(len(labels))
        right.bar(
            positions,
            values,
            yerr=[lows, highs],
            capsize=4,
            color=["k" if label == "base" else "tab:red" for label in labels],
            alpha=0.85,
        )
        for position, count in zip(positions, counts, strict=False):
            right.text(position, 0.02, count, ha="center", color="w", fontsize=9, rotation=90)
        right.set_xticks(list(positions))
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
