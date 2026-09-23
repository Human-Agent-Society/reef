"""SAO vs GRPO(+DIS) at the paper's batch: held-out accuracy and training dynamics against optimizer step.

    python plot_paper.py out.png --evals evals/ --records sao-a=records/sao-a.jsonl grpo-a=records/grpo-a.jsonl

Top row: held-out accuracy of the kept checkpoints (``eval-<run>-<step>-<set>.jsonl`` from
evaluate.py) on AIME 2025, HMMT Feb 2025 and IMO-AnswerBench, 95% Wilson intervals, the
untrained model as a dotted line. Bottom row: the mean response length and mean reward of the
training rollouts, grouped by the number of optimizer steps the trainer had completed when each
rollout was scored (``--steps run=slime-driver.log`` supplies the step timestamps; without it,
rollouts are grouped by position in batches of 128), smoothed with a trailing four-step mean.
Held-out accuracy
and training reward are different quantities and stay in separate panels. Runs whose name
starts with ``sao`` are red, ``grpo`` blue; run-1 arms are dashed.
"""

from __future__ import annotations

import argparse
import bisect
import glob
import os
import re
from collections import defaultdict
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation_stats import load_jsonl, wilson_interval

SETS = [
    ("aime2025", "AIME 2025 (30 problems x 8)"),
    ("hmmt_feb2025", "HMMT Feb 2025 (30 problems x 8)"),
    ("imo_answerbench", "IMO-AnswerBench (400 problems)"),
]

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP_LINE = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\].*'train/step': (\d+)")


def accuracy(path: str, expected: int) -> tuple[float, int, float, float] | None:
    """(accuracy, rows, low, high) of one evaluation file, or None while it is still incomplete."""
    rows = [row for row in load_jsonl(path) if "score" in row]  # rows with "error" are retried by evaluate.py
    if len(rows) < expected:
        return None
    correct = sum(1 for row in rows if row["score"] == 1)
    low, high = wilson_interval(correct, len(rows))
    return correct / len(rows), len(rows), low, high


def step_times(log_path: str) -> list[float]:
    """First log time (UTC epoch) of each optimizer step, actor or critic-only, from the slime driver log.

    Checkpoints are numbered by the same counter (hf/<step>), so SAO's critic warmup steps count."""
    seen: dict[int, float] = {}
    with open(log_path, errors="replace") as handle:
        for line in handle:
            match = STEP_LINE.search(ANSI.sub("", line))
            if match:
                stamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp()
                seen.setdefault(int(match.group(2)), stamp)
    return [seen[step] for step in sorted(seen)]


def smooth(values: list[float], window: int = 4) -> list[float]:
    """Trailing mean over ``window`` steps; a 128-rollout step is noisy on its own."""
    return [
        sum(values[max(0, index - window + 1) : index + 1]) / len(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]


def style(run: str, overrides: dict[str, dict]) -> dict:
    """A run's line style: its ``--style`` entry, else red for SAO and blue for GRPO, dashed for run-1 arms."""
    if run in overrides:
        return overrides[run]
    color = "tab:red" if run.startswith("sao") or "-sao" in run else "tab:blue"
    line_style = "--" if run.startswith("run1") else "-"
    marker = "o" if line_style == "-" else "s"
    return {"color": color, "ls": line_style, "marker": marker}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--evals", required=True)
    parser.add_argument("--records", nargs="*", default=[], help="run=path pairs of stream.py record files")
    parser.add_argument(
        "--steps", nargs="*", default=[], help="run=slime-driver.log pairs giving optimizer-step completion times"
    )
    parser.add_argument("--labels", nargs="*", default=[], help="run=display name pairs")
    parser.add_argument("--style", nargs="*", default=[], help="run=color:linestyle:marker, e.g. sao-h2=salmon:-:^")
    parser.add_argument(
        "--offset", nargs="*", default=[], help="run=steps pairs added to a run's step numbers (continuations)"
    )
    parser.add_argument(
        "--imo-rows", type=int, default=800, help="rows needed for an IMO-AnswerBench point (400 problems x runs)"
    )
    parser.add_argument(
        "--title", default="SAO vs GRPO(+DIS), Qwen3-30B-A3B-Thinking-2507, 128 rollouts per optimizer step"
    )
    args = parser.parse_args()
    labels = dict(pair.split("=", 1) for pair in args.labels)
    step_logs = dict(pair.split("=", 1) for pair in args.steps)
    offsets = {run: int(shift) for run, shift in (pair.split("=", 1) for pair in args.offset)}
    styles: dict[str, dict] = {}
    for pair in args.style:
        run, spec = pair.split("=", 1)
        color, line_style, marker = spec.rsplit(":", 2)
        styles[run] = {"color": color, "ls": line_style, "marker": marker}
    expected = {"aime2025": 240, "hmmt_feb2025": 240, "imo_answerbench": args.imo_rows}

    fig, axes = plt.subplots(2, 3, figsize=(19, 9.5))

    runs: set[str] = set()
    for axis, (bench, title) in zip(axes[0], SETS, strict=True):
        base_path = os.path.join(args.evals, f"eval-base-{bench}.jsonl")
        base_point = accuracy(base_path, expected[bench]) if os.path.exists(base_path) else None
        if base_point:
            axis.axhline(base_point[0], color="k", ls=":", lw=1.5, label=f"untrained {base_point[0]:.3f}")
            axis.axhspan(base_point[2], base_point[3], color="k", alpha=0.06)
        points: dict[str, list] = defaultdict(list)
        for path in glob.glob(os.path.join(args.evals, f"eval-*-{bench}.jsonl")):
            match = re.search(rf"eval-(.+)-(\d+)-{bench}\.jsonl$", os.path.basename(path))
            if not match:
                continue
            point = accuracy(path, expected[bench])
            if point:
                run = match.group(1)
                points[run].append((int(match.group(2)) + offsets.get(run, 0), *point))
        for run in sorted(points):
            runs.add(run)
            run_points = sorted(points[run])
            steps = [point[0] for point in run_points]
            accuracies = [point[1] for point in run_points]
            error_bars = [[point[1] - point[3] for point in run_points], [point[4] - point[1] for point in run_points]]
            axis.errorbar(
                steps, accuracies, yerr=error_bars, capsize=3, lw=2, label=labels.get(run, run), **style(run, styles)
            )
        axis.set_title(title)
        axis.set_xlabel("optimizer step")
        axis.set_ylabel("held-out accuracy, 95% interval")
        axis.set_xlim(left=-3)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7.5, loc="lower left")

    length_axis, reward_axis, truncation_axis = axes[1]
    for pair in args.records:
        run, path = pair.split("=", 1)
        rows = [row for record_path in sorted(glob.glob(path)) for row in load_jsonl(record_path)]
        if not rows:
            continue
        by_step: dict[int, list] = defaultdict(list)
        if run in step_logs:
            times = step_times(step_logs[run])
            for row in rows:
                by_step[bisect.bisect_right(times, row["recorded_at"])].append(row)
        else:
            for index, row in enumerate(sorted(rows, key=lambda record: record.get("recorded_at", 0))):
                by_step[index // 128].append(row)
        steps = sorted(step for step in by_step if len(by_step[step]) >= 32)
        shift = offsets.get(run, 0)
        length = [sum(row["completion_tokens"] for row in by_step[step]) / len(by_step[step]) for step in steps]
        reward = [sum(row["score"] for row in by_step[step]) / len(by_step[step]) for step in steps]
        truncation = [
            sum(1 for row in by_step[step] if row.get("finish_reason") == "length") / len(by_step[step])
            for step in steps
        ]
        run_style = style(run, styles)
        length, reward, truncation = (smooth(values) for values in (length, reward, truncation))
        steps = [step + shift for step in steps]
        length_axis.plot(
            steps, length, lw=1.8, label=labels.get(run, run), color=run_style["color"], ls=run_style["ls"]
        )
        reward_axis.plot(
            steps, reward, lw=1.8, label=labels.get(run, run), color=run_style["color"], ls=run_style["ls"]
        )
        truncation_axis.plot(
            steps, truncation, lw=1.8, label=labels.get(run, run), color=run_style["color"], ls=run_style["ls"]
        )
    length_axis.set_title("training rollouts: mean response length")
    length_axis.set_ylabel("tokens")
    reward_axis.set_title("training rollouts: mean reward")
    reward_axis.set_ylabel("fraction correct on the training pool")
    truncation_axis.set_title("training rollouts: truncated at the window")
    truncation_axis.set_ylabel("fraction hitting the generation cap")
    for axis in axes[1]:
        axis.set_xlabel("optimizer steps completed when the rollout was scored")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7.5)
    fig.suptitle(args.title)
    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    print("wrote", args.out, "runs:", sorted(runs))


if __name__ == "__main__":
    main()
