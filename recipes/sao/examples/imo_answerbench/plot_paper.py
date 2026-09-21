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
import json
import math
import os
import re
from collections import defaultdict
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SETS = [
    ("aime2025", "AIME 2025 (30 problems x 8)"),
    ("hmmt_feb2025", "HMMT Feb 2025 (30 problems x 8)"),
    ("imo_answerbench", "IMO-AnswerBench (400 problems)"),
]


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def accuracy(path: str, expected: int) -> tuple[float, int, float, float] | None:
    rows = [r for r in load(path) if "score" in r]  # rows with "error" are retried by evaluate.py
    if len(rows) < expected:
        return None
    k = sum(1 for r in rows if r["score"] == 1)
    lo, hi = wilson(k, len(rows))
    return k / len(rows), len(rows), lo, hi


ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP_LINE = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\].*'train/step': (\d+)")


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
    return [seen[k] for k in sorted(seen)]


def smooth(values: list[float], window: int = 4) -> list[float]:
    """Trailing mean over ``window`` steps; a 128-rollout step is noisy on its own."""
    return [
        sum(values[max(0, i - window + 1) : i + 1]) / len(values[max(0, i - window + 1) : i + 1])
        for i in range(len(values))
    ]


STYLES: dict[str, dict] = {}


def style(run: str) -> dict:
    if run in STYLES:
        return STYLES[run]
    color = "tab:red" if run.startswith("sao") or "-sao" in run else "tab:blue"
    ls = "--" if run.startswith("run1") else "-"
    marker = "o" if ls == "-" else "s"
    return {"color": color, "ls": ls, "marker": marker}


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
    offsets = {k: int(v) for k, v in (pair.split("=", 1) for pair in args.offset)}
    for pair in args.style:
        run, spec = pair.split("=", 1)
        color, ls, marker = spec.rsplit(":", 2)
        STYLES[run] = {"color": color, "ls": ls, "marker": marker}
    expected = {"aime2025": 240, "hmmt_feb2025": 240, "imo_answerbench": args.imo_rows}

    fig, axes = plt.subplots(2, 3, figsize=(19, 9.5))

    runs: set[str] = set()
    for ax, (bench, title) in zip(axes[0], SETS, strict=True):
        base = os.path.join(args.evals, f"eval-base-{bench}.jsonl")
        b = accuracy(base, expected[bench]) if os.path.exists(base) else None
        if b:
            ax.axhline(b[0], color="k", ls=":", lw=1.5, label=f"untrained {b[0]:.3f}")
            ax.axhspan(b[2], b[3], color="k", alpha=0.06)
        points: dict[str, list] = defaultdict(list)
        for path in glob.glob(os.path.join(args.evals, f"eval-*-{bench}.jsonl")):
            match = re.search(rf"eval-(.+)-(\d+)-{bench}\.jsonl$", os.path.basename(path))
            if not match:
                continue
            acc = accuracy(path, expected[bench])
            if acc:
                run = match.group(1)
                points[run].append((int(match.group(2)) + offsets.get(run, 0), *acc))
        for run in sorted(points):
            runs.add(run)
            pts = sorted(points[run])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            err = [[p[1] - p[3] for p in pts], [p[4] - p[1] for p in pts]]
            ax.errorbar(xs, ys, yerr=err, capsize=3, lw=2, label=labels.get(run, run), **style(run))
        ax.set_title(title)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("held-out accuracy, 95% interval")
        ax.set_xlim(left=-3)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower left")

    ax_len, ax_rew, ax_trunc = axes[1]
    for pair in args.records:
        run, path = pair.split("=", 1)
        rows = [r for p in sorted(glob.glob(path)) for r in load(p)]
        if not rows:
            continue
        by_step: dict[int, list] = defaultdict(list)
        if run in step_logs:
            times = step_times(step_logs[run])
            for r in rows:
                by_step[bisect.bisect_right(times, r["recorded_at"])].append(r)
        else:
            for i, r in enumerate(sorted(rows, key=lambda r: r.get("recorded_at", 0))):
                by_step[i // 128].append(r)
        steps = sorted(s for s in by_step if len(by_step[s]) >= 32)
        shift = offsets.get(run, 0)
        length = [sum(r["completion_tokens"] for r in by_step[s]) / len(by_step[s]) for s in steps]
        reward = [sum(r["score"] for r in by_step[s]) / len(by_step[s]) for s in steps]
        trunc = [sum(1 for r in by_step[s] if r.get("finish_reason") == "length") / len(by_step[s]) for s in steps]
        st = style(run)
        length, reward, trunc = (smooth(v) for v in (length, reward, trunc))
        steps = [s + shift for s in steps]
        ax_len.plot(steps, length, lw=1.8, label=labels.get(run, run), color=st["color"], ls=st["ls"])
        ax_rew.plot(steps, reward, lw=1.8, label=labels.get(run, run), color=st["color"], ls=st["ls"])
        ax_trunc.plot(steps, trunc, lw=1.8, label=labels.get(run, run), color=st["color"], ls=st["ls"])
    ax_len.set_title("training rollouts: mean response length")
    ax_len.set_ylabel("tokens")
    ax_rew.set_title("training rollouts: mean reward")
    ax_rew.set_ylabel("fraction correct on the training pool")
    ax_trunc.set_title("training rollouts: truncated at the window")
    ax_trunc.set_ylabel("fraction hitting the generation cap")
    for ax in axes[1]:
        ax.set_xlabel("optimizer steps completed when the rollout was scored")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    fig.suptitle(args.title)
    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    print("wrote", args.out, "runs:", sorted(runs))


if __name__ == "__main__":
    main()
