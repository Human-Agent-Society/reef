"""Draw the skill stream's Figure 3: both skills' test accuracy against gradient steps, SDFT beside the SFT control.

Reads the trace rows reef-eval stored for the stream (every judge score:
the step and both skills' accuracies, tagged ``arm`` and ``position``) and
the control run's scores from ``--control`` (a CSV in the same columns,
recorded by a run outside this example), writes ``figure3.png`` and
``accuracy.csv`` under ``--out``, in the figure style the repository's
results share with the docs site.

    uv run --no-project --python 3.12 --with reef-eval --with matplotlib plot.py \\
        --lab work/lab --control control.csv --out figure3
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import pandas as pd
from reef_eval import Lab

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = ("tooluse", "science")
SKILL_NAMES = {"tooluse": "Tool Use", "science": "Science Q&A"}
ARM_NAMES = {"sdft": "SDFT", "sft": "SFT"}
# Figure style shared with the docs site (paper surface, warm ink, hairline
# grid, and the site's brand red / blue as the fixed series order).
PAPER = "#fbfaf8"
INK = "#302c28"
MUTED = "#716b65"
HAIRLINE = "#ddd8d0"
ARM_COLORS = {"sdft": "#a03729", "sft": "#2e5fa6"}
STYLE = {
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "text.color": INK,
    "axes.titlecolor": INK,
    "axes.titleweight": "normal",
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": HAIRLINE,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": HAIRLINE,
    "grid.linewidth": 1.0,
    "axes.axisbelow": True,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "legend.frameon": False,
    "legend.labelcolor": INK,
    "font.size": 11,
}


def stream_scores(lab_root: Path, stream: str) -> pd.DataFrame:
    """One row per judge score: arm, position, stage, step, cumulative step, both accuracies."""
    trace = Lab(lab_root).store.df("trace")
    if trace.empty or "stream" not in trace.columns:
        raise FileNotFoundError(f"no trace rows under {lab_root}")
    rows = trace[trace["stream"] == stream].copy()
    if rows.empty:
        raise FileNotFoundError(f"no trace rows for stream {stream!r} under {lab_root}")
    rows["position"] = rows["position"].astype(int)
    rows["step"] = rows["step"].astype(int)
    rows["stage"] = rows["position"].map(lambda position: STAGES[position])
    # The x axis runs across stages: a stage's steps start where the previous stage's ended.
    stage_lengths = rows.groupby(["arm", "position"])["step"].max()
    offsets = {
        (arm, position): int(sum(stage_lengths.get((arm, earlier), 0) for earlier in range(position)))
        for arm, position in stage_lengths.index
    }
    rows["cumulative_step"] = [
        offsets[(arm, position)] + step
        for arm, position, step in zip(rows["arm"], rows["position"], rows["step"], strict=True)
    ]
    columns = ["arm", "position", "stage", "step", "cumulative_step", *STAGES]
    return rows[columns].sort_values(["arm", "cumulative_step"]).reset_index(drop=True)


def draw(scores: pd.DataFrame, out: Path) -> None:
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, len(STAGES), figsize=(11, 4), sharex=True)
        # Where training switched tasks: the last step of each earlier stage, the
        # later stage's span shaded so the switch reads at a glance. Both arms
        # run the same schedule, so the switch is one line per stage.
        switches = scores.groupby("position")["cumulative_step"].agg(["min", "max"])
        for axis, skill in zip(axes, STAGES, strict=True):
            for arm, group in scores.groupby("arm"):
                axis.plot(group["cumulative_step"], group[skill] * 100, color=ARM_COLORS[arm], label=ARM_NAMES[arm])
            for position, (start, end) in switches.iloc[1:].iterrows():
                axis.axvspan(start, end, color=HAIRLINE, alpha=0.45, linewidth=0)
                axis.axvline(start, color=INK, linestyle=(0, (5, 3)), linewidth=1.4)
                axis.annotate(
                    f"{SKILL_NAMES[STAGES[position]]}\ntraining starts",
                    xy=(start, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(5, -4),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    fontsize=9,
                    color=INK,
                )
            axis.set_title(f"{SKILL_NAMES[skill]} accuracy (%)")
            axis.set_xlabel("gradient steps (Tool Use, then Science Q&A)")
        axes[0].legend(loc="lower right")
        figure.tight_layout()
        figure.savefig(out / "figure3.png", dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", type=Path, default=Path("work") / "lab", help="the Lab directory run.py wrote")
    parser.add_argument("--stream", default="figure3", help="the stream's name in the Lab store")
    parser.add_argument("--out", type=Path, required=True, help="results directory for the figure and accuracy.csv")
    parser.add_argument("--control", type=Path, action="append", default=[], help="a CSV of another arm's scores")
    arguments = parser.parse_args()
    scores = stream_scores(arguments.lab, arguments.stream)
    for control in arguments.control:
        scores = pd.concat([scores, pd.read_csv(control)[scores.columns]], ignore_index=True)
    scores = scores.sort_values(["arm", "cumulative_step"]).reset_index(drop=True)
    arguments.out.mkdir(parents=True, exist_ok=True)
    with (arguments.out / "accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(scores.columns)
        writer.writerows(scores.itertuples(index=False))
    draw(scores, arguments.out)
    print(scores.groupby(["arm", "position"]).tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
