#!/usr/bin/env python3
"""Build the paper-comparison table from collected run JSON.

Usage: compare.py <results_dir> > table.md

Grader scores are converted back to the metric each paper table reports
(the conversion is the inverse of each task grader's normalization). Paper
reference values are from arXiv:2604.01658 Table 1 (SOTA column) and Table 2
(OpenCode + MiniMax M2.5 rows, 1-agent and 4-agent).
"""
import contextlib
import glob
import json
import os
import sys

# task -> (display, direction, score->metric, paper SOTA, paper M2.5 1-agent, paper M2.5 4-agent)
TASKS = {
    "circle_packing": ("Circle-Pack.", "up", lambda s: s * 2.635977, 2.6359, 2.3531, 2.5391),
    "signal_processing": ("Signal Proc.", "up", lambda s: s, 0.7429, 0.7174, 0.7383),
    "erdos": ("Erdos Over.", "down", lambda s: 0.38092303510845016 / s, 0.38088, 0.39237, 0.38311),
    "mmd_16_2": ("MMD-16-2", "down", lambda s: 12.889266112 / s, 12.89, 12.91, 12.89),
    "mmd_14_3": ("MMD-14-3", "down", lambda s: 4.165849767 / s, 4.16, 4.53, 4.19),
    "third_autocorr": ("3rd-Autocorr.", "down", lambda s: 1.4556427953745406 / s, 1.4557, 1.5337, 1.4931),
    "eplb": ("EPLB", "up", lambda s: s, 0.145, 0.128, 0.129),
    "prism": ("PRISM", "up", lambda s: s, 26.26, 25.85, 26.26),
    "llm_sql": ("LLM-SQL", "up", lambda s: s, 0.730, 0.693, 0.730),
    "txn_scheduling": ("Txn Sched.", "up", lambda s: s, 4348, 3704, 3774),
    "cloudcast": ("Cloudcast", "down", lambda s: 1.0 / s - 1.0, 632.7, 849.4, 672.8),
}
ARMS = ["base", "ttt", "base4", "ttt4"]

# Grader outputs that can only come from breaking the evaluator rather than
# solving the task. cloudcast: total_cost=0 means the program emitted no
# transfer at all (score 1.0 = 1/(1+0)); real solutions cost hundreds.
# txn_scheduling: the evaluator trusts the makespan the program reports and
# only checks the schedule's structure, so a jump far past SOTA (4348) cannot
# be verified; treat anything 1.5x SOTA as unverified.
INVALID = {"cloudcast": lambda s: s >= 0.01, "txn_scheduling": lambda s: s >= 6500}


def fmt(v, task):
    if v is None:
        return "-"
    if task in ("erdos",):
        return f"{v:.5f}"
    if task in ("circle_packing",):
        return f"{v:.4f}"
    if task in ("third_autocorr", "signal_processing"):
        return f"{v:.4f}"
    if task in ("eplb", "llm_sql"):
        return f"{v:.3f}"
    if task in ("txn_scheduling",):
        return f"{v:.0f}"
    return f"{v:.2f}"


def load(results_dir):
    runs = {}
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        with open(f) as fh:
            d = json.load(fh)
        runs[(d["task"], d["arm"])] = d
    return runs


def best_metric(d, task):
    invalid = INVALID.get(task, lambda s: False)
    scores = [
        a["score"] for a in d["attempts"] if a["score"] is not None and a["score"] > 0 and not invalid(a["score"])
    ]
    if not scores:
        return None, 0, 0
    best = max(scores)  # every grader normalizes so higher is better
    return TASKS[task][2](best), len(scores), len(d["attempts"])


def main(results_dir):
    runs = load(results_dir)
    print(
        "| Task | Dir | SOTA | paper M2.5 1-agent | paper M2.5 4-agent | ours 1-agent frozen | ours 1-agent TTT | ours 4-agent frozen | ours 4-agent TTT | evals (frozen/TTT, 1-agent) | evals (4-agent) | TTT steps (1/4-agent) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for task, (name, direction, _, sota, p1, p4) in TASKS.items():
        cells, evals1, evals4, steps = [], [], [], []
        for arm in ARMS:
            d = runs.get((task, arm))
            if d is None:
                cells.append("-")
                (evals1 if arm in ("base", "ttt") else evals4).append("-")
                if arm.startswith("ttt"):
                    steps.append("-")
                continue
            m, _scored, graded = best_metric(d, task)
            cells.append(fmt(m, task))
            (evals1 if arm in ("base", "ttt") else evals4).append(str(graded))
            if arm.startswith("ttt"):
                st = None
                with contextlib.suppress(KeyError, StopIteration, TypeError):
                    st = next(iter(d["reef_status"]["scenarios"].values()))["scenario_step"]
                steps.append(str(st) if st is not None else "-")
        arrow = "↑" if direction == "up" else "↓"
        print(
            f"| {name} | {arrow} | {fmt(sota, task)} | {fmt(p1, task)} | {fmt(p4, task)} | "
            + " | ".join(cells)
            + f" | {'/'.join(evals1)} | {'/'.join(evals4)} | {'/'.join(steps)} |"
        )


if __name__ == "__main__":
    main(sys.argv[1])
