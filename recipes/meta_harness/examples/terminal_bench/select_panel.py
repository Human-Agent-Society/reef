"""Select the reliability panel from a seed evaluation.

This partitions a seed evaluation; it does not rank tasks by usefulness. A
task the seed always passes can only show a regression. A task the seed
always fails is headroom, not dead weight -- it is precisely where a better
candidate has room to win, and in the run recorded in RESULTS.md five such
tasks supplied most of one iteration's gain. Do not use the always-fail
bucket to prune a task set.

Run it against a population mirror written by ``run.py``::

    python -m recipes.meta_harness.examples.terminal_bench.select_panel \\
        --population /path/to/population.json --tasks-file /path/to/tasks.txt

Reef emits scores in pairing order, task-major with repeats inside each task,
so the mirror plus the task list is enough to recover per-task results.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .tasks import read_tasks


def per_task_scores(scores: Sequence[float], tasks: Sequence[str]) -> dict[str, list[float]]:
    """Split a flat score vector back into per-task results."""
    if not tasks:
        raise ValueError("no tasks given")
    if len(scores) % len(tasks):
        raise ValueError(f"{len(scores)} scores do not divide across {len(tasks)} tasks")
    repeats = len(scores) // len(tasks)
    return {task: list(scores[index * repeats : (index + 1) * repeats]) for index, task in enumerate(tasks)}


def partition(results: dict[str, list[float]]) -> tuple[list[str], list[str], list[str]]:
    """Split tasks into always-pass, always-fail, and the panel between them."""
    always_pass = [task for task, values in results.items() if values and all(value == 1.0 for value in values)]
    always_fail = [task for task, values in results.items() if values and all(value == 0.0 for value in values)]
    decided = set(always_pass) | set(always_fail)
    return always_pass, always_fail, [task for task in results if task not in decided]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="select-panel", description=__doc__)
    parser.add_argument("--population", required=True, help="population.json written by run.py")
    parser.add_argument("--tasks-file", required=True, help="task ids, in the order they were run")
    parser.add_argument("--out", help="write the panel here as a comma-separated list")
    arguments = parser.parse_args(argv)

    tasks = list(read_tasks(arguments.tasks_file))
    population = json.loads(Path(arguments.population).read_text(encoding="utf-8"))
    seeds = [candidate for candidate in population.get("candidates", []) if candidate.get("parent_id") is None]
    if not seeds or not seeds[0].get("scores"):
        print("the population carries no scored seed; run at least one iteration first", file=sys.stderr)
        return 2

    results = per_task_scores(seeds[0]["scores"], tasks)
    always_pass, always_fail, panel = partition(results)
    total = len(results)
    print(f"seed mean {sum(seeds[0]['scores']) / len(seeds[0]['scores']):.3f} over {total} tasks")
    print(f"  always pass: {len(always_pass)}")
    print(f"  always fail: {len(always_fail)}")
    print(f"  panel:       {len(panel)}  ({100 * len(panel) // total}% of the tasks carry the signal)")
    for task in panel:
        print(f"    {task}: {results[task]}")
    if arguments.out:
        Path(arguments.out).write_text(",".join(panel) + "\n", encoding="utf-8")
        print(f"panel written to {arguments.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
