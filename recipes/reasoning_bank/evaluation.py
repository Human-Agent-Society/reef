"""Measure the contribution of a bank on paired evaluation runs."""

from collections.abc import Sequence

from reef.core.values import JsonValue
from reef.evaluation.harness import RunRecord
from reef.harness.tree import HarnessTree
from reef.train.harness_backend import StepContext


def measure_memory(context: StepContext, tree: HarnessTree, without_memory: HarnessTree) -> dict[str, JsonValue]:
    """Score both trees on the step's tasks and compare matching successful (task, repeat) pairs.

    Report an inconclusive result when no pair ran successfully. Infrastructure failures never count as zero.
    The history retains both sets of runs. Existing scores for the same tree and evaluation context are reused.
    Raise ValueError before any run when the step has no evaluation tasks or scorer.
    """
    tasks = context.workspace.tasks()
    repeats = context.workspace.evaluation_repeats()
    if not tasks or repeats < 1:
        raise ValueError("memory ablation requires evaluation tasks and a scorer, also with policy: none")
    if tree == without_memory:
        return {"status": "empty", "pairs": 0, "delta": None}
    with_runs = context.services.evaluate(tree, tasks, repeats=repeats)
    without_runs = context.services.evaluate(without_memory, tasks, repeats=repeats)
    return paired_result(with_runs, without_runs)


def paired_result(with_runs: Sequence[RunRecord], without_runs: Sequence[RunRecord]) -> dict[str, JsonValue]:
    """Means and their difference on the common, non-failed run pairs; positive delta favors memory.

    Each side must have unique (task_key, repeat) pairs, and both sides must use one evaluation context.
    Unmatched or failed pairs are excluded from both means. Run IDs identify all scored and failed attempts.
    """
    sides = [{(run.task_key, run.repeat): run for run in runs} for runs in (with_runs, without_runs)]
    if len(sides[0]) != len(with_runs) or len(sides[1]) != len(without_runs):
        raise ValueError("memory ablation runs must have unique task and repeat pairs")
    if len({run.context for run in (*with_runs, *without_runs)}) > 1:
        raise ValueError("memory ablation must use the same evaluation context on both sides")
    pairs = []
    for key in sorted(sides[0].keys() & sides[1].keys()):
        with_score, without_score = sides[0][key].score.value, sides[1][key].score.value
        if with_score is not None and without_score is not None:
            pairs.append((with_score, without_score))
    with_mean: float | None
    without_mean: float | None
    delta: float | None
    if pairs:
        with_mean = sum(pair[0] for pair in pairs) / len(pairs)
        without_mean = sum(pair[1] for pair in pairs) / len(pairs)
        delta = with_mean - without_mean
        if delta > 0:
            status = "helpful"
        elif delta < 0:
            status = "harmful"
        else:
            status = "unchanged"
    else:
        with_mean = without_mean = delta = None
        status = "inconclusive"
    return {
        "status": status,
        "pairs": len(pairs),
        "excluded_pairs": len(sides[0].keys() | sides[1].keys()) - len(pairs),
        "with_memory": with_mean,
        "without_memory": without_mean,
        "delta": delta,
        "with_runs": [run.run_id for run in with_runs],
        "without_runs": [run.run_id for run in without_runs],
    }
