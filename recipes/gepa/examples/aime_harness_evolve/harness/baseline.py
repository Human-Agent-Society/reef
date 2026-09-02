"""The pre-search gate: the seed prompt must score alike through both arms."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .adapter import AIMEExample
from .config import BASELINE_CRITICAL_VALUE, BASELINE_MARGIN, BASELINE_REPETITIONS, PI_TASK_MAX_TOKENS
from .data import RULES_SEED_CANDIDATE
from .files import write_json
from .heldout import BatchAdapter, CheckpointedEvaluator
from .reference import OFFICIAL_SEED_CANDIDATE


def run_baseline_alignment(
    *,
    official: BatchAdapter,
    reef: BatchAdapter,
    valset: Sequence[AIMEExample],
    output_dir: Path,
    workers: int,
) -> dict[str, Any]:
    """Score the seed prompt on repeated validation problems through both arms.

    Prompt equality is exact by construction; live scores are stochastic, so the
    gate is a one-sided non-inferiority check over per-problem mean differences.
    It also fails on any Reef episode that did not exit cleanly, so a shared zero
    cannot pass as agreement. On failure it raises before any search starts.
    """
    comparison = [example for _ in range(BASELINE_REPETITIONS) for example in valset]
    official_result = CheckpointedEvaluator(
        official, output_dir / "official-checkpoints", batch_size=workers
    ).evaluate("seed-validation", comparison, OFFICIAL_SEED_CANDIDATE)
    reef_result = CheckpointedEvaluator(reef, output_dir / "reef-checkpoints", batch_size=workers).evaluate(
        "seed-validation", comparison, RULES_SEED_CANDIDATE
    )
    official_scores = [float(score) for score in official_result.scores]
    reef_scores = [float(score) for score in reef_result.scores]
    failed_reef = [
        index
        for index, output in enumerate(reef_result.outputs)
        if not isinstance(output, dict) or output.get("exit_code") != 0 or output.get("residue")
    ]
    difference = sum(reef_scores) / len(reef_scores) - sum(official_scores) / len(official_scores)
    per_problem = [
        sum(
            reef_scores[rep * len(valset) + i] - official_scores[rep * len(valset) + i]
            for rep in range(BASELINE_REPETITIONS)
        )
        / BASELINE_REPETITIONS
        for i in range(len(valset))
    ]
    standard_error = 0.0
    if len(per_problem) > 1:
        variance = sum((value - difference) ** 2 for value in per_problem) / (len(per_problem) - 1)
        standard_error = math.sqrt(variance / len(per_problem))
    lower_bound = difference - BASELINE_CRITICAL_VALUE * standard_error
    aligned = not failed_reef and lower_bound > -BASELINE_MARGIN
    result = {
        "prompt": RULES_SEED_CANDIDATE["rules"],
        "task_max_tokens": PI_TASK_MAX_TOKENS,
        "validation_examples": len(valset),
        "repetitions": BASELINE_REPETITIONS,
        "evaluations_per_arm": len(comparison),
        "official_scores": official_scores,
        "reef_scores": reef_scores,
        "official_score": sum(official_scores) / len(official_scores),
        "reef_score": sum(reef_scores) / len(reef_scores),
        "reef_minus_official_score": difference,
        "per_problem_score_differences": per_problem,
        "standard_error": standard_error,
        "one_sided_confidence": 0.95,
        "lower_confidence_bound": lower_bound,
        "noninferiority_margin": BASELINE_MARGIN,
        "failed_reef_indices": failed_reef,
        "baseline_aligned": aligned,
        "usage": {"official": official.usage.snapshot(), "reef": reef.usage.snapshot()},
    }
    write_json(output_dir / "result.json", result)
    if not aligned:
        raise RuntimeError(
            f"baseline alignment failed: Reef-minus-official lower bound {lower_bound:.2%} does not exceed "
            f"-{BASELINE_MARGIN:.2%}, or {len(failed_reef)} Reef episodes failed; no GEPA search was started"
        )
    return result
