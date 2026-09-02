"""Per-cell and aggregate result files."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .files import read_json, write_json
from .models import REFLECTION_MODEL_PRICE, TASK_MODEL_PRICE
from .search import SealedSearchOutcome, pareto_candidate_indices


def write_search_report(
    *,
    output_dir: Path,
    cell: str,
    seed: int,
    outcome: SealedSearchOutcome,
    config: Mapping[str, Any],
    task_usage: Mapping[str, int],
    reflection_usage: Mapping[str, int],
) -> None:
    result = outcome.result
    task_cost = TASK_MODEL_PRICE.estimate(task_usage)
    reflection_cost = REFLECTION_MODEL_PRICE.estimate(reflection_usage)
    summary = {
        "cell": cell,
        "seed": seed,
        "promotion": asdict(outcome.promotion),
        "frozen_test_score": outcome.frozen_test_score,
        "selected_test_score": outcome.selected_test_score,
        "test_delta": outcome.selected_test_score - outcome.frozen_test_score,
        "frozen_test_scores": outcome.frozen_test_scores,
        "selected_test_scores": outcome.selected_test_scores,
        "pareto_candidate_indices": pareto_candidate_indices(result),
        "num_candidates": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "wall_time_s": outcome.wall_time_s,
        "usage": {"task": dict(task_usage), "reflection": dict(reflection_usage)},
        "estimated_cost_usd": {"task": task_cost, "reflection": reflection_cost, "total": task_cost + reflection_cost},
        "pricing": {"task": asdict(TASK_MODEL_PRICE), "reflection": asdict(REFLECTION_MODEL_PRICE)},
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "raw_result.json", result.to_dict())
    write_json(
        output_dir / "heldout.json",
        {
            "frozen_scores": outcome.frozen_test_scores,
            "frozen_outputs": outcome.frozen_test_outputs,
            "selected_scores": outcome.selected_test_scores,
            "selected_outputs": outcome.selected_test_outputs,
        },
    )
    write_json(output_dir / "selected_candidate.json", result.candidates[outcome.promotion.candidate_idx])
    write_json(
        output_dir / "learning_curve.json",
        [
            {
                "candidate_idx": idx,
                "metric_calls_at_discovery": result.discovery_eval_counts[idx],
                "validation_score": result.val_aggregate_scores[idx],
                "parents": result.parents[idx],
            }
            for idx in range(result.num_candidates)
        ],
    )
    (output_dir / "candidate-tree.dot").write_text(result.candidate_tree_dot(), encoding="utf-8")


def write_aggregate_report(*, output_dir: Path, cells: Sequence[str], seeds: Sequence[int]) -> None:
    """Aggregate completed cell summaries into one comparison; neutral and negative deltas stay as they are."""
    aggregates: dict[str, Any] = {}
    for cell in cells:
        runs = [
            _normalize_run(cell, seed, read_json(output_dir / cell / f"seed-{seed}" / "summary.json"))
            for seed in seeds
        ]
        selected_scores = [run["selected_test_score"] for run in runs]
        promotions = [run["promoted"] for run in runs if run["promoted"] is not None]
        aggregates[cell] = {
            "runs": runs,
            "frozen_test_score_mean": statistics.fmean(run["frozen_test_score"] for run in runs),
            "selected_test_score_mean": statistics.fmean(selected_scores),
            "selected_test_score_sample_stdev": (
                statistics.stdev(selected_scores) if len(selected_scores) > 1 else None
            ),
            "test_delta_mean": statistics.fmean(run["test_delta"] for run in runs),
            "promotion_rate": statistics.fmean(promotions) if promotions else None,
            "estimated_cost_usd_total": sum(run["estimated_cost_usd"] for run in runs),
            "wall_time_s_total": sum(run["wall_time_s"] for run in runs),
        }
    write_json(output_dir / "results.json", {"cells": aggregates})


def _normalize_run(cell: str, seed: int, summary: Mapping[str, Any]) -> dict[str, Any]:
    if cell == "frozen":
        frozen_score = selected_score = float(summary["test_score"])
        promoted = None
        estimated_cost = float(summary["estimated_cost_usd"])
        started, finished = summary["started_at"], summary["finished_at"]
        wall_time = (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()
    else:
        frozen_score = float(summary["frozen_test_score"])
        selected_score = float(summary["selected_test_score"])
        promoted = bool(summary["promotion"]["selected"])
        estimated_cost = float(summary["estimated_cost_usd"]["total"])
        wall_time = float(summary["wall_time_s"])
    return {
        "seed": seed,
        "frozen_test_score": frozen_score,
        "selected_test_score": selected_score,
        "test_delta": selected_score - frozen_score,
        "promoted": promoted,
        "estimated_cost_usd": estimated_cost,
        "wall_time_s": wall_time,
    }
