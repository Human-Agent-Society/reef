"""Upstream GEPA search, a validation-only promotion gate, then the sealed test split."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gepa
from gepa.core.adapter import GEPAAdapter
from gepa.core.result import GEPAResult

from .adapter import AIMEExample
from .heldout import CheckpointedEvaluator


@dataclass(frozen=True)
class PromotionDecision:
    """Made from validation scores alone, before the test split is unsealed."""

    selected: bool
    candidate_idx: int
    seed_score: float
    candidate_score: float
    reason: str


@dataclass(frozen=True)
class SealedSearchOutcome:
    result: GEPAResult[Any, Any]
    promotion: PromotionDecision
    frozen_test_scores: tuple[float, ...]
    selected_test_scores: tuple[float, ...]
    frozen_test_outputs: tuple[Any, ...]
    selected_test_outputs: tuple[Any, ...]
    wall_time_s: float

    @property
    def frozen_test_score(self) -> float:
        return statistics.fmean(self.frozen_test_scores) if self.frozen_test_scores else 0.0

    @property
    def selected_test_score(self) -> float:
        return statistics.fmean(self.selected_test_scores) if self.selected_test_scores else 0.0


def decide_promotion(result: GEPAResult[Any, Any]) -> PromotionDecision:
    """Promote only a strict aggregate validation improvement over the seed."""
    if not result.candidates or len(result.candidates) != len(result.val_aggregate_scores):
        raise ValueError("GEPA result candidates and validation scores are missing or misaligned")
    candidate_idx = result.best_idx
    seed_score = float(result.val_aggregate_scores[0])
    candidate_score = float(result.val_aggregate_scores[candidate_idx])
    selected = candidate_idx != 0 and candidate_score > seed_score
    reason = (
        f"candidate {candidate_idx} strictly improved aggregate validation score from {seed_score:.6f} to {candidate_score:.6f}"
        if selected
        else f"no candidate strictly improved the seed aggregate validation score {seed_score:.6f}"
    )
    return PromotionDecision(selected, candidate_idx if selected else 0, seed_score, candidate_score, reason)


def pareto_candidate_indices(result: GEPAResult[Any, Any]) -> tuple[int, ...]:
    """Every specialist retained on at least one per-instance front."""
    return tuple(sorted({idx for front in result.per_val_instance_best_candidates.values() for idx in front}))


def run_sealed_search(
    *,
    seed_candidate: Mapping[str, str],
    trainset: Sequence[AIMEExample],
    valset: Sequence[AIMEExample],
    testset: Sequence[AIMEExample],
    adapter: GEPAAdapter[AIMEExample, Any, Any],
    reflection_lm: Callable[[str | list[dict[str, Any]]], str] | None,
    max_metric_calls: int,
    seed: int,
    run_dir: Path,
    custom_candidate_proposer: Callable[..., dict[str, str]] | None = None,
    heldout_evaluator: CheckpointedEvaluator | None = None,
    skip_perfect_score: bool = True,
) -> SealedSearchOutcome:
    """Run upstream GEPA with the quickstart's settings; touch ``testset`` only after the gate.

    ``testset`` is never passed to GEPA. Both held-out evaluations happen after
    ``gepa.optimize`` returns and the promotion decision is fixed from validation.
    """
    if not trainset or not valset or not testset:
        raise ValueError("train, validation, and test splits must all be non-empty")
    started = time.monotonic()
    result = gepa.optimize(
        seed_candidate=dict(seed_candidate),
        trainset=list(trainset),
        valset=list(valset),
        adapter=adapter,
        reflection_lm=reflection_lm,
        custom_candidate_proposer=custom_candidate_proposer,
        candidate_selection_strategy="pareto",
        frontier_type="instance",
        module_selector="round_robin",
        max_metric_calls=max_metric_calls,
        run_dir=str(run_dir),
        seed=seed,
        track_best_outputs=True,
        cache_evaluation=False,
        skip_perfect_score=skip_perfect_score,
    )
    promotion = decide_promotion(result)
    selected_candidate = result.candidates[promotion.candidate_idx]
    if heldout_evaluator is None:
        frozen = adapter.evaluate(list(testset), dict(seed_candidate))
        selected = adapter.evaluate(list(testset), dict(selected_candidate))
    else:
        frozen = heldout_evaluator.evaluate("frozen", testset, seed_candidate)
        selected = heldout_evaluator.evaluate("selected", testset, selected_candidate)
    return SealedSearchOutcome(
        result=result,
        promotion=promotion,
        frozen_test_scores=tuple(frozen.scores),
        selected_test_scores=tuple(selected.scores),
        frozen_test_outputs=tuple(frozen.outputs),
        selected_test_outputs=tuple(selected.outputs),
        wall_time_s=time.monotonic() - started,
    )
