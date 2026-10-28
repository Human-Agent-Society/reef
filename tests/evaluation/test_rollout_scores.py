"""Rollout scores: a rollout's score or its failure, and the publication policies that read them."""

from __future__ import annotations

import math

import pytest

from reef.core.evaluation import RolloutScore, RolloutScores, mean_score
from reef.evaluation.evaluators import FloorCheck, ScoreComparison, check_floor, compare_scores

SCORED = RolloutScore.scored
FAILED = RolloutScore.failed()


def scores(*tasks: tuple[RolloutScore, ...]) -> RolloutScores:
    return RolloutScores(tuple((f"t{index}", runs) for index, runs in enumerate(tasks)))


@pytest.mark.unit
def test_a_rollout_score_is_a_finite_number_or_failed() -> None:
    assert SCORED(1).value == 1.0 and not SCORED(0.0).is_failed and FAILED.value is None and FAILED.is_failed
    for value in (math.nan, math.inf, True, "1"):
        with pytest.raises(ValueError, match="finite number"):
            RolloutScore(value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite value"):
        SCORED(None)  # type: ignore[arg-type]
    for score in (SCORED(0.5), FAILED):
        assert RolloutScore.from_json(score.to_json()) == score
    assert FAILED.to_json() == "failed"
    for value in (None, "no_answer", "not_run"):
        with pytest.raises(ValueError, match="a number or 'failed'"):
            RolloutScore.from_json(value)


@pytest.mark.unit
def test_rollout_scores_hold_the_same_number_of_rollouts_for_every_task() -> None:
    value = scores((SCORED(1.0), SCORED(0.0)), (FAILED, SCORED(0.0)))
    assert value.count == 2 and value.failed_count() == 1
    assert value.to_json() == [
        {"task": "t0", "scores": [1.0, 0.0]},
        {"task": "t1", "scores": ["failed", 0.0]},
    ]
    assert RolloutScores.from_json(value.to_json()) == value
    assert RolloutScores.by_task(["t0", "t1"], value.rollouts()) == value
    with pytest.raises(ValueError, match="same number of rollouts"):
        scores((SCORED(1.0),), (SCORED(1.0), SCORED(1.0)))
    with pytest.raises(ValueError, match="one or more tasks"):
        RolloutScores(())
    with pytest.raises(ValueError, match="divide evenly"):
        RolloutScores.by_task(["t0", "t1"], (SCORED(1.0),))


@pytest.mark.unit
def test_the_mean_leaves_out_a_failed_rollout() -> None:
    assert mean_score((SCORED(1.0), SCORED(0.0), FAILED)) == 0.5
    assert mean_score((FAILED,)) is None


@pytest.mark.unit
def test_compare_scores_drops_a_pair_with_a_failed_side_and_compares_the_rest_by_value() -> None:
    candidate = scores((SCORED(1.0), SCORED(0.0), FAILED, SCORED(0.0), SCORED(0.5)))
    current = scores((SCORED(0.0), SCORED(0.5), SCORED(1.0), SCORED(0.0), FAILED))
    assert compare_scores(candidate, current) == ScoreComparison(wins=1, losses=1, ties=1, dropped=2)
    with pytest.raises(ValueError, match="same tasks the same number of times"):
        compare_scores(candidate, scores((SCORED(1.0),)))


@pytest.mark.unit
def test_check_floor_passes_a_task_whose_rollouts_that_did_not_fail_reach_the_floor() -> None:
    candidate = scores(
        (SCORED(1.0), FAILED),  # passes: the rollout that ran met the floor
        (SCORED(1.0), SCORED(0.0)),  # fails: one rollout gave no answer, so it scores 0
        (FAILED, FAILED),  # fails: every rollout failed
        (SCORED(0.5), SCORED(1.0)),  # fails: one rollout is below the floor
    )
    assert check_floor(candidate, 1.0) == FloorCheck(passed=1, failed=3, failed_rollouts=3)
    assert check_floor(scores((SCORED(0.5),)), 0.5) == FloorCheck(passed=1, failed=0, failed_rollouts=0)
