"""Rollout scores: the three kinds of a rollout's score, and the publication policies that read them."""

from __future__ import annotations

import math

import pytest

from reef.core.evaluation import RolloutScore, RolloutScores, mean_score
from reef.evaluation.evaluators import FloorCheck, ScoreComparison, check_floor, compare_scores

SCORED = RolloutScore.scored
NO_ANSWER = RolloutScore.no_answer()
NOT_RUN = RolloutScore.not_run()


def scores(*tasks: tuple[RolloutScore, ...]) -> RolloutScores:
    return RolloutScores(tuple((f"t{index}", runs) for index, runs in enumerate(tasks)))


@pytest.mark.unit
def test_a_rollout_score_has_a_finite_value_exactly_when_it_was_scored() -> None:
    assert SCORED(1).value == 1.0 and NO_ANSWER.value is None and NOT_RUN.value is None
    for value in (math.nan, math.inf, True, "1", None):
        with pytest.raises(ValueError, match="finite value"):
            RolloutScore("scored", value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="has no value"):
        RolloutScore("not_run", 0.0)
    with pytest.raises(ValueError, match="kind must be one of"):
        RolloutScore("failed")  # type: ignore[arg-type]
    for score in (SCORED(0.5), NO_ANSWER, NOT_RUN):
        assert RolloutScore.from_json(score.to_json()) == score
    with pytest.raises(ValueError, match="a number, no_answer or not_run"):
        RolloutScore.from_json(None)


@pytest.mark.unit
def test_rollout_scores_hold_the_same_number_of_rollouts_for_every_task() -> None:
    value = scores((SCORED(1.0), NO_ANSWER), (NOT_RUN, SCORED(0.0)))
    assert value.count == 2 and value.kind_count("not_run") == 1
    assert value.to_json() == [
        {"task": "t0", "scores": [1.0, "no_answer"]},
        {"task": "t1", "scores": ["not_run", 0.0]},
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
def test_the_mean_counts_a_rollout_without_an_answer_as_zero_and_leaves_out_one_that_did_not_run() -> None:
    assert mean_score((SCORED(1.0), NO_ANSWER, NOT_RUN)) == 0.5
    assert mean_score((NOT_RUN,)) is None


@pytest.mark.unit
def test_compare_scores_drops_a_pair_that_did_not_run_and_ranks_no_answer_below_every_score() -> None:
    candidate = scores((SCORED(1.0), NO_ANSWER, NOT_RUN, SCORED(0.0), NO_ANSWER))
    current = scores((SCORED(0.0), SCORED(0.0), SCORED(1.0), SCORED(0.0), NO_ANSWER))
    assert compare_scores(candidate, current) == ScoreComparison(wins=1, losses=1, ties=2, dropped=1)
    with pytest.raises(ValueError, match="same tasks the same number of times"):
        compare_scores(candidate, scores((SCORED(1.0),)))


@pytest.mark.unit
def test_check_floor_fails_a_task_without_an_answer_but_not_one_with_a_rollout_that_did_not_run() -> None:
    candidate = scores(
        (SCORED(1.0), NOT_RUN),  # passes: the rollout that ran met the floor
        (SCORED(1.0), NO_ANSWER),  # fails: one rollout gave no answer
        (NOT_RUN, NOT_RUN),  # fails: no rollout ran
        (SCORED(0.5), SCORED(1.0)),  # fails: one rollout is below the floor
    )
    assert check_floor(candidate, 1.0) == FloorCheck(passed=1, failed=3, not_run=3)
    assert check_floor(scores((SCORED(0.5),)), 0.5) == FloorCheck(passed=1, failed=0, not_run=0)
