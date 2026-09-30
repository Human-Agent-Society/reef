"""The paired confidence selection (#698, #356): faults decide their pairs, repeats average within a task, and a
candidate is selected only when an exact sign test over tasks passes and the bootstrap lower bound clears
``min_effect``."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence

import pytest

from reef.train.cordis_backend import ScoreComparisonPlugin
from reef.train.evaluation import (
    EvaluationResult,
    PairedConfidenceMixin,
    PairedConfidenceSettings,
    SelectionDecision,
    UpdateCandidate,
    compare_pairs,
)

CANDIDATE = UpdateCandidate("job-698")


class DecideOnlyPlugin(PairedConfidenceMixin):
    """The policy alone: these cases exercise ``decide``."""

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        raise AssertionError("this case exercises decide(), not evaluate()")


class DecideOnlyBackend:
    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        raise AssertionError("this case exercises decide(), not evaluate()")


def evaluation(
    candidate: Sequence[float | str], current: Sequence[float | str], *, repeats: int = 1
) -> EvaluationResult:
    """Pairs in task order, then repeat; an outcome is a score or the fault of an episode without one."""

    def scores(outcomes: Sequence[float | str]) -> tuple[float | None, ...]:
        return tuple(None if isinstance(outcome, str) else outcome for outcome in outcomes)

    def faults(outcomes: Sequence[float | str]) -> tuple[str | None, ...]:
        return tuple(outcome if isinstance(outcome, str) else None for outcome in outcomes)

    return EvaluationResult(
        evaluator="harness_episode_pairs",
        evaluator_version="1",
        metrics={
            "candidate_scores": scores(candidate),
            "current_scores": scores(current),
            "candidate_faults": faults(candidate),
            "current_faults": faults(current),
            "episode_repeats": repeats,
        },
    )


def decide(result: EvaluationResult, **settings: float) -> SelectionDecision:
    return DecideOnlyPlugin(PairedConfidenceSettings(**settings)).decide(CANDIDATE, result)


def test_an_unchanged_candidate_that_won_every_task_at_small_n_is_not_selected() -> None:
    """Four lucky wins give a bootstrap interval of [1, 1]; the sign test alone bounds the false select rate."""
    lucky = evaluation((1.0,) * 4, (0.0,) * 4)
    decision = decide(lucky)
    assert not decision.selected and decision.policy == "paired_confidence" and decision.policy_version == "1"
    assert decision.metrics["selection_result"] == "insufficient_confidence"
    assert decision.metrics["sign_test_p_value"] == 1 / 16
    assert (decision.metrics["interval_lower"], decision.metrics["interval_upper"]) == (1.0, 1.0)
    assert decision.reason == "insufficient confidence: sign test p 0.0625, interval lower bound 1, min_effect 0"
    # The score comparison selects the same scores: four wins, no loss.
    compared = ScoreComparisonPlugin(DecideOnlyBackend()).decide(CANDIDATE, lucky)
    assert compared.selected and (compared.metrics["wins"], compared.metrics["losses"]) == (4, 0)


def test_a_planted_null_is_selected_at_most_at_the_confidence_level() -> None:
    """An unchanged candidate on 30 tasks whose pass rates differ: both sides draw from the same rate."""
    generator = random.Random(698)
    plugin = DecideOnlyPlugin(PairedConfidenceSettings())
    comparison = ScoreComparisonPlugin(DecideOnlyBackend())
    selected = compared = 0
    for _ in range(100):
        rates = [generator.random() for _ in range(30)]
        candidate = [float(generator.random() < rate) for rate in rates]
        current = [float(generator.random() < rate) for rate in rates]
        null = evaluation(candidate, current)
        selected += plugin.decide(CANDIDATE, null).selected
        compared += comparison.decide(CANDIDATE, null).selected
    assert selected <= 5
    # The score comparison publishes the same noise far more often.
    assert compared > 25


def test_a_real_gain_is_selected_and_the_decision_records_its_test() -> None:
    decision = decide(evaluation((1.0,) * 20 + (0.5,) * 10, (0.0,) * 20 + (0.5,) * 10))
    assert decision.selected and decision.metrics["selection_result"] == "selected"
    metrics = decision.metrics
    assert (metrics["valid_pairs"], metrics["void_pairs"], metrics["tasks"]) == (30, 0, 30)
    assert (metrics["wins"], metrics["losses"], metrics["ties"]) == (20, 0, 10)
    assert metrics["sign_test_p_value"] == 2**-20
    assert metrics["mean_difference"] == pytest.approx(2 / 3)
    assert 0 < metrics["interval_lower"] < 2 / 3 < metrics["interval_upper"] <= 1
    assert (metrics["confidence_level"], metrics["min_effect"], metrics["min_valid_pairs"]) == (0.95, 0.0, 1)
    assert decision.reason.startswith("candidate gained 0.6667 per task (interval ")
    assert decision.reason.endswith(" over 30 valid pairs (0 void), sign test p 9.537e-07")
    json.loads(json.dumps(decision.to_dict(), allow_nan=False))
    # A gain below min_effect is not enough: every task is a win, and the lower bound stays under it.
    below = decide(evaluation((0.6,) * 30, (0.5,) * 30), min_effect=0.2)
    assert not below.selected and below.metrics["selection_result"] == "insufficient_confidence"
    assert (below.metrics["wins"], below.metrics["losses"]) == (30, 0)
    assert below.metrics["interval_lower"] == pytest.approx(0.1)


def test_min_effect_bounds_the_mean_gain_and_leaves_ties_out_of_the_sign_test() -> None:
    """Eight wins and twelve ties on 0/1 rewards: a mean gain of 0.4 clears a min_effect of 0.05."""
    decision = decide(evaluation((1.0,) * 8 + (0.0,) * 12, (0.0,) * 20), min_effect=0.05)
    assert decision.selected and decision.metrics["selection_result"] == "selected"
    assert (decision.metrics["wins"], decision.metrics["losses"], decision.metrics["ties"]) == (8, 0, 12)
    assert decision.metrics["sign_test_p_value"] == 1 / 256
    assert decision.metrics["interval_lower"] > 0.05
    # The same wins do not clear a min_effect above the lower bound.
    strict = decide(evaluation((1.0,) * 8 + (0.0,) * 12, (0.0,) * 20), min_effect=0.4)
    assert not strict.selected and strict.metrics["sign_test_p_value"] == 1 / 256


def test_void_pairs_count_as_candidate_losses() -> None:
    """A pair an infrastructure fault hit takes the worst difference the evaluation allows."""
    voided = decide(evaluation((1.0,) * 10, (0.0,) * 8 + ("infrastructure",) * 2))
    assert not voided.selected and voided.metrics["selection_result"] == "insufficient_confidence"
    assert (voided.metrics["valid_pairs"], voided.metrics["void_pairs"]) == (8, 2)
    assert (voided.metrics["wins"], voided.metrics["losses"]) == (8, 2)
    assert voided.metrics["sign_test_p_value"] == 56 / 1024
    # The fault is on either side: the candidate's own fault voids its pair too.
    comparison = compare_pairs(evaluation((1.0, "infrastructure"), (0.0, 0.5)).metrics, confidence_level=0.95)
    # The worst difference: the fault value, one score range below the lowest score, minus the highest.
    assert comparison.task_differences == (1.0, -2.0) and comparison.void_pairs == 1
    # Dropping the void pairs would have selected.
    dropped = decide(evaluation((1.0,) * 8, (0.0,) * 8))
    assert dropped.selected and dropped.metrics["sign_test_p_value"] == 1 / 256


def test_a_harness_fault_ranks_below_every_real_score_of_the_evaluation() -> None:
    # The scores run from 0.25 to 0.75, so a harness fault scores one range below the lowest: -0.25.
    comparison = compare_pairs(
        evaluation(("harness", 0.25, "harness"), (0.5, 0.75, "harness")).metrics, confidence_level=0.95
    )
    assert comparison.valid_pairs == 3 and comparison.void_pairs == 0
    assert comparison.task_differences == (-0.75, -0.5, 0.0)
    assert (comparison.wins, comparison.losses, comparison.ties) == (0, 2, 1)
    # A candidate that fixes a current tree which crashed on every task wins every task, even when every real
    # score is the same.
    fixed = decide(evaluation((1.0,) * 8, ("harness",) * 8))
    assert fixed.selected and (fixed.metrics["wins"], fixed.metrics["ties"]) == (8, 0)
    crashed = decide(evaluation(("harness",) * 8, (1.0,) * 8))
    assert not crashed.selected and crashed.metrics["losses"] == 8


def test_an_evaluation_without_enough_valid_pairs_is_invalid() -> None:
    every_pair_void = decide(evaluation((1.0, 1.0, 1.0), ("infrastructure",) * 3))
    assert not every_pair_void.selected and every_pair_void.metrics["selection_result"] == "invalid_evaluation"
    assert every_pair_void.reason == "invalid evaluation: 0 valid pairs, below min_valid_pairs 1 (3 void)"
    too_few = decide(evaluation((1.0,) * 8, (0.0,) * 8), min_valid_pairs=10)
    assert not too_few.selected and too_few.metrics["selection_result"] == "invalid_evaluation"
    assert too_few.reason == "invalid evaluation: 8 valid pairs, below min_valid_pairs 10 (0 void)"


def test_repeats_average_within_a_task_before_the_test() -> None:
    comparison = compare_pairs(
        evaluation((1.0, 0.0, 1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 1.0), repeats=3).metrics,
        confidence_level=0.95,
    )
    assert comparison.valid_pairs == 6
    assert comparison.task_differences == pytest.approx((2 / 3, -1 / 3))
    assert (comparison.wins, comparison.losses, comparison.ties) == (1, 1, 0)
    assert comparison.mean_difference == pytest.approx(1 / 6)


def test_the_decision_is_derived_again_from_its_record() -> None:
    generator = random.Random(7)
    scores = [round(generator.random(), 3) for _ in range(24)]
    recorded = evaluation(scores[:12], scores[12:])
    first, again = decide(recorded), decide(recorded)
    assert first.metrics == again.metrics and first.reason == again.reason
    # The commit record keeps the vectors as JSON lists; read back, they decide the same way.
    reread = EvaluationResult(**json.loads(json.dumps(recorded.to_dict())))
    assert decide(reread).metrics == first.metrics


def test_an_evaluation_the_rule_cannot_read_is_an_error() -> None:
    plain = evaluation((1.0, 0.0), (0.0, 0.0))
    without_faults = {key: value for key, value in plain.metrics.items() if key != "candidate_faults"}
    with pytest.raises(ValueError, match="needs 'candidate_faults'"):
        compare_pairs(without_faults, confidence_level=0.95)
    one_sided = {**plain.metrics, "current_scores": (), "current_faults": ()}
    with pytest.raises(ValueError, match="needs both sides evaluated on the same pairs"):
        compare_pairs(one_sided, confidence_level=0.95)
    for candidate_scores, candidate_faults in (((None, 0.0), (None, None)), ((1.0, 0.0), ("harness", None))):
        broken = {**plain.metrics, "candidate_scores": candidate_scores, "candidate_faults": candidate_faults}
        with pytest.raises(ValueError, match="needs a finite score or a fault"):
            compare_pairs(broken, confidence_level=0.95)
    with pytest.raises(ValueError, match="a multiple of episode_repeats 3"):
        compare_pairs({**plain.metrics, "episode_repeats": 3}, confidence_level=0.95)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"min_valid_pairs": 0}, "min_valid_pairs must be an integer of at least 1"),
        ({"min_valid_pairs": True}, "min_valid_pairs must be an integer of at least 1"),
        ({"min_valid_pairs": 1.5}, "min_valid_pairs must be an integer of at least 1"),
        ({"min_effect": -0.1}, "min_effect must be a finite number of at least 0"),
        ({"min_effect": float("inf")}, "min_effect must be a finite number of at least 0"),
        ({"confidence_level": 1}, "confidence_level must be a number above 0.5 and below 1"),
        ({"confidence_level": 0.5}, "confidence_level must be a number above 0.5 and below 1"),
        ({"confidence_level": True}, "confidence_level must be a number above 0.5 and below 1"),
    ],
)
def test_settings_outside_their_range_are_refused(settings: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PairedConfidenceSettings(**settings)
