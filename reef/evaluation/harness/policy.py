"""The publication policies of harness evolution: how a candidate is evaluated, and when it is published.

``evolution.evaluation.policy`` names one:

``compare``
    Score the candidate and the head tree on the same (task, repeat) pairs.
    ``statistic: wins`` publishes when wins minus losses exceed
    ``min_margin``; an episode that could not run ranks below every score.
    ``statistic: mean`` publishes when the candidate's mean exceeds the
    head's mean by more than ``min_margin``; an episode that could not run
    counts as 0. The head's scores usually come from the score cache.
``floor``
    Score the candidate alone; publish when every episode scores at least
    ``floor_score``.
``none``
    Run no episode; publish every candidate that passed validation.
"""

from __future__ import annotations

import math
from abc import ABC
from statistics import fmean
from typing import Any, ClassVar

from reef.core.evaluation import CandidateEvaluationPlugin, EvaluationResult, SelectionDecision, UpdateCandidate
from reef.evaluation.harness.evaluator import EVALUATION_SIDES, HarnessCandidateEvaluator

#: The ``evolution.evaluation.policy`` names, and the ``statistic`` names of ``compare``.
POLICIES = ("compare", "floor", "none")
STATISTICS = ("wins", "mean")


class EpisodePolicy(CandidateEvaluationPlugin, ABC):
    """A publication policy that decides by the scores of evaluation episodes.

    Values: ``evaluator`` scores the candidate; ``sides`` are the sides each
    evaluation scores, a non-empty subset of ``EVALUATION_SIDES`` in its order.
    Purpose: the harness recipe builds one per scenario from
    ``evolution.evaluation``; the trainer calls ``evaluate`` and then
    ``decide`` on each step's candidate.
    """

    sides: ClassVar[tuple[str, ...]] = EVALUATION_SIDES

    def __init__(self, evaluator: HarnessCandidateEvaluator) -> None:
        self.evaluator = evaluator

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        return self.evaluator.evaluate(candidate, sides=self.sides)


class ComparePolicy(EpisodePolicy):
    """Publish a candidate that scores better than the head tree on the same tasks.

    Values: ``statistic`` is one of ``STATISTICS``; ``min_margin`` ≥ 0 is
    the margin the candidate must exceed: in wins minus losses for ``wins``,
    in mean score for ``mean``.
    """

    def __init__(self, evaluator: HarnessCandidateEvaluator, *, statistic: str = "wins", min_margin: float = 0.0):
        if statistic not in STATISTICS:
            raise ValueError(f"the compare statistic must be one of {', '.join(STATISTICS)}")
        if isinstance(min_margin, bool) or not math.isfinite(min_margin) or min_margin < 0:
            raise ValueError("min_margin must be a finite number of at least 0")
        super().__init__(evaluator)
        self.statistic = statistic
        self.min_margin = float(min_margin)

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        candidate_scores = tuple(evaluation.metrics.get("candidate_scores", ()))
        current_scores = tuple(evaluation.metrics.get("current_scores", ()))
        if len(candidate_scores) != len(current_scores):
            raise ValueError("a comparison needs one current score per candidate score")
        if not candidate_scores:
            return SelectionDecision(False, "compare", "no evaluation task was scored", evaluation)
        metrics: dict[str, Any] = {"statistic": self.statistic, "min_margin": self.min_margin}
        if self.statistic == "wins":
            wins, losses = score_comparison_tally(candidate_scores, current_scores)
            metrics.update({"wins": wins, "losses": losses, "ties": len(candidate_scores) - wins - losses})
            selected = wins - losses > self.min_margin
            reason = (
                f"candidate won {wins} task pairings and lost {losses}"
                if selected
                else f"candidate did not exceed its {losses} losses with {wins} wins"
            )
        else:
            candidate_mean = fmean(0.0 if score is None else score for score in candidate_scores)
            current_mean = fmean(0.0 if score is None else score for score in current_scores)
            metrics.update({"candidate_mean": candidate_mean, "current_mean": current_mean})
            selected = candidate_mean - current_mean > self.min_margin
            reason = (
                f"candidate scored {candidate_mean:.4f} against {current_mean:.4f} "
                f"over {len(candidate_scores)} episodes"
            )
        return SelectionDecision(selected, "compare", reason, evaluation, metrics)


class FloorPolicy(EpisodePolicy):
    """Publish a candidate whose every episode scores at least ``floor_score``; the head tree does not run.

    Values: ``floor_score`` is a positive finite number. An episode that
    could not run misses the floor.
    """

    sides = ("candidate",)

    def __init__(self, evaluator: HarnessCandidateEvaluator, *, floor_score: float = 1.0) -> None:
        if isinstance(floor_score, bool) or not math.isfinite(floor_score) or floor_score <= 0:
            raise ValueError("floor_score must be a positive number")
        super().__init__(evaluator)
        self.floor_score = float(floor_score)

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        scores = tuple(evaluation.metrics.get("candidate_scores", ()))
        passed = sum(1 for score in scores if score is not None and score >= self.floor_score)
        failed = len(scores) - passed
        if not scores:
            reason = "no evaluation task was scored"
        elif failed:
            reason = f"candidate missed the floor on {failed} of {len(scores)} tasks"
        else:
            reason = f"candidate met the floor on all {len(scores)} tasks"
        return SelectionDecision(
            selected=bool(scores) and not failed,
            policy="floor",
            reason=reason,
            evaluation=evaluation,
            metrics={"passed": passed, "failed": failed, "floor_score": self.floor_score},
        )


class NoEvaluationPolicy(CandidateEvaluationPlugin):
    """Publish every candidate that passed validation, without running an episode.

    Purpose: the harness recipe builds it for ``policy: none``; its
    evaluation result holds no scores.
    """

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        return EvaluationResult(evaluator="none", evaluator_version="1", metrics={"evaluation_sides": []})

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        return SelectionDecision(True, "none", "the deployment publishes every valid candidate", evaluation)


def score_comparison_tally(candidate: tuple[float | None, ...], current: tuple[float | None, ...]) -> tuple[int, int]:
    """Wins and losses of ``candidate`` over ``current``, pair by pair; ``None`` ranks below every score."""
    wins = losses = 0
    for candidate_score, current_score in zip(candidate, current, strict=True):
        candidate_rank = candidate_score if candidate_score is not None else float("-inf")
        current_rank = current_score if current_score is not None else float("-inf")
        if candidate_rank > current_rank:
            wins += 1
        elif candidate_rank < current_rank:
            losses += 1
    return wins, losses


__all__ = [
    "POLICIES",
    "STATISTICS",
    "ComparePolicy",
    "EpisodePolicy",
    "FloorPolicy",
    "NoEvaluationPolicy",
    "score_comparison_tally",
]
