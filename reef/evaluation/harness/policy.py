"""The publication policies of harness evolution: how a candidate is evaluated, and when the step keeps it.

A kept candidate becomes a new artifact; the scenario's activation rule then
decides whether it serves. ``evolution.evaluation.policy`` names one policy:

``compare``
    Score the candidate and the head tree on the same (task, repeat) pairs.
    ``statistic: wins`` keeps the candidate when wins minus losses exceed
    ``min_margin``, by ``compare_scores``: a pair with a failed run is
    dropped. ``statistic: mean`` keeps it when its mean exceeds the head's
    mean by more than ``min_margin``, by ``mean_score``: each mean averages
    the runs that did not fail. The head's scores usually come from the
    score cache.
``floor``
    Score the candidate alone; keep it when every task meets ``floor_score``,
    by ``check_floor``: a failed run does not fail its task, but a task
    whose runs all failed does not pass.
``none``
    Run no task; keep every candidate that passed validation.
"""

from __future__ import annotations

import math
from abc import ABC
from typing import Any, ClassVar

from reef.core.evaluation import (
    CandidateEvaluationPlugin,
    EvaluationResult,
    RolloutScores,
    SelectionDecision,
    UpdateCandidate,
    mean_score,
)
from reef.evaluation.evaluators import check_floor, compare_scores
from reef.evaluation.harness.evaluator import EVALUATION_SIDES, HarnessCandidateEvaluator

#: The ``evolution.evaluation.policy`` names, and the ``statistic`` names of ``compare``.
POLICIES = ("compare", "floor", "none")
STATISTICS = ("wins", "mean")


class TaskScorePolicy(CandidateEvaluationPlugin, ABC):
    """A publication policy that decides by the scores of evaluation runs.

    Values: ``evaluator`` scores the candidate; ``sides`` are the sides each
    evaluation scores, a non-empty subset of ``EVALUATION_SIDES`` in its order.
    Purpose: the harness recipe builds one per scenario from
    ``evolution.evaluation``; the trainer calls ``evaluate`` and then
    ``decide`` on each step's candidate. ``decide`` reads each side's
    ``RolloutScores`` JSON, ``candidate_scores`` and ``current_scores``.
    """

    sides: ClassVar[tuple[str, ...]] = EVALUATION_SIDES

    def __init__(self, evaluator: HarnessCandidateEvaluator) -> None:
        self.evaluator = evaluator

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        return self.evaluator.evaluate(candidate, sides=self.sides)


class ComparePolicy(TaskScorePolicy):
    """Keep a candidate that scores better than the head tree on the same tasks.

    Values: ``statistic`` is one of ``STATISTICS``; ``min_margin`` ≥ 0 is
    the margin the candidate must exceed: in wins minus losses for ``wins``,
    in mean score for ``mean``.
    Operations: ``decide`` selects exactly when the evaluation scored at least
    one task and, for ``wins``, ``compare_scores`` gives wins minus losses
    above ``min_margin``; for ``mean``, both sides have a ``mean_score`` and
    the candidate's exceeds the head's by more than ``min_margin``.
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
        if not evaluation.metrics.get("candidate_scores"):
            return SelectionDecision(False, "compare", "no evaluation task was scored", evaluation)
        candidate_scores = RolloutScores.from_json(evaluation.metrics["candidate_scores"])
        current_scores = RolloutScores.from_json(evaluation.metrics["current_scores"])
        metrics: dict[str, Any] = {"statistic": self.statistic, "min_margin": self.min_margin}
        if self.statistic == "wins":
            comparison = compare_scores(candidate_scores, current_scores)
            wins, losses = comparison.wins, comparison.losses
            metrics.update({"wins": wins, "losses": losses, "ties": comparison.ties, "dropped": comparison.dropped})
            selected = wins - losses > self.min_margin
            reason = (
                f"candidate won {wins} task pairings and lost {losses}"
                if selected
                else f"candidate did not exceed its {losses} losses with {wins} wins"
            )
            if comparison.dropped:
                reason += f"; {comparison.dropped} pairings with a failed run were dropped"
            return SelectionDecision(selected, "compare", reason, evaluation, metrics)
        candidate_mean = mean_score(candidate_scores.rollouts())
        current_mean = mean_score(current_scores.rollouts())
        if candidate_mean is None or current_mean is None:
            side = "candidate" if candidate_mean is None else "current tree"
            return SelectionDecision(
                False, "compare", f"every run of the {side} failed, so it has no mean", evaluation, metrics
            )
        metrics.update({"candidate_mean": candidate_mean, "current_mean": current_mean})
        selected = candidate_mean - current_mean > self.min_margin
        reason = (
            f"candidate scored {candidate_mean:.4f} against {current_mean:.4f} "
            f"over {len(candidate_scores.rollouts())} runs"
        )
        return SelectionDecision(selected, "compare", reason, evaluation, metrics)


class FloorPolicy(TaskScorePolicy):
    """Keep a candidate whose every task meets ``floor_score``; the head tree does not run.

    Values: ``floor_score`` is a positive finite number.
    Operations: ``decide`` selects exactly when the evaluation scored at least
    one task and ``check_floor`` fails none: each task has a run that did
    not fail, and every such run scores at least ``floor_score``.
    """

    sides = ("candidate",)

    def __init__(self, evaluator: HarnessCandidateEvaluator, *, floor_score: float = 1.0) -> None:
        if isinstance(floor_score, bool) or not math.isfinite(floor_score) or floor_score <= 0:
            raise ValueError("floor_score must be a positive number")
        super().__init__(evaluator)
        self.floor_score = float(floor_score)

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        if not evaluation.metrics.get("candidate_scores"):
            return SelectionDecision(
                False, "floor", "no evaluation task was scored", evaluation, {"floor_score": self.floor_score}
            )
        floor = check_floor(RolloutScores.from_json(evaluation.metrics["candidate_scores"]), self.floor_score)
        tasks = floor.passed + floor.failed
        if floor.failed:
            reason = f"candidate missed the floor on {floor.failed} of {tasks} tasks"
        else:
            reason = f"candidate met the floor on all {tasks} tasks"
        return SelectionDecision(
            selected=not floor.failed,
            policy="floor",
            reason=reason,
            evaluation=evaluation,
            metrics={
                "passed": floor.passed,
                "failed": floor.failed,
                "failed_rollouts": floor.failed_rollouts,
                "floor_score": self.floor_score,
            },
        )


class NoEvaluationPolicy(CandidateEvaluationPlugin):
    """Keep every candidate that passed validation, without running a run.

    Purpose: the harness recipe builds it for ``policy: none``; its
    evaluation result holds no scores.
    """

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        return EvaluationResult(evaluator="none", evaluator_version="1", metrics={"evaluation_sides": []})

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        return SelectionDecision(True, "none", "the deployment keeps every valid candidate", evaluation)


__all__ = [
    "POLICIES",
    "STATISTICS",
    "ComparePolicy",
    "FloorPolicy",
    "NoEvaluationPolicy",
    "TaskScorePolicy",
]
