"""The rules that read rollout scores.

The harness training component builds its publication policy from ``evolution.evaluation``
(``reef.evaluation.harness.policy``). The policies decide by the rules here:
``compare_scores`` pairs two sides' rollouts and ``check_floor`` checks one
side against a floor score.
"""

from __future__ import annotations

from dataclasses import dataclass

from reef.core.evaluation import RolloutScores


@dataclass(frozen=True)
class ScoreComparison:
    """How a candidate's rollouts compare with the current snapshot's, pair by pair.

    Values: ``wins``, ``losses``, ``ties`` and ``dropped`` are non-negative ints, and their sum is the number of
    pairs. Purpose: ``compare_scores`` makes it; the ``compare`` policy reads it.
    """

    wins: int
    losses: int
    ties: int
    dropped: int

    def __post_init__(self) -> None:
        if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in self.counts()):
            raise ValueError("ScoreComparison counts are non-negative ints")

    def counts(self) -> tuple[int, int, int, int]:
        return self.wins, self.losses, self.ties, self.dropped


def compare_scores(candidate: RolloutScores, current: RolloutScores) -> ScoreComparison:
    """Compare two snapshots' rollouts pair by pair: the same task and the same run on each side.

    The rollouts of a pair rank by their values. A pair with a failed side is dropped: the environment, not
    either policy, decided it. ``ValueError`` when the two did not run the same tasks the same number of times.
    """
    if [task for task, _ in candidate.tasks] != [
        task for task, _ in current.tasks
    ] or candidate.count != current.count:
        raise ValueError("compared releases run the same tasks the same number of times")

    wins = losses = ties = dropped = 0
    for mine, theirs in zip(candidate.rollouts(), current.rollouts(), strict=True):
        if mine.value is None or theirs.value is None:
            dropped += 1
        elif mine.value > theirs.value:
            wins += 1
        elif mine.value < theirs.value:
            losses += 1
        else:
            ties += 1
    return ScoreComparison(wins, losses, ties, dropped)


@dataclass(frozen=True)
class FloorCheck:
    """Which tasks of a candidate met a floor score.

    Values: ``passed`` and ``failed`` are non-negative ints whose sum is the number of tasks, and
    ``failed_rollouts`` is the non-negative number of rollouts that failed. Purpose: ``check_floor`` makes it;
    the ``floor`` policy reads it.
    """

    passed: int
    failed: int
    failed_rollouts: int

    def __post_init__(self) -> None:
        if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in self.counts()):
            raise ValueError("FloorCheck counts are non-negative ints")

    def counts(self) -> tuple[int, int, int]:
        return self.passed, self.failed, self.failed_rollouts


def check_floor(candidate: RolloutScores, floor_score: float) -> FloorCheck:
    """Check each task of ``candidate`` against ``floor_score``.

    A task passes when at least one of its rollouts did not fail and every such rollout is at least
    ``floor_score``. A failed rollout does not fail its task, but a task whose rollouts all failed does not pass.
    """
    passed = 0
    for _, scores in candidate.tasks:
        values = [score.value for score in scores if score.value is not None]
        if values and all(value >= floor_score for value in values):
            passed += 1
    return FloorCheck(passed, len(candidate.tasks) - passed, candidate.failed_count())


__all__ = [
    "FloorCheck",
    "ScoreComparison",
    "check_floor",
    "compare_scores",
]
