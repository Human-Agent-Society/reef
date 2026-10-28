"""Built-in candidate-evaluation mixins, the default plugin, and the selection plugin base.

A harness recipe's ``evolution.selection`` names a :class:`SelectionPlugin`
class, which the recipe builds for each scenario with ``from_config``.

A :class:`~reef.core.evaluation.CandidateEvaluationPlugin` is a single
class that both measures a candidate (``evaluate``) and decides whether to
publish it (``decide``). Rather than composing an evaluator object with a
selector object, a plugin mixes in the methods it needs: a *decide* mixin
(:class:`AlwaysSelectMixin`, :class:`RegressionCheckMixin`) and, when the
measurement comes from the candidate backend rather than the plugin's own code,
the :class:`BackendEvaluateMixin`.

Every mixin inherits the plugin contract and implements one half of it, so the
other half stays abstract: a mixin cannot be instantiated on its own, and a
plugin is only concrete once both halves are supplied.

    class MyPlugin(RegressionCheckMixin):
        def __init__(self, ...):
            super().__init__(metric="clean_rate", margin=0.17)
            ...
        def evaluate(self, candidate): ...   # its own measurement

    class BackendScored(ScoreMixin, BackendEvaluateMixin):
        def __init__(self, candidate_backend, **kw):
            super().__init__(**kw)           # the decide mixin's config
            self._candidate_backend = candidate_backend          # BackendEvaluateMixin reads this
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from reef.core.evaluation import (
    CandidateEvaluationPlugin,
    CandidateEvaluator,
    CheckpointCandidate,
    EvaluationResult,
    RolloutScores,
    SelectionDecision,
    UpdateCandidate,
)
from reef.core.plugins import Plugin, PluginConfigError
from reef.evaluation.config import ConfiguredEvaluationPlugin
from reef.runtime.interfaces import InferenceRuntime


@dataclass(frozen=True)
class ScoreComparison:
    """How a candidate's rollouts compare with the current release's, pair by pair.

    Values: ``wins``, ``losses``, ``ties`` and ``dropped`` are non-negative ints, and their sum is the number of
    pairs. Purpose: ``compare_scores`` makes it; the ``score_comparison`` policy reads it.
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
    """Compare two releases' rollouts pair by pair: the same task and the same run on each side.

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


class SelectionPlugin(CandidateEvaluationPlugin, Plugin):
    """A candidate evaluation plugin that a harness recipe's ``evolution.selection`` names.

    The recipe builds one for each scenario with ``from_config``, from the
    setting's configuration and the scenario's candidate backend.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping[str, Any], *, candidate_backend: CandidateEvaluator) -> Self: ...


class AlwaysSelectMixin(CandidateEvaluationPlugin):
    """Give a plugin a ``decide()`` that publishes every evaluated candidate.

    Abstract on its own: ``evaluate`` is inherited unimplemented, so this
    mixes into a plugin rather than standing up as one.
    """

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        return SelectionDecision(
            selected=True,
            policy="always",
            reason="the method selects every successfully evaluated candidate",
            evaluation=evaluation,
        )


class RegressionCheckMixin(CandidateEvaluationPlugin):
    """Give a plugin a best-checkpoint ``decide()`` over one scalar metric.

    Where :class:`AlwaysSelectMixin` publishes every evaluated candidate, this
    reads one scalar metric from the evaluation and selects a candidate only
    while that score stays within ``margin`` of the best score selected so far;
    otherwise it rejects, and serving holds the last selected weights. That makes
    it best-checkpoint selection made online: an objective that has passed its
    peak cannot compound regressing steps into serving. The bar is seeded by the
    first candidate, so the initial climb is always admitted and the check
    applies once a peak exists to regress from.

    The metric is whatever ``evaluate`` records — a held-out score, an accuracy,
    a clean-output rate. ``higher_is_better=False`` checks a metric that improves
    as it falls (a loss, an error rate).
    """

    def __init__(self, *, metric: str, margin: float = 0.0, higher_is_better: bool = True) -> None:
        if not isinstance(metric, str) or not metric:
            raise ValueError("RegressionCheckMixin needs a non-empty metric name")
        if margin < 0:
            raise ValueError("RegressionCheckMixin margin must be non-negative")
        super().__init__()
        self._metric = metric
        self._margin = float(margin)
        self._higher_is_better = bool(higher_is_better)
        #: The best oriented score selected so far; ``None`` until the first.
        self._best: float | None = None

    def _oriented(self, value: float) -> float:
        """The score in higher-is-better orientation, so one comparison serves both."""
        return value if self._higher_is_better else -value

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        try:
            raw = float(evaluation.metrics[self._metric])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"the regression check's metric {self._metric!r} is missing or non-numeric in the evaluation"
            ) from exc
        score = self._oriented(raw)
        bar = score if self._best is None else self._best - self._margin
        best_raw = raw if self._best is None else (self._best if self._higher_is_better else -self._best)
        metrics = {"metric": self._metric, "value": raw, "best": best_raw, "margin": self._margin}
        if score >= bar:
            self._best = score if self._best is None else max(self._best, score)
            new_best_raw = self._best if self._higher_is_better else -self._best
            return SelectionDecision(
                selected=True,
                policy="regression-check",
                reason=f"{self._metric} {raw:g} within margin of best {new_best_raw:g}",
                evaluation=evaluation,
                metrics=metrics,
            )
        return SelectionDecision(
            selected=False,
            policy="regression-check",
            reason=(
                f"{self._metric} {raw:g} regressed past margin {self._margin:g} below best {best_raw:g}; "
                "holding the last selected weights"
            ),
            evaluation=evaluation,
            metrics=metrics,
        )


class BackendEvaluateMixin(CandidateEvaluationPlugin):
    """Give a plugin an ``evaluate()`` that delegates to ``self._candidate_backend``.

    For methods whose measurement is the candidate backend's own candidate
    evaluation rather than code in the plugin. The plugin sets ``self._candidate_backend``
    in its ``__init__``; this mixin holds no state of its own.
    """

    _candidate_backend: Any

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        return self._candidate_backend.evaluate(candidate)


class BackendAlwaysSelectPlugin(AlwaysSelectMixin, BackendEvaluateMixin, SelectionPlugin):
    """The default plugin of a harness trainer: evaluate via the candidate backend, publish every candidate.

    A harness trainer that configures no evaluation gets it, and a harness
    recipe's ``always`` selection names it. A weight trainer's default is
    ``AlwaysSelectCheckpointPlugin``.
    """

    def __init__(self, candidate_backend: Any) -> None:
        super().__init__()
        self._candidate_backend = candidate_backend

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, candidate_backend: CandidateEvaluator) -> Self:
        if config:
            raise PluginConfigError(f"the always selection takes no config, got {', '.join(sorted(map(str, config)))}")
        return cls(candidate_backend)


class AlwaysSelectCheckpointPlugin(AlwaysSelectMixin):
    """The default candidate evaluation of a weight trainer: serve every checkpoint that a job saves.

    ``evaluate`` measures nothing: its result has no metrics, since the commit of the decision already carries
    the checkpoint's training metrics. It takes a ``CheckpointCandidate`` and raises ``TypeError`` for another
    candidate.
    """

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        if not isinstance(candidate, CheckpointCandidate):
            raise TypeError(f"{type(self).__name__} evaluates a CheckpointCandidate, got {type(candidate).__name__}")
        return EvaluationResult(evaluator="training_job", evaluator_version="1", metrics={})


class ManualPromotionPlugin(ConfiguredEvaluationPlugin):
    """Activate no checkpoint: a person activates each artifact (``POST /reef/scenarios/{s}/activate``).

    ``evaluate`` measures nothing, as ``AlwaysSelectCheckpointPlugin`` does, and ``decide`` selects no
    candidate, so each checkpoint is an inactive artifact until a person activates it. A deployment names it
    in its ``evaluation`` section, which takes no config. It takes a ``CheckpointCandidate`` and raises
    ``TypeError`` for another candidate. The name stays until RFC #965 step 3 makes this the activation rule.
    """

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, runtime: InferenceRuntime, scenario: str, environ: Mapping[str, str]
    ) -> Self:
        if config:
            raise PluginConfigError(f"manual promotion takes no config, got {', '.join(sorted(map(str, config)))}")
        return cls()

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        if not isinstance(candidate, CheckpointCandidate):
            raise TypeError(f"{type(self).__name__} evaluates a CheckpointCandidate, got {type(candidate).__name__}")
        return EvaluationResult(evaluator="manual_promotion", evaluator_version="1", metrics={})

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        return SelectionDecision(
            selected=False, policy="manual", reason="a person activates each release", evaluation=evaluation
        )


__all__ = [
    "AlwaysSelectCheckpointPlugin",
    "AlwaysSelectMixin",
    "BackendAlwaysSelectPlugin",
    "BackendEvaluateMixin",
    "FloorCheck",
    "ManualPromotionPlugin",
    "RegressionCheckMixin",
    "ScoreComparison",
    "SelectionPlugin",
    "check_floor",
    "compare_scores",
]
