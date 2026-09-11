"""Built-in candidate evaluators and reusable selection policies."""

from __future__ import annotations

from reef.train.evaluation.contracts import (
    CandidateEvaluationPlugin,
    CandidateEvaluator,
    CandidateSelector,
    EvaluationResult,
    SelectionDecision,
    UpdateCandidate,
)


class AlwaysSelect(CandidateSelector):
    """Select every successfully evaluated candidate."""

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        return SelectionDecision(
            outcome="select",
            policy="always",
            policy_version="1",
            reason="the method selects every successfully evaluated candidate",
            evaluation=evaluation,
        )


class RegressionGateMixin:
    """Mixin giving a :class:`CandidateEvaluationPlugin` best-checkpoint ``decide()``.

    Where :class:`AlwaysSelect` publishes every evaluated candidate, a plugin that
    mixes this in reads one scalar metric from the evaluation and selects a
    candidate only while that score stays within ``margin`` of the best score
    selected so far; otherwise it rejects, and serving holds the last selected
    weights. That makes it best-checkpoint selection made online: an objective
    that has passed its peak cannot compound regressing steps into serving. The
    bar is seeded by the first candidate, so the initial climb is always admitted
    and the gate only bites once a peak exists to regress from.

    Combine it with a plugin that supplies the measurement::

        class MyPlugin(RegressionGateMixin, CandidateEvaluationPlugin):
            def __init__(self, ...):
                super().__init__(metric="clean_rate", margin=0.17)
                ...
            def evaluate(self, candidate): ...

    The metric is whatever ``evaluate`` records — a held-out score, an accuracy,
    a clean-output rate. ``higher_is_better=False`` gates a metric that improves
    as it falls (a loss, an error rate).
    """

    def __init__(self, *, metric: str, margin: float = 0.0, higher_is_better: bool = True) -> None:
        if not isinstance(metric, str) or not metric:
            raise ValueError("RegressionGateMixin needs a non-empty metric name")
        if margin < 0:
            raise ValueError("RegressionGateMixin margin must be non-negative")
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
                f"the regression gate's metric {self._metric!r} is missing or non-numeric in the evaluation"
            ) from exc
        score = self._oriented(raw)
        bar = score if self._best is None else self._best - self._margin
        best_raw = raw if self._best is None else (self._best if self._higher_is_better else -self._best)
        metrics = {"metric": self._metric, "value": raw, "best": best_raw, "margin": self._margin}
        if score >= bar:
            self._best = score if self._best is None else max(self._best, score)
            new_best_raw = self._best if self._higher_is_better else -self._best
            return SelectionDecision(
                outcome="select",
                policy="regression-gate",
                policy_version="1",
                reason=f"{self._metric} {raw:g} within margin of best {new_best_raw:g}",
                evaluation=evaluation,
                metrics=metrics,
            )
        return SelectionDecision(
            outcome="reject",
            policy="regression-gate",
            policy_version="1",
            reason=(
                f"{self._metric} {raw:g} regressed past margin {self._margin:g} below best {best_raw:g}; "
                "holding the last selected weights"
            ),
            evaluation=evaluation,
            metrics=metrics,
        )


class DefaultCandidateEvaluationPlugin(CandidateEvaluationPlugin):
    """Combine an evaluator with a selector that defaults to ``AlwaysSelect``."""

    def __init__(
        self,
        evaluator: CandidateEvaluator,
        selector: CandidateSelector | None = None,
    ) -> None:
        self._evaluator = evaluator
        self._selector = AlwaysSelect() if selector is None else selector

    @property
    def evaluator(self) -> CandidateEvaluator:
        """The component that supplies the candidate measurements."""
        return self._evaluator

    @property
    def selector(self) -> CandidateSelector:
        """The component that decides whether to publish the candidate."""
        return self._selector

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        return self._evaluator.evaluate(candidate)

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        return self._selector.decide(candidate, evaluation)


__all__ = ["AlwaysSelect", "DefaultCandidateEvaluationPlugin", "RegressionGateMixin"]
