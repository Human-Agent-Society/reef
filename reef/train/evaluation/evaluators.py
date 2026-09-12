"""Built-in candidate-evaluation mixins and the default plugin.

A :class:`~reef.core.evaluation.CandidateEvaluationPlugin` is a single
class that both measures a candidate (``evaluate``) and decides whether to
publish it (``decide``). Rather than composing an evaluator object with a
selector object, a plugin mixes in the methods it needs: a *decide* mixin
(:class:`AlwaysSelectMixin`, :class:`RegressionGateMixin`) and, when the
measurement comes from the training backend rather than the plugin's own code,
the :class:`BackendEvaluateMixin`.

Every mixin inherits the plugin contract and implements one half of it, so the
other half stays abstract: a mixin cannot be instantiated on its own, and a
plugin is only concrete once both halves are supplied.

    class MyPlugin(RegressionGateMixin):
        def __init__(self, ...):
            super().__init__(metric="clean_rate", margin=0.17)
            ...
        def evaluate(self, candidate): ...   # its own measurement

    class BackendScored(ScoreMixin, BackendEvaluateMixin):
        def __init__(self, backend, **kw):
            super().__init__(**kw)           # the decide mixin's config
            self._backend = backend          # BackendEvaluateMixin reads this
"""

from __future__ import annotations

from typing import Any

from reef.core.evaluation import CandidateEvaluationPlugin, EvaluationResult, SelectionDecision, UpdateCandidate


class AlwaysSelectMixin(CandidateEvaluationPlugin):
    """Give a plugin a ``decide()`` that publishes every evaluated candidate.

    Abstract on its own: ``evaluate`` is inherited unimplemented, so this
    mixes into a plugin rather than standing up as one.
    """

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        return SelectionDecision(
            outcome="select",
            policy="always",
            policy_version="1",
            reason="the method selects every successfully evaluated candidate",
            evaluation=evaluation,
        )


class RegressionGateMixin(CandidateEvaluationPlugin):
    """Give a plugin a best-checkpoint ``decide()`` over one scalar metric.

    Where :class:`AlwaysSelectMixin` publishes every evaluated candidate, this
    reads one scalar metric from the evaluation and selects a candidate only
    while that score stays within ``margin`` of the best score selected so far;
    otherwise it rejects, and serving holds the last selected weights. That makes
    it best-checkpoint selection made online: an objective that has passed its
    peak cannot compound regressing steps into serving. The bar is seeded by the
    first candidate, so the initial climb is always admitted and the gate only
    bites once a peak exists to regress from.

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


class BackendEvaluateMixin(CandidateEvaluationPlugin):
    """Give a plugin an ``evaluate()`` that delegates to ``self._backend``.

    For methods whose measurement is the training backend's own candidate
    evaluation rather than code in the plugin. The plugin sets ``self._backend``
    in its ``__init__``; this mixin holds no state of its own.
    """

    _backend: Any

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        return self._backend.evaluate(candidate)


class BackendAlwaysSelectPlugin(AlwaysSelectMixin, BackendEvaluateMixin):
    """The default plugin: evaluate via the training backend, publish every candidate.

    What a weight-training deployment gets when it configures no evaluation — the
    same behaviour reef had before candidate gating existed.
    """

    def __init__(self, backend: Any) -> None:
        super().__init__()
        self._backend = backend


__all__ = [
    "AlwaysSelectMixin",
    "BackendAlwaysSelectPlugin",
    "BackendEvaluateMixin",
    "RegressionGateMixin",
]
