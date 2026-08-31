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


__all__ = ["AlwaysSelect", "DefaultCandidateEvaluationPlugin"]
