"""External-style activation rules and lookalikes used by the plugin seam tests."""

from __future__ import annotations

from reef.core.activation import ActivationRule, MadeArtifact
from reef.evaluation import CandidateEvaluator, EvaluationResult, UpdateCandidate


class ComponentRule(ActivationRule):
    """Activates the artifacts of the configured component only."""

    def __init__(self, component: str) -> None:
        self.component = component

    @classmethod
    def from_config(cls, config):
        return cls(str(config["component"]))

    def activates(self, made: MadeArtifact) -> bool:
        return made.component == self.component


class EvaluatorOnly(CandidateEvaluator):
    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        return EvaluationResult(evaluator="incomplete", evaluator_version="1", metrics={})


class DuckPlugin:
    def evaluate(self, candidate):
        raise AssertionError("a structural lookalike must be rejected before evaluation")

    def decide(self, candidate, evaluation):
        raise AssertionError("a structural lookalike must be rejected before selection")

    def activates(self, made):
        raise AssertionError("a structural lookalike must be rejected before activation")


class IncompleteRule(ActivationRule):
    pass


def plain_function(config):
    raise AssertionError("a function must not be invoked as a plugin")
