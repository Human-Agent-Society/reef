"""External-style checkpoint evaluator used by the plugin seam tests."""

from __future__ import annotations

from dataclasses import dataclass

from reef.core.evaluation import CheckpointCandidate
from reef.evaluation import (
    CandidateEvaluator,
    ConfiguredEvaluationPlugin,
    EvaluationResult,
    SelectionDecision,
    UpdateCandidate,
)


@dataclass(frozen=True)
class CheckpointEvaluator(ConfiguredEvaluationPlugin):
    score: float
    threshold: float
    scenario: str
    token: str | None

    @classmethod
    def from_config(cls, config, *, runtime, scenario, environ):
        token_env = config.get("token_env")
        return cls(
            score=float(config["score"]),
            threshold=float(config["threshold"]),
            scenario=scenario,
            token=environ.get(token_env) if token_env else None,
        )

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        if not isinstance(candidate, CheckpointCandidate):
            raise TypeError("checkpoint evaluation requires a CheckpointCandidate")
        return EvaluationResult(
            evaluator="external_checkpoint",
            evaluator_version="1",
            metrics={"score": self.score, "release_path": candidate.release_path},
            metadata={"scenario": self.scenario, "token_present": self.token is not None},
        )

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        del candidate
        score = float(evaluation.metrics["score"])
        selected = score >= self.threshold
        return SelectionDecision(
            selected=selected,
            policy="external_threshold",
            reason=f"score {score} {'met' if selected else 'missed'} threshold {self.threshold}",
            evaluation=evaluation,
            metrics={"threshold": self.threshold},
        )


@dataclass(frozen=True)
class EvaluatorOnly(CandidateEvaluator):
    score: float

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        del candidate
        return EvaluationResult(
            evaluator="incomplete",
            evaluator_version="1",
            metrics={"score": self.score},
        )


class DuckPlugin:
    def evaluate(self, candidate):
        raise AssertionError("a structural lookalike must be rejected before evaluation")

    def decide(self, candidate, evaluation):
        raise AssertionError("a structural lookalike must be rejected before selection")


class IncompletePlugin(ConfiguredEvaluationPlugin):
    pass


def plain_function(config, *, runtime, scenario, environ):
    raise AssertionError("a function must not be invoked as a plugin")
