"""External-style checkpoint evaluator used by the plugin seam tests."""

from __future__ import annotations

from dataclasses import dataclass

from reef.runtime.interfaces import ModelCandidate
from reef.train.evaluation import (
    CandidateEvaluationPlugin,
    CandidateEvaluationPluginFactory,
    CandidateEvaluator,
    EvaluationResult,
    SelectionDecision,
    UpdateCandidate,
)


@dataclass(frozen=True)
class CheckpointEvaluator(CandidateEvaluationPlugin):
    score: float
    threshold: float
    scenario: str
    token: str | None

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        if not isinstance(candidate, ModelCandidate):
            raise TypeError("checkpoint evaluation requires a ModelCandidate")
        return EvaluationResult(
            evaluator="external_checkpoint",
            evaluator_version="1",
            metrics={"score": self.score, "checkpoint_path": candidate.checkpoint_path},
            metadata={"scenario": self.scenario, "token_present": self.token is not None},
        )

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        del candidate
        score = float(evaluation.metrics["score"])
        selected = score >= self.threshold
        return SelectionDecision(
            outcome="select" if selected else "reject",
            policy="external_threshold",
            policy_version="1",
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


class CheckpointFactory(CandidateEvaluationPluginFactory):
    def build(self, config, *, runtime, training_runtime, scenario, environ):
        del runtime
        token_env = config.get("token_env")
        return CheckpointEvaluator(
            score=float(config["score"]),
            threshold=float(config["threshold"]),
            scenario=scenario,
            token=environ.get(token_env) if token_env else None,
        )


class EvaluatorOnlyFactory(CandidateEvaluationPluginFactory):
    def build(self, config, *, runtime, training_runtime, scenario, environ):
        del runtime, scenario, environ
        return EvaluatorOnly(score=float(config["score"]))


class InvalidFactory(CandidateEvaluationPluginFactory):
    def build(self, config, *, runtime, training_runtime, scenario, environ):
        del config, runtime, scenario, environ
        return object()


class DuckPlugin:
    def evaluate(self, candidate):
        raise AssertionError("a structural lookalike must be rejected before evaluation")

    def decide(self, candidate, evaluation):
        raise AssertionError("a structural lookalike must be rejected before selection")


class DuckFactory(CandidateEvaluationPluginFactory):
    def build(self, config, *, runtime, training_runtime, scenario, environ):
        return DuckPlugin()


class IncompleteFactory(CandidateEvaluationPluginFactory):
    pass


def plain_factory(config, *, runtime, training_runtime, scenario, environ):
    raise AssertionError("a callable without the factory contract must not be invoked")
