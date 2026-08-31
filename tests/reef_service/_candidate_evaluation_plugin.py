"""External-style checkpoint evaluator used by the plugin seam tests."""

from __future__ import annotations

from dataclasses import dataclass

from reef.runtime import ModelCandidate
from reef.train.evaluation import (
    CandidateEvaluationPlugin,
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


def build_evaluator(config, *, runtime, scenario, environ):
    del runtime
    token_env = config.get("token_env")
    return CheckpointEvaluator(
        score=float(config["score"]),
        threshold=float(config["threshold"]),
        scenario=scenario,
        token=environ.get(token_env) if token_env else None,
    )


def build_evaluator_only(config, *, runtime, scenario, environ):
    del runtime, scenario, environ
    return EvaluatorOnly(score=float(config["score"]))


def build_invalid(config, *, runtime, scenario, environ):
    del config, runtime, scenario, environ
    return object()
