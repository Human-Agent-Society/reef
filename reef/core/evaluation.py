"""Artifact-agnostic contracts for evaluating and selecting candidate updates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class UpdateCandidate:
    """One update that has been produced but not selected for serving.

    ``candidate_id`` is the stable idempotency and audit key for the attempt.
    """

    candidate_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")


@dataclass(frozen=True)
class EvaluationResult:
    """Measurements produced by evaluating one candidate against the incumbent."""

    evaluator: str
    evaluator_version: str
    metrics: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.evaluator, str) or not self.evaluator:
            raise ValueError("evaluator must be a non-empty string")
        if not isinstance(self.evaluator_version, str) or not self.evaluator_version:
            raise ValueError("evaluator_version must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator": self.evaluator,
            "evaluator_version": self.evaluator_version,
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SelectionDecision:
    """The durable, explainable decision for one evaluated candidate."""

    selected: bool
    policy: str
    reason: str
    evaluation: EvaluationResult
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.selected, bool):
            raise ValueError("selected must be a boolean")
        if not isinstance(self.policy, str) or not self.policy:
            raise ValueError("policy must be a non-empty string")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "policy": self.policy,
            "reason": self.reason,
            "evaluation": self.evaluation.to_dict(),
            "metrics": dict(self.metrics),
        }


class CandidateSelector(ABC):
    """Turn an evaluation result into a select-or-reject decision.

    Implementations inherit this contract explicitly. Evaluation and selection
    remain independent capabilities; a complete plugin implements both.
    """

    @abstractmethod
    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision: ...


class CandidateEvaluator(ABC):
    """Measure a candidate without deciding whether to publish it."""

    @abstractmethod
    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult: ...


class CandidateEvaluationPlugin(CandidateEvaluator, CandidateSelector):
    """Measure a candidate and decide whether Reef serves it."""

    def evaluate_and_decide(self, candidate: UpdateCandidate) -> SelectionDecision:
        """Evaluate ``candidate`` and decide on that evaluation.

        Raise ``ValueError`` when the decision does not carry the evaluation
        that ``evaluate`` returned, since Reef records the two together.
        """
        evaluation = self.evaluate(candidate)
        decision = self.decide(candidate, evaluation)
        if decision.evaluation is not evaluation:
            raise ValueError("candidate evaluator must retain the evaluation result supplied by Reef")
        return decision


__all__ = [
    "CandidateEvaluationPlugin",
    "CandidateEvaluator",
    "CandidateSelector",
    "EvaluationResult",
    "SelectionDecision",
    "UpdateCandidate",
]
