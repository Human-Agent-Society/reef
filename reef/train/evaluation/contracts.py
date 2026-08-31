"""Artifact-agnostic contracts for evaluating and selecting candidate updates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class UpdateCandidate:
    """One update that has been produced but not selected for serving.

    ``candidate_id`` is the stable idempotency and audit key for the attempt.
    ``current_version`` names the incumbent when the producing layer knows it;
    local artifact methods may leave it ``None`` and bind the version later in
    the scenario coordinator.
    """

    candidate_id: str
    current_version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if self.current_version is not None and (
            not isinstance(self.current_version, str) or not self.current_version
        ):
            raise ValueError("current_version must be a non-empty string or None")


@dataclass(frozen=True)
class EvaluationResult:
    """Measurements produced by evaluating one candidate against its current version."""

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

    outcome: Literal["select", "reject"]
    policy: str
    policy_version: str
    reason: str
    evaluation: EvaluationResult
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome not in {"select", "reject"}:
            raise ValueError("selection outcome must be 'select' or 'reject'")
        if not isinstance(self.policy, str) or not self.policy:
            raise ValueError("policy must be a non-empty string")
        if not isinstance(self.policy_version, str) or not self.policy_version:
            raise ValueError("policy_version must be a non-empty string")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")

    @property
    def selected(self) -> bool:
        return self.outcome == "select"

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "policy": self.policy,
            "policy_version": self.policy_version,
            "reason": self.reason,
            "evaluation": self.evaluation.to_dict(),
            "metrics": dict(self.metrics),
        }


@runtime_checkable
class CandidateSelector(Protocol):
    """Turn an evaluation result into a select-or-reject decision."""

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision: ...


@runtime_checkable
class CandidateEvaluator(Protocol):
    """Measure a candidate without deciding whether to publish it."""

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult: ...


@runtime_checkable
class CandidateEvaluationPlugin(CandidateEvaluator, CandidateSelector, Protocol):
    """Measure a candidate and decide whether it passes the publication gate."""


__all__ = [
    "CandidateEvaluationPlugin",
    "CandidateEvaluator",
    "CandidateSelector",
    "EvaluationResult",
    "SelectionDecision",
    "UpdateCandidate",
]
