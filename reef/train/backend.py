"""Backend contract for producing and selecting training candidates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from reef.train.evaluation.contracts import CandidateEvaluator, EvaluationResult, SelectionDecision, UpdateCandidate
from reef.train.types import TrainingBatch, TrainStepResult


@dataclass(frozen=True)
class PreparedStep:
    """One backend preparation, with or without a candidate to select.

    ``state`` and ``metrics`` are final for a skipped step. For a candidate
    step they are the common preparation values that settlement may extend.
    """

    outcome: Literal["candidate", "skip", "retry", "drop"]
    state: Mapping[str, Any]
    metrics: Mapping[str, Any] = field(default_factory=dict)
    candidate: UpdateCandidate | None = None
    storage: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.outcome not in {"candidate", "skip", "retry", "drop"}:
            raise ValueError("prepared step outcome must be 'candidate', 'skip', 'retry', or 'drop'")
        if self.outcome == "candidate" and self.candidate is None:
            raise ValueError("a candidate step requires a candidate")
        if self.outcome != "candidate" and self.candidate is not None:
            raise ValueError(f"a {self.outcome} step cannot carry a candidate")
        if self.outcome == "retry" and self.storage is None:
            raise ValueError("a retry step requires its blocking storage status")
        if self.outcome != "retry" and self.storage is not None:
            raise ValueError(f"a {self.outcome} step cannot carry storage status")

    @classmethod
    def with_candidate(
        cls,
        candidate: UpdateCandidate,
        *,
        state: Mapping[str, Any],
        metrics: Mapping[str, Any] | None = None,
    ) -> PreparedStep:
        return cls("candidate", state, metrics or {}, candidate)

    @classmethod
    def skipped(
        cls,
        *,
        state: Mapping[str, Any],
        metrics: Mapping[str, Any] | None = None,
    ) -> PreparedStep:
        return cls("skip", state, metrics or {})

    @classmethod
    def retrying(
        cls,
        *,
        state: Mapping[str, Any],
        storage: Mapping[str, Any],
        metrics: Mapping[str, Any] | None = None,
    ) -> PreparedStep:
        return cls("retry", state, metrics or {}, storage=storage)

    @classmethod
    def dropped(
        cls,
        *,
        state: Mapping[str, Any],
        metrics: Mapping[str, Any] | None = None,
    ) -> PreparedStep:
        return cls("drop", state, metrics or {})


@dataclass(frozen=True)
class StepExecution:
    """One backend attempt returned to the dispatcher."""

    outcome: Literal["commit", "retry", "drop"]
    result: TrainStepResult | None = None
    storage: Mapping[str, Any] | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome == "commit" and self.result is None:
            raise ValueError("a committed execution requires a training result")
        if self.outcome != "commit" and self.result is not None:
            raise ValueError(f"a {self.outcome} execution cannot carry a training result")
        if self.outcome == "retry" and self.storage is None:
            raise ValueError("a retry execution requires its blocking storage status")
        if self.outcome != "retry" and self.storage is not None:
            raise ValueError(f"a {self.outcome} execution cannot carry storage status")


class TrainingBackend(CandidateEvaluator, ABC):
    """Prepare and evaluate updates while Reef owns candidate selection.

    The backend owns method-specific candidate construction and settlement,
    and supplies the default evaluator. A recipe may inject a cohesive
    :class:`reef.train.evaluation.CandidateEvaluationPlugin`; otherwise the trainer
    wraps this evaluator in :class:`reef.train.evaluation.DefaultCandidateEvaluationPlugin`.
    Every backend therefore follows the same evaluate-then-decide lifecycle
    between preparation and settlement.
    """

    @abstractmethod
    def initial_state(self) -> Mapping[str, Any]: ...

    @property
    def dispatched(self) -> bool:
        """Whether the dispatcher must run this backend outside scenario locks."""
        return False

    def recover_pending_step(
        self,
        scenario_step: int,
        *,
        committed_training_job_id: str | None,
        committed_training_without_job_id: bool,
    ) -> None:
        """Finish or roll back backend work left pending across a restart."""
        return

    def acknowledge_commit(self, scenario_step: int, training_job_id: str) -> None:
        """Acknowledge that Reef durably committed a backend training job."""
        return

    def experiment_config(self) -> Mapping[str, Any]:
        """Non-secret backend identity/config attached to experiment runs."""
        return {}

    @abstractmethod
    def prepare_step(
        self,
        batch: TrainingBatch,
        state: Mapping[str, Any],
        scenario_step: int,
    ) -> PreparedStep: ...

    @abstractmethod
    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult: ...

    @abstractmethod
    def settle_step(
        self,
        prepared: PreparedStep,
        decision: SelectionDecision,
    ) -> TrainStepResult: ...

    @abstractmethod
    def abort_step(self, prepared: PreparedStep) -> None:
        """Restore backend-local state after evaluation or settlement fails."""


__all__ = ["PreparedStep", "StepExecution", "TrainingBackend"]
