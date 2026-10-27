"""The candidate lifecycle of harness evolution backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from reef.core.evaluation import CandidateEvaluator, EvaluationResult, SelectionDecision, UpdateCandidate
from reef.train.types import JobData, TrainStepResult


@dataclass(frozen=True)
class PreparedStep:
    """One backend preparation, with or without a candidate to select.

    ``outcome`` is ``"candidate"`` exactly when ``candidate`` is given.
    ``state`` and ``metrics`` are final for a skipped step. For a candidate
    step they are the common preparation values that settlement may extend.
    """

    outcome: Literal["candidate", "skip"]
    state: Mapping[str, Any]
    metrics: Mapping[str, Any] = field(default_factory=dict)
    candidate: UpdateCandidate | None = None

    def __post_init__(self) -> None:
        if self.outcome not in {"candidate", "skip"}:
            raise ValueError("prepared step outcome must be 'candidate' or 'skip'")
        if (self.outcome == "candidate") != (self.candidate is not None):
            raise ValueError("a candidate step, and only a candidate step, carries a candidate")

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


class CandidateBackend(CandidateEvaluator, ABC):
    """Prepare and evaluate updates while Reef owns candidate selection.

    The backend owns method-specific candidate construction and settlement,
    and supplies the default evaluator. A recipe may inject a cohesive
    :class:`reef.evaluation.CandidateEvaluationPlugin`; otherwise the trainer
    wraps this evaluator in :class:`reef.evaluation.BackendAlwaysSelectPlugin`.
    Every backend therefore follows the same evaluate-then-decide lifecycle
    between preparation and settlement.
    """

    @abstractmethod
    def initial_state(self) -> Mapping[str, Any]: ...

    def close(self) -> None:
        """Release resources owned by this backend; safe to call repeatedly."""
        return

    @property
    def harness_node_paths(self) -> Mapping[str, str] | None:
        """Root-relative render paths of the harness node kinds; ``None`` for a backend that evolves no harness."""
        return None

    @property
    def harness_adapter(self) -> str | None:
        """The name of the harness adapter the backend evolves; ``None`` for a backend that evolves no harness."""
        return None

    def commit_applied(self, state: Mapping[str, Any]) -> None:
        """Observe algorithm state only after its scenario commit is durable.

        Backends normally need no side effect here. A backend that maintains
        a derived local mirror may update it from ``state`` without letting
        speculative prepare/evaluate work escape Reef's commit boundary.
        """
        return

    def shipped_content_update(self, state: Mapping[str, Any], published_tree: Path) -> TrainStepResult | None:
        """A result that republishes the content this Reef ships, when the served tree no longer carries it.

        Content a backend ships with Reef itself (the harness backend's reef-owned
        entries) is fixed when a scenario is created and carried unchanged by every
        later step, so a scenario opened by an upgraded Reef compares it against
        ``published_tree``, the served release's files. The result carries the
        refreshed ``state`` and a durable ``artifact``; it is committed without an
        evaluation. The default ships nothing.
        """
        return None

    def experiment_config(self) -> Mapping[str, Any]:
        """Non-secret backend identity/config attached to experiment runs."""
        return {}

    def operational_metrics(self) -> Mapping[str, float | int]:
        """Nonblocking process-local measurements, independent of successful commits."""
        return {}

    def failed_step_metrics(self) -> Mapping[str, Any]:
        """Metadata to retain when the current instruction fails before producing a result."""
        return {}

    @abstractmethod
    def prepare_step(
        self,
        data: JobData,
        state: Mapping[str, Any],
        scenario_step: int,
    ) -> PreparedStep: ...

    @abstractmethod
    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        """Measure ``candidate``; the trainer's default plugin calls this with the candidate alone.

        A backend may accept further keyword-only options with defaults (the
        harness backend's ``sides``) for the plugins it ships; this contract
        stays the one every plugin can rely on.
        """

    @abstractmethod
    def settle_step(
        self,
        prepared: PreparedStep,
        decision: SelectionDecision,
    ) -> TrainStepResult: ...

    @abstractmethod
    def abort_step(self, prepared: PreparedStep) -> None:
        """Restore backend-local state after evaluation or settlement fails."""


__all__ = ["CandidateBackend", "PreparedStep"]
