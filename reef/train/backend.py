"""Candidate lifecycle shared by weight training and harness evolution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from reef.core.evaluation import CandidateEvaluator, EvaluationResult, SelectionDecision, UpdateCandidate
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


@dataclass(frozen=True)
class ReleaseFiles:
    """One component's files in one committed release.

    Values: ``release_id`` is the non-empty id of the release; ``files`` maps
    each relative path of the component's view of that release to its UTF-8
    text, without the repository's own files.
    Purpose: ``CommittedReleases`` makes it; a candidate backend builds its
    head, or a rollback target, from ``files``.
    """

    release_id: str
    files: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.release_id, str) or not self.release_id:
            raise ValueError("ReleaseFiles.release_id must be a non-empty string")


class CommittedReleases(ABC):
    """The releases a scenario's commits recorded, read for one component at a time.

    Purpose: a candidate backend that builds each candidate from the committed
    head reads it here, and the scenario layer implements it over its commit
    history and artifact repository. A component's head is the release of its
    latest commit that published one (a pending release included) or that
    moved the head (a rollback or a promote). Before any such commit, the head
    is the base release.
    """

    @abstractmethod
    def head(self, component: str) -> ReleaseFiles:
        """``component``'s files in its head release."""

    @abstractmethod
    def release(self, release_id: str, component: str) -> ReleaseFiles:
        """``component``'s files in release ``release_id``; ``ArtifactNotFound`` for an unknown release."""


@dataclass(frozen=True)
class ComponentReleases:
    """One component's view of a scenario's ``CommittedReleases``.

    Values: ``releases`` is the scenario's reader, and ``component`` names a
    component of the scenario's release, non-empty.
    Purpose: ``Recipe.build_trainers`` makes one per component and hands it
    to ``Recipe.build``; a candidate backend reads its head through it.
    """

    releases: CommittedReleases
    component: str

    def __post_init__(self) -> None:
        if not isinstance(self.component, str) or not self.component:
            raise ValueError("ComponentReleases.component must be a non-empty string")

    def head(self) -> ReleaseFiles:
        return self.releases.head(self.component)

    def release(self, release_id: str) -> ReleaseFiles:
        return self.releases.release(release_id, self.component)


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

    def retire_scenario(self, scenario: str) -> None:
        """The scenario is being deleted: release what the backend holds for it beyond Reef's own state.

        The default releases nothing. A backend that keeps per-scenario
        residency in a serving engine or publication history in a training job overrides
        this to let those go; until it does, the deletion is Reef-side only
        and the engine keeps the scenario's adapter until it is evicted or
        the group restarts.
        """

    def close(self) -> None:
        """Release resources owned by this backend; safe to call repeatedly."""
        return

    @property
    def dispatched(self) -> bool:
        """Whether the dispatcher must run this backend outside scenario locks."""
        return False

    @property
    def colocated(self) -> bool:
        """Whether a dispatched job holds the served engine, so no other component can evaluate meanwhile."""
        return False

    def recover_pending_step(self, scenario_step: int, *, committed_training_job_id: str | None) -> None:
        """Finish or roll back backend work left pending across a restart."""
        return

    def acknowledge_commit(self, scenario_step: int, training_job_id: str) -> None:
        """Acknowledge that Reef durably committed a backend training job."""
        return

    def commit_applied(self, state: Mapping[str, Any]) -> None:
        """Observe algorithm state only after its scenario commit is durable.

        Backends normally need no side effect here. A backend that maintains
        a derived local mirror may update it from ``state`` without letting
        speculative prepare/evaluate work escape Reef's commit boundary.
        """
        return

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
        batch: TrainingBatch,
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


__all__ = [
    "CandidateBackend",
    "CommittedReleases",
    "ComponentReleases",
    "PreparedStep",
    "ReleaseFiles",
    "StepExecution",
]
