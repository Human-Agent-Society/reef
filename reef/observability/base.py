"""Provider-neutral experiment logging contracts for Reef scenarios."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.core.artifact_ref import ArtifactRef


@dataclass(frozen=True, slots=True)
class TrainingExperimentContext:
    """Stable context known before a Reef training result is committed."""

    scenario: str
    recipe: str
    step: int
    source_artifact_ref: ArtifactRef
    run_segment: int = 0
    run_step: int = 0
    backend: str | None = None
    backend_config: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class TrainingExperimentEvent:
    """One durably committed training result and its final Reef version."""

    context: TrainingExperimentContext
    produced_artifact_ref: ArtifactRef
    metrics: Mapping[str, Any]
    outcome: str
    training_job_id: str | None = None
    source_weight_version: str | None = None
    produced_weight_version: str | None = None
    checkpoint_path: str | None = None


@dataclass(frozen=True, slots=True)
class RollbackExperimentEvent:
    """A committed rollback that ends one training-curve segment."""

    scenario: str
    recipe: str
    step: int
    run_segment: int
    source_artifact_ref: ArtifactRef
    produced_artifact_ref: ArtifactRef
    target_artifact_version: str


class ExperimentLogger:
    """Scenario-scoped, provider-neutral metric logger.

    Recipe and processor code choose a namespace but never import a concrete
    tracking SDK. Implementations must isolate provider failures internally;
    observability is never part of a scenario's update transaction.
    """

    def log(self, metrics: Mapping[str, Any], *, namespace: str) -> None:
        return None


class NullExperimentLogger(ExperimentLogger):
    """No-op logger used when experiment tracking is disabled."""


class ExperimentTracker:
    """Process-level provider that binds scenario loggers and commit events.

    Implementations must treat tracking as a side effect: Dispatcher also
    isolates provider failures so an observer can never become part of the
    durable update transaction.
    """

    def bind_scenario(
        self,
        *,
        scenario: str,
        recipe: str,
        source_artifact_ref: ArtifactRef,
        run_segment: int,
    ) -> ExperimentLogger:
        return NullExperimentLogger()

    def correlation_metrics(self, context: TrainingExperimentContext) -> Mapping[str, Any]:
        return {}

    def record(self, event: TrainingExperimentEvent) -> None:
        return None

    def record_rollback(self, event: RollbackExperimentEvent) -> None:
        return None

    def close(self) -> None:
        pass


class NullExperimentTracker(ExperimentTracker):
    """No-op tracker used when experiment tracking is disabled."""


__all__ = [
    "ExperimentLogger",
    "ExperimentTracker",
    "NullExperimentLogger",
    "NullExperimentTracker",
    "RollbackExperimentEvent",
    "TrainingExperimentContext",
    "TrainingExperimentEvent",
]
