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
    #: The release component whose trainer made the step, when the scenario runs several trainers.
    component: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingExperimentEvent:
    """One durably committed training result and its final Reef version."""

    context: TrainingExperimentContext
    produced_artifact_ref: ArtifactRef
    metrics: Mapping[str, Any]
    outcome: str
    training_job_id: str | None = None
    checkpoint_path: str | None = None


@dataclass(frozen=True, slots=True)
class TrainFromExperimentEvent:
    """A committed new training start, which ends one training-curve segment and opens the next.

    ``step`` is the step of the ``"train_from"`` commit, which numbers the
    next segment, and ``run_segment`` is the segment it ends.
    ``start_artifact_ref`` is the new training start. ``component`` names
    the trainer in a scenario of several components.
    The commit observer makes the value and the tracker reads it in
    :meth:`ExperimentTracker.record_train_from`.
    """

    scenario: str
    recipe: str
    step: int
    run_segment: int
    start_artifact_ref: ArtifactRef
    component: str | None = None


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

    Implementations must treat tracking as a side effect. The storage observer
    isolates event failures, and commit preparation isolates correlation failures,
    so tracking never becomes part of the durable update transaction.
    """

    @property
    def operational_metrics_enabled(self) -> bool:
        """Whether the dispatcher should periodically sample operational metrics."""
        return False

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

    def record_train_from(self, event: TrainFromExperimentEvent) -> None:
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
    "TrainFromExperimentEvent",
    "TrainingExperimentContext",
    "TrainingExperimentEvent",
]
