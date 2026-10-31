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


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """The run of an experiment tracker that records the training of one scenario segment.

    Values: ``tracker`` names the tracker, such as ``"wandb"``; ``project`` and ``run_id`` name the run in it.
    These three are non-empty strings. ``entity`` is the tracker's account or team, and ``url`` the run's web page;
    each is None or a non-empty string, None when the tracker does not know it.

    Purpose: ``ExperimentTracker.experiment_run`` makes it. Control writes ``to_dict()`` into the work item of
    each training job that the run records (``reef.train.job_queue.JobExperimentRecorder``), and the job listing
    reads it back with ``from_dict``.
    """

    tracker: str
    project: str
    run_id: str
    entity: str | None = None
    url: str | None = None

    def __post_init__(self) -> None:
        for value, name in ((self.tracker, "tracker"), (self.project, "project"), (self.run_id, "run_id")):
            if not isinstance(value, str) or not value:
                raise ValueError(f"an experiment run's {name} is a non-empty string")
        for optional, name in ((self.entity, "entity"), (self.url, "url")):
            if optional is not None and (not isinstance(optional, str) or not optional):
                raise ValueError(f"an experiment run's {name} is a non-empty string or None")

    def to_dict(self) -> dict[str, str | None]:
        """The run as ``{tracker, project, entity, run_id, url}``; ``from_dict`` of the result is an equal value."""
        return {
            "tracker": self.tracker,
            "project": self.project,
            "entity": self.entity,
            "run_id": self.run_id,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ExperimentRun:
        """Read the object that ``to_dict`` wrote; ``ValueError`` for another key set or a malformed value."""
        keys = {"tracker", "project", "entity", "run_id", "url"}
        if set(value) != keys:
            raise ValueError(f"an experiment run has the keys {sorted(keys)}, not {sorted(value)}")
        tracker, project, run_id = value["tracker"], value["project"], value["run_id"]
        entity, url = value["entity"], value["url"]
        if not isinstance(tracker, str) or not isinstance(project, str) or not isinstance(run_id, str):
            raise ValueError("an experiment run's tracker, project and run_id are strings")
        if (entity is not None and not isinstance(entity, str)) or (url is not None and not isinstance(url, str)):
            raise ValueError("an experiment run's entity and url are strings or None")
        return cls(tracker, project, run_id, entity, url)


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

    def experiment_run(self, context: TrainingExperimentContext) -> ExperimentRun | None:
        """The run that records the training of ``context``, when this tracker has it open in this process; None
        otherwise. The default tracker records nothing, so it returns None."""
        return None

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
    "ExperimentRun",
    "ExperimentTracker",
    "NullExperimentLogger",
    "NullExperimentTracker",
    "TrainFromExperimentEvent",
    "TrainingExperimentContext",
    "TrainingExperimentEvent",
]
