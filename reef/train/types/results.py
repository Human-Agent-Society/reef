from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.artifact.artifact import Artifact


@dataclass(frozen=True)
class NoArtifactPublication:
    """The publication of a commit that publishes nothing. It has one value."""


@dataclass(frozen=True)
class SavedArtifactPublication:
    """The publication of a commit that publishes durable bytes.

    Values: ``artifact`` is the release to publish, and ``component`` is None
    or the non-empty name of the release component that ``artifact`` replaces;
    the committer carries the other components forward. Purpose:
    ``TrainStepResult.publication`` makes the value, and the scenario committer
    publishes it. The published version, not a runtime load ID, names it.
    """

    artifact: Artifact
    component: str | None = None


# A closed sum type, deliberately not a base class: an empty base would be
# instantiable and would leave every ``isinstance`` chain open, so a type
# checker could not prove a new member is handled everywhere it must be.
ArtifactPublication = NoArtifactPublication | SavedArtifactPublication


@dataclass(frozen=True)
class TrainStepResult:
    """One commit's worth of training: the trainer state to commit, its metrics and what it publishes.

    ``state`` is the trainer state the commit records, a mapping, or ``None``
    for a commit outside a training step that keeps the current state; a
    training step's result needs a mapping (``Trainer.prepare_commit``
    refuses ``None``). ``metrics`` is a mapping. ``artifact`` holds the bytes to publish, or ``None`` for a
    commit that publishes nothing, such as a job submission or a job's end.
    ``training_job_id`` names the training job the commit is about, if any.

    ``pending`` publishes the artifact without activating it: it stays an
    inactive artifact until a person activates it. The field keeps its name
    until RFC #965 step 3 moves this decision into the activation rule. It
    travels as a field rather than as a metrics key: only durable bytes can
    stay inactive, and a step that publishes nothing is rejected here
    instead of having the request silently dropped.

    ``component`` names the release component ``artifact`` replaces. A scenario
    serving several components requires it; the committer carries the other
    components from the active release. A flat scenario may leave
    it unset.
    """

    state: Mapping[str, Any] | None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifact: Artifact | None = None
    training_job_id: str | None = None
    pending: bool = False
    component: str | None = None

    def __post_init__(self) -> None:
        if self.state is not None and not isinstance(self.state, Mapping):
            raise TypeError("state must be a mapping or None")
        if not isinstance(self.metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        if self.component is not None:
            if not isinstance(self.component, str) or not self.component:
                raise ValueError("component must be a non-empty string or None")
            if self.artifact is None:
                raise ValueError("component names the artifact a step publishes; set artifact")
        if self.training_job_id is not None and (
            not isinstance(self.training_job_id, str) or not self.training_job_id
        ):
            raise ValueError("training_job_id must be a non-empty string or None")
        if not isinstance(self.pending, bool):
            raise ValueError("pending must be a boolean")
        if self.pending and self.artifact is None:
            raise ValueError("a pending step must carry durable bytes: set artifact")

    @property
    def publication(self) -> ArtifactPublication:
        if self.artifact is not None:
            return SavedArtifactPublication(self.artifact, self.component)
        return NoArtifactPublication()
