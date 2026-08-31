from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.artifact.artifact import Artifact


@dataclass(frozen=True)
class NoArtifactPublication:
    pass


@dataclass(frozen=True)
class LiveWeightPublication:
    runtime_load_id: str


@dataclass(frozen=True)
class SavedArtifactPublication:
    """A materialized publication identified by its durable release.

    Engine-local ``runtime_load_id`` tokens deliberately stop at checkpoint
    export; once bytes are materialized, the published version is the source
    of identity.
    """

    artifact: Artifact


@dataclass(frozen=True)
class DurableWeightsPublication:
    """A backend train step whose weights are live and also exported on disk.

    Both publication shapes are still open at this point: the commit protocol
    owns checkpoint policy, so it decides whether to import ``checkpoint_path``
    as a durable version or to publish ``runtime_load_id`` as live weights and
    leave the export unreferenced. Producers must not pre-empt that choice.
    """

    checkpoint_path: str
    runtime_load_id: str


# A closed sum type, deliberately not a base class: an empty base would be
# instantiable and would leave every ``isinstance`` chain open, so a type
# checker could not prove a new member is handled everywhere it must be.
ArtifactPublication = (
    NoArtifactPublication | LiveWeightPublication | SavedArtifactPublication | DurableWeightsPublication
)


@dataclass(frozen=True)
class TrainStepResult:
    """One training step's outcome, including what it wants published.

    The publication fields are a small ordered precedence rather than four
    independent flags: ``artifact`` (bytes already materialized) outranks
    ``checkpoint_path`` (bytes the backend exported, still to be imported),
    which outranks a bare ``runtime_load_id`` (live weights only). Setting both
    ``artifact`` and ``runtime_load_id`` is normal for a durable train step; the
    version travels in the artifact's metadata. Combinations with no meaning
    are rejected here instead of being silently dropped by ``publication``.
    """

    state: Mapping[str, Any] | None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifact: Artifact | None = None
    runtime_load_id: str | None = None
    checkpoint_path: str | None = None
    training_job_id: str | None = None
    source_runtime_load_id: str | None = None

    def __post_init__(self) -> None:
        if self.checkpoint_path is not None and self.runtime_load_id is None:
            raise ValueError("checkpoint_path requires the runtime_load_id it was exported from")
        if self.training_job_id is not None:
            if not isinstance(self.training_job_id, str) or not self.training_job_id:
                raise ValueError("training_job_id must be a non-empty string or None")
            if self.runtime_load_id is None:
                raise ValueError("training_job_id requires the runtime_load_id produced by the job")
        if self.source_runtime_load_id is not None and (
            not isinstance(self.source_runtime_load_id, str) or not self.source_runtime_load_id
        ):
            raise ValueError("source_runtime_load_id must be a non-empty string or None")

    @property
    def has_model_update(self) -> bool:
        """True when this step produced a new model weight update."""
        return self.runtime_load_id is not None

    @property
    def publication(self) -> ArtifactPublication:
        if self.artifact is not None:
            return SavedArtifactPublication(self.artifact)
        if self.runtime_load_id is not None:
            if self.checkpoint_path is not None:
                return DurableWeightsPublication(self.checkpoint_path, self.runtime_load_id)
            return LiveWeightPublication(self.runtime_load_id)
        return NoArtifactPublication()
