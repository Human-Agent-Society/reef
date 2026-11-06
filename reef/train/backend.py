"""The values of one harness step: a prepared step, and the committed snapshots that a step reads its head from."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from reef.core.evaluation import UpdateCandidate
from reef.harness.tree.tree import normalized_path, require_text


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


@dataclass(frozen=True)
class ArtifactTextFiles:
    """One committed component artifact's ID and text files.

    Values: ``artifact_id`` is the non-empty opaque ID of the component artifact; ``files`` maps
    each tree path (``reef.harness.tree.normalized_path``) to its UTF-8 text. The mapping is
    immutable and excludes repository metadata.
    Purpose: ``CommittedArtifacts`` makes it; a harness step builds its
    head from ``files``.
    """

    artifact_id: str
    files: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not self.artifact_id:
            raise ValueError("ArtifactTextFiles.artifact_id must be a non-empty string")
        if not isinstance(self.files, Mapping):
            raise TypeError("artifact text files must be a mapping")
        files = {normalized_path(path): require_text(path, text) for path, text in self.files.items()}
        object.__setattr__(self, "files", MappingProxyType(files))


class CommittedArtifacts(ABC):
    """The artifacts a scenario's commits recorded, read for one component at a time.

    Purpose: a harness trainer reads the committed head here, as the artifact of
    its next job's start, and the scenario layer implements it over its commit
    history and artifact repository. A component's head is its training
    start: the snapshot of the later of its newest ``train_from`` record and
    its newest training commit that made an artifact, active or not. An
    ``activate`` record moves what serves, not the head. Before any such
    record, the head is the scenario's root snapshot.
    """

    @abstractmethod
    def head(self, component: str) -> ArtifactTextFiles:
        """``component``'s files in its head artifact."""

    @abstractmethod
    def artifact(self, artifact_id: str, component: str) -> ArtifactTextFiles:
        """Return a committed component artifact; raise ``ArtifactNotFound`` when its ID is unknown."""


@dataclass(frozen=True)
class ComponentArtifacts:
    """One component's view of a scenario's ``CommittedArtifacts``.

    Values: ``artifacts`` is the scenario's reader, and ``component`` names a
    component of the scenario, non-empty.
    Purpose: ``Recipe.build_trainers`` makes one per component and hands it
    to ``Recipe.build``; a harness trainer reads its head through it.
    """

    artifacts: CommittedArtifacts
    component: str

    def __post_init__(self) -> None:
        if not isinstance(self.component, str) or not self.component:
            raise ValueError("ComponentArtifacts.component must be a non-empty string")

    def head(self) -> ArtifactTextFiles:
        return self.artifacts.head(self.component)

    def artifact(self, artifact_id: str) -> ArtifactTextFiles:
        return self.artifacts.artifact(artifact_id, self.component)


__all__ = [
    "ArtifactTextFiles",
    "CommittedArtifacts",
    "ComponentArtifacts",
    "PreparedStep",
]
