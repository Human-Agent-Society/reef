"""The values of one harness step: a prepared step, and the committed releases that a step reads its head from."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from reef.core.evaluation import UpdateCandidate


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
class ReleaseFiles:
    """One component's files in one committed release.

    Values: ``release_id`` is the non-empty id of the release; ``files`` maps
    each relative path of the component's view of that release to its UTF-8
    text, without the repository's own files.
    Purpose: ``CommittedReleases`` makes it; a harness step builds its
    head from ``files``.
    """

    release_id: str
    files: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.release_id, str) or not self.release_id:
            raise ValueError("ReleaseFiles.release_id must be a non-empty string")


class CommittedReleases(ABC):
    """The releases a scenario's commits recorded, read for one component at a time.

    Purpose: a harness trainer reads the committed head here, as the release of
    its next job's start, and the scenario layer implements it over its commit
    history and artifact repository. A component's head is its training
    start: the release of the later of its newest ``train_from`` record and
    its newest training commit that made an artifact, active or not. An
    ``activate`` record moves what serves, not the head. Before any such
    record, the head is the base release.
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
    to ``Recipe.build``; a harness trainer reads its head through it.
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


__all__ = [
    "CommittedReleases",
    "ComponentReleases",
    "PreparedStep",
    "ReleaseFiles",
]
