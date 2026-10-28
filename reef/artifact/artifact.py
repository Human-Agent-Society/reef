from __future__ import annotations

import shutil
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

# Re-export the pure identity types while keeping ``reef.core`` independent of
# storage implementations.
from reef.core.artifact_ref import ArtifactRef, decode_artifact_ref, encode_artifact_ref
from reef.core.components import ReleaseComponents, release_components
from reef.core.errors import ReefError

__all__ = [
    "LOCAL_RELEASE_PREFIX",
    "Artifact",
    "ArtifactConflict",
    "ArtifactError",
    "ArtifactMaterializationError",
    "ArtifactNotFound",
    "ArtifactPublicationError",
    "ArtifactRef",
    "ArtifactRepository",
    "ArtifactSourceError",
    "ArtifactValidator",
    "decode_artifact_ref",
    "encode_artifact_ref",
    "is_local_release",
]

#: Release prefix for process-local (staged, never published) artifacts.
#: ``Artifact.local`` and repository staging both mint releases under it;
#: ``is_local_release`` is the shared predicate for recognizing them.
LOCAL_RELEASE_PREFIX = "local:"
#: Files a repository adds to a materialized release; a release with nothing else holds no weights.
REPOSITORY_FILES = frozenset({".git", ".gitattributes", "reef-artifact.json"})


def is_local_release(release_id: str) -> bool:
    """True when ``release_id`` names a process-local artifact without durable bytes."""
    return release_id.startswith(LOCAL_RELEASE_PREFIX)


class ArtifactError(ReefError):
    """Base class for artifact storage and release-chain errors."""


class ArtifactSourceError(ArtifactError):
    pass


class ArtifactNotFound(ArtifactError):
    pass


class ArtifactConflict(ArtifactError):
    pass


class ArtifactMaterializationError(ArtifactError):
    pass


class ArtifactPublicationError(ArtifactError):
    pass


class ArtifactRepository(ABC):
    """Minimal repository capability needed by an artifact."""

    @abstractmethod
    def materialize(self, ref: ArtifactRef) -> Artifact: ...


class ArtifactValidator(ABC):
    """Artifact admission checks selected for one scenario."""

    @abstractmethod
    def validate(self, artifact: Artifact) -> None: ...


class Artifact:
    """A concrete artifact bound to one scenario-scoped repository."""

    def __init__(
        self,
        ref: ArtifactRef,
        repository: ArtifactRepository | None,
        *,
        local_path: Path | None = None,
        metadata: Mapping[str, object] | None = None,
        artifact_id: str | None = None,
    ) -> None:
        self._ref = ref
        self._repository = repository
        self._local_path = None if local_path is None else Path(local_path)
        self._metadata = dict(metadata or {})
        self._artifact_id = artifact_id

    @property
    def ref(self) -> ArtifactRef:
        return self._ref

    @property
    def repository(self) -> ArtifactRepository | None:
        return self._repository

    @property
    def local_path(self) -> Path | None:
        return self._local_path

    @property
    def metadata(self) -> Mapping[str, object]:
        return MappingProxyType(self._metadata)

    @property
    def artifact_id(self) -> str:
        """The artifact that this value holds: the artifact of a component view, or the only component of a release.

        A release without a manifest is one artifact. The release made it, so
        its ``release_id`` names it. A manifest of one component names the
        artifact (``ReleaseComponents.artifact_ids``). For a release of several
        components, raise ``ArtifactNotFound``: each component view holds one
        of its artifacts.
        """
        if self._artifact_id is not None:
            return self._artifact_id
        manifest = self.materialize().components
        if manifest is None:
            return self.ref.release_id
        if not manifest.single:
            raise ArtifactNotFound(
                f"release {self.ref.release_id!r} holds components {list(manifest.names)}; name one of them"
            )
        return next(iter(manifest.artifact_ids(self.ref.release_id).values()))

    @property
    def components(self) -> ReleaseComponents | None:
        """The release's component manifest; ``None`` for a release published without one."""
        return release_components(self._metadata)

    def component(self, name: str) -> Artifact:
        """The named component of this release, materialized.

        A release without a manifest, or with one component, is that component
        itself. A multi-component release keeps each component in a directory
        named after it; the view carries the component's own content id and
        metadata under the release's identity.
        """
        materialized = self.materialize()
        manifest = materialized.components
        if manifest is None:
            return materialized
        if name not in manifest.entries:
            raise ArtifactNotFound(f"release {self.ref.release_id!r} binds no component {name!r}")
        if manifest.single:
            return materialized
        if materialized.local_path is None:
            raise ArtifactMaterializationError(f"release {self.ref.release_id!r} has no local component directories")
        entry = manifest.entries[name]
        directory = materialized.local_path / manifest.relative_path(name)
        # A component the base seeded nothing for has no directory in the
        # release (git keeps none for an empty tree); it is empty, not missing.
        # Only the leaf is created, and only inside a release tree that is
        # there: a release directory that is gone must not come back empty.
        if materialized.local_path.is_dir() and not directory.exists():
            # Two readers of one release race here; the second finds the leaf made.
            directory.mkdir(exist_ok=True)
        return Artifact(
            ArtifactRef(entry.content_id, self.ref.release_id, self.ref.parent_release_id),
            None,
            local_path=directory,
            metadata=entry.metadata,
            artifact_id=manifest.artifact_ids(self.ref.release_id)[name],
        )

    @classmethod
    def local(
        cls,
        path: Path,
        *,
        metadata: Mapping[str, object] | None = None,
        repository: ArtifactRepository | None = None,
    ) -> Artifact:
        token = uuid.uuid4().hex
        return cls(
            ArtifactRef(
                content_id=f"content:{token}",
                release_id=f"{LOCAL_RELEASE_PREFIX}{token}",
                parent_release_id=None,
            ),
            repository,
            local_path=path,
            metadata=metadata,
        )

    def with_repository(self, repository: ArtifactRepository) -> Artifact:
        return Artifact(
            self.ref,
            repository,
            local_path=self.local_path,
            metadata=self.metadata,
            artifact_id=self._artifact_id,
        )

    def materialize(self) -> Artifact:
        if self.local_path is not None:
            return self
        if self.repository is None:
            raise RuntimeError("artifact has no repository")
        return self.repository.materialize(self.ref)

    def discard(self) -> None:
        if self._local_path is not None:
            shutil.rmtree(self._local_path, ignore_errors=True)
