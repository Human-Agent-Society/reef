"""Artifact backends that a deployment names in ``storage.artifact_backend``, for assembly tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from reef.artifact import (
    CachedRepositoryBackendFactory,
    ConfigurableRepositoryBackendFactory,
    InMemoryRepositoryBackend,
    RepositoryBackendContext,
)


class MemoryArtifactBackend(CachedRepositoryBackendFactory, ConfigurableRepositoryBackendFactory):
    """In-memory scenario backends whose snapshots live under the context's cache directory."""

    def __init__(self, context: RepositoryBackendContext, *, label: str) -> None:
        super().__init__()
        self.context = context
        self.label = label
        self.archived: set[str] = set()

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: RepositoryBackendContext) -> Self:
        return cls(context, label=str(config["label"]))

    def _build_backend(self, scenario: str) -> InMemoryRepositoryBackend:
        return InMemoryRepositoryBackend(scenario, directory=self.context.cache_dir / "snapshots")

    def _archive_persisted_registration(self, scenario: str) -> tuple[str, ...]:
        self.archived.add(scenario)
        return ()

    def has_archived_registration(self, scenario: str) -> bool:
        return scenario in self.archived


class UnreachableArtifactBackend(CachedRepositoryBackendFactory, ConfigurableRepositoryBackendFactory):
    """A backend whose store cannot be reached when the service starts."""

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: RepositoryBackendContext) -> Self:
        raise ConnectionError(f"artifact store {Path(context.work_dir).name} is unreachable")

    def _build_backend(self, scenario: str) -> InMemoryRepositoryBackend:
        raise NotImplementedError

    def has_archived_registration(self, scenario: str) -> bool:
        raise NotImplementedError
