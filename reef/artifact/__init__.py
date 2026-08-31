"""Storage for the version chain: every committed version as an immutable artifact.

The package owns bytes and heads — persisted, staged, materialized on
demand. *When* a head moves is the scenario commit protocol's decision, one
level up. Boundaries this package holds:

- Repository backends are scenario-agnostic. A backend stores one
  repository; the factory owns the scenario-name-to-backend mapping. Nothing
  here knows what a scenario, trainer, or commit log is;
  ``ArtifactVersionChain``'s caller decides transaction ordering.
- Versions are immutable and heads move only by compare-and-swap
  (``advance_current`` takes ``expected``; ``publish`` takes
  ``expected_parent``), so concurrent publication conflicts loudly
  (``ArtifactConflict``) instead of losing a version.
- A ``local:`` version has identity but no durable bytes; recovery and
  rollback refuse it as a restore source (``VersionNotRestorable``).

Adding a storage backend: implement ``RepositoryBackend`` and expose it
through a ``CachedRepositoryBackendFactory`` subclass; the dispatcher takes
any ``RepositoryBackendFactory``. ``tests/reef_service/test_reef_git_lfs.py``
and ``test_reef_artifacts.py`` show the contract a backend must satisfy.
"""

from reef.artifact.artifact import (
    Artifact,
    ArtifactConflict,
    ArtifactError,
    ArtifactMaterializationError,
    ArtifactNotFound,
    ArtifactPublicationError,
    ArtifactRef,
    ArtifactSourceError,
    LiveWeightArtifactRef,
)
from reef.artifact.git_lfs import GitLFSRepositoryBackend
from reef.artifact.memory import InMemoryRepositoryBackend
from reef.artifact.peft import AdapterArtifactError, PEFTValidator, read_peft_config
from reef.artifact.repository import (
    CachedRepositoryBackendFactory,
    EnumerableRepositoryBackendFactory,
    RegistrationAwareRepositoryBackendFactory,
    Repository,
    RepositoryBackend,
    RepositoryBackendFactory,
)
from reef.artifact.sources import (
    ArtifactSource,
    DownloadedSnapshot,
    GitVersionSource,
    HuggingFaceSource,
    download_huggingface_snapshot,
    parse_artifact_source,
)
from reef.artifact.version_chain import ArtifactVersionChain, VersionNotRestorable

__all__ = [
    "AdapterArtifactError",
    "Artifact",
    "ArtifactConflict",
    "ArtifactError",
    "ArtifactMaterializationError",
    "ArtifactNotFound",
    "ArtifactPublicationError",
    "ArtifactRef",
    "ArtifactSource",
    "ArtifactSourceError",
    "ArtifactVersionChain",
    "CachedRepositoryBackendFactory",
    "DownloadedSnapshot",
    "EnumerableRepositoryBackendFactory",
    "GitLFSRepositoryBackend",
    "GitVersionSource",
    "HuggingFaceSource",
    "InMemoryRepositoryBackend",
    "LiveWeightArtifactRef",
    "PEFTValidator",
    "RegistrationAwareRepositoryBackendFactory",
    "Repository",
    "RepositoryBackend",
    "RepositoryBackendFactory",
    "VersionNotRestorable",
    "download_huggingface_snapshot",
    "parse_artifact_source",
    "read_peft_config",
]
