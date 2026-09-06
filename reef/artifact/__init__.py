"""Storage for the release chain: every publication as an immutable release.

The package owns bytes and heads — persisted, staged, materialized on
demand. *When* a head moves is the scenario commit protocol's decision, one
level up. Boundaries this package holds:

- Repository backends are scenario-agnostic. A backend stores one
  repository; the factory owns the scenario-name-to-backend mapping. Nothing
  here knows what a scenario, trainer, or commit log is;
  ``ArtifactReleaseChain``'s caller decides transaction ordering.
- Releases are immutable and heads move only by compare-and-swap
  (``advance_current`` takes ``expected``; ``publish`` takes
  ``expected_parent``), so concurrent publication conflicts loudly
  (``ArtifactConflict``) instead of losing a release.
- A ``local:`` release has identity but no durable bytes; recovery and
  rollback refuse it as a restore source (``ReleaseNotRestorable``).

Adding a storage backend: implement ``RepositoryBackend`` and expose it
through a ``CachedRepositoryBackendFactory`` subclass; the dispatcher takes
any ``RepositoryBackendFactory``. ``tests/reef_service/test_reef_git_lfs.py``
and ``test_reef_artifacts.py`` show the contract a backend must satisfy.

Backends used with a scenario commit log must subclass
``StagedReleaseRepositoryBackend``. ``publish`` must accept
``advance_head=False`` and persist resolvable bytes without advancing its head.
After the scenario commit log is durable, ``commit_release(ref, expected_parent=...)``
advances that pointer. It must be idempotent when ``ref`` is already current and
reject an unrelated head. A failed pointer update is repaired from the commit log
on restart; the pointer never overrides committed scenario state. The memory
and Git LFS backends implement this contract. Backends without this capability
are rejected before scenario creation or recovery when a commit log is configured.
They remain usable without a commit log.
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
from reef.artifact.release_chain import ArtifactReleaseChain, ReleaseNotRestorable
from reef.artifact.repository import (
    CachedRepositoryBackendFactory,
    EnumerableRepositoryBackendFactory,
    RegistrationAwareRepositoryBackendFactory,
    Repository,
    RepositoryBackend,
    RepositoryBackendFactory,
    StagedReleaseRepositoryBackend,
)
from reef.artifact.sources import (
    ArtifactSource,
    DownloadedSnapshot,
    GitVersionSource,
    HuggingFaceSource,
    download_huggingface_snapshot,
    parse_artifact_source,
)

__all__ = [
    "AdapterArtifactError",
    "Artifact",
    "ArtifactConflict",
    "ArtifactError",
    "ArtifactMaterializationError",
    "ArtifactNotFound",
    "ArtifactPublicationError",
    "ArtifactRef",
    "ArtifactReleaseChain",
    "ArtifactSource",
    "ArtifactSourceError",
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
    "ReleaseNotRestorable",
    "Repository",
    "RepositoryBackend",
    "RepositoryBackendFactory",
    "StagedReleaseRepositoryBackend",
    "download_huggingface_snapshot",
    "parse_artifact_source",
    "read_peft_config",
]
