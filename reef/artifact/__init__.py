"""Storage for the release chain: every publication as an immutable release.

The package owns bytes and heads — persisted, staged, materialized on
demand. *When* a head moves is the scenario committer's decision, one
level up. Boundaries this package holds:

- Repository backends are scenario-agnostic. A backend stores one
  repository; the factory owns the scenario-name-to-backend mapping. Nothing
  here knows what a scenario, trainer, or commit history is;
  ``ArtifactReleaseChain``'s caller decides transaction ordering.
- Releases are immutable and heads move only by compare-and-swap
  (``activate`` takes ``expected``; ``publish`` takes
  ``expected_parent``), so concurrent publication conflicts loudly
  (``ArtifactConflict``) instead of losing a release.
- A ``local:`` release has identity but no durable bytes; activation and
  a new training start refuse it as a source (``ReleaseNotRestorable``).

Adding a storage backend: implement ``RepositoryBackend`` and expose it
through a ``CachedRepositoryBackendFactory`` subclass; the dispatcher takes
any ``RepositoryBackendFactory``. A deployment selects a factory that also
inherits ``ConfigurableRepositoryBackendFactory`` in ``storage.artifact_backend``;
Git LFS is the built-in one. ``tests/artifact/test_reef_git_lfs.py``
and ``test_reef_artifacts.py`` show the contract a backend must satisfy.

Every repository starts from a base artifact. A configured backend imports
``RepositoryBackendContext.base`` (a ``BaseArtifactSource``) into a
repository that has none, with ``base.content_id`` as its content ID, and
refuses to open a repository whose base was imported from other
``base.inputs``.

Backends used with a scenario commit history must subclass
``StagedReleaseRepositoryBackend``. ``publish`` must accept
``advance_head=False`` and persist resolvable bytes without advancing its head.
After the scenario commit history is durable, ``commit_release(ref, expected_parent=...)``
advances that pointer. It must be idempotent when ``ref`` is already current and
reject an unrelated head. A failed pointer update is repaired from the commit history
on restart; the pointer never overrides committed scenario state. The memory
and Git LFS backends implement this contract. Backends without this capability
are rejected before scenario creation or recovery when a commit history is configured.
They remain usable without a commit history.
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
    ArtifactValidator,
)
from reef.artifact.composite import compose_release
from reef.artifact.git_lfs import GitLFSRepositoryBackend, GitLFSRepositoryBackendFactory
from reef.artifact.memory import InMemoryRepositoryBackend
from reef.artifact.peft import AdapterArtifactError, PEFTValidator, read_peft_config
from reef.artifact.release_chain import ArtifactReleaseChain, ReleaseNotRestorable
from reef.artifact.repository import (
    BaseArtifactSource,
    CachedRepositoryBackendFactory,
    ConfigurableRepositoryBackendFactory,
    EnumerableRepositoryBackendFactory,
    RegistrationAwareRepositoryBackendFactory,
    Repository,
    RepositoryBackend,
    RepositoryBackendContext,
    RepositoryBackendFactory,
    StagedReleaseRepositoryBackend,
)
from reef.artifact.sources import (
    DownloadedSnapshot,
    HuggingFaceSource,
    download_huggingface_snapshot,
    parse_huggingface_source,
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
    "ArtifactSourceError",
    "ArtifactValidator",
    "BaseArtifactSource",
    "CachedRepositoryBackendFactory",
    "ConfigurableRepositoryBackendFactory",
    "DownloadedSnapshot",
    "EnumerableRepositoryBackendFactory",
    "GitLFSRepositoryBackend",
    "GitLFSRepositoryBackendFactory",
    "HuggingFaceSource",
    "InMemoryRepositoryBackend",
    "PEFTValidator",
    "RegistrationAwareRepositoryBackendFactory",
    "ReleaseNotRestorable",
    "Repository",
    "RepositoryBackend",
    "RepositoryBackendContext",
    "RepositoryBackendFactory",
    "StagedReleaseRepositoryBackend",
    "compose_release",
    "download_huggingface_snapshot",
    "parse_huggingface_source",
    "read_peft_config",
]
