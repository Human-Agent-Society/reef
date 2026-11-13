"""Storage for the snapshot chain: every publication as an immutable snapshot.

The package owns bytes and heads — persisted, staged, materialized on
demand. *When* a head moves is the scenario committer's decision, one
level up. Boundaries this package holds:

- Repository backends are scenario-agnostic. A backend stores one
  repository; the factory owns the scenario-name-to-backend mapping. Nothing
  here knows what a scenario, trainer, or commit history is;
  ``ArtifactSnapshotChain``'s caller decides transaction ordering.
- Snapshots are immutable and heads move only by compare-and-swap
  (``activate`` takes ``expected``; ``publish`` takes
  ``expected_parent``), so concurrent publication conflicts loudly
  (``ArtifactConflictError``) instead of losing a snapshot.
- A ``local:`` snapshot has identity but no durable bytes; activation and
  a new training start refuse it as a source (``SnapshotNotRestorableError``).

Adding a storage backend: implement ``RepositoryBackend`` and expose it
through a ``CachedRepositoryBackendFactory`` subclass; the dispatcher takes
any ``RepositoryBackendFactory``. A deployment selects a factory that also
inherits ``ConfigurableRepositoryBackendFactory`` in ``storage.artifact_backend``;
Git LFS is the built-in one. ``tests/artifact/test_reef_git_lfs.py``
and ``test_reef_artifacts.py`` show the contract a backend must satisfy.

Every scenario owns its root snapshot, a snapshot with no parent.
``RepositoryBackend.register`` stores it and makes it the scenario's head in
one atomic step; a second caller gets the winner's head. Scenarios share no
snapshot. A root that no artifact item starts holds the deployment's
``DefaultStart``: a few small text files that the configuration decides, and
no weight file.

An artifact of a weight component has one of three forms: its own weight files, a
reference to a Hugging Face commit that holds them (``REFERENCE_FILE``, RFC #995), or
a base root, which holds ``BASE_FILE`` alone and stands for the deployment's base
model (``reef.artifact.base_root``). A new scenario of a backend that trains a
base model starts from a base root, which the backend's facts write
(``TrainingBackendFacts.default_start``).

Backends used with a scenario commit history must subclass
``StagedSnapshotRepositoryBackend``. ``publish`` must accept
``advance_head=False`` and persist resolvable bytes without advancing its head.
After the scenario commit history is durable, ``commit_snapshot(ref, expected_parent=...)``
advances that pointer. It must be idempotent when ``ref`` is already current and
reject an unrelated head. A failed pointer update is repaired from the commit history
on restart; the pointer never overrides committed scenario state. The memory
and Git LFS backends implement this contract. Backends without this capability
are rejected before scenario creation or recovery when a commit history is configured.
They remain usable without a commit history.

Every backend also holds stored content: named file trees that are no snapshot.
The training service stores the files of a checkpoint with ``store_content``, and
control commits them by name with ``publish_stored``, which makes the snapshot of
files that the backend holds already (``StoredFiles``). The process that commits
reads no file. ``discard_content`` removes content; a snapshot keeps its files.
"""

from reef.artifact.artifact import (
    Artifact,
    ArtifactConflictError,
    ArtifactError,
    ArtifactMaterializationError,
    ArtifactNotFoundError,
    ArtifactPublicationError,
    ArtifactSourceError,
    ArtifactValidator,
    SnapshotRef,
)
from reef.artifact.base_root import BASE_FILE, BaseRoot, base_root_of, read_base_root
from reef.artifact.composite import compose_snapshot
from reef.artifact.files import ArtifactFiles, DirectoryFiles, directory_checksum
from reef.artifact.git_lfs import GitLFSRepositoryBackend, GitLFSRepositoryBackendFactory
from reef.artifact.memory import InMemoryRepositoryBackend
from reef.artifact.peft import AdapterArtifactError, PEFTValidator, read_peft_config
from reef.artifact.repository import (
    CachedRepositoryBackendFactory,
    ConfigurableRepositoryBackendFactory,
    DefaultStart,
    EnumerableRepositoryBackendFactory,
    RegistrationAwareRepositoryBackendFactory,
    Repository,
    RepositoryBackend,
    RepositoryBackendContext,
    RepositoryBackendFactory,
    StagedSnapshotRepositoryBackend,
    StoredFiles,
)
from reef.artifact.snapshot_chain import ArtifactSnapshotChain, SnapshotNotRestorableError
from reef.artifact.sources import (
    REFERENCE_FILE,
    ArtifactSourceReader,
    ArtifactSources,
    DirectorySource,
    DownloadedSnapshot,
    FileReader,
    HuggingFaceReader,
    HuggingFaceRevision,
    HuggingFaceRevisionFiles,
    HuggingFaceSource,
    S3PrefixFiles,
    S3Reader,
    SourceFiles,
    download_huggingface_snapshot,
    parse_huggingface_source,
    read_reference,
)

__all__ = [
    "BASE_FILE",
    "REFERENCE_FILE",
    "AdapterArtifactError",
    "Artifact",
    "ArtifactConflictError",
    "ArtifactError",
    "ArtifactFiles",
    "ArtifactMaterializationError",
    "ArtifactNotFoundError",
    "ArtifactPublicationError",
    "ArtifactSnapshotChain",
    "ArtifactSourceError",
    "ArtifactSourceReader",
    "ArtifactSources",
    "ArtifactValidator",
    "BaseRoot",
    "CachedRepositoryBackendFactory",
    "ConfigurableRepositoryBackendFactory",
    "DefaultStart",
    "DirectoryFiles",
    "DirectorySource",
    "DownloadedSnapshot",
    "EnumerableRepositoryBackendFactory",
    "FileReader",
    "GitLFSRepositoryBackend",
    "GitLFSRepositoryBackendFactory",
    "HuggingFaceReader",
    "HuggingFaceRevision",
    "HuggingFaceRevisionFiles",
    "HuggingFaceSource",
    "InMemoryRepositoryBackend",
    "PEFTValidator",
    "RegistrationAwareRepositoryBackendFactory",
    "Repository",
    "RepositoryBackend",
    "RepositoryBackendContext",
    "RepositoryBackendFactory",
    "S3PrefixFiles",
    "S3Reader",
    "SnapshotNotRestorableError",
    "SnapshotRef",
    "SourceFiles",
    "StagedSnapshotRepositoryBackend",
    "StoredFiles",
    "base_root_of",
    "compose_snapshot",
    "directory_checksum",
    "download_huggingface_snapshot",
    "parse_huggingface_source",
    "read_base_root",
    "read_peft_config",
    "read_reference",
]
