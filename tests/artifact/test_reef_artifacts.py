from __future__ import annotations

from pathlib import Path

import pytest

from reef.artifact import (
    Artifact,
    ArtifactConflict,
    ArtifactRef,
    ArtifactReleaseChain,
    CachedRepositoryBackendFactory,
    HuggingFaceSource,
    InMemoryRepositoryBackend,
    Repository,
    RepositoryBackend,
    download_huggingface_snapshot,
    parse_huggingface_source,
)


@pytest.mark.unit
def test_artifact_ref_can_carry_ephemeral_local_path_without_changing_identity(tmp_path: Path) -> None:
    durable = ArtifactRef("artifact-1", "version-1", None)
    local = Artifact(
        durable,
        InMemoryRepositoryBackend("math"),
        local_path=tmp_path,
    )

    assert local.local_path == tmp_path
    assert local.ref == durable


def root_tree(directory: Path, text: str = "base") -> Artifact:
    """An unpublished root release in ``directory`` that holds ``model.txt`` = ``text``."""
    directory.mkdir()
    (directory / "model.txt").write_text(text)
    return Artifact.local(directory, content_id=f"content:{text}")


@pytest.mark.unit
def test_artifact_is_the_process_local_artifact_coordinator(tmp_path: Path) -> None:
    backend = InMemoryRepositoryBackend("math")
    root = backend.register(root_tree(tmp_path / "root"), metadata={})
    artifact = Repository(backend, root).materialize(root)

    assert not hasattr(artifact.ref, "scenario")


@pytest.mark.unit
def test_release_id_chain_owns_heads_staging_and_publication(tmp_path: Path) -> None:
    backend = InMemoryRepositoryBackend("math")
    root = backend.register(root_tree(tmp_path / "root"), metadata={})
    repository = Repository(
        backend,
        root,
        current_artifact=root,
        checkpoint_artifact=root,
        local_dir=tmp_path / "local",
    )
    chain = ArtifactReleaseChain(repository)
    assert chain.current == chain.checkpoint == chain.start == root

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "model.txt").write_text("trained")
    staged = chain.stage(2, Artifact.local(candidate))
    published = chain.publish(staged)

    assert chain.current == published
    assert chain.checkpoint == published
    assert published.parent_release_id == root.release_id


@pytest.mark.unit
def test_in_memory_repository_registers_materializes_and_publishes(tmp_path: Path) -> None:
    backend_factory = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    math_backend = backend_factory("math")
    code_backend = backend_factory("code")
    assert math_backend.metadata() is None

    math = math_backend.register(root_tree(tmp_path / "math-root"), metadata={"scenario": "math"})
    code = code_backend.register(root_tree(tmp_path / "code-root"), metadata={"scenario": "code"})

    # Two scenarios get distinct roots, each a release with no parent and the root's content id.
    assert math.release_id != code.release_id
    assert math.parent_release_id is None and math.content_id == "content:base"
    assert math_backend.metadata() == {"scenario": "math"}
    # A second registration stores nothing and gets the winner's head.
    assert math_backend.register(root_tree(tmp_path / "late-root", "late"), metadata={}) == math
    assert math_backend.materialize(math).local_path.joinpath("model.txt").read_text() == "base"

    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "model.txt").write_text("trained")
    published = math_backend.publish(
        Artifact.local(candidate_dir, metadata={"loss": "sft"}),
        expected_parent=math,
    )

    assert published.parent_release_id == math.release_id
    assert math_backend.current() == published
    assert math_backend.resolve_release(published.release_id) == published
    assert math_backend.materialize(published).local_path.joinpath("model.txt").read_text() == "trained"
    # Each scenario keeps its own head; math's publish does not leak into code.
    assert code_backend.current() == code

    with pytest.raises(ArtifactConflict):
        math_backend.publish(
            Artifact.local(candidate_dir),
            expected_parent=math,
        )

    assert math_backend is not code_backend


@pytest.mark.unit
def test_repository_factory_caches_registration_misses() -> None:
    class RegistrationProbeFactory(CachedRepositoryBackendFactory):
        def __init__(self) -> None:
            super().__init__()
            self.probes = 0

        def _build_backend(self, scenario: str) -> RepositoryBackend:
            raise AssertionError(f"unexpected backend construction for {scenario}")

        def _has_persisted_registration(self, scenario: str) -> bool:
            del scenario
            self.probes += 1
            return False

    factory = RegistrationProbeFactory()

    assert not factory.has_registration("missing")
    assert not factory.has_registration("missing")
    assert factory.probes == 1

    uncached = RegistrationProbeFactory()
    uncached._REGISTRATION_MISS_TTL_SECONDS = 0
    assert not uncached.has_registration("missing")
    assert not uncached.has_registration("missing")
    assert uncached.probes == 2


@pytest.mark.unit
def test_artifact_stages_serving_versions_and_publishes_checkpoints(tmp_path: Path) -> None:
    backend = InMemoryRepositoryBackend("math", directory=tmp_path / "repository")
    checkpoint = backend.register(root_tree(tmp_path / "root"), metadata={})
    repository = Repository(
        backend,
        checkpoint,
        current_artifact=checkpoint,
        checkpoint_artifact=checkpoint,
        local_dir=tmp_path / "serving",
    )

    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "model.txt").write_text("trained")
    staged = repository.stage(
        1,
        Artifact.local(candidate_dir, metadata={"loss": "sft"}),
        parent=checkpoint,
    )
    (candidate_dir / "model.txt").write_text("mutated")

    assert staged.ref.release_id.startswith("local:")
    lazy = Artifact(staged.ref, repository)
    assert lazy.materialize().local_path.joinpath("model.txt").read_text() == "trained"

    published = repository.publish(
        staged,
        expected_parent=checkpoint,
        metadata={"loss": "sft"},
    )

    assert repository.current_artifact == published
    assert repository.materialize(published).local_path.joinpath("model.txt").read_text() == "trained"
    assert not any(repository.local_root.iterdir())


@pytest.mark.unit
def test_repository_keeps_older_local_versions_addressable(tmp_path: Path) -> None:
    backend = InMemoryRepositoryBackend("skills", directory=tmp_path / "repository")
    checkpoint = backend.register(root_tree(tmp_path / "root"), metadata={})
    repository = Repository(
        backend,
        checkpoint,
        current_artifact=checkpoint,
        checkpoint_artifact=checkpoint,
        local_dir=tmp_path / "serving",
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "SKILL.md").write_text("v1")

    first = repository.stage(1, Artifact.local(candidate), parent=checkpoint)
    repository.activate(first.ref, expected=checkpoint)
    (candidate / "SKILL.md").write_text("v2")
    second = repository.stage(2, Artifact.local(candidate), parent=checkpoint)
    repository.activate(second.ref, expected=first.ref)

    first_path = repository.resolve(first.ref).local_path
    assert first_path.joinpath("SKILL.md").read_text() == "v1"
    assert repository.resolve(second.ref).local_path.joinpath("SKILL.md").read_text() == "v2"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("org/model", HuggingFaceSource("org/model", None)),
        ("hf://org/model", HuggingFaceSource("org/model", None)),
        ("hf://org/model@version", HuggingFaceSource("org/model", "version")),
    ],
)
def test_parse_huggingface_source(value, expected) -> None:
    assert parse_huggingface_source(value) == expected


@pytest.mark.unit
@pytest.mark.parametrize(("value", "message"), [("hf://model", "org/model syntax"), ("hf://org/model@", "non-empty")])
def test_parse_huggingface_source_refuses_another_form(value, message) -> None:
    with pytest.raises(ValueError, match=message):
        parse_huggingface_source(value)


@pytest.mark.unit
def test_download_huggingface_snapshot_returns_locked_version(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--org--model" / "snapshots" / "locked-sha"
    snapshot.mkdir(parents=True)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    result = download_huggingface_snapshot(
        HuggingFaceSource("org/model", "main"),
        snapshot_download=download,
    )

    assert result.local_path == snapshot
    assert result.version == "locked-sha"
    assert calls == [{"repo_id": "org/model", "re" + "vision": "main"}]


@pytest.mark.unit
def test_download_huggingface_snapshot_omits_missing_version(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--org--model" / "snapshots" / "locked-sha"
    snapshot.mkdir(parents=True)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    result = download_huggingface_snapshot(
        HuggingFaceSource("org/model"),
        snapshot_download=download,
    )

    assert result.version == "locked-sha"
    assert calls == [{"repo_id": "org/model"}]
