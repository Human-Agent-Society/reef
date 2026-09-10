from __future__ import annotations

from pathlib import Path

import pytest

from reef.artifact import (
    Artifact,
    ArtifactConflict,
    ArtifactRef,
    ArtifactReleaseChain,
    CachedRepositoryBackendFactory,
    GitVersionSource,
    HuggingFaceSource,
    InMemoryRepositoryBackend,
    LiveWeightArtifactRef,
    Repository,
    RepositoryBackend,
    download_huggingface_snapshot,
    parse_artifact_source,
)


@pytest.mark.unit
def test_live_weight_artifact_ref_specializes_the_generic_identity() -> None:
    durable = ArtifactRef("artifact-1", "checkpoint-1", None)
    live = LiveWeightArtifactRef("artifact-2", "live-2", "checkpoint-1", "engine:2")

    assert not hasattr(durable, "runtime_load_id")
    assert isinstance(live, ArtifactRef)
    assert live.runtime_load_id == "engine:2"
    with pytest.raises(ValueError, match="non-empty"):
        LiveWeightArtifactRef("artifact-3", "live-3", "checkpoint-1", "")


@pytest.mark.unit
def test_artifact_ref_can_carry_ephemeral_local_path_without_changing_identity(tmp_path: Path) -> None:
    durable = ArtifactRef("artifact-1", "version-1", None)
    local = Artifact(
        durable,
        InMemoryRepositoryBackend("math", tmp_path),
        local_path=tmp_path,
    )

    assert local.local_path == tmp_path
    assert local.ref == durable


@pytest.mark.unit
def test_artifact_is_the_process_local_artifact_coordinator(tmp_path: Path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()

    backend = InMemoryRepositoryBackend("math", initial)
    repository = Repository(backend, backend.resolve_release())
    artifact = repository.materialize(repository.fork())

    assert not hasattr(artifact.ref, "scenario")


@pytest.mark.unit
def test_release_id_chain_owns_heads_staging_and_publication(tmp_path: Path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "model.txt").write_text("base")
    backend = InMemoryRepositoryBackend("math", initial)
    base = backend.resolve_release()
    fork = backend.fork(base.release_id)
    repository = Repository(
        backend,
        base,
        current_artifact=fork,
        checkpoint_artifact=fork,
        local_dir=tmp_path / "local",
    )
    chain = ArtifactReleaseChain(repository, process_id="worker")

    expected, live = chain.prepare_live(step=1, runtime_load_id="engine-1")
    _, retry = chain.prepare_live(step=1, runtime_load_id="engine-1")
    assert expected == fork
    assert chain.current == fork
    assert retry == live
    assert live.release_id == "live:worker:engine-1:1"
    assert live.parent_release_id == fork.release_id

    chain.advance(live, expected=expected)
    assert chain.current == live
    assert chain.checkpoint == fork

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "model.txt").write_text("trained")
    staged = chain.stage(2, Artifact.local(candidate))
    published = chain.publish(staged)

    assert chain.current == published
    assert chain.checkpoint == published
    assert published.parent_release_id == fork.release_id


@pytest.mark.unit
def test_in_memory_repository_forks_materializes_and_publishes(tmp_path: Path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "model.txt").write_text("base")
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    math_backend = backend_factory("math")
    code_backend = backend_factory("code")
    code_initial_ref = code_backend.resolve_release()

    math = math_backend.fork()
    code = code_backend.fork()

    assert math.release_id != code.release_id
    assert math_backend.fork() == math
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
    assert math_backend.materialize(published).local_path.joinpath("model.txt").read_text() == "trained"
    assert code_backend.current() == code

    # Each scenario keeps its own latest; math's publish does not leak into code.
    # code has forked but not published, so its latest is still its own bootstrap.
    assert code_backend.resolve_release() == code_initial_ref
    assert math_backend.resolve_release() == published
    # A fresh backend bootstraps from the same source but has its own storage.
    from_initial_backend = backend_factory("from-initial")
    from_initial = from_initial_backend.fork()
    assert from_initial.parent_release_id == from_initial_backend.resolve_release().release_id
    assert from_initial_backend.materialize(from_initial).local_path.joinpath("model.txt").read_text() == "base"

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
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "model.txt").write_text("base")
    backend = InMemoryRepositoryBackend("math", initial, root=tmp_path / "repository")
    repository = Repository(
        backend,
        backend.resolve_release(),
        local_dir=tmp_path / "serving",
    )
    checkpoint = repository.fork()

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
    initial = tmp_path / "initial"
    initial.mkdir()
    backend = InMemoryRepositoryBackend("skills", initial, root=tmp_path / "repository")
    repository = Repository(backend, backend.resolve_release(), local_dir=tmp_path / "serving")
    checkpoint = repository.fork()
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "SKILL.md").write_text("v1")

    first = repository.stage(1, Artifact.local(candidate), parent=checkpoint)
    repository.advance_current(first.ref, expected=checkpoint)
    (candidate / "SKILL.md").write_text("v2")
    second = repository.stage(2, Artifact.local(candidate), parent=checkpoint)
    repository.advance_current(second.ref, expected=first.ref)

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
        ("git+lfs://abc123", GitVersionSource("abc123")),
    ],
)
def test_parse_artifact_source(value, expected) -> None:
    assert parse_artifact_source(value) == expected


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
