from __future__ import annotations

import base64
import json
import os
import shutil
import stat
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from reef.artifact import (
    Artifact,
    ArtifactConflictError,
    ArtifactMaterializationError,
    ArtifactNotFoundError,
    ArtifactPublicationError,
    ArtifactSourceError,
    DefaultStart,
    GitLFSRepositoryBackend,
    GitLFSRepositoryBackendFactory,
    InMemoryRepositoryBackend,
    RepositoryBackend,
    RepositoryBackendContext,
    SnapshotRef,
    StoredFiles,
)
from reef.artifact.composite import compose_snapshot, link_or_copy
from reef.artifact.git_client import GitClient
from reef.artifact.git_lfs import HUB_REFS, same_file, token_config
from reef.core.plugins import PluginConfigError


def link_model_files(snapshot: Path, directory: Path) -> None:
    """Put the model files of ``snapshot`` into ``directory`` as hard links where the file system allows, as an
    import of a local model directory stores them."""
    shutil.copytree(snapshot, directory, dirs_exist_ok=True, copy_function=link_or_copy)


def root_tree(directory: Path, files: dict[str, str] | None = None) -> Artifact:
    """An unpublished root snapshot in ``directory`` that holds ``files``."""
    directory.mkdir(parents=True)
    for relative, text in (files or {}).items():
        (directory / relative).parent.mkdir(parents=True, exist_ok=True)
        (directory / relative).write_text(text)
    return Artifact.local(directory)


def run_git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.mark.integration
def test_git_lfs_repository_publishes_with_a_relative_work_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_git_lfs: None,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = GitLFSRepositoryBackend(
        "math",
        tmp_path / "artifacts.git",
        work_dir=Path("work"),
        cache_dir=tmp_path / "cache",
    )
    parent = backend.register(root_tree(tmp_path / "root", {"rules.md": "old rules\n"}), metadata={})
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "rules.md").write_text("new rules\n")

    published = backend.publish(Artifact.local(candidate), expected_parent=parent)

    assert published.parent_revision == parent.revision
    assert backend.current() == published
    materialized = backend.materialize(published).local_path
    assert materialized is not None
    assert (materialized / "rules.md").read_text() == "new rules\n"
    assert not (tmp_path / "work" / "publish-index").exists()


@pytest.mark.integration
def test_artifact_identity_survives_publication_and_a_fresh_git_cache(tmp_path: Path, fake_git_lfs: None) -> None:
    remote = tmp_path / "artifacts.git"
    backend = GitLFSRepositoryBackend("math", remote, work_dir=tmp_path / "work", cache_dir=tmp_path / "cache")
    root = root_tree(tmp_path / "root", {"rules.md": "initial"})
    initial = backend.register(root, metadata={})
    candidate = root_tree(tmp_path / "candidate", {"rules.md": "updated"})
    identity = candidate.artifact_id
    published = backend.publish(candidate, expected_parent=initial)
    assert backend.materialize(initial).artifact_id == root.artifact_id
    assert backend.materialize(published).artifact_id == identity
    assert identity not in (candidate.ref.revision, published.revision)

    recovered = GitLFSRepositoryBackend(
        "math", remote, work_dir=tmp_path / "recovered-work", cache_dir=tmp_path / "recovered-cache"
    )
    assert recovered.materialize(recovered.current()).artifact_id == identity
    assert recovered.resolve_artifact(identity).artifact_id == identity
    assert recovered.resolve_artifact(root.artifact_id).artifact_id == root.artifact_id
    assert recovered.materialize(recovered.resolve_snapshot(initial.revision)).artifact_id == root.artifact_id


def test_a_batch_lookup_gives_each_binding_as_one_lookup_does(tmp_path: Path, fake_git_lfs: None) -> None:
    remote = tmp_path / "artifacts.git"
    backend = GitLFSRepositoryBackend("math", remote, work_dir=tmp_path / "work", cache_dir=tmp_path / "cache")
    root = root_tree(tmp_path / "root", {"rules.md": "initial"})
    head = backend.register(root, metadata={})
    trained = root_tree(tmp_path / "trained", {"rules.md": "updated"})
    backend.publish(trained, expected_parent=head)

    fresh = GitLFSRepositoryBackend("math", remote, work_dir=tmp_path / "fresh", cache_dir=tmp_path / "fresh-cache")
    found = fresh.artifact_snapshots({root.artifact_id, trained.artifact_id, "artifact:missing"})
    assert found == {
        root.artifact_id: backend.artifact_snapshot(root.artifact_id),
        trained.artifact_id: backend.artifact_snapshot(trained.artifact_id),
    }
    assert fresh.artifact_snapshots(set()) == {}


def test_the_default_batch_lookup_leaves_out_an_id_without_a_binding(tmp_path: Path) -> None:
    backend = InMemoryRepositoryBackend("math", directory=tmp_path / "stored")
    root = root_tree(tmp_path / "root", {"rules.md": "initial"})
    stored = backend.register(root, metadata={})
    assert backend.artifact_snapshots({root.artifact_id, "artifact:missing"}) == {root.artifact_id: stored}


@pytest.mark.parametrize("storage", ["memory", "git"])
def test_artifact_id_binding_is_immutable_and_scoped_to_its_scenario(
    tmp_path: Path, fake_git_lfs: None, storage: str
) -> None:
    backend: RepositoryBackend
    other: RepositoryBackend
    if storage == "git":
        remote = tmp_path / "artifacts.git"
        backend = GitLFSRepositoryBackend("one", remote, work_dir=tmp_path / "work", cache_dir=tmp_path / "cache")
        other = GitLFSRepositoryBackend(
            "two", remote, work_dir=tmp_path / "other-work", cache_dir=tmp_path / "other-cache"
        )
    else:
        backend = InMemoryRepositoryBackend("one", directory=tmp_path / "store")
        other = InMemoryRepositoryBackend("two", directory=tmp_path / "other-store")
    root = root_tree(tmp_path / "root", {"rules.md": "original"})
    head = backend.register(root, metadata={})
    candidate = root_tree(tmp_path / "candidate", {"rules.md": "updated"})
    published = backend.publish(candidate, expected_parent=head)
    repeated = backend.publish(candidate, expected_parent=published)
    resolved = backend.resolve_artifact(candidate.artifact_id)
    assert resolved.artifact_id == candidate.artifact_id
    assert resolved.local_path is not None
    assert (resolved.local_path / "rules.md").read_text() == "updated"
    with pytest.raises(ArtifactNotFoundError):
        other.resolve_artifact(candidate.artifact_id)
    changed = root_tree(tmp_path / "changed", {"rules.md": "different"})
    conflicting = Artifact(
        changed.ref,
        None,
        local_path=changed.local_path,
        metadata=candidate.metadata,
        artifact_id=candidate.artifact_id,
    )
    with pytest.raises(ArtifactConflictError, match="already names different"):
        backend.publish(conflicting, expected_parent=repeated)
    assert backend.current() == repeated
    assert (backend.resolve_artifact(candidate.artifact_id).local_path / "rules.md").read_text() == "updated"


@pytest.mark.integration
def test_each_scenario_registers_its_own_root_snapshot_with_no_parent(tmp_path: Path, fake_git_lfs: None) -> None:
    """A registration pushes a commit with no parent to the scenario's ref; a second registration of the scenario
    stores nothing and gets the winner's head; another scenario gets a distinct root. No shared ref exists."""
    remote = tmp_path / "artifacts.git"
    factory = GitLFSRepositoryBackend.factory(remote, work_dir=tmp_path / "work", cache_dir=tmp_path / "cache")
    first = factory("first")
    assert first.metadata() is None
    files = {"pi-agent/AGENTS.md": "seed rules\n", "pi-agent/skills/a/SKILL.md": "# a\n"}
    root = first.register(root_tree(tmp_path / "first-root", files), metadata={"m": 1})
    assert root.parent_revision is None
    assert run_git("--git-dir", str(remote), "rev-list", "--parents", "-n", "1", root.revision) == root.revision
    assert first.current() == root and first.metadata() == {"m": 1}
    tree = first.materialize(root).local_path
    assert tree is not None
    assert (tree / "pi-agent" / "AGENTS.md").read_text(encoding="utf-8") == "seed rules\n"
    assert (tree / "pi-agent" / "skills" / "a" / "SKILL.md").read_text(encoding="utf-8") == "# a\n"
    assert json.loads((tree / "reef-artifact.json").read_text(encoding="utf-8"))["source"] == {"kind": "root"}

    late = factory("first").register(root_tree(tmp_path / "late-root", {"x": "late"}), metadata={"m": 2})
    assert late == root and first.metadata() == {"m": 1}
    other = factory("second").register(root_tree(tmp_path / "second-root", files), metadata={})
    assert other.revision != root.revision and other.parent_revision is None
    assert "refs/reef/base" not in run_git("--git-dir", str(remote), "for-each-ref", "--format=%(refname)")
    assert first.resolve_snapshot(root.revision) == root


@pytest.mark.integration
def test_configured_factory_registers_a_scenario_from_its_context(tmp_path: Path, fake_git_lfs: None) -> None:
    context = RepositoryBackendContext(work_dir=tmp_path / "work", cache_dir=tmp_path / "cache")
    factory = GitLFSRepositoryBackendFactory.from_config(
        {"repository": str(tmp_path / "artifacts.git")}, context=context
    )

    backend = factory("math")
    assert factory.list_registrations() == ()
    root = backend.register(root_tree(tmp_path / "root", {"agent/AGENTS.md": "seed rules\n"}), metadata={})
    tree = backend.materialize(root).local_path
    assert tree is not None
    assert (tree / "agent" / "AGENTS.md").read_text(encoding="utf-8") == "seed rules\n"
    assert factory.list_registrations() == ("math",)


@pytest.mark.integration
def test_configured_factory_reads_a_relative_repository_below_the_top_of_a_git_work_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_git_lfs: None
) -> None:
    # An example directory of a Reef checkout: Git reads a relative remote path from the top of the work tree.
    example = tmp_path / "checkout" / "example"
    example.mkdir(parents=True)
    run_git("init", str(tmp_path / "checkout"))
    monkeypatch.chdir(example)
    context = RepositoryBackendContext(work_dir=Path("work/artifact-work"), cache_dir=Path("work/artifact-cache"))
    config = {"repository": "work/artifacts.git"}
    factory = GitLFSRepositoryBackendFactory.from_config(config, context=context)
    factory("math").register(root_tree(tmp_path / "root", {"rules.md": "seed rules\n"}), metadata={})

    restarted = GitLFSRepositoryBackendFactory.from_config(config, context=context)

    assert restarted.list_registrations() == ("math",)
    assert restarted.has_registration("math") and not restarted.has_registration("other")


@pytest.mark.integration
@pytest.mark.parametrize("repository", ["https://git.example/repository", "git@example:repository"])
def test_configured_factory_keeps_a_remote_location(repository: str, tmp_path: Path, fake_git_lfs: None) -> None:
    context = RepositoryBackendContext(work_dir=tmp_path / "work", cache_dir=tmp_path / "cache")

    factory = GitLFSRepositoryBackendFactory.from_config({"repository": repository}, context=context)

    assert factory._repository == repository


@pytest.mark.parametrize(
    ("config", "message"),
    [({}, "repository is required"), ({"repository": " "}, "non-empty"), ({"repository": "a", "url": "b"}, "url")],
)
def test_configured_factory_refuses_an_invalid_config(config: dict[str, str], message: str, tmp_path: Path) -> None:
    context = RepositoryBackendContext(work_dir=tmp_path / "work", cache_dir=tmp_path / "cache")

    with pytest.raises(PluginConfigError, match=message):
        GitLFSRepositoryBackendFactory.from_config(config, context=context)
    assert not (tmp_path / "work").exists()


@pytest.mark.integration
def test_local_repository_bootstrap_recovers_after_missing_git_lfs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_git = shutil.which("git")
    assert real_git is not None
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    git = executable_dir / "git"
    git.write_text(f'#!/bin/sh\nexec {real_git} "$@"\n')
    git.chmod(0o755)
    monkeypatch.setenv("PATH", str(executable_dir))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)

    remote = tmp_path / "artifacts.git"
    with pytest.raises(ArtifactSourceError, match="git lfs version"):
        GitLFSRepositoryBackend.factory(
            remote,
            work_dir=tmp_path / "failed-work",
            cache_dir=tmp_path / "failed-cache",
        )
    assert not remote.exists()
    assert not (tmp_path / "failed-work").exists()
    assert not (tmp_path / "failed-cache").exists()

    # Recreate the empty directory left by constructors from before this fix.
    subprocess.run([real_git, "init", "--bare", str(remote)], check=True, capture_output=True)

    git_lfs = executable_dir / "git-lfs"
    git_lfs.write_text("#!/bin/sh\nexit 0\n")
    git_lfs.chmod(0o755)
    backend_factory = GitLFSRepositoryBackend.factory(
        remote,
        work_dir=tmp_path / "retry-work",
        cache_dir=tmp_path / "retry-cache",
    )
    backend = backend_factory("math")

    root = backend.register(root_tree(tmp_path / "root"), metadata={})
    assert run_git("--git-dir", str(remote), "rev-parse", backend.ref_name) == root.revision


@pytest.mark.integration
def test_local_repository_bootstrap_creates_the_repository(tmp_path: Path, fake_git_lfs: None) -> None:
    remote = tmp_path / "artifacts.git"
    first = GitLFSRepositoryBackend("math", remote, work_dir=tmp_path / "first-work", cache_dir=tmp_path / "cache")
    assert remote.is_dir()
    clone = tmp_path / "first-work" / "repository"
    assert run_git("-C", str(clone), "config", "--local", "core.hooksPath") == str(clone / ".git" / "reef-hooks")
    root = first.register(root_tree(tmp_path / "root"), metadata={})

    recovered = GitLFSRepositoryBackend("math", remote, work_dir=tmp_path / "second-work", cache_dir=tmp_path / "c2")

    assert recovered.current() == root


@pytest.mark.integration
def test_concurrent_registrations_of_one_scenario_make_one_root(tmp_path: Path, fake_git_lfs: None) -> None:
    remote = tmp_path / "artifacts.git"
    GitLFSRepositoryBackend("bootstrap", remote, work_dir=tmp_path / "work-bootstrap", cache_dir=tmp_path / "cache")

    def register(index: int):
        backend = GitLFSRepositoryBackend(
            "scenario", remote, work_dir=tmp_path / f"work-{index}", cache_dir=tmp_path / f"cache-{index}"
        )
        return backend.register(root_tree(tmp_path / f"root-{index}", {"n": str(index)}), metadata={"n": index})

    with ThreadPoolExecutor(max_workers=8) as executor:
        heads = tuple(executor.map(register, range(8)))

    assert len({head.revision for head in heads}) == 1
    encoded = base64.urlsafe_b64encode(b"scenario").decode().rstrip("=")
    assert run_git("--git-dir", str(remote), "rev-parse", f"refs/reef/scenarios/{encoded}") == heads[0].revision


@pytest.mark.integration
def test_metadata_and_snapshot_reads_share_one_git_workspace_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_git_lfs: None
) -> None:
    remote, work = tmp_path / "artifacts.git", tmp_path / "work"
    first = GitLFSRepositoryBackend("scenario", remote, work_dir=work, cache_dir=tmp_path / "cache")
    root = first.register(root_tree(tmp_path / "root"), metadata={"generation": "root"})
    candidate = root_tree(tmp_path / "candidate", {"rules.md": "new rules"})
    candidate = Artifact(
        candidate.ref,
        None,
        artifact_id=candidate.artifact_id,
        local_path=candidate.local_path,
        metadata={"generation": "new"},
    )
    first.publish(candidate, expected_parent=root)
    second = GitLFSRepositoryBackend("scenario", remote, work_dir=work / ".", cache_dir=tmp_path / "cache")
    fetched, release, attempted, second_fetch = Event(), Event(), Event(), Event()
    fetch_version = first._workspace.fetch_version
    git = second._git_client.git

    def hold_metadata_fetch(version: str) -> str:
        resolved = fetch_version(version)
        fetched.set()
        assert release.wait(5)
        return resolved

    def observe_other_fetch(*args: str) -> str:
        if args[0] == "fetch":
            second_fetch.set()
        return git(*args)

    def read_root():
        attempted.set()
        return second.resolve_snapshot(root.revision)

    monkeypatch.setattr(first._workspace, "fetch_version", hold_metadata_fetch)
    monkeypatch.setattr(second._git_client, "git", observe_other_fetch)
    with ThreadPoolExecutor(max_workers=2) as executor:
        metadata = executor.submit(first.metadata)
        try:
            assert fetched.wait(5)
            resolved = executor.submit(read_root)
            assert attempted.wait(5)
            assert not second_fetch.wait(0.1), "A concurrent fetch replaced metadata's FETCH_HEAD"
        finally:
            release.set()
        assert metadata.result(timeout=5) == {"generation": "new"}
        assert resolved.result(timeout=5) == root


@pytest.mark.integration
def test_git_lfs_repository_registers_publishes_and_materializes(
    tmp_path: Path,
    fake_git_lfs: None,
) -> None:
    remote = tmp_path / "artifacts.git"
    run_git("init", "--bare", str(remote))

    snapshot = tmp_path / "models--org--model" / "snapshots" / "upstream-sha"
    snapshot.mkdir(parents=True)
    blob = tmp_path / "hub-cache" / "blob"
    blob.parent.mkdir()
    blob.write_text("base")
    (snapshot / "model.safetensors").symlink_to(blob)
    (snapshot / "config.json").write_text("{}")

    backend_factory = GitLFSRepositoryBackend.factory(remote, work_dir=tmp_path / "work", cache_dir=tmp_path / "cache")

    def default_root(name: str) -> Artifact:
        (tmp_path / name).mkdir()
        link_model_files(snapshot, tmp_path / name)
        return Artifact.local(tmp_path / name)

    backend = backend_factory("math")
    code_backend = backend_factory("code")
    math = backend.register(default_root("math-root"), metadata={"scenario_commit_record": {"recipe": "openclawrl"}})
    code = code_backend.register(default_root("code-root"), metadata={})
    manifest = json.loads(run_git("--git-dir", str(remote), "show", f"{math.revision}:reef-artifact.json"))
    assert math.revision != code.revision
    assert backend.materialize(math).artifact_id != code_backend.materialize(code).artifact_id
    assert "content_id" not in manifest
    assert "scenario" not in manifest
    assert backend.metadata() == {"scenario_commit_record": {"recipe": "openclawrl"}}
    assert run_git("--git-dir", str(remote), "rev-parse", backend.ref_name) == math.revision

    fresh_factory = GitLFSRepositoryBackend.factory(
        remote, work_dir=tmp_path / "fresh-work", cache_dir=tmp_path / "fresh-cache"
    )
    assert fresh_factory.has_registration("math")
    assert not fresh_factory.has_registration("unregistered")
    assert not (tmp_path / "fresh-work").exists()
    assert not (tmp_path / "fresh-cache").exists()

    materialized = backend.materialize(math)
    assert materialized.local_path.joinpath("model.safetensors").read_text() == "base"
    assert not materialized.local_path.joinpath("model.safetensors").is_symlink()
    assert "*.safetensors filter=lfs" in materialized.local_path.joinpath(".gitattributes").read_text()

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "adapter.safetensors").write_text("trained")
    published = backend.publish(
        Artifact.local(candidate, metadata={"loss": "sft"}),
        expected_parent=math,
    )

    assert published.parent_revision == math.revision
    assert backend.current() == published
    assert code_backend.current() == code
    assert backend.materialize(published).local_path.joinpath("adapter.safetensors").read_text() == "trained"
    fresh = fresh_factory("math")
    with ThreadPoolExecutor(max_workers=8) as executor:
        copies = tuple(executor.map(fresh.materialize, (published,) * 8))
    assert all(copy.local_path.joinpath("adapter.safetensors").read_text() == "trained" for copy in copies)

    with pytest.raises(ArtifactConflictError):
        backend.publish(Artifact.local(candidate), expected_parent=math)


def trained_weights(directory: Path, contents: bytes) -> Artifact:
    (directory / "weights").mkdir(parents=True)
    (directory / "weights" / "adapter_model.safetensors").write_bytes(contents)
    return Artifact.local(directory)


@pytest.mark.integration
def test_a_fresh_workspace_publishes_with_real_lfs_over_objects_it_never_fetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restarted service's fresh workspace fetches no LFS objects, so neither a pending snapshot nor a publish from
    it may ask Git LFS to upload the trained version, which the remote already holds.

    A global ``core.hooksPath`` and a Git template both install user hooks that fail; Reef's clones must run their own
    hooks only and leave the user's configuration and hooks unchanged.
    """
    global_config = tmp_path / "gitconfig"
    global_config.touch()
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    hooks = tmp_path / "template" / "hooks"
    hooks.mkdir(parents=True)
    hook_contents = "#!/bin/sh\necho 'unrelated user hook must not run' >&2\nexit 1\n"
    for name in ("pre-push", "post-checkout"):
        (hooks / name).write_text(hook_contents)
        (hooks / name).chmod(0o755)
    run_git("config", "--global", "core.hooksPath", str(hooks))
    monkeypatch.setenv("GIT_TEMPLATE_DIR", str(hooks.parent))
    original_config = global_config.read_text()
    remote = tmp_path / "artifacts.git"
    first = GitLFSRepositoryBackend("scenario-a", remote, work_dir=tmp_path / "work-a", cache_dir=tmp_path / "cache-a")
    root = first.register(trained_weights(tmp_path / "root", b"adapter v0" * 100), metadata={"scenario": "a"})
    trained = first.publish(trained_weights(tmp_path / "trained-1", b"adapter v1" * 100), expected_parent=root)
    assert trained.parent_revision == root.revision
    assert run_git("--git-dir", str(remote), "rev-parse", first.ref_name) == trained.revision

    # A restarted service materializes, saves a pending snapshot and publishes from a fresh workspace.
    fresh = GitLFSRepositoryBackend("scenario-a", remote, work_dir=tmp_path / "fresh-work", cache_dir=tmp_path / "c2")
    materialized = fresh.materialize(trained).local_path
    assert materialized is not None
    assert (materialized / "weights" / "adapter_model.safetensors").read_bytes() == b"adapter v1" * 100
    assert not (materialized / ".git").exists()
    # A pending snapshot pushes to a new ref and leaves the head alone: the LFS check still stops at history the
    # remote already holds.
    pending = fresh.publish(
        trained_weights(tmp_path / "trained-2", b"adapter v2" * 100), expected_parent=trained, advance_head=False
    )
    assert pending.parent_revision == trained.revision and fresh.current() == trained
    published = fresh.publish(trained_weights(tmp_path / "trained-3", b"adapter v3" * 100), expected_parent=trained)
    assert published.parent_revision == trained.revision and fresh.current() == published

    assert global_config.read_text() == original_config
    for name in ("pre-push", "post-checkout"):
        assert (hooks / name).read_text() == hook_contents
        assert (tmp_path / "fresh-work" / "repository" / ".git" / "hooks" / name).read_text() == hook_contents


@pytest.mark.integration
def test_archiving_a_scenario_renames_its_ref_and_drops_its_work_clone(
    tmp_path: Path, fake_git_lfs: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The factory's archive moves ``refs/reef/scenarios/<name>`` under ``refs/reef/archived`` and removes
    the scenario's work clone; other scenarios stay, and the name is free again."""
    # Fix both Git timestamps so identical roots cannot differ only because time passed.
    monkeypatch.setenv("GIT_AUTHOR_DATE", "2026-01-01T00:00:00Z")
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2026-01-01T00:00:00Z")
    remote = tmp_path / "artifacts.git"
    factory = GitLFSRepositoryBackend.factory(remote, work_dir=tmp_path / "work", cache_dir=tmp_path / "cache")
    kept_ref = factory("kept").register(root_tree(tmp_path / "kept-root"), metadata={})
    doomed_ref = factory("doomed").register(root_tree(tmp_path / "doomed-root"), metadata={})
    assert kept_ref.revision != doomed_ref.revision
    assert set(factory.list_registrations()) == {"doomed", "kept"}
    encoded = base64.urlsafe_b64encode(b"doomed").decode().rstrip("=")
    assert (tmp_path / "work" / encoded).is_dir()

    archived = factory.archive_registration("doomed")

    assert factory.list_registrations() == ("kept",)
    assert not factory.has_registration("doomed")
    assert len(archived) == 2 and archived[0].startswith("refs/reef/archived/")
    assert run_git("--git-dir", str(remote), "rev-parse", archived[0]) == doomed_ref.revision
    assert not (tmp_path / "work" / encoded).exists()
    # The name creates fresh: a new backend registers a new root rather than continuing the archived chain.
    fresh = factory("doomed")
    assert fresh.metadata() is None
    again = fresh.register(root_tree(tmp_path / "again-root"), metadata={})
    assert again.parent_revision is None and again.revision != doomed_ref.revision


@pytest.mark.unit
def test_a_default_start_writes_its_text_files_and_refuses_other_values(tmp_path: Path) -> None:
    start = DefaultStart(
        files={"harness/AGENTS.md": "seed rules\n", "weights/base.json": "{}\n"},
        inputs={"weights.model": "org/model"},
    )
    tree = tmp_path / "root"
    tree.mkdir()
    start.write(tree)

    assert (tree / "harness" / "AGENTS.md").read_text() == "seed rules\n"
    assert (tree / "weights" / "base.json").read_text() == "{}\n"
    assert sorted(path.name for path in tree.iterdir()) == ["harness", "weights"]
    with pytest.raises(ValueError, match="relative path inside the snapshot"):
        DefaultStart(files={"../x": "text"})
    with pytest.raises(ValueError, match="must hold text"):
        DefaultStart(files={"model.safetensors": b"weights"})
    with pytest.raises(ValueError, match="non-empty string"):
        DefaultStart(inputs={"model": ""})


@pytest.mark.integration
def test_a_published_file_is_linked_into_the_snapshot_cache_not_copied(tmp_path: Path, fake_git_lfs: None) -> None:
    """Stored files are immutable, so a publish links them into the snapshot's cache entry instead of cloning it."""
    remote = tmp_path / "artifacts.git"
    backend = GitLFSRepositoryBackend("agent", remote, work_dir=tmp_path / "work", cache_dir=tmp_path / "cache")
    head = backend.register(root_tree(tmp_path / "root"), metadata={})
    source = tmp_path / "snapshot"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"\x00" * 64)
    published = backend.publish(Artifact.local(source), expected_parent=head)
    cached = tmp_path / "cache" / published.revision
    assert (cached / "weights.bin").samefile(source / "weights.bin")
    assert not (tmp_path / "work" / "repository" / "weights.bin").exists()
    materialized = backend.materialize(published)
    assert materialized.local_path == cached
    assert json.loads((cached / "reef-artifact.json").read_text())["parent_revision"] == head.revision


@pytest.mark.unit
def test_same_file_names_one_inode_and_a_missing_path_names_none(tmp_path: Path) -> None:
    (tmp_path / "file").write_text("x")
    os.link(tmp_path / "file", tmp_path / "link")
    (tmp_path / "copy").write_text("x")
    assert same_file(tmp_path / "file", tmp_path / "link")
    assert not same_file(tmp_path / "file", tmp_path / "copy")
    assert not same_file(tmp_path / "file", tmp_path / "missing")


@pytest.mark.integration
def test_a_carried_component_is_neither_hashed_nor_copied_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snapshot that carries the parent's files forward as links reuses their blobs; only the changed files are read."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    remote = tmp_path / "artifacts.git"
    backend = GitLFSRepositoryBackend("agent", remote, work_dir=tmp_path / "work", cache_dir=tmp_path / "cache")
    head = backend.register(root_tree(tmp_path / "root"), metadata={})
    first = tmp_path / "first"
    (first / "weights").mkdir(parents=True)
    (first / "harness").mkdir()
    (first / "weights" / "model.safetensors").write_bytes(b"weights v1")
    (first / "harness" / "AGENTS.md").write_text("rules v1\n")
    (first / "harness" / "old.md").write_text("dropped\n")
    parent = backend.publish(Artifact.local(first), expected_parent=head)
    parent_tree = backend.materialize(parent).local_path
    assert parent_tree is not None

    second = tmp_path / "second"
    shutil.copytree(parent_tree / "weights", second / "weights", copy_function=os.link)
    (second / "harness").mkdir()
    (second / "harness" / "AGENTS.md").write_text("rules v2\n")
    carried = second / "weights" / "model.safetensors"
    # A carried file that were hashed again would fail to read.
    carried.chmod(0)
    try:
        published = backend.publish(Artifact.local(second), expected_parent=parent)
    finally:
        carried.chmod(0o644)

    def blob(revision: str, path: str) -> str:
        return run_git("--git-dir", str(remote), "rev-parse", f"{revision}:{path}")

    assert blob(published.revision, "weights/model.safetensors") == blob(parent.revision, "weights/model.safetensors")
    assert run_git("--git-dir", str(remote), "show", f"{published.revision}:weights/model.safetensors").startswith(
        "version https://git-lfs.github.com/spec/v1"
    )
    cached = backend.materialize(published).local_path
    assert cached is not None
    assert (cached / "weights" / "model.safetensors").samefile(parent_tree / "weights" / "model.safetensors")
    # The linked cache entry holds what a clone of the snapshot holds.
    cloned = GitLFSRepositoryBackend(
        "agent", remote, work_dir=tmp_path / "fresh-work", cache_dir=tmp_path / "fresh-cache"
    ).materialize(published)
    assert cloned.local_path is not None

    def tree(root: Path) -> dict[str, bytes]:
        return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    assert tree(cached) == tree(cloned.local_path)
    assert tree(cached)["harness/AGENTS.md"] == b"rules v2\n"
    assert "harness/old.md" not in tree(cached)


@pytest.mark.integration
def test_the_work_tree_never_writes_through_a_link(tmp_path: Path, fake_git_lfs: None) -> None:
    """A linked file shares its bytes with the source, so the files Reef rewrites are written to a fresh inode."""
    remote = tmp_path / "artifacts.git"
    run_git("init", "--bare", str(remote))
    snapshot = tmp_path / "models--org--model" / "snapshots" / "upstream-sha"
    snapshot.mkdir(parents=True)
    blobs = tmp_path / "hub-cache"
    blobs.mkdir()
    (blobs / "attributes").write_text("*.bin filter=lfs\n")
    (blobs / "weights").write_text("base")
    (snapshot / ".gitattributes").symlink_to(blobs / "attributes")
    (snapshot / "model.safetensors").symlink_to(blobs / "weights")
    backend = GitLFSRepositoryBackend.factory(remote, work_dir=tmp_path / "work", cache_dir=tmp_path / "cache")(
        "agent"
    )
    (tmp_path / "root").mkdir()
    link_model_files(snapshot, tmp_path / "root")
    (tmp_path / "root" / "AGENTS.md").write_text("seed\n")
    head = backend.register(Artifact.local(tmp_path / "root"), metadata={})
    # The registration rewrote .gitattributes in the work tree, not the Hugging Face blob behind the snapshot.
    assert (blobs / "attributes").read_text() == "*.bin filter=lfs\n"
    assert (blobs / "weights").read_text() == "base"
    assert "*.safetensors filter=lfs" in backend.materialize(head).local_path.joinpath(".gitattributes").read_text()

    source = tmp_path / "snapshot"
    source.mkdir()
    (source / "reef-artifact.json").write_text("{}")
    (source / ".gitattributes").write_text("stale\n")
    (source / "weights.bin").write_bytes(b"\x01" * 8)
    backend.publish(Artifact.local(source), expected_parent=head)
    assert (source / "reef-artifact.json").read_text() == "{}"
    assert (source / ".gitattributes").read_text() == "stale\n"


@pytest.mark.integration
def test_a_broken_symlink_below_the_top_level_is_refused_by_name(tmp_path: Path, fake_git_lfs: None) -> None:
    remote = tmp_path / "artifacts.git"
    backend = GitLFSRepositoryBackend("agent", remote, work_dir=tmp_path / "work", cache_dir=tmp_path / "cache")
    head = backend.register(root_tree(tmp_path / "root"), metadata={})
    source = tmp_path / "snapshot"
    (source / "sub").mkdir(parents=True)
    (source / "sub" / "gone").symlink_to(tmp_path / "nowhere")
    with pytest.raises(ArtifactSourceError, match=r"broken symlink: .*sub/gone"):
        backend.publish(Artifact.local(source), expected_parent=head)


def hub_like_remote(tmp_path: Path) -> Path:
    """A bare repository with a ``main`` branch of its own, as a new Hugging Face repo has."""
    remote = tmp_path / "hub.git"
    run_git("init", "--bare", "-q", str(remote))
    seed = tmp_path / "hub-seed"
    run_git("init", "-q", str(seed))
    (seed / ".gitattributes").write_text("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
    run_git("-C", str(seed), "add", ".gitattributes")
    run_git("-C", str(seed), "-c", "user.name=hub", "-c", "user.email=hub@example", "commit", "-qm", "initial commit")
    run_git("-C", str(seed), "push", "-q", str(remote), "HEAD:refs/heads/main")
    return remote


@pytest.mark.integration
def test_the_hub_layout_keeps_snapshots_on_tags_and_moving_refs_on_branches(
    tmp_path: Path, fake_git_lfs: None
) -> None:
    remote = hub_like_remote(tmp_path)
    factory = GitLFSRepositoryBackend.factory(
        remote, work_dir=tmp_path / "work", cache_dir=tmp_path / "cache", refs=HUB_REFS
    )
    # The repo's own main branch is no ref of the backend's, so the repository counts as new.
    backend = factory("math")
    root = backend.register(root_tree(tmp_path / "root"), metadata={})
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "AGENTS.md").write_text("be brief\n")
    published = backend.publish(Artifact.local(candidate), expected_parent=root)
    pending = backend.publish(Artifact.local(candidate), expected_parent=published, advance_head=False)

    encoded = base64.urlsafe_b64encode(b"math").decode().rstrip("=")
    refs = dict(line.split()[::-1] for line in run_git("--git-dir", str(remote), "show-ref").splitlines())
    artifact_refs = {name for name in refs if name.startswith(f"refs/tags/reef/artifacts/{encoded}/")}
    assert len(artifact_refs) == 3
    assert refs.keys() - artifact_refs == {
        "refs/heads/main",
        f"refs/heads/reef/scenarios/{encoded}",
        f"refs/tags/reef/releases/{pending.revision}",
    }
    assert refs[f"refs/heads/reef/scenarios/{encoded}"] == published.revision
    # A snapshot's ID is its commit, so a snapshot on the Hub is the revision of that commit.
    assert backend.resolve_snapshot(pending.revision) == pending
    assert factory.has_registration("math") and factory.list_registrations() == ("math",)

    # Stored content is replaced and deleted, so its ref is a branch; a snapshot of it is a tag as any other.
    backend.store_content("job:1", candidate)
    (content_ref,) = set(run_git("--git-dir", str(remote), "show-ref").split()[1::2]) - refs.keys()
    assert content_ref.startswith(f"refs/heads/reef/contents/{encoded}/")
    by_name = backend.publish_stored(
        StoredFiles(content="job:1"), metadata={"artifact_id": "a1"}, expected_parent=published
    )
    assert by_name.parent_revision == published.revision and backend.current() == by_name
    backend.discard_content("job:1")
    assert content_ref not in run_git("--git-dir", str(remote), "show-ref")
    assert factory.list_registrations() == ("math",)

    # A repository keeps one layout: the other is refused instead of hiding the snapshots.
    with pytest.raises(ArtifactSourceError, match="keeps its refs under refs/heads/reef/ and refs/tags/reef/"):
        GitLFSRepositoryBackend("math", remote, work_dir=tmp_path / "reef-work", cache_dir=tmp_path / "reef-cache")


def test_a_token_authorizes_its_https_host_only_and_stays_out_of_commands(tmp_path: Path) -> None:
    token = "hf_secret_token"
    config = token_config("https://huggingface.co/org/artifacts", token)
    client = GitClient(tmp_path, config)
    header = run_git_with(
        client, "config", "--get-urlmatch", "http.extraHeader", "https://huggingface.co/org/artifacts"
    )
    assert header == f"Authorization: Basic {base64.b64encode(f'reef:{token}'.encode()).decode()}"
    assert run_git_with(client, "config", "--get-urlmatch", "http.extraHeader", "https://example.com/other") == ""
    with pytest.raises(ArtifactSourceError) as refused:
        client.run(("git", "config", "--get", "no.such.key"), source_error=True)
    assert token not in str(refused.value)
    with pytest.raises(ValueError, match="https:// repository"):
        token_config("git@huggingface.co:org/artifacts", token)


def run_git_with(client: GitClient, *args: str) -> str:
    """``client``'s output of ``git args``, or empty when git finds nothing."""
    try:
        return client.run(("git", *args), source_error=True)
    except ArtifactSourceError:
        return ""


@pytest.mark.integration
def test_configured_factory_reads_the_ref_layout_and_the_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_git_lfs: None
) -> None:
    context = RepositoryBackendContext(work_dir=tmp_path / "work", cache_dir=tmp_path / "cache")
    config = {
        "repository": "https://huggingface.co/org/artifacts",
        "refs": "branches-and-tags",
        "token_env": "HF_TOKEN",
    }
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ArtifactSourceError, match="needs the variable HF_TOKEN"):
        GitLFSRepositoryBackendFactory.from_config(config, context=context)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_token")
    factory = GitLFSRepositoryBackendFactory.from_config(config, context=context)
    assert factory._refs == HUB_REFS
    assert factory._git_config == token_config(config["repository"], "hf_secret_token")
    with pytest.raises(PluginConfigError, match="refs must be one of"):
        GitLFSRepositoryBackendFactory.from_config({"repository": "a", "refs": "custom"}, context=context)


def files_of(directory: Path) -> dict[str, str]:
    """The text files under ``directory`` by relative path, without the files that a backend adds to a snapshot."""
    return {
        str(path.relative_to(directory)): path.read_text()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in (".gitattributes", "reef-artifact.json")
    }


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["memory", "git"])
def test_a_backend_publishes_a_snapshot_of_stored_content_and_bound_artifacts_by_name(
    kind: str, tmp_path: Path, fake_git_lfs: None
) -> None:
    def backend(name: str) -> RepositoryBackend:
        if kind == "memory":
            return memory
        return GitLFSRepositoryBackend(
            "agent", tmp_path / "artifacts.git", work_dir=tmp_path / name / "work", cache_dir=tmp_path / name / "cache"
        )

    memory = InMemoryRepositoryBackend("agent", directory=tmp_path / "memory")
    # One process stores, another publishes by name, and a third reads.
    training, control, reader = backend("training"), backend("control"), backend("reader")
    manifest = {
        "weights": {"metadata": {"rank": 8}, "artifact_id": "w0"},
        "harness": {"metadata": {}, "artifact_id": "h0"},
    }
    # The root keeps no file of the weights component.
    root = control.register(
        root_tree(tmp_path / "root", {"harness/AGENTS.md": "seed"}), metadata={"components": manifest}
    )

    checkpoint = root_tree(tmp_path / "one", {"AGENTS.md": "first", "docs/notes.md": "n"}).local_path
    assert checkpoint is not None
    training.store_content("job:0", checkpoint)
    # Storing a name again replaces its files; a symlink is stored as the file that it names.
    (checkpoint / "docs" / "notes.md").unlink()
    (checkpoint / "AGENTS.md").write_text("evolved")
    (tmp_path / "linked.md").write_text("linked")
    (checkpoint / "link.md").symlink_to(tmp_path / "linked.md")
    training.store_content("job:0", checkpoint)
    shutil.rmtree(checkpoint)
    evolved = {"AGENTS.md": "evolved", "link.md": "linked"}
    assert files_of(reader.materialize_content("job:0")) == evolved

    # A snapshot of several components: stored content for one, and the bound artifact of the other.
    files = {"weights": StoredFiles(artifact_id="w0"), "harness": StoredFiles(content="job:0")}
    metadata = {"components": {**manifest, "harness": {"metadata": {"step": 1}, "artifact_id": "h1"}}}
    with pytest.raises(ArtifactConflictError):
        control.publish_stored(files, metadata=metadata, expected_parent=SnapshotRef("other", None))
    pending = control.publish_stored(files, metadata=metadata, expected_parent=root, advance_head=False)
    assert pending.parent_revision == root.revision and control.current() == root
    assert control.snapshot_metadata(pending)["components"] == metadata["components"]
    assert reader.artifact_snapshot("h1") == pending
    assert files_of(reader.materialize(pending).local_path) == {
        f"harness/{path}": text for path, text in evolved.items()
    }
    control.commit_snapshot(pending, expected_parent=root)
    assert reader.current() == pending

    # A flat snapshot of stored content, and of an artifact that a snapshot of several components binds.
    flat = control.publish_stored(
        StoredFiles(content="job:0"), metadata={"artifact_id": "flat-1"}, expected_parent=pending
    )
    assert control.current() == flat and files_of(reader.materialize(flat).local_path) == evolved
    # An artifact ID never names other files or metadata.
    with pytest.raises(ArtifactConflictError):
        control.publish_stored(StoredFiles(artifact_id="h1"), metadata={"artifact_id": "h1"}, expected_parent=flat)
    again = control.publish_stored(
        StoredFiles(artifact_id="h1"), metadata={"artifact_id": "flat-2"}, expected_parent=flat
    )
    assert files_of(reader.materialize(again).local_path) == evolved

    # Discarded content is gone, a second discard changes nothing, and every snapshot keeps its files.
    control.discard_content("job:0")
    training.discard_content("job:0")
    for missing in (StoredFiles(content="job:0"), StoredFiles(artifact_id="unknown")):
        with pytest.raises(ArtifactNotFoundError):
            control.publish_stored(missing, metadata={"artifact_id": "x"}, expected_parent=again)
    with pytest.raises(ArtifactNotFoundError):
        reader.materialize_content("job:0")
    assert control.current() == again
    assert files_of(backend("late").materialize(flat).local_path) == evolved


@pytest.mark.unit
def test_stored_files_name_content_or_an_artifact() -> None:
    assert StoredFiles(content="job:0") != StoredFiles(artifact_id="job:0")
    for values in ({}, {"content": ""}, {"content": "a", "artifact_id": "b"}, {"artifact_id": 3}):
        with pytest.raises(ValueError, match="exactly one of content and artifact_id"):
            StoredFiles(**values)
    with pytest.raises(ValueError, match="no local_path"):
        Artifact(SnapshotRef("local:x", None), None, local_path=Path("."), content="job:0")
    stored = Artifact.stored("job:0")
    assert (stored.content, stored.local_path) == ("job:0", None)
    assert stored.artifact_id != Artifact.stored("job:0").artifact_id


def copies_of(root: Path, byte_count: int) -> int:
    """How many files of ``byte_count`` bytes under ``root`` hold their own bytes: hard links count once."""
    return len(
        {path.stat().st_ino for path in root.rglob("*") if path.is_file() and path.stat().st_size == byte_count}
    )


@pytest.mark.integration
def test_a_checkpoint_on_one_host_is_stored_and_served_with_one_copy_of_its_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a local repository, the trainer's export is copied once, into the LFS storage. The repository and the
    materialized snapshot of every service on the host link that one object."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    byte_count = 4 * 1024 * 1024
    weights = os.urandom(byte_count)

    def backend(name: str) -> GitLFSRepositoryBackend:
        return GitLFSRepositoryBackend(
            "agent", tmp_path / "artifacts.git", work_dir=tmp_path / name / "work", cache_dir=tmp_path / name / "cache"
        )

    training, control, inference = backend("training"), backend("control"), backend("inference")
    root = control.register(root_tree(tmp_path / "root", {"config.json": "{}"}), metadata={})
    export = tmp_path / "export"
    export.mkdir()
    (export / "model.safetensors").write_bytes(weights)
    training.store_content("job:1", export)
    assert copies_of(tmp_path, byte_count) == 2
    published = control.publish_stored(
        StoredFiles(content="job:1"), metadata={"artifact_id": "a1"}, expected_parent=root
    )
    control.discard_content("job:1")
    # Two services read the snapshot: each cache entry is a link to the stored object, not a copy.
    for reader in (inference, training):
        served = reader.materialize(published).local_path
        assert served is not None and (served / "model.safetensors").read_bytes() == weights
    assert copies_of(tmp_path, byte_count) == 2
    stored = next((tmp_path / "artifacts.git" / "lfs" / "objects").rglob("*/*/*"))
    assert same_file(stored, served / "model.safetensors")
    # The control process still holds none of the bytes.
    assert copies_of(tmp_path / "control", byte_count) == 0


@pytest.mark.integration
def test_content_whose_upload_failed_is_not_stored_so_no_snapshot_names_its_missing_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``publish_stored`` pushes without the LFS upload, so the objects must be on the remote already. The push of
    ``store_content`` uploads them before it moves the content's ref: when the upload fails, the ref does not exist,
    and a snapshot of that content cannot be published."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    remote = tmp_path / "artifacts.git"
    training = GitLFSRepositoryBackend("agent", remote, work_dir=tmp_path / "t-work", cache_dir=tmp_path / "t-cache")
    control = GitLFSRepositoryBackend("agent", remote, work_dir=tmp_path / "c-work", cache_dir=tmp_path / "c-cache")
    root = control.register(root_tree(tmp_path / "root", {"config.json": "{}"}), metadata={})
    export = tmp_path / "export"
    export.mkdir()
    (export / "model.safetensors").write_bytes(os.urandom(4096))
    # The remote cannot take an LFS object.
    shutil.rmtree(remote / "lfs", ignore_errors=True)
    (remote / "lfs").write_text("not a directory")
    with pytest.raises(ArtifactPublicationError):
        training.store_content("job:1", export)
    assert "contents" not in run_git("--git-dir", str(remote), "show-ref")
    with pytest.raises(ArtifactNotFoundError, match="no content 'job:1'"):
        control.publish_stored(StoredFiles(content="job:1"), metadata={"artifact_id": "a1"}, expected_parent=root)
    assert control.current() == root

    # Once the upload works, the same call stores the content, and the snapshot's object is on the remote.
    (remote / "lfs").unlink()
    training.store_content("job:1", export)
    published = control.publish_stored(
        StoredFiles(content="job:1"), metadata={"artifact_id": "a1"}, expected_parent=root
    )
    served = control.materialize(published).local_path
    assert served is not None
    assert (served / "model.safetensors").read_bytes() == (export / "model.safetensors").read_bytes()


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["memory", "git"])
def test_stored_content_and_a_carried_component_keep_every_file_also_one_named_as_the_backends_own(
    kind: str, tmp_path: Path, fake_git_lfs: None
) -> None:
    """A checkpoint or a component can hold its own ``.gitattributes`` or ``reef-artifact.json``. Content holds
    exactly the directory's files, and a snapshot carries a component byte for byte, so its artifact ID keeps
    naming the same files and the next commit of the scenario binds it again."""
    memory = InMemoryRepositoryBackend("agent", directory=tmp_path / "memory")

    def backend(name: str) -> RepositoryBackend:
        if kind == "memory":
            return memory
        return GitLFSRepositoryBackend(
            "agent", tmp_path / "artifacts.git", work_dir=tmp_path / name / "work", cache_dir=tmp_path / name / "cache"
        )

    def every_file(directory: Path) -> dict[str, str]:
        return {str(path.relative_to(directory)): path.read_text() for path in directory.rglob("*") if path.is_file()}

    training, control, reader = backend("training"), backend("control"), backend("reader")
    own = {
        ".gitattributes": "*.md text\n",
        "reef-artifact.json": "{}",
        "AGENTS.md": "seed",
        "sub/.gitattributes": "x\n",
    }
    manifest = {"weights": {"metadata": {}, "artifact_id": "w0"}, "harness": {"metadata": {}, "artifact_id": "h0"}}
    root = control.register(
        root_tree(
            tmp_path / "root", {**{f"harness/{path}": text for path, text in own.items()}, "weights/w.txt": "0"}
        ),
        metadata={"components": manifest},
    )
    checkpoint = root_tree(tmp_path / "checkpoint", {**own, "AGENTS.md": "evolved"}).local_path
    assert checkpoint is not None
    training.store_content("job:1", checkpoint)
    assert every_file(reader.materialize_content("job:1")) == {**own, "AGENTS.md": "evolved"}
    empty = tmp_path / "empty"
    empty.mkdir()
    training.store_content("job:2", empty)
    assert every_file(reader.materialize_content("job:2")) == {}

    # The weights change, and the harness h0 is carried with its own files of those names.
    weights = {**manifest, "weights": {"metadata": {}, "artifact_id": "w1"}}
    first = control.publish_stored(
        {"weights": StoredFiles(content="job:2"), "harness": StoredFiles(artifact_id="h0")},
        metadata={"components": weights},
        expected_parent=root,
    )
    harness = {**weights, "harness": {"metadata": {}, "artifact_id": "h1"}}
    second = control.publish_stored(
        {"weights": StoredFiles(artifact_id="w1"), "harness": StoredFiles(content="job:1")},
        metadata={"components": harness},
        expected_parent=first,
    )
    for snapshot, text in ((first, "seed"), (second, "evolved")):
        local = reader.materialize(snapshot).local_path
        assert local is not None
        assert every_file(local / "harness") == {**own, "AGENTS.md": text}
        assert not (local / "weights").exists() or every_file(local / "weights") == {}


@pytest.mark.integration
def test_a_materialized_weight_file_cannot_be_written_into_the_repositorys_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A materialized LFS file of a local repository is a link to the repository's own object. It is read-only,
    so a write through the cache is refused and every later reader gets the stored bytes. An object of another
    size than its pointer names is not linked: ``git lfs pull`` checks it."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    remote = tmp_path / "artifacts.git"

    def backend(name: str) -> GitLFSRepositoryBackend:
        return GitLFSRepositoryBackend("agent", remote, work_dir=tmp_path / name / "work", cache_dir=tmp_path / name)

    training, reader = backend("training"), backend("reader")
    root = training.register(root_tree(tmp_path / "root", {"config.json": "{}"}), metadata={})
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"GOOD" * 1000)
    training.store_content("job:1", checkpoint)
    published = training.publish_stored(
        StoredFiles(content="job:1"), metadata={"artifact_id": "a1"}, expected_parent=root
    )
    served = reader.materialize(published).local_path
    assert served is not None
    stored = next(path for path in (remote / "lfs" / "objects").rglob("*") if path.is_file())
    assert same_file(stored, served / "model.safetensors")
    assert stat.S_IMODE(stored.stat().st_mode) == 0o444
    with pytest.raises(PermissionError):
        (served / "model.safetensors").open("r+b")
    assert (backend("late").materialize(published).local_path / "model.safetensors").read_bytes() == b"GOOD" * 1000

    # A damaged object is not linked, and the pull refuses it: no reader gets other bytes than the stored ones.
    stored.chmod(0o644)
    stored.write_bytes(b"short")
    with pytest.raises(ArtifactMaterializationError):
        backend("after-damage").materialize(published)


def lfs_files(directory: Path) -> list[str]:
    """The files under ``directory``, as paths under it, without Git's own."""
    return sorted(
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )


@pytest.mark.integration
@pytest.mark.parametrize("composite", [False, True])
def test_a_reader_of_named_files_fetches_those_files_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, composite: bool
) -> None:
    """``materialize_component_files`` of a few named files (Tinker's ``start_files``) checks out only those
    paths of the component: a named file that the component lacks is missing, a file of the same name in a
    subdirectory is not read, and no weight object reaches the reader's directories."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    weights = b"weight-bytes-that-the-reader-must-not-fetch:" + bytes(range(256)) * 32
    files = {"base.json": b'{"inputs": {}}', "model.safetensors": weights, "sub/tinker-checkpoint.json": b"nested"}
    component = tmp_path / "weights"
    for relative, data in files.items():
        (component / relative).parent.mkdir(parents=True, exist_ok=True)
        (component / relative).write_bytes(data)
    writer = GitLFSRepositoryBackend(
        "s", tmp_path / "artifacts.git", work_dir=tmp_path / "writer" / "work", cache_dir=tmp_path / "writer" / "cache"
    )
    if composite:
        harness = root_tree(tmp_path / "harness", {"base.json": "harness file"})
        snapshot = compose_snapshot(
            {"weights": Artifact.local(component), "harness": harness}, directory=tmp_path / "composed"
        )
        root = writer.register(snapshot, metadata=dict(snapshot.storage_metadata))
        name: str | None = "weights"
    else:
        root = writer.register(Artifact.local(component), metadata={})
        name = None
    reader_dir = tmp_path / "reader"
    reader = GitLFSRepositoryBackend(
        "s", tmp_path / "artifacts.git", work_dir=reader_dir / "work", cache_dir=reader_dir / "cache"
    )

    read = reader.materialize_component_files(root, name, ("base.json", "tinker-checkpoint.json"))

    assert read.local_path is not None
    assert lfs_files(read.local_path) == ["base.json"]
    assert (read.local_path / "base.json").read_bytes() == files["base.json"]
    assert reader.materialize_component_files(root, name, ()).local_path is not None
    for path in reader_dir.rglob("*"):
        assert not path.is_file() or weights[:64] not in path.read_bytes(), path
