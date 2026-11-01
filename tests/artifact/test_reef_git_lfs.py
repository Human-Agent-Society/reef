from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from reef.artifact import (
    Artifact,
    ArtifactConflict,
    ArtifactSourceError,
    DefaultStart,
    GitLFSRepositoryBackend,
    GitLFSRepositoryBackendFactory,
    ModelDirectory,
    RepositoryBackendContext,
)
from reef.artifact.git_client import GitClient
from reef.artifact.git_lfs import HUB_REFS, same_file, token_config
from reef.core.plugins import PluginConfigError


def root_tree(directory: Path, files: dict[str, str] | None = None, *, content_id: str = "content:root") -> Artifact:
    """An unpublished root release in ``directory`` that holds ``files``."""
    directory.mkdir(parents=True)
    for relative, text in (files or {}).items():
        (directory / relative).parent.mkdir(parents=True, exist_ok=True)
        (directory / relative).write_text(text)
    return Artifact.local(directory, content_id=content_id)


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

    assert published.parent_release_id == parent.release_id
    assert backend.current() == published
    materialized = backend.materialize(published).local_path
    assert materialized is not None
    assert (materialized / "rules.md").read_text() == "new rules\n"
    assert not (tmp_path / "work" / "publish-index").exists()


@pytest.mark.integration
def test_each_scenario_registers_its_own_root_release_with_no_parent(tmp_path: Path, fake_git_lfs: None) -> None:
    """A registration pushes a commit with no parent to the scenario's ref; a second registration of the scenario
    stores nothing and gets the winner's head; another scenario gets a distinct root. No shared ref exists."""
    remote = tmp_path / "artifacts.git"
    factory = GitLFSRepositoryBackend.factory(remote, work_dir=tmp_path / "work", cache_dir=tmp_path / "cache")
    first = factory("first")
    assert first.metadata() is None
    files = {"pi-agent/AGENTS.md": "seed rules\n", "pi-agent/skills/a/SKILL.md": "# a\n"}
    root = first.register(root_tree(tmp_path / "first-root", files, content_id="content:one"), metadata={"m": 1})
    assert root.parent_release_id is None and root.content_id == "content:one"
    assert run_git("--git-dir", str(remote), "rev-list", "--parents", "-n", "1", root.release_id) == root.release_id
    assert first.current() == root and first.metadata() == {"m": 1}
    tree = first.materialize(root).local_path
    assert tree is not None
    assert (tree / "pi-agent" / "AGENTS.md").read_text(encoding="utf-8") == "seed rules\n"
    assert (tree / "pi-agent" / "skills" / "a" / "SKILL.md").read_text(encoding="utf-8") == "# a\n"
    assert json.loads((tree / "reef-artifact.json").read_text(encoding="utf-8"))["source"] == {"kind": "root"}

    late = factory("first").register(root_tree(tmp_path / "late-root", {"x": "late"}), metadata={"m": 2})
    assert late == root and first.metadata() == {"m": 1}
    other = factory("second").register(root_tree(tmp_path / "second-root", files), metadata={})
    assert other.release_id != root.release_id and other.parent_release_id is None
    assert "refs/reef/base" not in run_git("--git-dir", str(remote), "for-each-ref", "--format=%(refname)")
    assert first.resolve_release(root.release_id) == root


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
    assert run_git("--git-dir", str(remote), "rev-parse", backend.ref_name) == root.release_id


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

    assert len({head.release_id for head in heads}) == 1
    encoded = base64.urlsafe_b64encode(b"scenario").decode().rstrip("=")
    assert run_git("--git-dir", str(remote), "rev-parse", f"refs/reef/scenarios/{encoded}") == heads[0].release_id


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
    start = DefaultStart(
        model_files=ModelDirectory(snapshot), inputs={"model": "org/model", "version": "upstream-sha"}
    )

    def default_root(name: str) -> Artifact:
        (tmp_path / name).mkdir()
        start.write(tmp_path / name)
        return Artifact.local(tmp_path / name, content_id=start.content_id)

    backend = backend_factory("math")
    code_backend = backend_factory("code")
    math = backend.register(default_root("math-root"), metadata={"scenario_commit_record": {"recipe": "openclawrl"}})
    code = code_backend.register(default_root("code-root"), metadata={})
    manifest = json.loads(run_git("--git-dir", str(remote), "show", f"{math.release_id}:reef-artifact.json"))
    assert math.release_id != code.release_id and math.content_id == code.content_id == start.content_id
    assert "scenario" not in manifest
    assert backend.metadata() == {"scenario_commit_record": {"recipe": "openclawrl"}}
    assert run_git("--git-dir", str(remote), "rev-parse", backend.ref_name) == math.release_id

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

    assert published.parent_release_id == math.release_id
    assert backend.current() == published
    assert code_backend.current() == code
    assert backend.materialize(published).local_path.joinpath("adapter.safetensors").read_text() == "trained"
    fresh = fresh_factory("math")
    with ThreadPoolExecutor(max_workers=8) as executor:
        copies = tuple(executor.map(fresh.materialize, (published,) * 8))
    assert all(copy.local_path.joinpath("adapter.safetensors").read_text() == "trained" for copy in copies)

    with pytest.raises(ArtifactConflict):
        backend.publish(Artifact.local(candidate), expected_parent=math)


def trained_weights(directory: Path, contents: bytes) -> Artifact:
    (directory / "weights").mkdir(parents=True)
    (directory / "weights" / "adapter_model.safetensors").write_bytes(contents)
    return Artifact.local(directory)


@pytest.mark.integration
def test_a_fresh_workspace_publishes_with_real_lfs_over_objects_it_never_fetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restarted service's fresh workspace fetches no LFS objects, so neither a pending release nor a publish from
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
    assert trained.parent_release_id == root.release_id
    assert run_git("--git-dir", str(remote), "rev-parse", first.ref_name) == trained.release_id

    # A restarted service materializes, saves a pending release and publishes from a fresh workspace.
    fresh = GitLFSRepositoryBackend("scenario-a", remote, work_dir=tmp_path / "fresh-work", cache_dir=tmp_path / "c2")
    materialized = fresh.materialize(trained).local_path
    assert materialized is not None
    assert (materialized / "weights" / "adapter_model.safetensors").read_bytes() == b"adapter v1" * 100
    assert not (materialized / ".git").exists()
    # A pending release pushes to a new ref and leaves the head alone: the LFS check still stops at history the
    # remote already holds.
    pending = fresh.publish(
        trained_weights(tmp_path / "trained-2", b"adapter v2" * 100), expected_parent=trained, advance_head=False
    )
    assert pending.parent_release_id == trained.release_id and fresh.current() == trained
    published = fresh.publish(trained_weights(tmp_path / "trained-3", b"adapter v3" * 100), expected_parent=trained)
    assert published.parent_release_id == trained.release_id and fresh.current() == published

    assert global_config.read_text() == original_config
    for name in ("pre-push", "post-checkout"):
        assert (hooks / name).read_text() == hook_contents
        assert (tmp_path / "fresh-work" / "repository" / ".git" / "hooks" / name).read_text() == hook_contents


@pytest.mark.integration
def test_archiving_a_scenario_renames_its_ref_and_drops_its_work_clone(tmp_path: Path, fake_git_lfs: None) -> None:
    """The factory's archive moves ``refs/reef/scenarios/<name>`` under ``refs/reef/archived`` and removes
    the scenario's work clone; other scenarios stay, and the name is free again."""
    remote = tmp_path / "artifacts.git"
    factory = GitLFSRepositoryBackend.factory(remote, work_dir=tmp_path / "work", cache_dir=tmp_path / "cache")
    factory("kept").register(root_tree(tmp_path / "kept-root"), metadata={})
    doomed_ref = factory("doomed").register(root_tree(tmp_path / "doomed-root"), metadata={})
    assert set(factory.list_registrations()) == {"doomed", "kept"}
    encoded = base64.urlsafe_b64encode(b"doomed").decode().rstrip("=")
    assert (tmp_path / "work" / encoded).is_dir()

    archived = factory.archive_registration("doomed")

    assert factory.list_registrations() == ("kept",)
    assert not factory.has_registration("doomed")
    assert len(archived) == 2 and archived[0].startswith("refs/reef/archived/")
    assert run_git("--git-dir", str(remote), "rev-parse", archived[0]) == doomed_ref.release_id
    assert not (tmp_path / "work" / encoded).exists()
    # The name creates fresh: a new backend registers a new root rather than continuing the archived chain.
    fresh = factory("doomed")
    assert fresh.metadata() is None
    again = fresh.register(root_tree(tmp_path / "again-root"), metadata={})
    assert again.parent_release_id is None and again.release_id != doomed_ref.release_id


@pytest.mark.unit
def test_a_default_start_writes_the_model_under_its_component_beside_its_files(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--org--model" / "snapshots" / "upstream-sha"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_text("weights")
    start = DefaultStart(
        files={"harness/AGENTS.md": "seed rules\n"},
        model_files=ModelDirectory(snapshot),
        model_component="weights",
        inputs={"weights.model": "org/model"},
    )
    tree = tmp_path / "root"
    tree.mkdir()
    start.write(tree)

    assert (tree / "weights" / "config.json").read_text() == "{}"
    assert (tree / "weights" / "model.safetensors").samefile(snapshot / "model.safetensors")
    assert (tree / "harness" / "AGENTS.md").read_text() == "seed rules\n"
    assert not (tree / "config.json").exists()
    assert start.content_id == DefaultStart(inputs={"weights.model": "org/model"}).content_id
    with pytest.raises(ValueError, match="directory name"):
        DefaultStart(model_files=ModelDirectory(snapshot), model_component="../x")


def test_a_model_directory_exists_and_a_model_component_needs_model_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        ModelDirectory(tmp_path / "missing")
    with pytest.raises(ValueError, match="requires model_files"):
        DefaultStart(model_component="weights")


@pytest.mark.integration
def test_a_published_file_is_linked_into_the_release_cache_not_copied(tmp_path: Path, fake_git_lfs: None) -> None:
    """Released files are immutable, so a publish links them into the release's cache entry instead of cloning it."""
    remote = tmp_path / "artifacts.git"
    backend = GitLFSRepositoryBackend("agent", remote, work_dir=tmp_path / "work", cache_dir=tmp_path / "cache")
    head = backend.register(root_tree(tmp_path / "root"), metadata={})
    source = tmp_path / "release"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"\x00" * 64)
    published = backend.publish(Artifact.local(source), expected_parent=head)
    cached = tmp_path / "cache" / published.release_id
    assert (cached / "weights.bin").samefile(source / "weights.bin")
    assert not (tmp_path / "work" / "repository" / "weights.bin").exists()
    materialized = backend.materialize(published)
    assert materialized.local_path == cached
    assert json.loads((cached / "reef-artifact.json").read_text())["parent_release_id"] == head.release_id


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
    """A release that carries the parent's files forward as links reuses their blobs; only the changed files are read."""
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

    def blob(release: str, path: str) -> str:
        return run_git("--git-dir", str(remote), "rev-parse", f"{release}:{path}")

    assert blob(published.release_id, "weights/model.safetensors") == blob(
        parent.release_id, "weights/model.safetensors"
    )
    assert run_git("--git-dir", str(remote), "show", f"{published.release_id}:weights/model.safetensors").startswith(
        "version https://git-lfs.github.com/spec/v1"
    )
    cached = backend.materialize(published).local_path
    assert cached is not None
    assert (cached / "weights" / "model.safetensors").samefile(parent_tree / "weights" / "model.safetensors")
    # The linked cache entry holds what a clone of the release holds.
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
    start = DefaultStart(files={"AGENTS.md": "seed\n"}, model_files=ModelDirectory(snapshot))
    (tmp_path / "root").mkdir()
    start.write(tmp_path / "root")
    head = backend.register(Artifact.local(tmp_path / "root", content_id=start.content_id), metadata={})
    # The registration rewrote .gitattributes in the work tree, not the Hugging Face blob behind the snapshot.
    assert (blobs / "attributes").read_text() == "*.bin filter=lfs\n"
    assert (blobs / "weights").read_text() == "base"
    assert "*.safetensors filter=lfs" in backend.materialize(head).local_path.joinpath(".gitattributes").read_text()

    source = tmp_path / "release"
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
    source = tmp_path / "release"
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
def test_the_hub_layout_keeps_releases_on_tags_and_moving_refs_on_branches(tmp_path: Path, fake_git_lfs: None) -> None:
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
    assert refs.keys() == {
        "refs/heads/main",
        f"refs/heads/reef/scenarios/{encoded}",
        f"refs/tags/reef/releases/{pending.release_id}",
    }
    assert refs[f"refs/heads/reef/scenarios/{encoded}"] == published.release_id
    # A release's ID is its commit, so a release on the Hub is the revision of that commit.
    assert backend.resolve_release(pending.release_id) == pending
    assert factory.has_registration("math") and factory.list_registrations() == ("math",)

    # A repository keeps one layout: the other is refused instead of hiding the releases.
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
