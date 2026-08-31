from __future__ import annotations

import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from reef.artifact import Artifact, ArtifactConflict, ArtifactSourceError, GitLFSRepositoryBackend


def run_git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.mark.integration
def test_git_lfs_repository_initializes_default_local_repository(
    tmp_path: Path,
    fake_git_lfs: None,
) -> None:
    remote = tmp_path / "artifacts.git"

    backend = GitLFSRepositoryBackend(
        "math",
        remote,
        work_dir=tmp_path / "work",
        cache_dir=tmp_path / "cache",
    )

    initial = backend.resolve_version()
    assert remote.is_dir()
    assert run_git("--git-dir", str(remote), "rev-parse", "refs/reef/initial") == initial.version
    assert run_git("-C", str(tmp_path / "work" / "repository"), "config", "--local", "core.hooksPath") == str(
        tmp_path / "work" / "repository" / ".git" / "hooks"
    )
    assert not hasattr(backend.fork(), "scenario")


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

    initial = backend.resolve_version()
    assert run_git("--git-dir", str(remote), "rev-parse", "refs/reef/initial") == initial.version
    assert run_git("--git-dir", str(remote), "rev-parse", "refs/reef/latest") == initial.version


@pytest.mark.integration
def test_local_repository_bootstrap_repairs_missing_latest(
    tmp_path: Path,
    fake_git_lfs: None,
) -> None:
    remote = tmp_path / "artifacts.git"
    first = GitLFSRepositoryBackend(
        "math",
        remote,
        work_dir=tmp_path / "first-work",
        cache_dir=tmp_path / "first-cache",
    )
    initial = first.resolve_version()
    run_git("--git-dir", str(remote), "update-ref", "-d", "refs/reef/latest")

    recovered = GitLFSRepositoryBackend(
        "math",
        remote,
        work_dir=tmp_path / "recovered-work",
        cache_dir=tmp_path / "recovered-cache",
    )

    assert recovered.resolve_version() == initial
    assert run_git("--git-dir", str(remote), "rev-parse", "refs/reef/latest") == initial.version


@pytest.mark.integration
def test_local_repository_bootstrap_is_idempotent_across_constructors(
    tmp_path: Path,
    fake_git_lfs: None,
) -> None:
    remote = tmp_path / "artifacts.git"

    def build(index: int):
        backend = GitLFSRepositoryBackend(
            f"scenario-{index}",
            remote,
            work_dir=tmp_path / f"work-{index}",
            cache_dir=tmp_path / f"cache-{index}",
        )
        return backend.resolve_version()

    with ThreadPoolExecutor(max_workers=8) as executor:
        versions = tuple(executor.map(build, range(8)))

    assert len({version.version for version in versions}) == 1
    assert run_git("--git-dir", str(remote), "rev-parse", "refs/reef/initial") == versions[0].version
    assert run_git("--git-dir", str(remote), "rev-parse", "refs/reef/latest") == versions[0].version


@pytest.mark.integration
def test_git_lfs_repository_imports_forks_publishes_and_materializes(
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

    backend_factory = GitLFSRepositoryBackend.factory(
        remote,
        "org/model@main",
        work_dir=tmp_path / "work",
        cache_dir=tmp_path / "cache",
        snapshot_download=lambda **kwargs: str(snapshot),
    )

    backend = backend_factory("math")
    code_backend = backend_factory("code")
    initial = backend.resolve_version()
    assert run_git("--git-dir", str(remote), "rev-parse", "refs/reef/initial") == initial.version
    assert run_git("--git-dir", str(remote), "rev-parse", "refs/reef/latest") == initial.version

    math = backend.fork(metadata={"scenario_snapshot": {"recipe": "openclawrl"}})
    code = code_backend.fork()
    manifest = json.loads(run_git("--git-dir", str(remote), "show", f"{math.version}:reef-artifact.json"))
    assert math.version != code.version
    assert "scenario" not in manifest
    assert backend.fork() == math
    assert backend.metadata() == {"scenario_snapshot": {"recipe": "openclawrl"}}
    assert run_git("--git-dir", str(remote), "rev-parse", backend.ref_name) == math.version

    fresh_factory = GitLFSRepositoryBackend.factory(
        remote,
        work_dir=tmp_path / "fresh-work",
        cache_dir=tmp_path / "fresh-cache",
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

    assert published.parent_version == math.version
    assert backend.current() == published
    assert code_backend.current() == code
    assert backend.materialize(published).local_path.joinpath("adapter.safetensors").read_text() == "trained"
    fresh = fresh_factory("math")
    with ThreadPoolExecutor(max_workers=8) as executor:
        copies = tuple(executor.map(fresh.materialize, (published,) * 8))
    assert all(copy.local_path.joinpath("adapter.safetensors").read_text() == "trained" for copy in copies)

    with pytest.raises(ArtifactConflict):
        backend.publish(Artifact.local(candidate), expected_parent=math)


@pytest.mark.integration
def test_fresh_scenario_forks_latest_artifact_with_real_lfs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    remote = tmp_path / "artifacts.git"
    first = GitLFSRepositoryBackend(
        "scenario-a",
        remote,
        work_dir=tmp_path / "work-a",
        cache_dir=tmp_path / "cache-a",
    )
    initial = first.fork()
    candidate = tmp_path / "trained"
    candidate.mkdir()
    weights = candidate / "model-00001-of-00001.safetensors"
    weights.write_bytes(b"trained weights")
    trained = first.publish(Artifact.local(candidate), expected_parent=initial)

    second = GitLFSRepositoryBackend(
        "scenario-b",
        remote,
        work_dir=tmp_path / "work-b",
        cache_dir=tmp_path / "cache-b",
    )
    forked = second.fork(metadata={"source_scenario": "scenario-a"})

    assert forked.parent_version == trained.version
    assert second.metadata() == {"source_scenario": "scenario-a"}
    assert run_git("--git-dir", str(remote), "rev-parse", second.ref_name) == forked.version
    inherited_weights = tmp_path / "work-b" / "repository" / weights.name
    assert inherited_weights.read_text().startswith("version https://git-lfs.github.com/spec/v1")

    fresh = GitLFSRepositoryBackend(
        "scenario-b",
        remote,
        work_dir=tmp_path / "fresh-work",
        cache_dir=tmp_path / "fresh-cache",
    )
    materialized = fresh.materialize(forked)
    assert materialized.local_path is not None
    assert materialized.local_path.joinpath(weights.name).read_bytes() == b"trained weights"
