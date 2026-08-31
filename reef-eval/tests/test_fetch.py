"""fetch_pinned_tasks: pinned-commit fetching, tested against a local git repo."""

import subprocess
from pathlib import Path

import pytest

from reef_eval.fetch import fetch_pinned_tasks


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture()
def upstream(tmp_path):
    """A local 'published benchmark' repo: two root tasks, one nested dataset
    task, and a non-task folder that must never be copied."""
    repo = tmp_path / "upstream"
    for name in ("alpha", "beta"):
        (repo / name).mkdir(parents=True)
        (repo / name / "task.toml").write_text(f'name = "{name}"\n')
        (repo / name / "instruction.md").write_text("do the thing\n")
    (repo / "docs").mkdir()
    (repo / "docs" / "notes.md").write_text("not a task\n")
    (repo / "datasets" / "swe" / "gamma").mkdir(parents=True)
    (repo / "datasets" / "swe" / "gamma" / "task.toml").write_text('name = "gamma"\n')
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    # What GitHub allows, granted locally: fetch by SHA, with a blob filter.
    _git(repo, "config", "uploadpack.allowAnySHA1InWant", "true")
    _git(repo, "config", "uploadpack.allowFilter", "true")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "pin")
    sha = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "tag", "v9.9.9")  # releases are pinned by tag, not by sha
    return f"file://{repo}", sha


def test_fetches_a_pin_given_as_a_tag(upstream, tmp_path):
    """Committed benchmarks pin to a release tag (fetch.TASKS_REF), not a sha.
    Fetching a tag leaves no local ref behind, so naming the tag again would
    fail even though the objects are there."""
    url, _ = upstream
    dest = tmp_path / "dest"
    assert fetch_pinned_tasks(url, "v9.9.9", dest) == ["alpha", "beta"]
    assert (dest / "alpha" / "task.toml").exists()


def test_fetches_root_tasks_and_skips_non_tasks(upstream, tmp_path):
    url, sha = upstream
    dest = tmp_path / "dest"
    assert fetch_pinned_tasks(url, sha, dest) == ["alpha", "beta"]
    assert (dest / "alpha" / "instruction.md").exists()
    assert not (dest / "docs").exists()  # no task.toml
    assert not (dest / "datasets").exists()  # nested, not a root task


def test_subdir_scopes_the_scan(upstream, tmp_path):
    url, sha = upstream
    dest = tmp_path / "dest"
    assert fetch_pinned_tasks(url, sha, dest, subdir="datasets/swe") == ["gamma"]
    assert (dest / "gamma" / "task.toml").exists()


def test_only_selects_and_unknown_names_are_loud(upstream, tmp_path):
    url, sha = upstream
    dest = tmp_path / "dest"
    assert fetch_pinned_tasks(url, sha, dest, only=["beta"]) == ["beta"]
    assert not (dest / "alpha").exists()
    with pytest.raises(ValueError, match="unknown task"):
        fetch_pinned_tasks(url, sha, dest, only=["nope"])


def test_limit_keeps_the_first_n(upstream, tmp_path):
    url, sha = upstream
    assert fetch_pinned_tasks(url, sha, tmp_path / "dest", limit=1) == ["alpha"]


def test_refetch_replaces_local_edits(upstream, tmp_path):
    url, sha = upstream
    dest = tmp_path / "dest"
    fetch_pinned_tasks(url, sha, dest)
    (dest / "alpha" / "task.toml").write_text("tampered = true\n")
    fetch_pinned_tasks(url, sha, dest)
    assert (dest / "alpha" / "task.toml").read_text() == 'name = "alpha"\n'


def test_empty_scope_is_loud(upstream, tmp_path):
    url, sha = upstream
    with pytest.raises(RuntimeError, match="no task folders"):
        fetch_pinned_tasks(url, sha, tmp_path / "dest", subdir="docs")


# ---------------------------------------------------------- benchmark cache


def _benchmark_upstream(tmp_path):
    """A local stand-in for the reef-eval repo: one benchmark with one task."""
    repo = tmp_path / "reef-eval-repo"
    task = repo / "reef-eval" / "tasks" / "autoresearch" / "first-party" / "alpha"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('name = "alpha"\n')
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "uploadpack.allowAnySHA1InWant", "true")
    _git(repo, "config", "uploadpack.allowFilter", "true")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "pin")
    return repo


def test_benchmark_downloads_once_then_caches(tmp_path, monkeypatch):
    from reef_eval import fetch

    repo = _benchmark_upstream(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD").strip()
    monkeypatch.setenv("REEF_EVAL_TASKS_REPO", f"file://{repo}")
    monkeypatch.setenv("REEF_EVAL_TASKS_REF", sha)
    monkeypatch.setenv("REEF_EVAL_CACHE", str(tmp_path / "cache"))

    dest = fetch.benchmark("first-party")
    assert (dest / "alpha" / "task.toml").exists()
    assert str(tmp_path / "cache") in str(dest) and sha in str(dest)

    # Second call is served from the cache: works even without the source.
    import shutil

    shutil.rmtree(repo)
    assert fetch.benchmark("first-party") == dest


def test_benchmark_unknown_name_is_loud():
    from reef_eval import fetch

    with pytest.raises(ValueError, match="unknown benchmark"):
        fetch.benchmark("nope")


def test_resolve_targets_downloads_known_benchmarks(tmp_path, monkeypatch):
    from reef_eval import fetch
    from reef_eval.cli import resolve_targets

    bench = tmp_path / "bench"
    (bench / "alpha").mkdir(parents=True)
    (bench / "alpha" / "task.toml").write_text('name = "alpha"\n')
    monkeypatch.setattr(fetch, "benchmark", lambda name: bench)

    assert resolve_targets(["first-party"], None) == [str(bench / "alpha")]
    assert resolve_targets(["first-party/alpha"], None) == [str(bench / "alpha")]
    # Unregistered names still pass through as Harbor registry ids.
    assert resolve_targets(["org/some-task"], None) == ["org/some-task"]


# ------------------------------------------------------------- register


def test_register_makes_a_benchmark_downloadable(upstream, tmp_path, monkeypatch):
    from reef_eval import fetch

    url, sha = upstream
    monkeypatch.setenv("REEF_EVAL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setitem(fetch.REGISTRY, "my-bench", fetch.Source(url, sha))

    dest = fetch.benchmark("my-bench")
    assert (dest / "alpha" / "task.toml").exists()
    assert "my-bench" in fetch.known_benchmarks()


def test_register_resolves_in_the_cli(upstream, tmp_path, monkeypatch):
    from reef_eval import fetch
    from reef_eval.cli import resolve_targets

    url, sha = upstream
    monkeypatch.setenv("REEF_EVAL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setitem(fetch.REGISTRY, "my-bench", fetch.Source(url, sha))

    hits = resolve_targets(["my-bench/beta"], None)
    assert len(hits) == 1 and hits[0].endswith("my-bench/beta")


def test_register_overrides_a_builtin():
    """The registry wins over the built-in table, so a fork can take a name."""
    from reef_eval import fetch

    fetch.register("cl-bench", "https://example.com/fork.git", "abc123")
    try:
        assert fetch.REGISTRY["cl-bench"] == fetch.Source(
            "https://example.com/fork.git", "abc123"
        )
    finally:
        del fetch.REGISTRY["cl-bench"]
