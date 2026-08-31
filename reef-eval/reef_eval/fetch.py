"""Fetch stock Harbor tasks from a benchmark's pinned git commit.

Stream benchmarks publish their tasks as ordinary Harbor task directories
in a git repository, pinned to one commit. Fetching is a shallow fetch of
that commit plus copying the wanted task folders out, with no conversion.
Blobs are fetched lazily, so pulling a few tasks from a large dataset
repository downloads roughly those tasks and nothing else.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


def fetch_pinned_tasks(
    git_url: str,
    commit: str,
    dest: Path | str,
    *,
    subdir: str = "",
    only: list[str] | None = None,
    limit: int | None = None,
) -> list[str]:
    """Copy the task folders published at ``git_url@commit`` into *dest*.

    A task folder is any directory directly under *subdir* (the repo root
    by default) containing a ``task.toml``. ``only`` restricts the fetch
    to the named tasks and raises on unknown names; ``limit`` keeps the
    first N alphabetically. Existing copies in *dest* are replaced.
    Returns the copied names, sorted.
    """
    dest = Path(dest)
    prefix = f"{subdir.rstrip('/')}/" if subdir else ""
    with tempfile.TemporaryDirectory(prefix="reef-eval-fetch-") as tmp_str:
        tmp = Path(tmp_str)
        _git(tmp, "init", "-q")
        _git(tmp, "remote", "add", "origin", git_url)
        print(f"fetching {git_url} @ {commit[:12]} ...", flush=True)
        _git(tmp, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit)
        # Fetching a tag or branch name leaves no local ref behind, so name
        # what came back rather than the pin: only FETCH_HEAD resolves for
        # every kind of ref, a commit sha included.
        pinned = _git(tmp, "rev-parse", "FETCH_HEAD").strip()

        found: set[str] = set()
        for line in _git(tmp, "ls-tree", "-r", "--name-only", pinned).splitlines():
            if not line.startswith(prefix):
                continue
            rest = line[len(prefix) :]
            if rest.count("/") == 1 and rest.endswith("/task.toml"):
                found.add(rest.split("/", 1)[0])
        names = sorted(found)
        if not names:
            raise RuntimeError(
                f"no task folders under '{subdir or '.'}' at {git_url}@{commit[:12]}"
            )
        if only:
            unknown = sorted(set(only) - set(names))
            if unknown:
                raise ValueError(
                    f"unknown task(s) {unknown}; the pin has {len(names)} tasks: "
                    f"{', '.join(names[:8])}, ..."
                )
            names = [n for n in names if n in set(only)]
        if limit is not None:
            names = names[:limit]

        print(f"copying {len(names)} task folder(s) ...", flush=True)
        _git(tmp, "checkout", "-q", pinned, "--", *(prefix + n for n in names))
        dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            src = tmp / subdir / name if subdir else tmp / name
            target = dest / name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)
    return names


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    if proc.returncode:
        raise RuntimeError(f"git {args[0]} failed: {proc.stderr.strip()[-500:]}")
    return proc.stdout


# ---- benchmark downloads ----
#
# The package ships code only; benchmark tasks download on first use, the
# way dataset libraries do. Committed benchmarks come from this repo at a
# release ref; SWE-bench Verified comes from its upstream source because
# its dataset repo has no license to redistribute under.

TASKS_REPO = "https://github.com/Human-Agent-Society/reef.git"
# REEF Eval now lives in the Reef monorepo. Use main until Reef adopts a
# dedicated REEF Eval release-tag convention, then pin this to that tag.
TASKS_REF = "main"

BENCHMARKS = {
    "first-party": "reef-eval/tasks/autoresearch/first-party",
    "edgebench": "reef-eval/tasks/autoresearch/edgebench",
    "frontier-cs": "reef-eval/tasks/autoresearch/frontier-cs",
    "terminal-bench": "reef-eval/tasks/continual-learning/terminal-bench",
    "cl-bench": "reef-eval/tasks/continual-learning/cl-bench",
}


@dataclass(frozen=True)
class Source:
    """Where a benchmark's task folders live: a repo, a pinned ref, a subdir."""

    repo: str
    ref: str
    subdir: str = ""


REGISTRY: dict[str, Source] = {
    "swebench-verified": Source(
        "https://github.com/laude-institute/harbor-datasets.git",
        "86723674f04e4209ac479d0fb75d9d9f44b4377e",
        "datasets/swebench-verified",
    ),
}


def register(name: str, repo: str, ref: str, *, subdir: str = "") -> None:
    """Make ``benchmark(name)`` resolve a benchmark hosted in any git repo.

    A benchmark is a directory whose immediate children are Harbor task
    folders; *subdir* points at it inside the repo (the root by default).
    Pin *ref* to a commit or tag so the tasks are reproducible. The
    registry is per process, so ship the ``register`` call in your
    package's import, the way gym environments register. Registering an
    existing name replaces it, which is how a fork takes over a built-in.
    """
    REGISTRY[name] = Source(repo, ref, subdir)


def known_benchmarks() -> list[str]:
    """Every name ``benchmark`` accepts right now, sorted."""
    return sorted({*BENCHMARKS, *REGISTRY})


def cache_home() -> Path:
    if "REEF_EVAL_CACHE" in os.environ:
        return Path(os.environ["REEF_EVAL_CACHE"])
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "reef-eval"


def benchmark(name: str, *, limit: int | None = None) -> Path:
    """Return a local directory with the benchmark's tasks, downloading on
    first use.

    Tasks are cached under ``~/.cache/reef-eval`` (override with ``REEF_EVAL_CACHE``)
    per release ref, so a given reef-eval version always sees the same tasks. A
    repo checkout does not need this: its ``tasks/`` folder already has
    everything.
    """
    if name in REGISTRY:
        source = REGISTRY[name]
        repo, ref, subdir = source.repo, source.ref, source.subdir
    elif name in BENCHMARKS:
        repo = os.environ.get("REEF_EVAL_TASKS_REPO", TASKS_REPO)
        ref = os.environ.get("REEF_EVAL_TASKS_REF", TASKS_REF)
        subdir = BENCHMARKS[name]
    else:
        raise ValueError(
            f"unknown benchmark {name!r}; known: {', '.join(known_benchmarks())}"
        )

    dest = cache_home() / "tasks" / ref / name
    if dest.is_dir() and any(dest.iterdir()) and limit is None:
        return dest
    fetch_pinned_tasks(repo, ref, dest, subdir=subdir, limit=limit)
    return dest
