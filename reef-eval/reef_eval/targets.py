"""Turning a target into the list of tasks it names.

A target is a task directory, a folder of tasks (a benchmark or a whole
regime), a known benchmark name that downloads on first use, or a Harbor
registry id (``org/name``) that passes through untouched. `reef-eval run`,
`reef-eval stream`, and :func:`tasks` all accept the same ones and expand them
the same way, so a benchmark names the same tasks in the same order from
the CLI and from a script.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_tasks_root(tasks_dir: str | Path | None = None) -> Path | None:
    """Locate the tasks catalog: explicit arg → ``$REEF_EVAL_TASKS_DIR`` → ``./tasks``
    → the checkout the reef-eval package itself lives in."""
    if tasks_dir:
        return Path(tasks_dir)
    env = os.environ.get("REEF_EVAL_TASKS_DIR")
    if env:
        return Path(env)
    for base in (Path.cwd(), Path(__file__).parent.parent):
        candidate = base / "tasks"
        if candidate.is_dir():
            return candidate
    return None


def is_task_dir(path: Path) -> bool:
    return (path / "task.toml").is_file()


def tasks_under(path: Path) -> list[Path]:
    """Every task folder inside *path*, skipping ``_``- and ``.``-prefixed
    directories such as ``_template``.

    The skip test looks only at the parts below *path*: the search root
    itself may sit anywhere, including under a dot-directory (benchmarks
    download to ``~/.cache/reef-eval`` by default).
    """
    found = []
    for task_toml in path.glob("**/task.toml"):
        parts = task_toml.relative_to(path).parts
        if any(part.startswith(("_", ".")) for part in parts):
            continue
        found.append(task_toml.parent)
    return sorted(found, key=lambda p: p.as_posix())


def _expand(target: str, candidate: Path) -> list[str] | None:
    """A task dir resolves to itself; a folder expands to the tasks inside."""
    if is_task_dir(candidate):
        return [str(candidate)]
    if candidate.is_dir():
        inside = tasks_under(candidate)
        if not inside:
            raise ValueError(
                f"'{target}' is a directory but contains no task.toml. "
                "Fetch its tasks first (see its README, or `reef-eval fetch`)."
            )
        return [str(t) for t in inside]
    return None


def _fetch_known_benchmark(target: str) -> list[str] | None:
    """Download a known benchmark on first use (pip installs have no tasks/)."""
    from reef_eval import fetch

    name = target.split("/", 1)[0]
    if name not in fetch.BENCHMARKS and name not in fetch.REGISTRY:
        return None
    root = fetch.benchmark(name)
    parts = target.split("/", 1)
    candidate = root / parts[1] if len(parts) == 2 else root
    return _expand(target, candidate)


def resolve(targets: list[str], tasks_root: Path | None) -> list[str]:
    """Expand *targets* into runnable task references.

    Local paths win, then the tasks catalog (at either level), then known
    benchmarks download into the cache, and anything else passes through
    to Harbor as a registry id. Raises ``ValueError`` on a target that
    names nothing runnable.
    """
    resolved: list[str] = []
    for target in targets:
        candidates = [Path(target)]
        if tasks_root is not None:
            candidates.append(tasks_root / target)
            # Benchmarks live one level down (tasks/<regime>/<benchmark>), so
            # bare names like "edgebench" or "terminal-bench" resolve too.
            candidates.extend(sorted(tasks_root.glob(f"*/{target}")))
        for candidate in candidates:
            hit = _expand(target, candidate)
            if hit is not None:
                resolved.extend(hit)
                break
        else:
            hit = _fetch_known_benchmark(target)
            if hit is not None:
                resolved.extend(hit)
            elif "/" in target:
                resolved.append(target)  # a Harbor registry id, org/name
            else:
                # Harbor registry ids are always org/name, so a bare word that
                # matched nothing local cannot be one. Saying so here beats
                # letting Harbor fail on it with a schema error.
                from reef_eval import fetch

                raise ValueError(
                    f"'{target}' is not a task directory, a folder of tasks, or "
                    f"a known benchmark ({', '.join(fetch.known_benchmarks())}). "
                    "Harbor registry ids look like 'org/name'. "
                    "Run `reef-eval list` to see what is available here."
                )
    return resolved


def tasks(*targets: str, tasks_dir: str | Path | None = None) -> list[str]:
    """The tasks *targets* name, as a list of task references.

    Takes what the CLI takes and returns it in the CLI's order, so
    ``tasks("cl-bench")`` is the list `reef-eval stream cl-bench` runs. It is an
    ordinary list: filter it, reorder it, or repeat an entry before handing
    it to :class:`reef_eval.Stream`, which runs exactly the list it is given.

    ``tasks_dir`` points at a tasks catalog to search; by default the same
    one the CLI uses (``$REEF_EVAL_TASKS_DIR``, ``./tasks``, or the checkout).
    """
    return resolve(list(targets), find_tasks_root(tasks_dir))
