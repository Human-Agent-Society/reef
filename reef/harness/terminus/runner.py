"""Run one Terminal-Bench task for one episode, on behalf of ``run_episode``.

This is the only module that imports the evaluation stack, and nothing on the
render path imports it, so the adapter registry stays cheap and a deployment
without the ``terminus`` extra can still load the descriptor.

Reef stays the outer loop. An episode launches ``reef-terminus --task <task>``;
this module runs that one task through reef-eval, the same primitive every
recipe under ``recipes/`` uses to reach Harbor, driving Harbor's own
``terminus-2`` agent. Nothing of Reef's runs inside the agent: the tree
reaches it as Harbor's native inputs, so no evolved code is executed in this
process and the adapter is not coupled to Terminus 2 internals.

It writes one trial file under ``REEF_TERMINUS_SESSION_DIR`` holding the
verifier's rewards and the ATIF steps, which the adapter's
``terminus-atif-json`` reader turns into the episode's trajectory. Harbor's
own trial tree lands in ``REEF_TERMINUS_TRIALS_DIR``, keeping the session
directory the reader walks to files Reef wrote.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

from reef.harness.terminus.tree import TerminusTreeError, instruction_paths, load_tree, skill_roots, terminus_kwargs

#: Harbor's own Terminus 2. Reef contributes configuration, not code.
AGENT_NAME = "terminus-2"
SESSION_DIR_ENV = "REEF_TERMINUS_SESSION_DIR"
TREE_DIR_ENV = "REEF_TERMINUS_DIR"
TRIALS_DIR_ENV = "REEF_TERMINUS_TRIALS_DIR"


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise TerminusTreeError(f"{name} must name a path for the terminus runner")
    return value


def trial_slug(task: str) -> str:
    """A filename for one task's trial, safe under the session directory.

    The task reaches this runner from the episode prompt, and a promoted task
    can originate in recorded traffic, so it never becomes a path component
    unchecked: only the last segment is kept, and only if it is an ordinary
    name. Anything else would let a trial file land outside the directory the
    trajectory reader walks.
    """
    name = PurePosixPath(task.replace("\\", "/")).name
    if not name or name.startswith(".") or set(name) & set('/:*?"<>|'):
        raise TerminusTreeError(f"terminus task {task!r} does not name a trial file")
    return name


def agent_spec(root: str, tree: dict[str, str]) -> dict[str, Any]:
    """The Harbor agent this tree configures.

    Constructor arguments come from the tree's config node, skills and
    commands from its skill roots. The model is the one Reef's binding
    rendered; a tree that carries none cannot reach a model at all.
    """
    knobs = dict(terminus_kwargs(tree))
    model = knobs.pop("model_name", None)
    if not model:
        raise TerminusTreeError("the terminus tree carries no model_name; Reef's model binding sets it")
    spec: dict[str, Any] = {"name": AGENT_NAME, "model_name": str(model)}
    if knobs:
        spec["kwargs"] = knobs
    roots = skill_roots(root, tree)
    if roots:
        spec["skills"] = [str(path) for path in roots]
    return spec


def _continuation_index(path: Path) -> tuple[str, int]:
    """Order a trial's dumps the way Terminus 2 wrote them.

    ``trajectory.json`` is the first segment and each summarization pass opens
    ``trajectory.cont-N.json``. Name order gets this wrong twice: ``cont-1``
    sorts before the base file, and ``cont-10`` before ``cont-2``.
    """
    stem = path.name[len("trajectory") : -len(".json")]
    return (path.parent.as_posix(), int(stem.removeprefix(".cont-")) if stem.startswith(".cont-") else 0)


def atif_steps(trial_dir: Path) -> list[dict[str, Any]]:
    """Every ATIF step Terminus 2 dumped for one trial, in continuation order."""
    steps: list[dict[str, Any]] = []
    for path in sorted(Path(trial_dir).rglob("trajectory*.json"), key=_continuation_index):
        try:
            trajectory = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue  # a torn dump costs its steps, not the trial's reward
        if isinstance(trajectory, dict) and isinstance(trajectory.get("steps"), list):
            steps.extend(step for step in trajectory["steps"] if isinstance(step, dict))
    return steps


def trial_record(task: str, rewards: Any, trials_dir: Path, error: str = "") -> dict[str, Any]:
    """One trial as the ``terminus-atif-json`` reader expects it.

    ``error`` is recorded rather than dropped: an episode whose container never
    built scores nothing, and a bare zero would read as a candidate the agent
    failed instead of a run that never happened.
    """
    scores = dict(rewards or {})
    return {
        "task": task,
        "rewards": scores,
        "reward": next(iter(scores.values()), None),
        "failed": not scores,
        "error": error,
        "steps": atif_steps(trials_dir),
    }


def write_trial(record: dict[str, Any], sessions: Path) -> Path:
    """Write the trial where the adapter's trajectory reader will find it."""
    sessions.mkdir(parents=True, exist_ok=True)
    target = sessions / f"{trial_slug(str(record['task']))}.json"
    target.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return target


def run(task: str) -> int:
    """Run one task and return the process exit code.

    Non-zero when the verifier produced no reward: the episode then reports a
    failed run rather than a scoreless success.
    """
    # Lazy: reef-eval pulls Harbor's dependency tree, which the render path
    # and its tests must never need. Harbor requires Python 3.12, above Reef's
    # own floor, so the extra carries that marker and can be absent on a
    # supported interpreter; say so rather than raising a bare ImportError.
    try:
        from reef_eval import Lab
    except ImportError as exc:
        raise TerminusTreeError(
            "the terminus runner needs reef-eval: pip install 'reef-infra[terminus]' on Python 3.12 or newer"
        ) from exc

    trial_slug(task)  # refuse a task that cannot name its own trial file
    root = _required_env(TREE_DIR_ENV)
    tree = load_tree(root)
    sessions = Path(_required_env(SESSION_DIR_ENV))
    trials = Path(os.environ.get(TRIALS_DIR_ENV) or sessions.parent / "trials")
    trials.mkdir(parents=True, exist_ok=True)

    row = asyncio.run(
        Lab(trials).run(
            task,
            agent_spec(root, tree),
            extra_instruction_paths=instruction_paths(root, tree),
        )
    )
    error = str((getattr(row, "tags", None) or {}).get("error") or "")
    record = trial_record(task, getattr(row, "rewards", None), trials, error)
    write_trial(record, sessions)
    return 1 if record["failed"] else 0
