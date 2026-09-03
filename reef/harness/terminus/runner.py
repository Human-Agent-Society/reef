"""Run one Terminal-Bench task for one episode, on behalf of ``run_episode``.

This is the only module that imports the evaluation stack, and nothing on the
render path imports it, so the adapter registry stays cheap and a deployment
without the ``terminus`` extra can still load the descriptor.

Reef stays the outer loop. An episode launches ``reef-terminus --task <task>``;
this module runs that one task through reef-eval, the same primitive every
recipe under ``recipes/`` uses to reach Harbor, with
:class:`~reef.harness.terminus.agent.ReefTerminus` named as the agent. It then
writes one trial file under ``REEF_TERMINUS_SESSION_DIR``: the verifier's
rewards beside the ATIF steps Terminus 2 dumped. The adapter's
``terminus-atif-json`` reader turns that into the episode's trajectory, so a
scorer reads the reward and a method reads the turns from one place.

Harbor's own trial tree lands in ``REEF_TERMINUS_TRIALS_DIR`` instead, keeping
the session directory the reader walks to files Reef wrote.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from reef.harness.terminus.tree import TerminusTreeError, load_tree, terminus_kwargs

#: How reef-eval names an agent: the import path of the class to construct.
AGENT_IMPORT_PATH = "reef.harness.terminus.agent:ReefTerminus"
DATASET_ENV = "REEF_TERMINUS_DATASET"
SESSION_DIR_ENV = "REEF_TERMINUS_SESSION_DIR"
TREE_DIR_ENV = "REEF_TERMINUS_DIR"
TRIALS_DIR_ENV = "REEF_TERMINUS_TRIALS_DIR"


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise TerminusTreeError(f"{name} must name a path for the terminus runner")
    return value


def agent_spec(tree: dict[str, str], tree_dir: str) -> dict[str, Any]:
    """The agent reef-eval should construct for this tree.

    The tree owns the model: Reef's model binding rendered ``model_name`` into
    the config, and the agent reads the rest of the tree from disk itself, so
    the only thing passed through here is where to look.
    """
    model = terminus_kwargs(tree).get("model_name")
    if not model:
        raise TerminusTreeError("the terminus tree carries no model_name; Reef's model binding sets it")
    return {"name": AGENT_IMPORT_PATH, "model_name": str(model), "kwargs": {"reef_dir": tree_dir}}


def task_path(task: str) -> str:
    """Where the named task lives, under the configured dataset."""
    dataset = Path(_required_env(DATASET_ENV))
    candidate = dataset / task
    if candidate.is_dir():
        return str(candidate)
    if dataset.name == task and dataset.is_dir():
        return str(dataset)
    raise TerminusTreeError(f"dataset {dataset} holds no task {task!r}")


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
    target = sessions / f"{record['task']}.json"
    target.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return target


def run(task: str) -> int:
    """Run one task and return the process exit code.

    Non-zero when the verifier produced no reward: the episode then reports a
    failed run rather than a scoreless success.
    """
    # Lazy: reef-eval pulls Harbor's dependency tree, which the render path
    # and its tests must never need.
    from reef_eval import Lab

    tree_dir = _required_env(TREE_DIR_ENV)
    tree = load_tree(tree_dir)
    sessions = Path(_required_env(SESSION_DIR_ENV))
    trials = Path(os.environ.get(TRIALS_DIR_ENV) or sessions.parent / "trials")
    trials.mkdir(parents=True, exist_ok=True)

    row = asyncio.run(Lab(trials).run(task_path(task), agent_spec(tree, tree_dir)))
    error = str((getattr(row, "tags", None) or {}).get("error") or "")
    record = trial_record(task, getattr(row, "rewards", None), trials, error)
    write_trial(record, sessions)
    return 1 if record["failed"] else 0
