"""Drive one harbor job for one task, on behalf of ``run_episode``.

This is the only module that imports harbor, and the adapter never imports it
at render time, so the registry stays cheap and a deployment without the
``terminus`` extra can still load the descriptor.

Reef stays the outer loop: an episode launches ``reef-terminus --task <name>``,
this module runs that one task through harbor with :class:`ReefTerminus` bound
to the rendered tree, and writes the trial's ATIF trajectory and the verifier's
reward under ``REEF_TERMINUS_SESSION_DIR``, where the adapter's
``terminus-atif-json`` reader picks them up.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from reef.harness.terminus.tree import TerminusTreeError, load_tree, terminus_kwargs

AGENT_IMPORT_PATH = "reef.harness.terminus.agent:ReefTerminus"
DATASET_ENV = "REEF_TERMINUS_DATASET"
SESSION_DIR_ENV = "REEF_TERMINUS_SESSION_DIR"
TREE_DIR_ENV = "REEF_TERMINUS_DIR"


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise TerminusTreeError(f"{name} must name a path for the terminus runner")
    return value


def _job_config(task: str, tree: dict[str, str], sessions: Path) -> Any:
    """Build the harbor job for one task, one attempt, this tree's agent."""
    from harbor.job import AgentConfig, DatasetConfig, JobConfig

    knobs = terminus_kwargs(tree)
    model = knobs.get("model_name")
    if not model:
        raise TerminusTreeError("the terminus tree carries no model_name; Reef's model binding sets it")
    return JobConfig(
        job_name=f"reef-terminus-{task}",
        jobs_dir=sessions,
        n_attempts=1,
        n_concurrent_trials=1,
        quiet=True,
        agents=[
            AgentConfig(
                name="reef-terminus-2",
                import_path=AGENT_IMPORT_PATH,
                model_name=str(model),
                # The agent reads the tree from disk itself, so the only thing
                # it needs here is where to look.
                kwargs={"reef_dir": os.environ[TREE_DIR_ENV]},
            )
        ],
        datasets=[DatasetConfig(path=Path(_required_env(DATASET_ENV)), task_names=[task])],
    )


def _write_trial(sessions: Path, task: str, result: Any) -> None:
    """Record the ATIF trajectory and the verifier's reward for the reader."""
    steps = getattr(result, "steps", None)
    trial = {
        "task": task,
        "reward": getattr(result, "reward", None),
        "resolved": getattr(result, "resolved", None),
        "steps": list(steps) if isinstance(steps, list) else [],
    }
    sessions.mkdir(parents=True, exist_ok=True)
    (sessions / f"{task}.json").write_text(json.dumps(trial, indent=2, default=str) + "\n", encoding="utf-8")


def run(task: str) -> int:
    """Run one task and return the process exit code."""
    from harbor.job import Job

    tree = load_tree(_required_env(TREE_DIR_ENV))
    sessions = Path(_required_env(SESSION_DIR_ENV))
    job = Job.create(_job_config(task, tree, sessions))
    result = job.run()
    _write_trial(sessions, task, result)
    return 0
