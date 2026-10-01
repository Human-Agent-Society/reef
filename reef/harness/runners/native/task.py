"""``reef-native task --task <dir>``: one Harbor task for one episode of the native_harbor adapter.

The runner hands the task to Harbor through reef-eval, as the terminus runner does, with the Harbor agent in
``reef.harness.runners.native.harbor``: the native loop runs in this process and its tools run in the task
container. It then writes one flat ``verifier`` row, the shape the terminus runner writes, to ``verifier.jsonl`` in
the session directory: the task, the verifier's rewards, and whether the trial failed with its error. A failure of
the runner itself, before or after the trial, is a failed row too, so the episode scores as one that never ran
instead of a zero. Harbor's trial tree lands in ``REEF_NATIVE_HARBOR_TRIALS_DIR``, outside the session directory.

Nothing here imports Harbor or reef-eval at module level.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from reef.core.errors import ReefError
from reef.harness.episodes.trajectory import primary_reward
from reef.harness.runners.harbor_trial import HarborTrialError, infrastructure_error, infrastructure_markers, run_trial
from reef.harness.runners.native import LoadError, binding_from, output_token_limit_from
from reef.harness.runners.native.control import episode_token_limit

TRIALS_DIR_ENV = "REEF_NATIVE_HARBOR_TRIALS_DIR"
#: Where Harbor runs the task: ``docker`` (the default) or ``e2b``.
ENVIRONMENT_ENV = "REEF_NATIVE_HARBOR_ENVIRONMENT"
AGENT_IMPORT_PATH = "reef.harness.runners.native.harbor:NativeTeamAgent"
VERIFIER_FILE = "verifier.jsonl"


def verifier_row(task: str, rewards: Mapping[str, float], *, is_failed: bool, error: str) -> dict[str, object]:
    """The row ``verifier_reward`` scores; its fields sit at the top level, where a session event keeps ``data``."""
    return {
        "type": "verifier",
        "task": task,
        "rewards": dict(rewards),
        "reward": primary_reward(rewards),
        "failed": is_failed,
        "error": error,
    }


def write_verifier_row(session_path: Path, row: Mapping[str, object]) -> None:
    session_path.mkdir(parents=True, exist_ok=True)
    with (session_path / VERIFIER_FILE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def run_task(task: str) -> int:
    """Run one task and write its verifier row; 1 when the trial or the runner failed, else 0."""
    root = Path(os.environ.get("REEF_NATIVE_DIR") or "native").absolute()
    sessions = Path(os.environ.get("REEF_NATIVE_SESSION_DIR") or root / "sessions").absolute()
    try:
        trials_text = os.environ.get(TRIALS_DIR_ENV)
        if not trials_text:
            raise HarborTrialError(f"{TRIALS_DIR_ENV} must name the directory Harbor writes its trials in")
        trials = Path(trials_text)
        trials.mkdir(parents=True, exist_ok=True)
        environment = os.environ.get(ENVIRONMENT_ENV, "docker")
        if environment not in ("docker", "e2b"):
            raise HarborTrialError(f"{ENVIRONMENT_ENV}={environment!r} names no Harbor environment; use docker or e2b")
        markers = infrastructure_markers(os.environ)
        max_completion_tokens = output_token_limit_from(root / "models.json")
        binding = binding_from(root / "models.json")
        agent = {
            "import_path": AGENT_IMPORT_PATH,
            "model_name": binding.model,
            # Harbor writes these into the trial config; the key stays in models.json, which the agent reads.
            "kwargs": {
                "tree_path": str(root),
                "session_path": str(sessions),
                "max_completion_tokens": max_completion_tokens,
                "token_limit": episode_token_limit(os.environ),
            },
        }
        result = run_trial(task, agent, trials_path=trials, environment=environment, runner_name="native_harbor")
        marker_error = infrastructure_error(result.trial_path, markers)
    except (ReefError, LoadError, OSError, ValueError) as exc:
        print(f"[reef-native] {exc}", file=sys.stderr)
        write_verifier_row(sessions, verifier_row(task, {}, is_failed=True, error=str(exc)))
        return 1
    is_failed = not result.rewards or marker_error is not None
    write_verifier_row(
        sessions, verifier_row(task, result.rewards, is_failed=is_failed, error=marker_error or result.error)
    )
    return 1 if is_failed else 0


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="reef-native task", description="Run one Harbor task with the native agent for a Reef episode."
    )
    parser.add_argument("--task", required=True, help="the Harbor task directory or registry id")
    return run_task(parser.parse_args(list(argv)).task)
