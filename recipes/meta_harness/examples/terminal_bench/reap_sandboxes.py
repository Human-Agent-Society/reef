"""Kill e2b sandboxes that outlived the episode that created them.

A harness that dies between creating a sandbox and tearing it down leaks it,
and e2b caps an account at a fixed number of concurrent sandboxes. The leak is
silent until the cap is reached, at which point new episodes fail to start and
score zero -- which reads as a bad candidate rather than an outage.

This e2b account is shared with other teams, and this project's slice of it is
32 concurrent sandboxes out of the account's 100. A sandbox this run did not
create belongs to someone else's job, so the reaper kills only sandboxes whose
``environment_name`` names a terminal-bench task, and there is no flag to turn
that off. Age is the second guard: the longest task in terminal-bench@2.0
declares a 12000s cap, so a sandbox older than that plus a margin cannot
belong to an episode still allowed to run.

Age alone is not enough. Reaping on age alone here killed 97 sandboxes, most
of them other teams' running work.

    python -m recipes.meta_harness.examples.terminal_bench.reap_sandboxes --older-than 14400
    python -m recipes.meta_harness.examples.terminal_bench.reap_sandboxes --all --yes
"""

from __future__ import annotations

import argparse
import datetime
import sys
from collections.abc import Sequence
from pathlib import Path

from .tasks import read_tasks

# 12000s is the longest agent timeout in terminal-bench@2.0; the margin covers
# verifier time and teardown after the agent itself is cut off.
LONGEST_TASK_SECONDS = 12000
DEFAULT_MARGIN_SECONDS = 2400


def _sandboxes(module):
    """Read every page of the sandbox listing."""
    paginator = module.list()
    found = []
    while True:
        batch = paginator.next_items()
        if not batch:
            return found
        found.extend(batch)
        if not paginator.has_next:
            return found


def _environment(sandbox) -> str:
    """The task a sandbox was created for, or empty when it carries no metadata."""
    metadata = getattr(sandbox, "metadata", None) or {}
    return str(metadata.get("environment_name", ""))


def select_doomed(alive, ours, now, older_than, ignore_age=False):
    """Choose which sandboxes to kill.

    A sandbox survives unless it both belongs to this benchmark and is older
    than any episode is allowed to run. There is deliberately no override for
    the ownership half: this account is shared, so a sandbox that is not ours
    is someone's running work, and no flag should be able to reach it.
    """
    return [
        sandbox
        for sandbox in alive
        if _environment(sandbox) in ours
        and (ignore_age or (now - sandbox.started_at).total_seconds() > older_than)
    ]


def _kill(module, sandbox_id: str) -> bool:
    """Kill one sandbox, treating one that already ended as done rather than failed."""
    try:
        module.kill(sandbox_id)
    except Exception as exc:
        print(f"  {sandbox_id}: {exc}", file=sys.stderr)
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reap-sandboxes", description=__doc__)
    parser.add_argument(
        "--older-than",
        type=int,
        default=LONGEST_TASK_SECONDS + DEFAULT_MARGIN_SECONDS,
        help="kill sandboxes older than this many seconds",
    )
    parser.add_argument(
        "--tasks-file",
        default=str(Path(__file__).with_name("tasks-89.txt")),
        help="only reap sandboxes whose environment_name is one of these tasks",
    )
    parser.add_argument("--all", action="store_true", help="ignore age (still filtered by owner)")
    parser.add_argument("--yes", action="store_true", help="kill rather than report what would be killed")
    arguments = parser.parse_args(argv)

    try:
        from e2b import Sandbox
    except ImportError:
        print("the reaper needs e2b: pip install e2b", file=sys.stderr)
        return 2

    now = datetime.datetime.now(datetime.timezone.utc)
    alive = _sandboxes(Sandbox)
    ours = {task.split("/")[-1] for task in read_tasks(arguments.tasks_file)}
    doomed = select_doomed(
        alive,
        ours,
        now,
        older_than=arguments.older_than,
        ignore_age=arguments.all,
    )
    foreign = sum(1 for sandbox in alive if _environment(sandbox) not in ours)
    print(f"{len(alive)} alive, {foreign} not from this benchmark, {len(doomed)} to reap")
    if not arguments.yes:
        for sandbox in doomed:
            age = (now - sandbox.started_at).total_seconds() / 60
            print(f"  would kill {sandbox.sandbox_id} ({age:.0f} min)")
        return 0

    killed = sum(_kill(Sandbox, sandbox.sandbox_id) for sandbox in doomed)
    print(f"killed {killed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
