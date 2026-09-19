"""The task's engine session: start it for one episode, stop it for the verifier.

Runs inside the task container with the pinned checkout's interpreter
(``cd /opt/ceobench && .venv/bin/python /opt/ceobench-engine.py ...``) and
prints one JSON line per command.

``start`` does what the benchmark's runner (``saas_bench.agents.bash_agent
.run_test``) does before the agent's first turn: the run directory and the
agent's workspace (the published ``docs/`` and ``novamind-operation`` and
nothing else), the session through the zipapp, the engine as a separate
process in server mode, and ``config.json`` where the verifier reads it.
``commit`` snapshots the workspace at each week boundary the way the runner
does (the agent can see the repository it plays in), and ``stop`` does what
the runner does after the last turn: reads the final status, stops the
engine, and copies the session's ``world.nmdb`` beside ``config.json``. The
runner's checkpoints and timing logs are not reproduced: they serve resume
and tamper checks, and the agent never sees them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

CHECKOUT = Path("/opt/ceobench")
PUBLIC = CHECKOUT / "public"
STATUS_FALLBACK = {"day": 0, "cash": 0, "subscribers": 0, "timed_out": False}
GITIGNORE = """\
sessions/
_engine/
*.nmdb
*.db
*.db-journal
*.db-wal
*.db-shm
__pycache__/
*.pyc
.pytest_cache/
.venv/
"""


def emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def http_get(port: int, path: str, timeout: float = 30) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as response:
        return json.loads(response.read())


def server_environment() -> dict[str, str]:
    """The zipapp dispatches to the engine's server entry under this variable."""
    return {**os.environ, "NOVAMIND_SERVER_MODE": "1"}


def zipapp(workspace: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PUBLIC / "novamind-operation"), "--base", str(workspace), *arguments],
        capture_output=True,
        text=True,
        env=server_environment(),
    )


def chown_tree(root: Path, user: str) -> None:
    for directory_entry in os.walk(root):
        directory, files = directory_entry[0], directory_entry[2]
        shutil.chown(directory, user, user)
        for name in files:
            shutil.chown(os.path.join(directory, name), user, user)


def git(workspace: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *arguments], cwd=str(workspace), capture_output=True, text=True)


def git_init(workspace: Path) -> None:
    """The runner's repository in the agent's workspace: main branch, its author, its ignore list."""
    if (workspace / ".git").exists():
        return
    git(workspace, "init", "-q", "-b", "main")
    git(workspace, "config", "user.email", "bash-agent@bossbench.local")
    git(workspace, "config", "user.name", "BashAgent")
    if not (workspace / ".gitignore").exists():
        (workspace / ".gitignore").write_text(GITIGNORE)


def git_commit(workspace: Path, message: str, once_key: str | None = None) -> None:
    """Commit the workspace as the runner does; a ``once_key`` makes the commit idempotent."""
    if not (workspace / ".git").exists():
        return
    if once_key is not None:
        existing = git(workspace, "log", "--grep", f"[{once_key}]", "--fixed-strings", "--oneline")
        if existing.returncode == 0 and existing.stdout.strip():
            return
        message = f"{message} [{once_key}]"
    git(workspace, "add", "-A")
    status = git(workspace, "status", "--porcelain")
    if status.returncode == 0 and not status.stdout.strip():
        git(workspace, "commit", "--allow-empty", "-q", "-m", message)
    else:
        git(workspace, "commit", "-q", "-m", message)


def cmd_start(args: argparse.Namespace) -> None:
    run_id = uuid.uuid4().hex[:8]
    run_dir = Path(args.runs_dir) / f"run_{run_id}"
    workspace = run_dir / "agent_workspace"
    logs = run_dir / "logs"
    for path in (workspace, logs, workspace / "daily_scripts"):
        path.mkdir(parents=True, exist_ok=True)
    total_days = (args.days // 7) * 7  # the runner rounds down to whole weeks

    git_init(workspace)
    # docs/ and the zipapp are the only two things the published layout gives the agent.
    if (PUBLIC / "docs").exists():
        shutil.copytree(PUBLIC / "docs", workspace / "docs", ignore=shutil.ignore_patterns("__pycache__"))
    operation = workspace / "novamind-operation"
    shutil.copy2(PUBLIC / "novamind-operation", operation)
    operation.chmod(operation.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    created = zipapp(
        workspace, "new-session", "--days", str(total_days), "--seed", str(args.seed), "--cash", str(args.cash)
    )
    if created.returncode != 0:
        raise RuntimeError(f"novamind-operation new-session failed:\n{created.stderr}\n{created.stdout}")
    session_id = json.loads(created.stdout)["session_id"]
    git_commit(workspace, "Initial workspace setup (day 0)")
    if args.tool_user:
        chown_tree(workspace, args.tool_user)  # the agent's tools run as this user

    # The engine in server mode, detached from this command. Its stdout and
    # stderr go to files: a pipe nobody drains would wedge it, and the port
    # is on the first stdout line.
    stdout_path, stderr_path = logs / "api_server_stdout.log", logs / "api_server_stderr.log"
    with open(stdout_path, "ab", buffering=0) as stdout, open(stderr_path, "ab", buffering=0) as stderr:
        subprocess.Popen(
            [
                sys.executable,
                str(PUBLIC / "novamind-operation"),
                "--base",
                str(workspace),
                "start-server",
                "--session",
                session_id,
            ],
            stdout=stdout,
            stderr=stderr,
            env=server_environment(),
            start_new_session=True,
        )
    port = None
    for _ in range(120):
        first_line = stdout_path.read_text(errors="replace").splitlines()[:1]
        if first_line:
            port = int(json.loads(first_line[0])["port"])
            break
        time.sleep(0.5)
    if port is None:
        raise RuntimeError(f"engine did not start:\n{stderr_path.read_text(errors='replace')[-4000:]}")
    for _ in range(60):
        try:
            http_get(port, "/health", timeout=2)
            break
        except (OSError, ValueError):
            time.sleep(0.5)
    else:
        raise RuntimeError("engine did not answer /health after 30s")

    config = {
        "run_id": run_id,
        "model": args.model,
        "provider": "openai",
        "reasoning_effort": args.reasoning_effort,
        "seed": args.seed,
        "scenario": "default",
        "total_days": total_days,
        "initial_cash": args.cash,
        "agent_type": "bash_agent",
        "api_server_port": port,
        "session_id": session_id,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    emit(
        {
            "run_dir": str(run_dir),
            "workspace": str(workspace),
            "session_id": session_id,
            "port": port,
            "total_days": total_days,
            "initial_cash": args.cash,
        }
    )


def cmd_commit(args: argparse.Namespace) -> None:
    """One commit per simulated week reached, as the runner tags its timeline."""
    workspace = Path(args.run_dir) / "agent_workspace"
    for week in range(1, args.day // 7 + 1):
        git_commit(workspace, f"Week {week} (day {week * 7})", once_key=f"week-{week}")
    emit({"committed_through_week": args.day // 7})


def cmd_stop(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    workspace = run_dir / "agent_workspace"
    port, session_id = int(config["api_server_port"]), str(config["session_id"])
    try:
        final = http_get(port, "/game-status")
    except (OSError, ValueError):
        final = dict(STATUS_FALLBACK)
    stopped = zipapp(workspace, "stop-server", "--session", session_id)
    time.sleep(2.0)  # the engine drains its pending writes on shutdown
    live = workspace / "sessions" / session_id / "world.nmdb"
    copied = False
    if live.exists():
        shutil.copy2(live, run_dir / "world.nmdb")
        copied = True
    emit({"final": final, "copied": copied, "stop_output": stopped.stdout.strip()[:400]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("start")
    start.add_argument("--runs-dir", required=True)
    start.add_argument("--seed", type=int, required=True)
    start.add_argument("--days", type=int, required=True)
    start.add_argument("--cash", type=float, default=1_000_000.0)
    start.add_argument("--model", default="reef")
    start.add_argument("--reasoning-effort", default="none")
    start.add_argument("--tool-user", default=None)
    start.set_defaults(run=cmd_start)

    commit = commands.add_parser("commit")
    commit.add_argument("--run-dir", required=True)
    commit.add_argument("--day", type=int, required=True)
    commit.set_defaults(run=cmd_commit)

    stop = commands.add_parser("stop")
    stop.add_argument("--run-dir", required=True)
    stop.set_defaults(run=cmd_stop)

    args = parser.parse_args()
    try:
        args.run(args)
    except (RuntimeError, OSError, ValueError, urllib.error.URLError) as error:
        emit({"error": f"{type(error).__name__}: {error}"})
        sys.exit(1)


if __name__ == "__main__":
    main()
