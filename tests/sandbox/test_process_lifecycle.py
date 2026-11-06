"""Local process ownership, termination causes and cleanup contracts."""

from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from reef.execution.processes import ProcessRequest, ProcessScope, ProcessScopeClosed

pytestmark = pytest.mark.unit


def request(tmp_path: Path, source: str, **changes: object) -> ProcessRequest:
    base = ProcessRequest(argv=(sys.executable, "-c", source), cwd=tmp_path, env=dict(os.environ), timeout_seconds=5.0)
    return replace(base, **changes)


def wait_for_file(path: Path) -> None:
    deadline = time.monotonic() + 5
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists()


def test_execution_keeps_output_exit_status_and_explicit_environment(tmp_path: Path) -> None:
    scope = ProcessScope()
    result = scope.run(
        request(
            tmp_path,
            "import os, sys; print(os.environ['SETTING']); print(sys.stdin.read(), file=sys.stderr); sys.exit(3)",
            env={"SETTING": "value"},
            stdin_text="input",
        )
    )
    assert (result.reason, result.exit_code, result.stdout, result.stderr) == ("exited", 3, "value\n", "input\n")
    assert scope.pids() == ()
    assert not scope.wait_closed(0)
    scope.request_stop()
    assert scope.wait_closed(0)


def test_timeout_preserves_the_cause_when_term_handler_exits_zero(tmp_path: Path) -> None:
    source = (
        "import signal, sys, time; "
        "signal.signal(signal.SIGTERM, lambda *args: (print('cleaned', flush=True), sys.exit(0))); "
        "time.sleep(60)"
    )
    result = ProcessScope().run(request(tmp_path, source, timeout_seconds=0.5, grace_seconds=1.0))
    assert result.reason == "timed_out"
    assert result.exit_code == 0
    assert result.stdout == "cleaned\n"


def test_stop_waits_for_cleanup_and_rejects_new_launches(tmp_path: Path) -> None:
    scope = ProcessScope()
    source = (
        "import pathlib, signal, sys, time; "
        "signal.signal(signal.SIGTERM, lambda *args: (time.sleep(.15), pathlib.Path('cleaned').touch(), sys.exit(0))); "
        "pathlib.Path('ready').touch(); time.sleep(60)"
    )
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(scope.run, request(tmp_path, source, grace_seconds=1.0))
        wait_for_file(tmp_path / "ready")
        (pid,) = scope.pids()
        scope.request_stop()
        assert not scope.wait_closed(0)
        with pytest.raises(ProcessScopeClosed):
            scope.run(request(tmp_path, "raise SystemExit(0)"))
        scope.request_stop(grace_seconds=0)  # repeated closure must not shorten the accepted grace
        assert scope.wait_closed(5)
        result = pending.result(timeout=5)
    assert result.reason == "stopped" and result.exit_code == 0
    assert (tmp_path / "cleaned").exists()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_ignoring_term_forces_group_kill_after_grace(tmp_path: Path) -> None:
    scope = ProcessScope()
    source = "import pathlib, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path('ready').touch(); time.sleep(60)"
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(scope.run, request(tmp_path, source, grace_seconds=0.15))
        wait_for_file(tmp_path / "ready")
        (pid,) = scope.pids()
        scope.request_stop()
        assert not scope.wait_closed(0.02)
        assert scope.wait_closed(5)
        result = pending.result(timeout=5)
    assert result.reason == "stopped" and result.exit_code == -9
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)


def test_child_scope_is_stopped_with_its_owner(tmp_path: Path) -> None:
    parent, child = ProcessScope(), ProcessScope()
    parent.attach(child)
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(
            child.run, request(tmp_path, "import pathlib,time; pathlib.Path('ready').touch(); time.sleep(60)")
        )
        wait_for_file(tmp_path / "ready")
        parent.request_stop()
        assert parent.wait_closed(5)
        assert pending.result(timeout=5).reason == "stopped"
    with pytest.raises(ProcessScopeClosed):
        child.run(request(tmp_path, "pass"))
    with pytest.raises(ValueError, match="cycle"):
        child.attach(parent)


def test_concurrent_spawn_and_close_cannot_lose_a_child(tmp_path: Path) -> None:
    scope = ProcessScope()
    start = threading.Barrier(2)

    def run() -> str:
        start.wait()
        try:
            return scope.run(request(tmp_path, "import time; time.sleep(60)")).reason
        except ProcessScopeClosed:
            return "refused"

    with ThreadPoolExecutor() as pool:
        pending = pool.submit(run)
        start.wait()
        scope.request_stop()
        assert scope.wait_closed(5)
        assert pending.result(timeout=5) in ("stopped", "refused")
    assert scope.pids() == ()


def test_launch_failure_leaves_no_registered_process(tmp_path: Path) -> None:
    scope = ProcessScope()
    with pytest.raises(FileNotFoundError):
        scope.run(request(tmp_path, "pass", argv=(str(tmp_path / "absent"),)))
    assert scope.pids() == ()
    scope.request_stop()
    assert scope.wait_closed(0)


@pytest.mark.parametrize(
    "changes", [{"argv": ()}, {"timeout_seconds": 0}, {"grace_seconds": float("nan")}, {"env": {"BAD=NAME": "value"}}]
)
def test_request_rejects_invalid_values(tmp_path: Path, changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        request(tmp_path, "pass", **changes)


def test_execution_contract_controls_different_owners(tmp_path: Path) -> None:
    import asyncio

    from reef.execution.jobs import JobContext
    from reef.execution.lifecycle import Execution
    from reef.execution.sandbox.environment import ExecutorEnvironment, ProgramExecution
    from reef.execution.sandbox.process import LocalExecutor, ProgramLaunchError

    context = JobContext()
    program = ProgramExecution(ExecutorEnvironment(LocalExecutor(), root=tmp_path, workspace=tmp_path), ["true"])
    executions: tuple[Execution, ...] = (ProcessScope(), context, program)
    for execution in executions:
        assert not execution.wait_closed(0)
        with pytest.raises(ValueError):
            execution.wait_closed(-1)
        execution.request_stop()
        execution.request_stop()
    assert not context.wait_closed(0)  # Cancellation does not complete runner cleanup.
    asyncio.run(context.finish())
    for execution in executions:
        assert execution.wait_closed(0)
        assert execution.wait_closed(None)
    with pytest.raises(ProgramLaunchError, match="stopped"):
        program.run()
