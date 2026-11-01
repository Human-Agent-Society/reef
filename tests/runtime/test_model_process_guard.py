"""Owned native process cleanup survives forced parent death and forked children."""

import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest

from reef.runtime.executor.process_guard import DEPLOYMENT_ENV, _directory, retire


def wait_for(path):
    deadline = time.monotonic() + 10
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"missing {path}")
        time.sleep(0.01)


@pytest.mark.parametrize("forced", [False, True])
def test_process_lease_retires_forked_children_without_touching_other_groups(tmp_path, forced):
    psutil = pytest.importorskip("psutil")
    token = uuid4().hex
    info = tmp_path / "child.json"
    script = """
import json, os, time, sys
from pathlib import Path
from reef.runtime.executor.process_guard import install
install()
pid = os.fork()
if pid == 0:
    time.sleep(60)
else:
    # Publish the pid atomically: the test polls for the file and must never read it half-written.
    pending = Path(sys.argv[1] + ".tmp")
    pending.write_text(json.dumps({"child": pid}))
    os.replace(pending, sys.argv[1])
    time.sleep(60)
"""
    owner = subprocess.Popen([sys.executable, "-c", script, str(info)], env={**os.environ, DEPLOYMENT_ENV: token})
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        wait_for(info)
        child = json.loads(info.read_text())["child"]
        if forced:
            os.kill(owner.pid, signal.SIGKILL)
        retire(token)
        owner.wait(timeout=10)
        assert not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE
        assert unrelated.poll() is None
        assert list(_directory(token).glob("*.done"))
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait()
        unrelated.terminate()
        unrelated.wait()


@pytest.mark.parametrize("worker_alive", [True, False])
def test_a_missing_guard_retires_its_group_once_the_worker_is_gone(tmp_path, monkeypatch, worker_alive):
    from reef.runtime.executor import process_guard

    pytest.importorskip("psutil")
    retired: list[tuple[int, float | None]] = []
    monkeypatch.setattr(process_guard, "_directory", lambda token: tmp_path)
    monkeypatch.setattr(process_guard, "_worker_alive", lambda group: group == 12345 and worker_alive)
    monkeypatch.setattr(process_guard, "_retire_group", lambda group, started: retired.append((group, started)))
    # Ray kills an exiting worker's children, its guard among them, so the guard can be gone before it confirms.
    (tmp_path / "12345.json").write_text(json.dumps({"guard_pid": 99999999, "guard_started": 0}))
    if worker_alive:
        with pytest.raises(RuntimeError, match="before confirming cleanup"):
            retire("unused")
        assert retired == []
    else:
        retire("unused")
        # The worker's other processes, such as compile workers, outlive it and its guard: retire them here.
        assert retired == [(12345, None)] and (tmp_path / "12345.done").exists()


def test_a_group_whose_leader_exited_is_retired_whatever_its_leader_started_at(monkeypatch):
    from reef.runtime.executor import process_guard

    psutil = pytest.importorskip("psutil")
    signals = []

    def gone(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", gone)
    monkeypatch.setattr(process_guard.os, "killpg", lambda group, sig: signals.append((group, sig)))
    monkeypatch.setattr(process_guard, "_group_alive", lambda group: False)
    process_guard._retire_group(12345, None)
    assert signals == [(12345, signal.SIGKILL)]


@pytest.mark.parametrize("alive", [False, True])
def test_group_retirement_handles_exit_during_identity_read_but_requires_no_live_members(monkeypatch, alive):
    from reef.runtime.executor import process_guard

    psutil = pytest.importorskip("psutil")

    def disappeared():
        raise psutil.NoSuchProcess(12345)

    def kill(group, sig):
        assert group == 12345 and sig == signal.SIGKILL
        raise PermissionError("group concurrently retired")

    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(create_time=disappeared))
    monkeypatch.setattr(process_guard.os, "killpg", kill)
    monkeypatch.setattr(process_guard, "_group_alive", lambda group: alive)
    if alive:
        with pytest.raises(PermissionError):
            process_guard._retire_group(12345, 1)
    else:
        process_guard._retire_group(12345, 1)


def test_group_retirement_never_signals_reused_owner_pid(monkeypatch):
    from reef.runtime.executor import process_guard

    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 2))
    monkeypatch.setattr(process_guard.os, "killpg", lambda *args: pytest.fail("unrelated group signaled"))
    process_guard._retire_group(12345, 1)
