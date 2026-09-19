"""The lease guard reaps its child before killing its own process group."""

import os
import signal
import subprocess
import threading
from types import SimpleNamespace

import pytest

from reef.service.deploy import guard


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
@pytest.mark.parametrize("ignores_term", [False, True])
def test_guard_reaps_child_before_final_group_kill(monkeypatch, ignores_term):
    events = []
    clock = [0.0]
    stopping = threading.Event()
    stopping.set()

    class Child:
        def poll(self):
            return None

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            if ignores_term and timeout is not None:
                clock[0] += timeout
                raise subprocess.TimeoutExpired("service", timeout)
            if not ignores_term:
                clock[0] += 0.25
            return -signal.SIGKILL if ignores_term else -signal.SIGTERM

        def kill(self):
            events.append("kill-child")

    monkeypatch.setattr(guard.sys, "argv", ["guard", "10", "service"])
    monkeypatch.setattr(guard.threading, "Event", lambda: stopping)
    monkeypatch.setattr(guard.threading, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(guard.signal, "signal", lambda *args: None)
    monkeypatch.setattr(guard.os, "getpid", lambda: 42)
    monkeypatch.setattr(guard.os, "getpgrp", lambda: 42)
    monkeypatch.setattr(guard.os, "killpg", lambda pid, sig: events.append((pid, sig)))
    monkeypatch.setattr(guard.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(guard.time, "sleep", lambda duration: events.append(("sleep", duration)))

    guard.main()

    expected = [(42, signal.SIGTERM), ("wait", 1)]
    if ignores_term:
        expected += ["kill-child", ("wait", None)]
    expected += [("sleep", 0 if ignores_term else 0.75), (42, signal.SIGKILL)]
    assert events == expected
