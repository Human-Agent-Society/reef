"""POSIX process leases for workers in one owned model deployment.

A separate watchdog survives SIGKILL of a Ray worker and retires its process
group, including native engine children. Workers and children must not detach
into another session. Node-local completion records let the owner confirm
cleanup before allocating a replacement deployment.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

DEPLOYMENT_ENV = "REEF_MODEL_PROCESS_LEASE"


def _directory(token: str) -> Path:
    if len(token) != 32 or any(character not in "0123456789abcdef" for character in token):
        raise ValueError("invalid model deployment process lease")
    return Path(tempfile.gettempdir()) / f"reef-model-{os.getuid()}-{token}"


def install() -> None:
    """Ray worker setup hook, enabled only in an owned model job runtime env."""
    token = os.environ.get(DEPLOYMENT_ENV)
    if not token:
        return
    directory = _directory(token)
    directory.mkdir(mode=0o700, exist_ok=True)
    if (directory / "closing").exists():
        raise RuntimeError("model deployment is closing")
    if os.getpgrp() != os.getpid():
        os.setsid()
    lease_read, write_descriptor = os.pipe()
    lease_write: int | None = write_descriptor

    def close_lease() -> None:
        nonlocal lease_write
        if lease_write is not None:
            os.close(lease_write)
            lease_write = None

    # Forked model children must not keep their parent's lease alive.
    # Clear the child copy so another fork cannot close a reused descriptor.
    os.register_at_fork(after_in_child=close_lease)
    try:
        guard = subprocess.Popen(
            [sys.executable, "-m", __name__, token, str(os.getpid()), str(lease_read)],
            pass_fds=(lease_read,),
            start_new_session=True,
        )
    finally:
        os.close(lease_read)
    record = directory / f"{os.getpid()}.json"
    deadline = time.monotonic() + 10
    while not record.exists():
        if guard.poll() is not None or time.monotonic() >= deadline:
            close_lease()
            raise RuntimeError("model worker process guard failed to start")
        time.sleep(0.01)
    if (directory / "closing").exists():
        close_lease()
        raise RuntimeError("model deployment closed during worker startup")
    # The write descriptor intentionally lives until this worker process exits.


def _in_group(process: Any, group: int) -> bool:
    import psutil

    try:
        return os.getpgid(process.pid) == group and process.status() != psutil.STATUS_ZOMBIE
    except (ProcessLookupError, PermissionError, psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _group_alive(group: int) -> bool:
    import psutil

    return any(_in_group(process, group) for process in psutil.process_iter())


def _retire_group(group: int, owner_started: float) -> None:
    import psutil

    try:
        leader_started = psutil.Process(group).create_time()
    except psutil.NoSuchProcess:
        leader_started = None
    # A live group reserves its numeric ID. A reused leader PID means that
    # the old group has vanished; never signal the new process's group.
    if leader_started is not None and leader_started != owner_started:
        return
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Darwin can report EPERM when concurrent cleanup leaves only zombies.
        if _group_alive(group):
            raise
    deadline = time.monotonic() + 20
    while _group_alive(group):
        if time.monotonic() >= deadline:
            raise TimeoutError("model worker process group remained alive after SIGKILL")
        time.sleep(0.05)


def retire(token: str, timeout_s: float = 30) -> None:
    """Retire and confirm all leased process groups on the current node.

    Called on each deployment node by an unguarded cleanup worker. A tombstone
    prevents workers that are still starting from joining this generation.
    Missing or failed guards fail cleanup closed instead of permitting reuse.
    """
    import psutil

    directory = _directory(token)
    directory.mkdir(mode=0o700, exist_ok=True)
    (directory / "closing").touch()
    deadline = time.monotonic() + timeout_s
    while True:
        pending = False
        for record in directory.glob("*.json"):
            if record.with_suffix(".done").exists():
                continue
            metadata = json.loads(record.read_text())
            try:
                guard = psutil.Process(metadata["guard_pid"])
                if guard.create_time() != metadata["guard_started"]:
                    raise RuntimeError("model process guard identity changed")
                guard.send_signal(signal.SIGTERM)
            except psutil.NoSuchProcess as exc:
                if record.with_suffix(".done").exists():
                    continue
                raise RuntimeError("model process guard disappeared before confirming cleanup") from exc
            pending = True
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("model process groups did not retire before the cleanup deadline")
        time.sleep(0.05)


def main() -> None:
    import psutil

    token, owner, descriptor = sys.argv[1:]
    group = int(owner)
    owner_started = psutil.Process(group).create_time()
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_args: stopping.set())
    signal.signal(signal.SIGINT, lambda *_args: stopping.set())
    record = _directory(token) / f"{group}.json"
    temporary = record.with_suffix(".tmp")
    temporary.write_text(json.dumps({"guard_pid": os.getpid(), "guard_started": psutil.Process().create_time()}))
    os.replace(temporary, record)

    def watch_owner() -> None:
        try:
            while os.read(int(descriptor), 1):
                pass
        finally:
            os.close(int(descriptor))
            stopping.set()

    threading.Thread(target=watch_owner, daemon=True).start()
    stopping.wait()
    # The worker established this private group before spawning any children.
    # Force retirement: a failed model process cannot safely finish a write.
    _retire_group(group, owner_started)
    record.with_suffix(".done").touch()


if __name__ == "__main__":
    main()
