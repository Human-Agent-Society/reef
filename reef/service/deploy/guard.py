"""A POSIX service lease: retire the process group if its Ray owner disappears.

The guard is the session leader. Its command shares that process group, so
ordinary shutdown and forced cleanup reach both. The lease descriptor is never
inherited by the service. This is not a general daemon/cgroup supervisor:
services must not detach themselves into another session.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time


def main() -> None:
    lease_fd = int(sys.argv[1])
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_args: stopping.set())
    signal.signal(signal.SIGINT, lambda *_args: stopping.set())

    def watch_owner() -> None:
        try:
            while os.read(lease_fd, 1):
                pass
        finally:
            os.close(lease_fd)
            stopping.set()

    watcher = threading.Thread(target=watch_owner, daemon=True)
    watcher.start()
    child = subprocess.Popen(sys.argv[2:], close_fds=True)
    while child.poll() is None and not stopping.wait(0.1):
        pass
    # This process was explicitly spawned as its own session leader.
    if os.getpgrp() != os.getpid():
        raise RuntimeError("service lease guard must own its process group")
    os.killpg(os.getpid(), signal.SIGTERM)
    # Keep the guard alive through escalation, even if the direct child exits
    # early and leaves grandchildren. SIGKILL also retires this guard.
    time.sleep(1)
    os.killpg(os.getpid(), signal.SIGKILL)


if __name__ == "__main__":
    main()
