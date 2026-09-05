"""Keep an episode's process group tied to its worker, including SIGKILL loss.

Run as a session leader. The command must remain in this process group; this
lease is lifecycle protection, not a substitute for a sandbox/container.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading


def main() -> None:
    lease_fd = int(sys.argv[1])

    def watch_owner() -> None:
        try:
            while os.read(lease_fd, 1):
                pass
        finally:
            os.close(lease_fd)
            if os.getpid() == os.getpgrp():
                os.killpg(os.getpid(), signal.SIGKILL)

    threading.Thread(target=watch_owner, daemon=True).start()
    child = subprocess.Popen(sys.argv[2:], close_fds=True)
    code = child.wait()
    sys.exit(code if code >= 0 else 128 - code)


if __name__ == "__main__":
    main()
