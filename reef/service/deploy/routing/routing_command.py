"""Run one Docker CLI command with a bounded lifetime and an owner pipe.

This helper receives no registry lock. Its parent owns the write end of the
pipe; even SIGKILL closes that end and makes this helper terminate the command.
A killed client does not prove a submitted Docker operation was rolled back.
"""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import sys
import time


def main() -> int:
    owner_descriptor = int(sys.argv[1])
    timeout_seconds = float(sys.argv[2])
    command = sys.argv[3:]
    stopping = False

    def request_stop(signum: int, frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    # The helper and CLI share a fresh group, so parent escalation also reaches
    # Compose plugins and any other descendants after an unresponsive helper.
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    deadline = time.monotonic() + timeout_seconds
    outcome = 0
    with selectors.DefaultSelector() as selector:
        selector.register(owner_descriptor, selectors.EVENT_READ)
        while process.poll() is None:
            if stopping:
                outcome = 125
                break
            if time.monotonic() >= deadline:
                outcome = 124
                break
            if selector.select(min(0.1, max(0, deadline - time.monotonic()))) and not os.read(owner_descriptor, 1):
                outcome = 125
                break
    os.close(owner_descriptor)
    if outcome:
        os.killpg(os.getpgrp(), signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            # Kill the whole isolated group, including this guard. The caller
            # treats every non-zero/timeout result as an uncertain mutation.
            os.killpg(os.getpgrp(), signal.SIGKILL)
        return outcome
    return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
