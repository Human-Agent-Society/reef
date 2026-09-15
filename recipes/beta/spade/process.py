"""Run a generated environment class in a child interpreter on the host, for the Designer's smoke test.

The child runs with the shipped loader and no site packages, in its own process group, and talks over a
pipe of its own, so a class that hangs, exits, forks or floods its standard streams costs one check, never
the caller. The child still runs with the caller's user, files and network: the Designer's reply is checked
for the imports the rules forbid before it gets here, and that check is the trust boundary on the host.
The same loader and rules run inside the task's container and its verifier.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType

from recipes.beta.spade import environment_loader

STEP_TIMEOUT_S = 30.0
RESET_TIMEOUT_S = 30.0
CLOSE_TIMEOUT_S = 5.0
OBSERVATION_CHARS = 20000
QUEUED_LINES = 64


class EnvironmentProcessError(RuntimeError):
    """The child that runs the environment died, hung or answered with something that is not an environment event."""


ENVIRONMENT_SERVER = r'''"""Serve one environment class over stdin and a pipe of its own, one JSON line per request and reply."""

import json
import os
import sys

from env_loader import load_environment_class, make_environment, step_result


def main(environment_path, max_turns, protocol):
    environment = make_environment(load_environment_class(environment_path), max_turns)
    for line in sys.stdin:
        request = json.loads(line)
        if request["op"] == "reset":
            observation, _ = environment.reset(seed=request["seed"])
            reply = {"observation": str(observation)}
        elif request["op"] == "step":
            observation, reward, terminated, truncated, error = step_result(environment, request["action"])
            reply = {"observation": observation, "reward": reward, "terminated": terminated, "truncated": truncated}
            if error is not None:
                reply["step_error"] = error
        else:
            reply = {"error": f"unknown op {request['op']!r}"}
        protocol.write(json.dumps(reply) + "\n")
        protocol.flush()


if __name__ == "__main__":
    channel = os.fdopen(int(sys.argv[3]), "w", encoding="utf-8")
    try:
        main(sys.argv[1], int(sys.argv[2]), channel)
    except BaseException as exc:
        channel.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"[:500]}) + "\n")
        channel.flush()
        raise SystemExit(1)
'''


class EnvironmentProcess:
    """One environment in a child interpreter: ``reset`` and ``step`` with a timeout each; kill the group on exit."""

    def __init__(self, code: str, *, max_turns: int) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="spade-env-")
        root = Path(self.temporary_directory.name)
        (root / "env.py").write_text(code, encoding="utf-8")
        (root / "env_loader.py").write_text(inspect.getsource(environment_loader), encoding="utf-8")
        (root / "server.py").write_text(ENVIRONMENT_SERVER, encoding="utf-8")
        read_end, write_end = os.pipe()
        try:
            # -E -s -S: no PYTHONPATH, no user site, no site packages, the way the task image has none.
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-E",
                    "-s",
                    "-S",
                    str(root / "server.py"),
                    str(root / "env.py"),
                    str(max_turns),
                    str(write_end),
                ],
                cwd=root,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                pass_fds=(write_end,),
                start_new_session=True,
                text=True,
                encoding="utf-8",
            )
        except OSError:
            os.close(read_end)
            os.close(write_end)
            self.temporary_directory.cleanup()
            raise
        os.close(write_end)
        self.protocol = os.fdopen(read_end, "r", encoding="utf-8", errors="replace")
        self.lines: queue.Queue[str | None] = queue.Queue(maxsize=QUEUED_LINES)
        self.reader = threading.Thread(target=self.read_lines, daemon=True)
        self.reader.start()

    def read_lines(self) -> None:
        with contextlib.suppress(ValueError, OSError):
            for line in self.protocol:
                self.lines.put(line)
        self.lines.put(None)

    def request(self, payload: Mapping[str, object], timeout_s: float) -> dict[str, object]:
        """Send one request and return its reply; a timeout kills the child, any other failure is a EnvironmentProcessError."""
        if self.process.stdin is None or self.process.poll() is not None:
            raise EnvironmentProcessError("the environment process is not running")
        try:
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise EnvironmentProcessError(f"the environment process closed its input: {exc}") from exc
        try:
            line = self.lines.get(timeout=timeout_s)
        except queue.Empty:
            self.close()
            raise EnvironmentProcessError(f"the environment did not answer within {timeout_s:g} s") from None
        if line is None:
            raise EnvironmentProcessError("the environment process exited")
        try:
            reply = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EnvironmentProcessError(
                f"the environment process wrote something that is not a reply: {line[:200]!r}"
            ) from exc
        if not isinstance(reply, dict):
            raise EnvironmentProcessError("the environment process wrote something that is not a reply")
        if "error" in reply:
            raise EnvironmentProcessError(str(reply["error"]))
        return reply

    def reset(self, seed: int, *, timeout_s: float = RESET_TIMEOUT_S) -> str:
        """The first observation; ``EnvironmentProcessError`` when the environment cannot start."""
        reply = self.request({"op": "reset", "seed": seed}, timeout_s)
        try:
            return str(reply["observation"])[:OBSERVATION_CHARS]
        except KeyError as exc:
            raise EnvironmentProcessError("the environment process wrote a reply without an observation") from exc

    def step(self, action: str, *, timeout_s: float = STEP_TIMEOUT_S) -> tuple[str, float, bool, bool]:
        """One step: (observation, reward, terminated, truncated); a step that broke is a EnvironmentProcessError."""
        reply = self.request({"op": "step", "action": action}, timeout_s)
        if "step_error" in reply:
            raise EnvironmentProcessError(f"step: {reply['step_error']}")
        try:
            observation, reward, terminated, truncated = (
                reply["observation"],
                reply["reward"],
                reply["terminated"],
                reply["truncated"],
            )
        except KeyError as exc:
            raise EnvironmentProcessError(f"the environment process wrote a reply without {exc}") from exc
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise EnvironmentProcessError(f"the environment process wrote a reward that is not a number: {reward!r}")
        return str(observation)[:OBSERVATION_CHARS], float(reward), bool(terminated), bool(truncated)

    def close(self) -> None:
        """Kill the child's whole process group and remove its directory; safe to call more than once."""
        if self.process.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.process.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.process.wait(timeout=CLOSE_TIMEOUT_S)
        if self.process.stdin is not None:
            self.process.stdin.close()
        # A grandchild that kept the pipe open would block a close of the protocol; the reader keeps it instead.
        self.reader.join(timeout=1.0)
        if not self.reader.is_alive():
            self.protocol.close()
        self.temporary_directory.cleanup()

    def __enter__(self) -> EnvironmentProcess:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, value: BaseException | None, trace: TracebackType | None
    ) -> None:
        self.close()
