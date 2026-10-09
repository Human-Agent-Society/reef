"""A stateful Python execution tool, run only inside the student Harbor sandbox."""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import signal
import socket
import traceback
from pathlib import Path

SOCKET_PATH = "/tmp/agentcl-kernel.sock"


class LimitedOutput(io.TextIOBase):
    """Bound captured stdout/stderr without stopping the student's computation."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.parts: list[str] = []
        self.length = 0

    def write(self, text: str) -> int:
        remaining = self.limit - self.length
        if remaining > 0:
            self.parts.append(text[:remaining])
            self.length += len(text[:remaining])
        return len(text)

    def text(self) -> str:
        return "".join(self.parts)


def execute(code: str, namespace: dict[str, object], timeout_seconds: int, output_limit: int) -> dict[str, object]:
    """Execute one action in the episode's namespace with a bounded wall clock."""
    output = LimitedOutput(output_limit)

    def timeout_handler(signum: int, frame: object) -> None:
        raise TimeoutError("code execution exceeded the tool time limit")

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    status = "ok"
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            exec(compile(code, "<student-tool>", "exec"), namespace)
    except TimeoutError:
        status = "timeout"
        output.write("\nCode execution timed out.\n")
    except (Exception, SystemExit, KeyboardInterrupt):
        status = "error"
        output.write(traceback.format_exc(limit=3))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    return {"status": status, "output": output.text()}


def serve() -> None:
    path = Path(SOCKET_PATH)
    path.unlink(missing_ok=True)
    namespace: dict[str, object] = {"__name__": "__student_tool__"}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(SOCKET_PATH)
        server.listen(1)
        while True:
            connection, _address = server.accept()
            with connection, connection.makefile("rb") as request_file:
                request = json.loads(request_file.readline(1_000_000))
                result = execute(request["code"], namespace, request["timeout_seconds"], request["output_limit"])
                connection.sendall((json.dumps(result) + "\n").encode())


def send(encoded_request: str) -> None:
    request = base64.b64decode(encoded_request, validate=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(SOCKET_PATH)
        connection.sendall(request + b"\n")
        with connection.makefile("rb") as response:
            print(response.readline(1_000_000).decode().rstrip("\n"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--serve", action="store_true")
    group.add_argument("--execute")
    arguments = parser.parse_args()
    if arguments.serve:
        serve()
    else:
        send(arguments.execute)


if __name__ == "__main__":
    main()
