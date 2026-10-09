"""Root-only, one-shot command execution through private request and result files."""

from __future__ import annotations

import json
import math
import os
import pwd
import signal
import stat
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID


@dataclass(frozen=True)
class CommandRequest:
    nonce: str
    command: str
    cwd: str | None
    env: dict[str, str]
    user: str | int | None
    deadline: float | None

    @classmethod
    def read(cls, path: Path) -> CommandRequest:
        request = json.loads(path.read_text())
        nonce = request["nonce"]
        if not isinstance(nonce, str) or UUID(nonce).hex != path.parent.name:
            raise ValueError("Invalid command nonce")
        command, cwd, env, user, deadline = (request[key] for key in ("command", "cwd", "env", "user", "deadline"))
        if not isinstance(command, str) or (cwd is not None and not isinstance(cwd, str)):
            raise ValueError("Invalid command or working directory")
        if not isinstance(env, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) or "=" in key or "\0" in key + value
            for key, value in env.items()
        ):
            raise ValueError("Invalid command environment")
        if user is not None and (isinstance(user, bool) or not isinstance(user, (str, int))):
            raise ValueError("Invalid command user")
        if deadline is not None and (
            isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline)
        ):
            raise ValueError("Invalid command deadline")
        return cls(nonce, command, cwd, env, user, float(deadline) if deadline is not None else None)


def write_atomic(path: Path, record: dict[str, object]) -> None:
    """Publish a complete root-private JSON record without exposing partial writes."""
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(record, stream)
    os.replace(temporary, path)


def run(request_path: Path) -> None:
    """Execute once as the requested account; only root publishes completion."""
    if os.geteuid() != 0:
        raise PermissionError("Command runner requires root")
    for directory in (request_path.parent, request_path.parent.parent):
        metadata = directory.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) != 0o700:
            raise PermissionError("Command control directory must be root-private")
    metadata = request_path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise PermissionError("Command request must be a root-private regular file")
    request = CommandRequest.read(request_path)
    # Exclusive ownership prevents a repeated detached launch from executing twice.
    ownership = os.open(request_path.parent / "claimed", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(ownership)
    user = "root" if request.user is None else str(request.user)
    if user in ("root", "student"):
        account = pwd.getpwnam(user)
    elif user.isdecimal():
        account = pwd.getpwuid(int(user))
    else:
        raise ValueError("Command user must be root, student, or a registered numeric UID")
    environment = {**os.environ, **request.env}
    remaining_seconds = request.deadline - time.time() if request.deadline is not None else None
    if (request_path.parent / "cancel.json").exists() or (remaining_seconds is not None and remaining_seconds <= 0):
        return
    child = subprocess.Popen(
        ["bash", "-c", request.command],
        cwd=request.cwd,
        env=environment,
        user=account.pw_uid,
        group=account.pw_gid,
        extra_groups=os.getgrouplist(account.pw_name, account.pw_gid),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        close_fds=True,
    )
    write_atomic(request_path.parent / "started.json", {"nonce": request.nonce, "pid": child.pid})
    timed_out = False
    try:
        while True:
            if (request_path.parent / "cancel.json").exists():
                raise subprocess.TimeoutExpired(child.args, 0)
            remaining_seconds = request.deadline - time.time() if request.deadline is not None else None
            if remaining_seconds is not None and remaining_seconds <= 0:
                raise subprocess.TimeoutExpired(child.args, 0)
            try:
                stdout, stderr = child.communicate(
                    timeout=min(0.05, remaining_seconds) if remaining_seconds is not None else 0.05
                )
                break
            except subprocess.TimeoutExpired:
                continue
    except subprocess.TimeoutExpired:
        timed_out = True
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        try:
            stdout, stderr = child.communicate(timeout=0.1)
        except subprocess.TimeoutExpired as error:
            # A setsid descendant can retain pipes; the container still owns it.
            stdout, stderr = error.output or b"", error.stderr or b""
            if child.stdout is not None:
                child.stdout.close()
            if child.stderr is not None:
                child.stderr.close()
            child.wait()
    write_atomic(
        request_path.parent / "result.json",
        {
            "nonce": request.nonce,
            "pid": child.pid,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
            "return_code": child.returncode,
            "timed_out": timed_out,
            "completed_at": time.time(),
        },
    )


if __name__ == "__main__":
    try:
        run(Path(sys.argv[1]))
    except (OSError, ValueError, KeyError, TypeError):
        # Requests can contain secrets; launch diagnostics must not print them.
        sys.exit(1)
