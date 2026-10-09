"""Detached Podman commands with root-private, atomic completion records."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
import stat
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import override
from uuid import uuid4

from harbor.environments.base import ExecResult
from harbor.environments.docker.docker import _sanitize_docker_compose_project_name

from .command_runner import write_atomic
from .no_network import NoNetworkDockerEnvironment


class DetachedNoNetworkEnvironment(NoNetworkDockerEnvironment):
    """Keep Harbor's sandbox lifecycle; do not trust Podman's exec wait status."""

    control_temp_dir: tempfile.TemporaryDirectory[str] | None = None
    container_id: str | None = None
    active_commands: dict[asyncio.Task[object], asyncio.Future[None]] | None = None

    @override
    def _write_mounts_compose_file(self) -> Path:
        path = super()._write_mounts_compose_file()
        if os.geteuid() != 0:
            raise PermissionError("Detached transport requires a root-local Podman host")
        if self.control_temp_dir is None:
            self.control_temp_dir = tempfile.TemporaryDirectory(prefix="agentcl-control-", dir="/tmp")
            control = Path(self.control_temp_dir.name)
            control.chmod(0o700)
            shutil.copyfile(Path(__file__).with_name("command_runner.py"), control / "command_runner.py")
            (control / "command_runner.py").chmod(0o644)
        compose = json.loads(path.read_text())
        compose["services"]["main"].setdefault("volumes", []).append(
            {"type": "bind", "source": self.control_temp_dir.name, "target": "/agentcl-control", "read_only": False}
        )
        path.write_text(json.dumps(compose))
        return path

    async def native_process(self, arguments: list[str]) -> asyncio.subprocess.Process:
        """Start a native client without logging command or environment values."""
        return await asyncio.create_subprocess_exec(
            "/usr/bin/podman",
            *arguments,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            close_fds=True,
        )

    async def main_container_id(self, deadline: float | None) -> str:
        if self.container_id is not None:
            return self.container_id
        project = _sanitize_docker_compose_project_name(self.session_id)
        process = await self.native_process(
            [
                "ps",
                "--no-trunc",
                "--filter",
                f"label=com.docker.compose.project={project}",
                "--filter",
                "label=com.docker.compose.service=main",
                "--format",
                "{{.ID}}",
            ]
        )
        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=max(0, deadline - time.time()) if deadline is not None else None
            )
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        identifiers = stdout.decode().split()
        if process.returncode != 0 or len(identifiers) != 1 or re.fullmatch(r"[0-9a-f]{64}", identifiers[0]) is None:
            raise RuntimeError("Detached transport requires exactly one labeled main container")
        self.container_id = identifiers[0]
        return self.container_id

    @override
    async def _compose_exec(
        self,
        command: str,
        *,
        service: str,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout_sec: int | None,
        user: str | int | None,
    ) -> ExecResult:
        if service != "main":
            return await super()._compose_exec(
                command, service=service, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user
            )
        timeout_seconds = timeout_sec
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("Detached command timeout must be positive")
        deadline = time.time() + timeout_seconds if timeout_seconds is not None else None
        if self.control_temp_dir is None:
            raise RuntimeError("Detached command control mount has not been prepared")
        nonce = uuid4().hex
        job = Path(self.control_temp_dir.name) / nonce
        job.mkdir(mode=0o700)
        write_atomic(
            job / "request.json",
            {"nonce": nonce, "command": command, "cwd": cwd, "env": env or {}, "user": user, "deadline": deadline},
        )
        process: asyncio.subprocess.Process | None = None
        acknowledgement: asyncio.Task[tuple[bytes, bytes]] | None = None
        actual_timeout = False
        command_task = asyncio.current_task()
        if command_task is None:
            raise RuntimeError("Detached command requires an active task")
        if self.active_commands is None:
            self.active_commands = {}
        command_finished: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.active_commands[command_task] = command_finished
        try:
            container_id = await self.main_container_id(deadline)
            process = await self.native_process(
                [
                    "exec",
                    "--detach",
                    "--user",
                    "root",
                    container_id,
                    "/usr/local/bin/python3",
                    "-I",
                    "-S",
                    "/agentcl-control/command_runner.py",
                    f"/agentcl-control/{nonce}/request.json",
                ]
            )
            acknowledgement = asyncio.create_task(process.communicate())
            while True:
                started_path, result_path = job / "started.json", job / "result.json"
                if started_path.exists() and result_path.exists():
                    for status_path in (started_path, result_path):
                        metadata = status_path.lstat()
                        if (
                            not stat.S_ISREG(metadata.st_mode)
                            or metadata.st_uid != 0
                            or stat.S_IMODE(metadata.st_mode) != 0o600
                        ):
                            raise ValueError("Detached completion must be a root-private regular file")
                    started, result = json.loads(started_path.read_text()), json.loads(result_path.read_text())
                    pid = started["pid"]
                    if (
                        started["nonce"] != nonce
                        or result["nonce"] != nonce
                        or type(pid) is not int
                        or pid <= 1
                        or type(result["pid"]) is not int
                        or result["pid"] != pid
                        or type(result["return_code"]) is not int
                        or type(result["timed_out"]) is not bool
                        or not isinstance(result["stdout"], str)
                        or not isinstance(result["stderr"], str)
                    ):
                        raise ValueError("Invalid detached command completion")
                    if result["timed_out"]:
                        actual_timeout = True
                        raise TimeoutError("Detached child command timed out")
                    completed_at = result["completed_at"]
                    if type(completed_at) not in (int, float) or not math.isfinite(completed_at):
                        raise ValueError("Invalid detached completion time")
                    if deadline is not None and completed_at > deadline:
                        raise TimeoutError("Detached completion exceeded the command deadline")
                    output = self._output_callback()
                    if output is not None:
                        await output(result["stdout"], "stdout")
                        await output(result["stderr"], "stderr")
                    return ExecResult(
                        stdout=result["stdout"] or None,
                        stderr=result["stderr"] or None,
                        return_code=result["return_code"],
                    )
                if deadline is not None and time.time() >= deadline:
                    raise TimeoutError("Detached command outcome is unknown")
                await asyncio.sleep(min(0.01, max(0, deadline - time.time())) if deadline is not None else 0.01)
        except (TimeoutError, ValueError, KeyError, TypeError) as error:
            # A missing or invalid result is never permission to replay the command.
            with suppress(RuntimeError):
                await self._run_docker_compose_command(
                    ["down", "--volumes", "--remove-orphans"], check=False, timeout_sec=2
                )
            if not isinstance(error, TimeoutError):
                raise RuntimeError("Invalid detached command completion; outcome is unknown") from None
            message = f"Command timed out after {timeout_seconds} seconds"
            if not actual_timeout:
                message += "; detached outcome is unknown"
            raise RuntimeError(message) from None
        finally:
            try:
                if not (job / "result.json").exists():
                    write_atomic(job / "cancel.json", {"nonce": nonce})
                if process is not None and process.returncode is None:
                    with suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()
                if acknowledgement is not None:
                    await acknowledgement
            finally:
                self.active_commands.pop(command_task)
                command_finished.set_result(None)

    @override
    async def stop(self, delete: bool) -> None:
        async def retire() -> None:
            commands = tuple((self.active_commands or {}).items())
            for command, _finished in commands:
                command.cancel()
            if commands:
                await asyncio.gather(*(finished for _command, finished in commands))
            await super(DetachedNoNetworkEnvironment, self).stop(delete)
            self.container_id = None
            if self.control_temp_dir is not None:
                self.control_temp_dir.cleanup()
                self.control_temp_dir = None

        # Keep the request mount until command finalizers and container teardown finish.
        teardown = asyncio.create_task(retire())
        cancellation: asyncio.CancelledError | None = None
        while not teardown.done():
            try:
                await asyncio.shield(teardown)
            except asyncio.CancelledError as error:
                if teardown.cancelled():
                    raise
                cancellation = error
        teardown.result()
        if cancellation is not None:
            raise cancellation
