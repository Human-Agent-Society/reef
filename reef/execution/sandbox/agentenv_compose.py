"""Run a task's Compose project in one AgentENV sandbox, with ``aenv compose up``.

A Harbor task can name more services next to ``main`` in ``environment/docker-compose.yaml``: ``main`` is the
container of the agent, and the other services are what the agent works against, such as the server of a CTF
challenge. An E2B sandbox is one machine of one image, so Harbor's E2B environment starts only the task's image, and
the agent reaches no other service. AgentENV can run a Compose project inside one sandbox: ``aenv compose up`` starts
a guest that runs Docker, attaches the image of each service as a drive, and starts the services on the project's
networks. The services then reach one another by name, as under Docker Compose.

This module holds the parts of that start that need no sandbox: the project that the sandbox runs
(``compose_project``), the name of the image of ``main`` (``main_image``), and the ``aenv`` commands
(``ComposeSandbox``, ``prepare_guest``, ``delete_sandbox``). ``KillOnTimeoutE2BEnvironment``
(``reef.execution.sandbox.e2b``) uses it for a task with a Compose file when ``REEF_E2B_COMPOSE`` is ``aenv``, and then
runs the trial's commands in ``main`` with ``AgentEnvComposeOps``.

A deployment selects this with ``REEF_E2B_COMPOSE`` (``compose_mode``), and the default is ``off``. ``aenv build
--image`` and ``aenv compose up`` write image tags and image descriptions to the repository of the AgentENV
deployment, which a deployment without Compose tasks never has. A deployment that manages that repository with its
own tools must make them read these records first.

The ``aenv`` CLI and its planner ``aenv-compose-plan`` must be on ``PATH``, in one directory.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import logging
import os
import re
import shlex
import tarfile
import tempfile
from abc import abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import override
from urllib.parse import quote

import aiohttp
import tomli_w
import yaml
from harbor.constants import MAIN_SERVICE_NAME
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.dind_compose import DinDComposeOps

from reef.core.errors import InternalError
from reef.execution.sandbox.agentenv_build import BUILD_OUTPUT_TAIL_CHARS, TRANSIENT_STATUSES, AgentEnvTemplates
from reef.execution.sandbox.process import SandboxCapacityError

logger = logging.getLogger(__name__)

#: The variable that selects how Reef's Harbor E2B class runs a task with a Compose file: ``off`` (only the task's own
#: image, as Harbor's class does; the default) or ``aenv`` (the project, with ``aenv compose up``).
COMPOSE_VARIABLE = "REEF_E2B_COMPOSE"
COMPOSE_MODES = ("off", "aenv")
#: The file of a task's environment that names its Compose services, as Harbor reads it.
COMPOSE_FILE = "docker-compose.yaml"
#: The Compose project name that ``aenv compose up`` gives every project.
PROJECT_NAME = "aenv"
#: The command of ``main`` when the task's Compose file sets none: Harbor's, which keeps the container running.
MAIN_COMMAND = ("sh", "-c", "sleep infinity")
#: The part of ``aenv compose up``'s error when the deployment has no image under a name of its own repository.
IMAGE_NOT_FOUND_TEXT = "was not found"
#: The guest of ``aenv compose up`` has no bash, and the E2B SDK starts each command with ``/bin/bash -l -c``. This
#: command writes a ``/bin/bash`` that runs the guest's ``sh`` with the same arguments, unless the guest has one.
GUEST_BASH_COMMAND = (
    "[ -x /bin/bash ] || { printf '%s\\n' '#!/bin/sh' 'exec /bin/sh \"$@\"' > /bin/bash && chmod 755 /bin/bash; }"
)


class ComposeStartError(InternalError):
    """``aenv compose up``, or the preparation of its guest, failed. The message has the tail of the output."""


class ImageNotFoundError(ComposeStartError):
    """``aenv compose up`` found no image under a name of the deployment's own repository."""


class TransientStartError(ComposeStartError, SandboxCapacityError):
    """``aenv compose up`` received a status of ``TRANSIENT_STATUSES`` from the AgentENV API: it has no capacity now."""


def compose_mode(environ: Mapping[str, str]) -> str:
    """The value of ``COMPOSE_VARIABLE`` in ``environ``: one of ``COMPOSE_MODES``, ``off`` when it is unset or empty.
    Raise ``ValueError`` for any other value."""
    value = environ.get(COMPOSE_VARIABLE, "").strip() or "off"
    if value not in COMPOSE_MODES:
        raise ValueError(f"{COMPOSE_VARIABLE} must be one of {', '.join(COMPOSE_MODES)}, not {value!r}")
    return value


def main_image(alias: str) -> tuple[str, str]:
    """The image of ``main`` for the environment whose template alias is ``alias``: the name that ``aenv build
    --image`` takes, and the reference that a Compose project gives ``aenv compose up``.

    An image name has lowercase letters, digits and single separators, and an alias can have other characters. So
    the name is the alias in that form, then the first 16 hex digits of the alias's SHA-256: two aliases have two
    names.
    """
    if not alias:
        raise ValueError("the image of a Compose project's main service needs a template alias")
    readable = re.sub(r"[^a-z0-9]+", "-", alias.lower()).strip("-")
    name = f"{readable}-{hashlib.sha256(alias.encode()).hexdigest()[:16]}".lstrip("-")
    return name, f"aenv/{name}:latest"


def compose_project(compose_text: str, image: str, environment: Mapping[str, str]) -> dict[str, object]:
    """The Compose project that a sandbox runs for a task whose Compose file has the text ``compose_text``.

    It is the task's project with these changes to its ``main`` service, which Harbor makes with its own Compose
    files under Docker: the image is ``image``; a ``build`` is removed, because ``image`` is the built task image;
    the command is ``MAIN_COMMAND`` when the file sets none; and each variable of ``environment`` is set, over a
    variable of the file with the same name. A project without ``main`` gets one.

    Raise ``ValueError`` when the text is not a Compose mapping with a ``services`` mapping, or when a service other
    than ``main`` has a ``build``: ``aenv compose up`` runs images only, and only the task's Dockerfile is built.
    """
    project = yaml.safe_load(compose_text)
    services = project.get("services") if isinstance(project, dict) else None
    if not isinstance(services, dict):
        raise ValueError(f"{COMPOSE_FILE} is not a Compose project with a services mapping")
    for name, service in services.items():
        if name != MAIN_SERVICE_NAME and isinstance(service, dict) and "build" in service:
            raise ValueError(
                f"the service {name} of {COMPOSE_FILE} has a build: an AgentENV sandbox runs a service from its image"
            )
    main = services.get(MAIN_SERVICE_NAME)
    if main is None:
        main = {}
    elif not isinstance(main, dict):
        raise ValueError(f"the service {MAIN_SERVICE_NAME} of {COMPOSE_FILE} is not a mapping")
    main = {key: value for key, value in main.items() if key != "build"}
    main["image"] = image
    main.setdefault("command", list(MAIN_COMMAND))
    declared = main.get("environment") or {}
    if isinstance(declared, list):
        # The list form: NAME=value, or NAME alone for a variable that Compose takes from its own environment.
        pairs = (str(entry).partition("=") for entry in declared)
        declared = {name: value if separator else None for name, separator, value in pairs}
    elif not isinstance(declared, dict):
        raise ValueError(f"the environment of the service {MAIN_SERVICE_NAME} is not a list or a mapping")
    if declared or environment:
        main["environment"] = {**declared, **environment}
    return {**project, "services": {**services, MAIN_SERVICE_NAME: main}}


@contextlib.contextmanager
def aenv_environment(templates: AgentEnvTemplates) -> Iterator[dict[str, str]]:
    """The process environment of an ``aenv`` command on ``templates``' deployment, for the time of the block.
    ``aenv`` reads the deployment from a credentials file in a new configuration directory, which the block removes."""
    with tempfile.TemporaryDirectory(prefix="reef-aenv-") as config_home:
        credentials = Path(config_home) / "aenv" / "credentials"
        credentials.parent.mkdir(mode=0o700)
        credentials.write_text(tomli_w.dumps({"url": templates.api_url, "api_key": templates.api_key}))
        credentials.chmod(0o600)
        yield {**os.environ, "XDG_CONFIG_HOME": config_home}


async def run_aenv(
    arguments: Sequence[str], templates: AgentEnvTemplates, standard_input: bytes = b""
) -> tuple[int, str, str]:
    """Run ``aenv`` with ``arguments`` on ``templates``' deployment, with ``standard_input`` as its input. Return its
    exit code, its standard output and its standard error. A cancelled run kills the process."""
    with aenv_environment(templates) as environment:
        process = await asyncio.create_subprocess_exec(
            "aenv",
            *arguments,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
        )
        try:
            output, errors = await process.communicate(standard_input)
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise
    return process.returncode or 0, output.decode(errors="replace"), errors.decode(errors="replace")


@dataclass(frozen=True)
class ComposeSandbox:
    """One ``aenv compose up`` of a Compose project.

    Values: ``project`` is the project, with an image for each service (``compose_project``). ``cpu_count`` and
    ``memory_mb`` are the sandbox's resources, None for AgentENV's default; all services share them.
    ``lifetime_seconds`` is the sandbox's lifetime once the services run, and ``startup_seconds`` the limit of the
    start: Docker's readiness, the registration of the images and the wait for the services. Both are at least 1.
    Purpose: ``KillOnTimeoutE2BEnvironment`` makes one for a trial of a task with a Compose file and starts it.
    """

    project: Mapping[str, object]
    cpu_count: int | None
    memory_mb: int | None
    lifetime_seconds: int
    startup_seconds: int

    def __post_init__(self) -> None:
        if self.lifetime_seconds < 1 or self.startup_seconds < 1:
            raise ValueError("a Compose sandbox needs a lifetime and a startup limit of at least 1 s")

    def arguments(self) -> list[str]:
        """The arguments of ``aenv`` for this start. The project is the command's standard input."""
        arguments = ["compose", "up", "--file", "-", "--timeout", str(self.lifetime_seconds)]
        arguments += ["--startup-timeout", str(self.startup_seconds)]
        if self.cpu_count is not None:
            arguments += ["--cpu", str(self.cpu_count)]
        if self.memory_mb is not None:
            arguments += ["--memory", str(self.memory_mb)]
        return arguments

    async def start(self, templates: AgentEnvTemplates) -> str:
        """Start the sandbox on ``templates``' deployment and return its ID, when every service runs. Raise
        ``ImageNotFoundError`` when the deployment has no image under a name of its own repository,
        ``TransientStartError`` for a temporary API failure, and ``ComposeStartError`` for any other failure.
        ``aenv`` deletes the sandbox of a start that fails."""
        code, output, errors = await run_aenv(self.arguments(), templates, json.dumps(self.project).encode())
        logger.debug("aenv compose up:\n%s", errors)
        if code == 0:
            # aenv prints only the sandbox ID on its standard output.
            return output.strip()
        message = f"aenv compose up failed with exit code {code}: {errors[-BUILD_OUTPUT_TAIL_CHARS:]}"
        error = ("\n" + errors).rpartition("\nError: ")[2]
        if error.startswith("HTTP ") and IMAGE_NOT_FOUND_TEXT in error:
            raise ImageNotFoundError(message)
        if any(error.startswith(f"HTTP {status}:") for status in TRANSIENT_STATUSES):
            raise TransientStartError(message)
        raise ComposeStartError(message)


async def prepare_guest(templates: AgentEnvTemplates, sandbox_id: str) -> None:
    """Give the guest of the Compose sandbox ``sandbox_id`` a ``/bin/bash`` (``GUEST_BASH_COMMAND``), so that the
    E2B SDK can run commands in it. Raise ``ComposeStartError`` when the command fails."""
    code, _, errors = await run_aenv(["exec", sandbox_id, "--", "sh", "-c", GUEST_BASH_COMMAND], templates)
    if code != 0:
        raise ComposeStartError(
            f"could not prepare the guest of the sandbox {sandbox_id} (exit code {code}): "
            f"{errors[-BUILD_OUTPUT_TAIL_CHARS:]}"
        )


async def delete_sandbox(templates: AgentEnvTemplates, sandbox_id: str) -> None:
    """Delete the sandbox ``sandbox_id``. A sandbox that is gone already counts as deleted."""
    url = f"{templates.api_url}/sandboxes/{quote(sandbox_id, safe='')}"
    async with (
        aiohttp.ClientSession(headers={"X-API-Key": templates.api_key}) as session,
        session.delete(url) as response,
    ):
        if response.status != 404:
            response.raise_for_status()


class ComposeHost(BaseEnvironment):
    """A Harbor environment whose sandbox is the guest of a Compose project: the machine that runs Docker, not the
    ``main`` container.

    Values: those of ``BaseEnvironment``. Each operation below acts on the guest's own filesystem and processes.
    Purpose: ``AgentEnvComposeOps`` reaches the guest through these operations, and ``KillOnTimeoutE2BEnvironment``
    implements them with the E2B SDK.
    """

    @abstractmethod
    async def host_exec(self, command: str, timeout_sec: int | None = None) -> ExecResult:
        """Run the shell command ``command`` in the guest as root, for at most ``timeout_sec`` seconds (no limit for
        None), and return its result."""

    @abstractmethod
    async def host_write(self, path: str, data: bytes) -> None:
        """Write ``data`` to the file ``path`` of the guest, with its parent directories."""

    @abstractmethod
    async def host_read(self, path: str) -> bytes:
        """The bytes of the file ``path`` of the guest."""


class AgentEnvComposeOps(DinDComposeOps):
    """Harbor's operations on the ``main`` service of a Compose project, for a project that ``aenv compose up``
    started.

    Values: ``host`` is the environment of the project's guest. Harbor's ``DinDComposeOps`` reads it as ``_env``.
    Purpose: ``KillOnTimeoutE2BEnvironment`` gives each command and file transfer of a trial to one, and it runs
    them in ``main`` with ``docker compose`` in the guest. A directory moves between this host and the guest as one
    tar archive, so an empty directory and a file's mode keep.
    """

    # Harbor's local Docker environment runs a command of ``main`` with ``bash -c``, which keeps the image's PATH.
    _MAIN_LOGIN_SHELL = False

    def __init__(self, host: ComposeHost) -> None:
        self._env = host
        self.host = host

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        *,
        service: str | None = None,
    ) -> ExecResult:
        # Docker finds a user name in the container's /etc/passwd, which the agent can change. Root goes by its UID.
        user = 0 if user == "root" else user
        return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user, service=service)

    @override
    async def _compose_exec(self, subcommand: list[str], timeout_sec: int | None = None) -> ExecResult:
        command = ["docker", "compose", "--project-name", PROJECT_NAME, *subcommand]
        return await self.host.host_exec(shlex.join(command), timeout_sec=timeout_sec)

    @override
    async def _host_exec(self, command: str, timeout_sec: int | None = None) -> ExecResult:
        return await self.host.host_exec(command, timeout_sec=timeout_sec)

    @override
    async def _stage_file_to_host(self, source_path: Path | str, host_path: str) -> None:
        await self.host.host_write(host_path, Path(source_path).read_bytes())

    @override
    async def _stage_dir_to_host(self, source_dir: Path | str, host_dir: str) -> None:
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            tar.add(str(source_dir), arcname=".")
        await self.host.host_write(f"{host_dir}.tar", archive.getvalue())
        directory, packed = shlex.quote(host_dir), shlex.quote(f"{host_dir}.tar")
        result = await self.host.host_exec(
            f"mkdir -p {directory} && tar -xf {packed} -C {directory}; code=$?; rm -f {packed}; exit $code",
            timeout_sec=self._CP_DIR_TIMEOUT_SEC,
        )
        if result.return_code != 0:
            raise RuntimeError(f"could not unpack {source_dir} in the Compose guest: {result.stdout} {result.stderr}")

    @override
    async def _fetch_file_from_host(self, host_path: str, target_path: Path | str) -> None:
        Path(target_path).write_bytes(await self.host.host_read(host_path))

    @override
    async def _fetch_dir_from_host(self, host_dir: str, target_dir: Path | str) -> None:
        directory, packed = shlex.quote(host_dir), shlex.quote(f"{host_dir}.tar")
        result = await self.host.host_exec(f"tar -cf {packed} -C {directory} .", timeout_sec=self._CP_DIR_TIMEOUT_SEC)
        if result.return_code != 0:
            raise RuntimeError(f"could not pack {host_dir} in the Compose guest: {result.stdout} {result.stderr}")
        try:
            data = await self.host.host_read(f"{host_dir}.tar")
        finally:
            await self.host.host_exec(f"rm -f {packed}", timeout_sec=10)
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            # The archive is the agent's files: the data filter refuses a path or a link that leaves target_dir.
            tar.extractall(target_dir, filter="data")


__all__ = [
    "COMPOSE_FILE",
    "COMPOSE_MODES",
    "COMPOSE_VARIABLE",
    "GUEST_BASH_COMMAND",
    "MAIN_COMMAND",
    "PROJECT_NAME",
    "AgentEnvComposeOps",
    "ComposeHost",
    "ComposeSandbox",
    "ComposeStartError",
    "ImageNotFoundError",
    "TransientStartError",
    "compose_mode",
    "compose_project",
    "delete_sandbox",
    "main_image",
    "prepare_guest",
    "run_aenv",
]
