"""Build a task's Dockerfile into an AgentENV template with AgentENV's BuildKit build, ``aenv build``.

AgentENV serves E2B's template API, but its E2B template builds take no files: it refuses each COPY step
("template build filesHash/COPY support is not implemented yet"). E2B's Dockerfile parser also refuses a multi-stage
Dockerfile. ``aenv build`` builds the Dockerfile with BuildKit in an AgentENV worker instead, with Docker's
semantics, and publishes the image as a template under an alias.

AgentENV binds the alias to a new template when a build starts, and keeps the binding when the build fails. It refuses
another build under that alias until the template is deleted. So ``ensure_aenv_template`` reads the state of the alias
before it builds: it uses a ready template, waits for a build that another trial runs, and deletes a failed template
before it builds again.

``aenv build`` makes the template, in the status ``waiting``, before it starts the build. An ``aenv build`` that stops
then, killed or refused by an overloaded API, leaves the template ``waiting`` for good, and the alias bound to it. So a
template that stays ``waiting`` for ``ORPHAN_WAITING_S`` is an orphan: ``ensure_aenv_template`` deletes it and builds.

``KillOnTimeoutE2BEnvironment`` (``reef.execution.sandbox.e2b``) uses this module when ``REEF_E2B_DOCKERFILE_BUILD`` is
``aenv``. The ``aenv`` CLI and its ``aenv-buildctl`` must be on ``PATH``.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self
from urllib.parse import quote

import aiohttp
import tomli_w

from reef.core.errors import InternalError

logger = logging.getLogger(__name__)

#: The variable that selects how Reef's Harbor E2B class builds a task's Dockerfile: ``e2b`` (E2B's template API, the
#: default) or ``aenv`` (``aenv build`` on an AgentENV deployment).
DOCKERFILE_BUILD_VARIABLE = "REEF_E2B_DOCKERFILE_BUILD"
DOCKERFILE_BUILDS = ("e2b", "aenv")
#: How long a wait for another trial's build sleeps between two reads of the build's status, on average: each sleep
#: is from half to one and a half times it, so that the trials that wait for one build do not read it together.
BUILD_POLL_INTERVAL_S = 15.0
#: How long a template can stay ``waiting`` before it counts as an orphan: ``aenv build`` starts its build at once
#: after it makes the template, so a live build leaves ``waiting`` within seconds.
ORPHAN_WAITING_S = 180.0
#: The HTTP statuses of an AgentENV API that is busy, down or overloaded for a while: a later request can pass. Each
#: AgentENV worker runs a fixed number of builds at once, and refuses another with 429 ("node concurrent build limit
#: reached; retry later").
TRANSIENT_STATUSES = (429, 502, 503, 504)
#: The statuses of an AgentENV template build, with E2B's names.
BUILD_STATUSES = ("waiting", "building", "ready", "error")
#: The part of ``aenv build``'s error when another build holds the alias (AgentENV's ``AliasConflict``).
ALIAS_CONFLICT_TEXT = "already points to"
#: How much of ``aenv build``'s output an error keeps.
BUILD_OUTPUT_TAIL_CHARS = 4000


def dockerfile_build(environ: Mapping[str, str]) -> str:
    """The value of ``DOCKERFILE_BUILD_VARIABLE`` in ``environ``: one of ``DOCKERFILE_BUILDS``, ``e2b`` when it is
    unset or empty. Raise ``ValueError`` for any other value."""
    value = environ.get(DOCKERFILE_BUILD_VARIABLE, "").strip() or "e2b"
    if value not in DOCKERFILE_BUILDS:
        raise ValueError(f"{DOCKERFILE_BUILD_VARIABLE} must be one of {', '.join(DOCKERFILE_BUILDS)}, not {value!r}")
    return value


def dockerfile_sets_start_command(dockerfile: str) -> bool:
    """Whether the last stage of ``dockerfile`` has a CMD or an ENTRYPOINT.

    E2B's template build starts only the command that the Dockerfile itself names. ``aenv build`` starts the image's
    ENTRYPOINT and CMD, which a base image can set. So a build of a Dockerfile without its own command disables the
    start command, as E2B's build has none.
    """
    # Harbor's e2b extra installs dockerfile-parse.
    from dockerfile_parse import DockerfileParser

    instructions = [
        entry["instruction"] for entry in DockerfileParser(fileobj=io.BytesIO(dockerfile.encode())).structure
    ]
    last_stage = len(instructions) - instructions[::-1].index("FROM") if "FROM" in instructions else 0
    return any(instruction in ("CMD", "ENTRYPOINT") for instruction in instructions[last_stage:])


class AgentEnvBuildError(InternalError):
    """``aenv build`` failed. The message has the tail of its output."""


class AliasConflictError(AgentEnvBuildError):
    """``aenv build`` found its alias bound to the template of another build."""


class TransientBuildError(AgentEnvBuildError):
    """``aenv build`` failed on a status of ``TRANSIENT_STATUSES`` from the AgentENV API: a later build can pass."""


@dataclass(frozen=True)
class TemplateState:
    """The template that an alias names on an AgentENV deployment.

    Values: ``template_id`` is the template's ID, which is also the ID of its one build. ``status`` is one of
    ``BUILD_STATUSES``. ``reason`` is the reason that AgentENV gives for an ``error`` status, else empty.
    Purpose: ``AgentEnvTemplates.state`` makes one; ``ensure_aenv_template`` decides from it.
    """

    template_id: str
    status: str
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.template_id:
            raise ValueError("a template state needs a template ID")
        if self.status not in BUILD_STATUSES:
            raise ValueError(f"unknown template build status {self.status!r} of template {self.template_id}")


@dataclass(frozen=True)
class AgentEnvTemplates:
    """The template endpoints of one AgentENV deployment.

    Values: ``api_url`` is the deployment's API URL without a trailing slash, and ``api_key`` its API key. Both are
    non-empty. The E2B SDK reads them from ``E2B_API_URL`` and ``E2B_API_KEY``.
    Purpose: ``ensure_aenv_template`` reads the state of an alias and deletes a failed template through it, and
    ``DockerfileBuild.run`` gives both values to ``aenv build``.
    """

    api_url: str
    api_key: str

    def __post_init__(self) -> None:
        if not self.api_url or self.api_url.endswith("/"):
            raise ValueError("an AgentENV API URL must be non-empty and have no trailing slash")
        if not self.api_key:
            raise ValueError("an AgentENV deployment needs an API key")

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> Self:
        """The deployment of ``E2B_API_URL`` and ``E2B_API_KEY`` in ``environ``. Raise ``ValueError`` when either
        is unset: an ``aenv`` build needs the deployment's own URL, not E2B's default."""
        api_url, api_key = environ.get("E2B_API_URL", "").strip(), environ.get("E2B_API_KEY", "").strip()
        if not api_url or not api_key:
            raise ValueError(f"{DOCKERFILE_BUILD_VARIABLE}=aenv needs E2B_API_URL and E2B_API_KEY")
        return cls(api_url.rstrip("/"), api_key)

    async def state(self, alias: str) -> TemplateState | None:
        """The template that ``alias`` names, or None when no template has the alias."""
        async with aiohttp.ClientSession(headers={"X-API-Key": self.api_key}) as session:
            async with session.get(f"{self.api_url}/templates/aliases/{quote(alias, safe='')}") as response:
                if response.status == 404:
                    return None
                response.raise_for_status()
                template_id = (await response.json())["templateID"]
            path = f"/templates/{quote(template_id, safe='')}/builds/{quote(template_id, safe='')}/status"
            async with session.get(f"{self.api_url}{path}") as response:
                if response.status == 404:
                    # Deleted after the alias was read.
                    return None
                response.raise_for_status()
                status = await response.json()
        reason = status.get("reason") or {}
        return TemplateState(template_id, status["status"], reason.get("message") or "")

    async def delete(self, template_id: str) -> None:
        """Delete the template ``template_id``. A template that is gone already counts as deleted."""
        url = f"{self.api_url}/templates/{quote(template_id, safe='')}"
        async with (
            aiohttp.ClientSession(headers={"X-API-Key": self.api_key}) as session,
            session.delete(url) as response,
        ):
            if response.status != 404:
                response.raise_for_status()


@dataclass(frozen=True)
class DockerfileBuild:
    """One ``aenv build`` of a task's Dockerfile into a template.

    Values: ``context_dir`` is the build context and ``dockerfile`` the Dockerfile's path. ``alias`` is the template's
    alias. ``cpu_count`` and ``memory_mb`` are the template's resources, None for AgentENV's default.
    ``timeout_seconds`` is the build's deadline, from 1 to 86400 seconds as ``aenv build`` takes it.
    ``start_command`` is False when the template starts no command (``dockerfile_sets_start_command``).
    Purpose: ``ensure_aenv_template`` runs it when the alias names no template.
    """

    context_dir: Path
    dockerfile: Path
    alias: str
    cpu_count: int | None
    memory_mb: int | None
    timeout_seconds: int
    start_command: bool

    def __post_init__(self) -> None:
        if not self.alias:
            raise ValueError("an aenv build needs a template alias")
        if not 1 <= self.timeout_seconds <= 86400:
            raise ValueError("an aenv build's timeout must be from 1 to 86400 seconds")

    def command(self) -> list[str]:
        """The ``aenv build`` command line."""
        command = ["aenv", "build", str(self.context_dir), "--file", str(self.dockerfile), "--name", self.alias]
        command += ["--progress", "plain"]
        command += ["--timeout", str(self.timeout_seconds)]
        if self.cpu_count is not None:
            command += ["--cpu", str(self.cpu_count)]
        if self.memory_mb is not None:
            command += ["--memory", str(self.memory_mb)]
        if not self.start_command:
            command += ["--start-cmd", ""]
        return command

    async def run(self, templates: AgentEnvTemplates) -> None:
        """Run ``aenv build`` until the template is ready. ``aenv`` reads the deployment from a credentials file in a
        new configuration directory. Raise ``AliasConflictError`` when another build holds the alias, and
        ``AgentEnvBuildError`` for any other failure. A cancelled run kills the ``aenv`` process; the deployment then
        ends the build at its deadline."""
        with tempfile.TemporaryDirectory(prefix="reef-aenv-") as config_home:
            credentials = Path(config_home) / "aenv" / "credentials"
            credentials.parent.mkdir(mode=0o700)
            credentials.write_text(tomli_w.dumps({"url": templates.api_url, "api_key": templates.api_key}))
            credentials.chmod(0o600)
            process = await asyncio.create_subprocess_exec(
                *self.command(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "XDG_CONFIG_HOME": config_home},
            )
            try:
                output, _ = await process.communicate()
            except asyncio.CancelledError:
                process.kill()
                await process.wait()
                raise
        text = output.decode(errors="replace")
        logger.debug("aenv build of %s:\n%s", self.alias, text)
        if process.returncode == 0:
            return
        tail = text[-BUILD_OUTPUT_TAIL_CHARS:]
        if ALIAS_CONFLICT_TEXT in text:
            raise AliasConflictError(f"another build holds the alias {self.alias}: {tail}")
        message = f"aenv build of {self.alias} failed with exit code {process.returncode}: {tail}"
        if any(f"HTTP {status}" in text for status in TRANSIENT_STATUSES):
            raise TransientBuildError(message)
        raise AgentEnvBuildError(message)


def poll_interval() -> float:
    """One sleep between two reads of a build's status: from half to one and a half times ``BUILD_POLL_INTERVAL_S``."""
    return BUILD_POLL_INTERVAL_S * random.uniform(0.5, 1.5)


def is_transient(error: aiohttp.ClientError | asyncio.TimeoutError) -> bool:
    """Whether ``error`` of a request to the AgentENV API can pass later: a status of ``TRANSIENT_STATUSES``, or no
    answer at all."""
    if isinstance(error, aiohttp.ClientResponseError):
        return error.status in TRANSIENT_STATUSES
    return isinstance(error, (aiohttp.ClientConnectionError, asyncio.TimeoutError))


async def ensure_aenv_template(templates: AgentEnvTemplates, build: DockerfileBuild) -> None:
    """Return when ``build.alias`` names a ready template on ``templates``' deployment.

    A ready template is used as it is. While another trial builds the template, this waits for it, at most
    ``build.timeout_seconds``, and then raises ``TimeoutError``. A failed template is deleted, and the alias is built
    again; so is a template that this wait saw ``waiting``, unchanged, for ``ORPHAN_WAITING_S`` (an orphan). When the
    alias names no template, ``build`` runs; when another trial starts a build under the alias first, this waits for
    that build. A failure of this trial's own build deletes the ``waiting`` template that it left. A build or a read
    that the API refuses for a while (``is_transient``, ``TransientBuildError``) is tried again until the deadline; any
    other failure of this trial's own build raises its ``AgentEnvBuildError``.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + build.timeout_seconds
    #: The waiting template that this wait saw first, and when.
    waiting_since: tuple[str, float] | None = None
    while True:
        try:
            state = await templates.state(build.alias)
        except (TimeoutError, aiohttp.ClientError) as error:
            if not is_transient(error):
                raise
            if loop.time() >= deadline:
                raise TimeoutError(f"the AgentENV API did not answer for the template {build.alias}") from error
            logger.warning("could not read the template %s; reading it again: %s", build.alias, error)
            await asyncio.sleep(poll_interval())
            continue
        if state is not None and state.status == "ready":
            return
        if loop.time() >= deadline:
            status = "missing" if state is None else state.status
            raise TimeoutError(f"the template {build.alias} was {status} after {build.timeout_seconds} s")
        if state is None:
            try:
                await build.run(templates)
                return
            except AliasConflictError:
                logger.info("another trial builds the template %s; waiting for it", build.alias)
            except AgentEnvBuildError as error:
                await delete_orphan(templates, build.alias)
                if not isinstance(error, TransientBuildError):
                    raise
                logger.warning("the AgentENV API refused the build of %s; building it again: %s", build.alias, error)
            await asyncio.sleep(poll_interval())
            continue
        orphan = False
        if state.status == "waiting":
            if waiting_since is None or waiting_since[0] != state.template_id:
                waiting_since = (state.template_id, loop.time())
            orphan = loop.time() - waiting_since[1] >= ORPHAN_WAITING_S
        if state.status == "error" or orphan:
            reason = state.reason if state.status == "error" else f"waiting for {ORPHAN_WAITING_S:.0f} s"
            logger.warning("deleting the template %s (%s): %s", build.alias, state.template_id, reason)
            try:
                await templates.delete(state.template_id)
                continue
            except (TimeoutError, aiohttp.ClientError) as error:
                if not is_transient(error):
                    raise
                logger.warning("could not delete the template %s; trying again: %s", build.alias, error)
        await asyncio.sleep(poll_interval())


async def delete_orphan(templates: AgentEnvTemplates, alias: str) -> None:
    """Delete the template that ``alias`` names when it is ``waiting``: a failed ``aenv build`` leaves it so. A read or a
    delete that fails is logged, and the orphan check of a later wait deletes the template."""
    try:
        state = await templates.state(alias)
        if state is not None and state.status == "waiting":
            await templates.delete(state.template_id)
    except (TimeoutError, aiohttp.ClientError) as error:
        logger.warning("could not delete the template %s that a failed build left: %s", alias, error)


__all__ = [
    "DOCKERFILE_BUILDS",
    "DOCKERFILE_BUILD_VARIABLE",
    "AgentEnvBuildError",
    "AgentEnvTemplates",
    "AliasConflictError",
    "DockerfileBuild",
    "TemplateState",
    "TransientBuildError",
    "dockerfile_build",
    "dockerfile_sets_start_command",
    "ensure_aenv_template",
]
