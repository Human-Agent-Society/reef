"""Reef's Harbor environment for E2B task trials, and the builds of their templates.

``KillOnTimeoutE2BEnvironment`` reuses Harbor's E2B environment for task trials. It requests deletion at timeout,
so AgentENV deletes an abandoned sandbox instead of pausing it. It also keeps the file modes of a task's
``COPY --chmod``, which E2B's Dockerfile parser ignores (:func:`dockerfile_with_copy_modes`). On AgentENV, whose E2B
template builds take no files, ``REEF_E2B_DOCKERFILE_BUILD=aenv`` makes it build a task's Dockerfile with
``aenv build`` (:mod:`reef.execution.sandbox.agentenv.build`). There, ``REEF_E2B_COMPOSE=aenv`` also makes
it run the Compose project of a task that has one in one sandbox, so that the agent reaches the task's other services
(:mod:`reef.execution.sandbox.agentenv.compose`). When a deployment supplies verified images for the task,
AgentENV can start each environment from its image (:mod:`reef.execution.sandbox.agentenv.images`).
The shared ``environment_config`` selects this class for Reef's Harbor rollouts and generator checks through
the ``import_path`` that saved trial configs keep,
``reef.execution.sandbox.e2b:KillOnTimeoutE2BEnvironment``. A standalone Harbor trial that selects only ``e2b`` uses
Harbor's own class.

This module can load without the optional E2B SDK. Opening a sandbox requires the SDK; the Harbor subclass also
requires Harbor 0.25, installed through ``reef-infra[e2b]``.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import os
import posixpath
import re
import shlex
import tempfile
import time
from pathlib import Path
from typing import Any, override

import aiohttp
from harbor.environments.base import ExecResult
from harbor.environments.definition import effective_exec_cwd
from harbor.environments.e2b import E2BEnvironment
from harbor.models.task.config import EnvironmentConfig, NetworkMode
from harbor.models.task.task import Task
from harbor.models.task.verifier_mode import resolve_verifier_environment_definition
from harbor.models.trial.paths import TrialPaths
from tenacity import retry, retry_if_exception_type, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from reef.core.errors import InternalError
from reef.execution.sandbox.agentenv.build import (
    DOCKERFILE_BUILD_VARIABLE,
    AgentEnvTemplates,
    DockerfileBuild,
    dockerfile_build,
    dockerfile_sets_start_command,
)
from reef.execution.sandbox.agentenv.compose import (
    COMPOSE_FILE,
    COMPOSE_VARIABLE,
    AgentEnvComposeOps,
    ComposeHost,
    ComposeSandbox,
    ImageNotFoundError,
    TransientStartError,
    compose_mode,
    compose_project,
    main_image,
)
from reef.execution.sandbox.agentenv.images import ColdSandbox, ColdSandboxError, is_missing_snapshot
from reef.execution.sandbox.prepared_images import (
    PreparedImage,
    PreparedImageJson,
    prepared_images_from_json,
    select_prepared_image,
)
from reef.execution.sandbox.process import SandboxCapacityError, SandboxRequestError

logger = logging.getLogger(__package__)

#: The end of the template alias of a task whose Dockerfile has ``--chmod``. Harbor's class builds that template
#: without the modes, so Reef's class does not use a template that Harbor's class built.
COPY_MODES_TEMPLATE_SUFFIX = "__copy-modes"
#: The end of the template alias that ``aenv build`` builds, so that it never names a template of E2B's template API,
#: which AgentENV builds without the task's files.
AENV_TEMPLATE_SUFFIX = "__aenv"
#: The shell command that waits, for at most about 130 s, until a task image whose first process is systemd emptied
#: ``/tmp`` at boot (``systemd-tmpfiles-setup.service``), or until the boot ended. A trial uploads its agent's files to
#: ``/tmp`` after the start, and that boot step deletes them. An image whose first process is not systemd returns at
#: once. Each ``systemctl`` call gets 5 s. When systemd does not answer in that time, the wait stops: in such an image
#: ``systemctl`` hangs, and the boot does not empty ``/tmp``. An image without ``timeout`` does not wait.
SYSTEMD_TMP_CLEANUP_WAIT = (
    'if [ "$(cat /proc/1/comm 2>/dev/null)" = systemd ] && command -v timeout >/dev/null 2>&1; then '
    "end=$(($(date +%s) + 120)); while :; do "
    "timeout 5 systemctl is-active --quiet systemd-tmpfiles-setup.service; code=$?; "
    "[ $code -eq 0 ] || [ $code -eq 124 ] && break; "
    "state=$(timeout 5 systemctl is-system-running 2>/dev/null); [ $? -eq 124 ] && break; "
    'case "$state" in running|degraded) break;; esac; '
    '[ "$(date +%s)" -ge "$end" ] && break; sleep 1; done; fi'
)
#: The seconds that ``KillOnTimeoutE2BEnvironment.start`` gives ``SYSTEMD_TMP_CLEANUP_WAIT``.
SYSTEMD_TMP_CLEANUP_WAIT_SECONDS = 150
#: The pause between two runs of a prepared image's ready command after a cold start, in seconds, as in AgentENV's
#: template runner.
READY_POLL_SECONDS = 2.0
#: The longest wait of a cold start for a prepared image's start to be ready, in seconds, as in AgentENV's template
#: runner. The environment's start time limit can end it sooner.
READY_LIMIT_SECONDS = 600.0
#: When the start of a prepared image without a ready command is ready, in seconds after it began: AgentENV's template
#: runner then uses ``/agentenv/bin/busybox sleep 20`` as the ready command.
DEFAULT_READY_SECONDS = 20.0
#: The longest wait for AgentENV's state of a prepared image's template, in seconds.
TEMPLATE_STATUS_WAIT_SECONDS = 5.0
#: The longest wait for the deletion of a sandbox whose start failed, in seconds.
DELETE_WAIT_SECONDS = 30.0


def dockerfile_with_copy_modes(dockerfile: str, context_dir: Path) -> str:
    """``dockerfile`` with each ``--chmod`` of a COPY or ADD done by a ``chmod`` after it.

    E2B's Dockerfile parser ignores ``--chmod``, so a copied file keeps the mode that it has in the build context.
    In a task, that can make a root-only file readable, or remove the setuid bit of a wrapper. In the result, each
    COPY or ADD with ``--chmod=MODE`` has no ``--chmod``, and a RUN after it sets MODE as Docker does. It sets MODE
    on each copied file. For a copied directory, it runs ``chmod -R`` on the destination, which also changes files
    that were in that destination before. The RUN runs as root. When an earlier USER names another user, the result
    sets ``USER root`` before the RUN and that user again after it.

    ``context_dir`` is the build context: it tells a file source from a directory source. A source with a glob
    pattern is each path that matches it. All other lines keep their text. A Dockerfile without ``--chmod`` is
    returned unchanged.
    """
    # Harbor's e2b extra installs dockerfile-parse, which E2B's own parser also uses.
    from dockerfile_parse import DockerfileParser

    lines = dockerfile.splitlines(keepends=True)
    user = "root"
    for instruction in DockerfileParser(fileobj=io.BytesIO(dockerfile.encode())).structure:
        if instruction["instruction"] == "USER":
            user = instruction["value"].strip()
            continue
        mode_flag = re.search(r"(?:^|\s)--chmod=(\S+)", instruction["value"])
        if instruction["instruction"] not in ("COPY", "ADD") or mode_flag is None:
            continue
        arguments = " ".join(
            (instruction["value"][: mode_flag.start()] + instruction["value"][mode_flag.end() :]).split()
        )
        paths = [argument for argument in shlex.split(arguments) if not argument.startswith("--")]
        destination = paths[-1]
        sources: list[Path] = []
        for pattern in paths[:-1]:
            if any(character in pattern for character in "*?["):
                sources.extend(sorted(context_dir.glob(pattern)))
            else:
                sources.append(context_dir / pattern)
        into_directory = destination.endswith("/") or len(sources) > 1
        mode = shlex.quote(mode_flag.group(1))
        commands = []
        for source in sources:
            if source.is_dir():
                commands.append(f"chmod -R {mode} {shlex.quote(destination)}")
            elif into_directory:
                commands.append(f"chmod {mode} {shlex.quote(posixpath.join(destination, source.name))}")
            else:
                commands.append(f"chmod {mode} {shlex.quote(destination)}")
        chmod_run = f"RUN {' && '.join(commands)}\n"
        as_root = user.split(":")[0] in ("root", "0")
        chmod_lines = chmod_run if as_root else f"USER root\n{chmod_run}USER {user}\n"
        # The instruction's lines become the new text; the later lines keep their places in ``lines``.
        lines[instruction["startline"]] = f"{instruction['instruction']} {arguments}\n{chmod_lines}"
        for index in range(instruction["startline"] + 1, instruction["endline"] + 1):
            lines[index] = ""
    return "".join(lines)


#: The part of a session ID that Harbor gives a separate verifier environment of a trial
#: (``Trial._separate_verifier_session_id`` in Harbor 0.25).
VERIFIER_SESSION_MARKER = "__verifier__"
#: The share of an environment's start time limit (``build_timeout_sec``) in which a verifier environment waits for
#: sandbox capacity. The rest is for its other start steps, so the wait ends before Harbor's limit.
VERIFIER_CAPACITY_WAIT_SHARE = 0.8
#: The first and the longest pause between two create calls of a verifier environment that waits for capacity.
VERIFIER_CAPACITY_FIRST_PAUSE_SECONDS = 15.0
VERIFIER_CAPACITY_MAX_PAUSE_SECONDS = 120.0


class TemplateNotPreparedError(InternalError):
    """A trial that builds no template (``KillOnTimeoutE2BEnvironment.builds_templates`` false) found its template
    missing, or not ready on AgentENV; or the prepared images of a task have no image of one of its environments."""


class KillOnTimeoutE2BEnvironment(E2BEnvironment, ComposeHost):
    """Harbor's ``E2BEnvironment`` whose sandbox is created with ``lifecycle={"on_timeout": "kill"}``, and whose
    template keeps the modes of ``COPY --chmod``.

    Values: those of ``E2BEnvironment``. Harbor makes one from a trial's environment config, so the sandbox's lifetime
    (``sandbox_timeout_secs``), its metadata, its startup environment and its network options are the parent's, and
    ``type()`` is ``e2b``. ``dockerfile_build`` is the value of ``REEF_E2B_DOCKERFILE_BUILD`` when it was made. The
    template is the parent's, with two differences for a task that builds its image from a Dockerfile:

    - With ``dockerfile_build`` ``aenv``, ``aenv build`` builds the template on the AgentENV deployment
      (``reef.execution.sandbox.agentenv.build``), and its alias ends with ``AENV_TEMPLATE_SUFFIX``.
    - Else, when the Dockerfile has ``--chmod``, the template is built from ``dockerfile_with_copy_modes`` of it, and
      its alias ends with ``COPY_MODES_TEMPLATE_SUFFIX``.

    ``builds_templates`` is the bool of the environment config's ``kwargs.builds_templates``, True when the config
    has none. When it is false, the environment builds no template: ``_create_template`` raises
    ``TemplateNotPreparedError``, and on AgentENV ``_does_template_exist`` is true only for a ready template.

    ``compose_ops`` is not None if and only if ``REEF_E2B_COMPOSE`` was ``aenv`` when the environment was made and
    the environment directory has ``docker-compose.yaml``. ``REEF_E2B_COMPOSE=aenv`` needs ``dockerfile_build``
    ``aenv``: the constructor raises ``ValueError`` without it. The environment then runs the task's Compose project
    in one AgentENV sandbox (``start_compose``), and it differs from the above in these points:

    - The sandbox is the guest of the project, and each command and file transfer of the trial goes to the ``main``
      service. The project's services reach one another by name.
    - The environment uses no template. ``aenv build --image`` builds the task's Dockerfile into the image of
      ``main`` (``main_image`` of the template alias); a task with a ``docker_image`` runs that image.
      ``_create_template`` builds the image, and ``_does_template_exist`` is false, because AgentENV does not tell
      whether it has an image. A start builds the image only after AgentENV refused the project for a missing image.
    - AgentENV pauses the sandbox at its timeout, and does not delete it: ``aenv compose up`` sets no other policy.
    - The start does not wait for systemd: the first process of ``main`` is the project's command.

    ``prepared_image`` is the descriptor of the config's ``kwargs.prepared_images`` for this environment: the one
    whose directory is this environment's in the task at ``kwargs.task_directory``, and whose source image is the
    task config's ``docker_image`` (``select_prepared_image``). A Compose project uses the verified digest for
    its main service. The constructor raises ``ValueError`` for an invalid list and ``TemplateNotPreparedError``
    for a missing environment. With a
    descriptor, the environment builds no template, also with ``force_build``, and ``_create_sandbox`` starts the
    image on AgentENV (``start_prepared_image``). When a later step of its ``start`` fails or is cancelled, the start
    deletes that sandbox.

    Purpose: Harbor's ``EnvironmentFactory`` makes one for an ``e2b`` trial whose environment config names this class
    as its ``import_path``, and one for each separate verifier environment of the trial, from the same config.
    ``start`` builds a missing template with ``_create_template``, creates the sandbox with ``_create_sandbox``, and
    ``stop`` deletes it, as before. Then ``start`` waits until the sandbox's systemd, if its first process is systemd,
    emptied ``/tmp`` (``SYSTEMD_TMP_CLEANUP_WAIT``). ``build_task_template`` makes one that builds templates.
    """

    def __init__(
        self,
        *args: Any,
        builds_templates: bool = True,
        prepared_images: list[PreparedImageJson] | None = None,
        task_directory: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(builds_templates, bool):
            raise TypeError("builds_templates is a bool")
        if (prepared_images is None) != (task_directory is None):
            raise ValueError("prepared_images and task_directory are given together")
        super().__init__(*args, **kwargs)
        self.builds_templates = builds_templates
        #: The ``time.monotonic()`` value when ``start`` began, or None before it.
        self.started_at: float | None = None
        self.dockerfile_build = dockerfile_build(os.environ)
        runs_compose = compose_mode(os.environ) == "aenv"
        if runs_compose and self.dockerfile_build != "aenv":
            raise ValueError(f"{COMPOSE_VARIABLE}=aenv needs {DOCKERFILE_BUILD_VARIABLE}=aenv")
        self.compose_ops: AgentEnvComposeOps | None = (
            AgentEnvComposeOps(self) if runs_compose and (self.environment_dir / COMPOSE_FILE).is_file() else None
        )
        self.prepared_image: PreparedImage | None = None
        if prepared_images is not None and task_directory is not None:
            images = prepared_images_from_json(prepared_images)
            task_path = Path(task_directory)
            missing = [
                f"{directory} (source image {config.docker_image or None})"
                for directory, config in task_environments(Task(task_path))
                if select_prepared_image(images, task_path, directory, config.docker_image or None) is None
            ]
            if missing:
                raise TemplateNotPreparedError(
                    f"the prepared images of {self.environment_name} have no image of {', '.join(missing)}, and a "
                    "task with prepared images builds none"
                )
            self.prepared_image = select_prepared_image(
                images, task_path, self.environment_dir, self.task_env_config.docker_image or None
            )
            if self.prepared_image is None:
                raise TemplateNotPreparedError(f"no prepared image for the environment at {self.environment_dir}")
        # Harbor builds the template from the task's image when it has one; then the Dockerfile is not read.
        if self.task_env_config.docker_image:
            return
        if self.dockerfile_build == "aenv":
            self._template_name += AENV_TEMPLATE_SUFFIX
            return
        dockerfile = self._environment_definition_path.read_text()
        if dockerfile_with_copy_modes(dockerfile, self.environment_dir) != dockerfile:
            self._template_name += COPY_MODES_TEMPLATE_SUFFIX

    @override
    async def start(self, force_build: bool) -> None:
        self.started_at = time.monotonic()
        try:
            if self.compose_ops is not None:
                await self.start_compose(force_build)
                return
            await super().start(force_build)
            # A task image can boot systemd, which empties /tmp some seconds after the start. The agent's install
            # uploads its archives there next, so it waits for that step. The wait is a best effort: its result is not
            # checked.
            await self.exec(SYSTEMD_TMP_CLEANUP_WAIT, timeout_sec=SYSTEMD_TMP_CLEANUP_WAIT_SECONDS, user="root")
        except BaseException:
            # The sandbox of a prepared image that a later step of the start did not finish, such as Harbor's
            # directories or uploads, is this start's own.
            if self.prepared_image is not None and self._sandbox is not None:
                await self.discard_sandbox(self._sandbox.sandbox_id)
            raise

    async def start_compose(self, force_build: bool) -> None:
        """Start the task's Compose project in one AgentENV sandbox, and make that sandbox this environment's.

        The project is ``compose_project`` of the task's Compose file, with the image of ``main`` and this
        environment's startup variables. The sandbox has the task's CPU and memory for all services, and this
        environment's lifetime; the start has the task's build time limit. ``force_build`` builds the image of
        ``main`` first. Else the image is built only when AgentENV has none, and then the start runs once more. A
        network policy other than ``public`` is applied to the sandbox after the start. A start that fails after
        AgentENV made the sandbox deletes it.

        Raise ``TemplateNotPreparedError`` when the image of ``main`` is missing and this environment builds no
        template, and ``ComposeStartError`` when AgentENV cannot start the project.
        """
        # Keep this optional SDK import inside sandbox creation.
        from e2b import AsyncSandbox

        from reef.execution.sandbox.e2b import delete_sandbox, prepare_guest

        templates = AgentEnvTemplates.from_environ(os.environ)
        image = (
            self.prepared_image.image if self.prepared_image else self.task_env_config.docker_image
        ) or main_image(self._template_name)[1]
        cpus = self._effective_cpus
        sandbox = ComposeSandbox(
            project=compose_project((self.environment_dir / COMPOSE_FILE).read_text(), image, self._startup_env()),
            cpu_count=None if cpus is None else int(cpus),
            memory_mb=self._effective_memory_mb,
            lifetime_seconds=self._sandbox_timeout,
            startup_seconds=min(max(math.ceil(self.task_env_config.build_timeout_sec), 1), 86400),
        )
        if self.prepared_image is not None:
            sandbox_id = await self.start_compose_sandbox(sandbox, templates)
        elif force_build:
            await self._create_template()
            sandbox_id = await self.start_compose_sandbox(sandbox, templates)
        else:
            try:
                sandbox_id = await self.start_compose_sandbox(sandbox, templates)
            except ImageNotFoundError:
                await self._create_template()
                sandbox_id = await self.start_compose_sandbox(sandbox, templates)
        try:
            await prepare_guest(templates, sandbox_id)
            self._sandbox: AsyncSandbox | None = await AsyncSandbox.connect(sandbox_id, timeout=self._sandbox_timeout)
        finally:
            # Harbor's stop deletes only the sandbox that this environment holds.
            if self._sandbox is None:
                await delete_sandbox(templates, sandbox_id)
        if self.network_policy.network_mode != NetworkMode.PUBLIC:
            await self._apply_network_policy(self.network_policy)
        await self.ensure_dirs(self._mount_targets(writable_only=True))
        await self._upload_environment_dir_after_start()

    @retry(
        retry=retry_if_exception_type(TransientStartError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    async def start_compose_sandbox(self, sandbox: ComposeSandbox, templates: AgentEnvTemplates) -> str:
        """``sandbox.start(templates)``, run again after a temporary failure of the AgentENV API, four times at
        most."""
        return await sandbox.start(templates)

    @override
    async def host_exec(self, command: str, timeout_sec: int | None = None) -> ExecResult:
        # Keep this optional SDK import inside the command.
        from e2b.sandbox.commands.command_handle import CommandExitException

        handle = await self._dispatch_command(command, cwd=None, env=None, timeout_sec=timeout_sec, user="root")
        try:
            result = await handle.wait()
        except CommandExitException as error:
            # A non-zero exit is a result, as in Harbor's exec.
            result = error
        return ExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.exit_code)

    @override
    async def host_write(self, path: str, data: bytes) -> None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        await self._sandbox.files.write(path, data)

    @override
    async def host_read(self, path: str) -> bytes:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        return bytes(await self._sandbox.files.read(path, format="bytes"))

    # With a Compose project, the trial's commands and files go to the main service. Else they are Harbor's.
    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if self.compose_ops is None:
            return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)
        return await self.compose_ops.exec(
            command,
            cwd=effective_exec_cwd(cwd, self.task_env_config.workdir, self._workdir),
            env=self._merge_env(env),
            timeout_sec=timeout_sec,
            user=self._resolve_user(user),
        )

    @override
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        if self.compose_ops is None:
            await super().upload_file(source_path, target_path)
        else:
            await self.compose_ops.upload_file(source_path, target_path)

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        if self.compose_ops is None:
            await super().upload_dir(source_dir, target_dir)
        else:
            await self.compose_ops.upload_dir(source_dir, target_dir)

    @override
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        if self.compose_ops is None:
            await super().download_file(source_path, target_path)
        else:
            await self.compose_ops.download_file(source_path, target_path)

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        if self.compose_ops is None:
            await super().download_dir(source_dir, target_dir)
        else:
            await self.compose_ops.download_dir(source_dir, target_dir)

    @override
    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        if self.compose_ops is None:
            return await super().is_dir(path, user=user)
        return await self.compose_ops.is_dir(path, user=self._resolve_user(user))

    @override
    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        if self.compose_ops is None:
            return await super().is_file(path, user=user)
        return await self.compose_ops.is_file(path, user=self._resolve_user(user))

    @property
    def builds_with_aenv(self) -> bool:
        """Whether ``aenv build`` builds this trial's template: the task builds from a Dockerfile, and
        ``REEF_E2B_DOCKERFILE_BUILD`` is ``aenv``."""
        return self.dockerfile_build == "aenv" and not self.task_env_config.docker_image

    @override
    async def _does_template_exist(self) -> bool:
        if self.prepared_image is not None:
            # A prepared image needs no template build; ``start_prepared_image`` checks its template.
            return True
        if self.compose_ops is not None:
            return False
        if not self.builds_with_aenv:
            return await super()._does_template_exist()
        if self.builds_templates:
            # An AgentENV alias can name a template that is still building or that failed. ``ensure_aenv_template``
            # in ``_create_template`` reads its status and returns at once for a ready template.
            return False
        state = await AgentEnvTemplates.from_environ(os.environ).state(self._template_name)
        return state is not None and state.status == "ready"

    # Harbor 0.25.0's template build, with five changes: an environment with a prepared image builds nothing; an
    # environment that builds no template refuses; with a Compose project, ``aenv build --image`` builds the image of
    # the main service; with ``builds_with_aenv``, ``aenv build`` builds the template; else the Dockerfile is
    # ``dockerfile_with_copy_modes`` of the task's Dockerfile. Harbor's retry policy applies to builds. A trial that
    # cannot build fails immediately.
    # The parity test in tests/sandbox/test_task_environment.py compares the build with Harbor's for a Dockerfile
    # without ``--chmod``.
    @retry(
        # tenacity gives every BaseException to the condition. A cancelled build must end, not build again.
        retry=retry_if_exception_type(Exception) & retry_if_not_exception_type(TemplateNotPreparedError),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    @override
    async def _create_template(self) -> None:
        from reef.execution.sandbox.e2b import build_image, ensure_aenv_template

        if self.prepared_image is not None:
            # Also for ``force_build``: the deployment prepared the image, so the task's Dockerfile is not built.
            return
        if not self.builds_templates:
            missing = (
                f"the template {self._template_name}"
                if self.compose_ops is None
                else f"the image {main_image(self._template_name)[1]} of the main service"
            )
            raise TemplateNotPreparedError(
                f"{missing} of {self.environment_name} is not ready, and this trial "
                "builds no template: prepare the task's templates before its trials"
            )
        if self.compose_ops is not None:
            if not self.task_env_config.docker_image:
                image_build = DockerfileBuild(
                    context_dir=self.environment_dir,
                    dockerfile=self._environment_definition_path,
                    alias=main_image(self._template_name)[0],
                    cpu_count=None,
                    memory_mb=None,
                    timeout_seconds=min(max(math.ceil(self.task_env_config.build_timeout_sec), 1), 86400),
                    start_command=True,
                    image=True,
                )
                await build_image(AgentEnvTemplates.from_environ(os.environ), image_build)
            return
        if self.builds_with_aenv:
            dockerfile = self._environment_definition_path.read_text()
            cpus = self._effective_cpus
            build = DockerfileBuild(
                context_dir=self.environment_dir,
                dockerfile=self._environment_definition_path,
                alias=self._template_name,
                cpu_count=None if cpus is None else int(cpus),
                memory_mb=self._effective_memory_mb,
                timeout_seconds=min(max(math.ceil(self.task_env_config.build_timeout_sec), 1), 86400),
                start_command=dockerfile_sets_start_command(dockerfile),
            )
            await ensure_aenv_template(AgentEnvTemplates.from_environ(os.environ), build)
            return
        # Keep this optional SDK import inside template creation.
        from e2b import AsyncTemplate, Template

        image = self.task_env_config.docker_image
        if image:
            template = Template().from_image(image=image)
        else:
            dockerfile = dockerfile_with_copy_modes(
                self._environment_definition_path.read_text(), self.environment_dir
            )
            template = Template(file_context_path=str(self.environment_dir)).from_dockerfile(
                dockerfile_content_or_path=dockerfile
            )
        # Harbor leaves out a resource that the task does not set; None is E2B's default for both.
        cpus = self._effective_cpus
        await AsyncTemplate.build(
            template=template,
            alias=self._template_name,
            cpu_count=None if cpus is None else int(cpus),
            memory_mb=self._effective_memory_mb,
        )

    # Harbor 0.25.0 names this method and the attributes that it reads; an override cannot rename them. The decorator
    # preserves Harbor's retry for transient failures. Bad requests and capacity refusals fail immediately:
    # verifiers already wait for capacity inside this method, and agent rollouts defer in the queue. A failed cold
    # start fails immediately too: its sandbox can exist after an answer that did not come.
    # The create call, in create_from_template, is Harbor's call with one more argument.
    # When Harbor adds an argument to the call, add it there too: the parity test in
    # tests/sandbox/test_task_environment.py compares the two calls. A refusal for want of capacity becomes
    # ``SandboxCapacityError``, so that an evaluation waits for capacity instead of failing, and a bad request becomes
    # ``SandboxRequestError``.
    @retry(
        retry=retry_if_exception_type(Exception)
        & retry_if_not_exception_type((SandboxCapacityError, SandboxRequestError, ColdSandboxError)),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    @override
    async def _create_sandbox(self) -> None:
        """Create the sandbox. A separate verifier environment runs after the agent, so it does not give up the
        agent's work at the first refusal for want of capacity: it calls again, with pauses of 15 s doubling to 2 min,
        until ``VERIFIER_CAPACITY_WAIT_SHARE`` of its start time limit has passed since ``start``. Then it raises the
        last ``SandboxCapacityError``. Another environment raises it at once. Any environment raises a
        ``SandboxRequestError`` at once."""
        verifier = VERIFIER_SESSION_MARKER in self.session_id
        started_at = time.monotonic() if self.started_at is None else self.started_at
        wait_ends_at = started_at + VERIFIER_CAPACITY_WAIT_SHARE * self.task_env_config.build_timeout_sec
        pause_seconds = VERIFIER_CAPACITY_FIRST_PAUSE_SECONDS
        while True:
            try:
                await self.create_sandbox_once()
                return
            except SandboxCapacityError as exc:
                if not verifier or time.monotonic() + pause_seconds > wait_ends_at:
                    raise
                logger.info("verifier sandbox %s waits %.0f s for capacity: %s", self.session_id, pause_seconds, exc)
                await asyncio.sleep(pause_seconds)
                pause_seconds = min(2 * pause_seconds, VERIFIER_CAPACITY_MAX_PAUSE_SECONDS)

    async def create_sandbox_once(self) -> None:
        """Ask the sandbox service for the sandbox once: from the prepared image when the environment has one
        (``start_prepared_image``), else from the environment's template (``create_from_template``)."""
        if self.prepared_image is not None:
            await self.start_prepared_image(self.prepared_image)
        else:
            await self.create_from_template(self._template_name)

    async def create_from_template(self, template: str) -> None:
        """Create the sandbox from the template ``template`` once. Raise ``SandboxCapacityError`` when the service
        refused it for want of capacity, and ``SandboxRequestError`` with the service's reason when it answered that
        the request is bad (HTTP 400)."""
        # Keep these optional SDK imports inside sandbox creation.
        import httpx
        from e2b import AsyncSandbox, RateLimitException, SandboxException, ServiceBusyException

        try:
            self._sandbox = await AsyncSandbox.create(
                template=template,
                metadata={"environment_name": self.environment_name, "session_id": self.session_id},
                envs=self._startup_env(),
                timeout=self._sandbox_timeout,
                allow_internet_access=self.network_policy.network_mode != NetworkMode.NO_NETWORK,
                network=self._sandbox_create_network_options(),
                lifecycle={"on_timeout": "kill"},
            )
        except (ServiceBusyException, RateLimitException) as exc:
            raise SandboxCapacityError(f"the sandbox service has no capacity now: {exc}") from exc
        except httpx.TimeoutException as exc:
            # The service holds the create call while it places the sandbox. A service that adds workers can place
            # it later than the SDK waits.
            raise SandboxCapacityError(f"the sandbox service did not place the sandbox in time: {exc}") from exc
        except json.JSONDecodeError as exc:
            # The SDK reads each answer of the create call as JSON. A gateway that timed out before the service placed
            # the sandbox answers 504 with an HTML body, and the SDK then fails here.
            raise SandboxCapacityError(
                "the sandbox service gave no JSON answer, as its gateway does when it times out"
            ) from exc
        except SandboxException as exc:
            # The SDK keeps the HTTP status of the service's answer. Only a bad request is definite and final; the
            # other answers, such as a credential error, stay the SDK's.
            if exc.status_code != 400:
                raise
            raise SandboxRequestError(f"the sandbox service refused the sandbox's request: {exc}") from exc

    async def start_prepared_image(self, prepared: PreparedImage) -> None:
        """Start the sandbox from ``prepared`` on AgentENV, without a template build.

        When AgentENV reads the template ``prepared.template_id`` as ready (``template_is_ready``), the sandbox starts
        from it, as from any template (``create_from_template``): its snapshot already ran the image's start. The image
        starts cold instead (``start_cold``) when it has no template, when the template is not ready, and when the
        create call answers that the worker has no snapshot of that template (``is_missing_snapshot``). AgentENV gives
        that answer before it allocates a sandbox, so the cold start makes no second sandbox. Every other failure of
        the template's start is raised as ``create_from_template`` raises it: a refusal for want of capacity waits, and
        no cold start follows another answer, another template's answer, or an answer that did not come.
        """
        # Keep this optional SDK import inside sandbox creation.
        from e2b import SandboxException

        templates = AgentEnvTemplates.from_environ(os.environ)
        if prepared.template_id is not None and await self.template_is_ready(templates, prepared.template_id):
            try:
                await self.create_from_template(prepared.template_id)
                return
            except (SandboxRequestError, SandboxException) as error:
                # create_from_template keeps the SDK's error of a refused request as the cause of its own.
                answer = error.__cause__ if isinstance(error, SandboxRequestError) else error
                if not isinstance(answer, SandboxException) or not is_missing_snapshot(
                    answer.status_code, str(answer), prepared.template_id
                ):
                    raise
            logger.info(
                "AgentENV has no snapshot of the template %s; starting %s cold", prepared.template_id, prepared.image
            )
        await self.start_cold(templates, prepared)

    async def template_is_ready(self, templates: AgentEnvTemplates, template_id: str) -> bool:
        """Whether AgentENV reads the template ``template_id`` as ready (``AgentEnvTemplates.is_ready``). A read that
        gives no answer in ``TEMPLATE_STATUS_WAIT_SECONDS``, or before the start time limit, counts as not ready: it
        allocates nothing. If the cache cannot serve the read, the verified image can start immediately.
        Credential and request errors still reach the caller."""
        try:
            return await templates.is_ready(
                template_id, min(TEMPLATE_STATUS_WAIT_SECONDS, max(self.start_seconds_left(), 1.0))
            )
        except aiohttp.ClientResponseError as error:
            if error.status != 429 and not 500 <= error.status < 600:
                raise
        except (TimeoutError, aiohttp.ClientConnectionError, aiohttp.ClientPayloadError):
            pass
        logger.warning("AgentENV could not report the state of template %s; starting from its image", template_id)
        return False

    def start_seconds_left(self) -> float:
        """The seconds that are left of the environment's start time limit (``build_timeout_sec``), counted from
        ``start``, or from now before it; 0 or less after the limit."""
        started_at = time.monotonic() if self.started_at is None else self.started_at
        return started_at + self.task_env_config.build_timeout_sec - time.monotonic()

    async def start_cold(self, templates: AgentEnvTemplates, prepared: PreparedImage) -> None:
        """Start ``prepared.image`` in a new sandbox on ``templates``' deployment, without a template, and then run its
        start as the template ran it before its snapshot (``run_image_start``).

        The request has the values of the template's create call: this environment's lifetime, network options,
        metadata and startup variables; and the task's CPU and memory, which a template build would set. AgentENV
        deletes the sandbox at the end of its lifetime, and does not pause it. The request waits for its answer until
        the end of the environment's start time limit (``start_seconds_left``), and is sent once. When the start fails
        after AgentENV made the sandbox, also when it is cancelled, this deletes the sandbox (``discard_sandbox``)
        before it raises.

        Raise what ``ColdSandbox.create`` raises, and ``ColdSandboxError`` for a later failure.
        """
        # Keep this optional SDK import inside sandbox creation.
        from e2b import AsyncSandbox

        cpus = self._effective_cpus
        request = ColdSandbox(
            image=prepared.image,
            timeout_seconds=self._sandbox_timeout,
            allow_internet_access=self.network_policy.network_mode != NetworkMode.NO_NETWORK,
            network=self._sandbox_create_network_options(),
            metadata={"environment_name": self.environment_name, "session_id": self.session_id},
            env_vars=self._startup_env(),
            cpu_count=None if cpus is None else int(cpus),
            memory_mb=self._effective_memory_mb,
        )
        sandbox_id = await request.create(templates, max(self.start_seconds_left(), 1.0))
        try:
            self._sandbox = await AsyncSandbox.connect(sandbox_id, timeout=self._sandbox_timeout)
            await self.run_image_start(prepared)
        except BaseException as error:
            await self.discard_sandbox(sandbox_id)
            if not isinstance(error, Exception) or isinstance(error, ColdSandboxError):
                raise
            raise ColdSandboxError(f"the sandbox {sandbox_id} of {prepared.image} did not start: {error}") from error

    async def discard_sandbox(self, sandbox_id: str) -> None:
        """Delete the sandbox ``sandbox_id``, which a start from a prepared image made and did not finish, and hold no
        sandbox: Harbor's ``stop`` deletes only a sandbox that the environment holds. The delete waits at most
        ``DELETE_WAIT_SECONDS``. A delete that fails is logged, and AgentENV deletes the sandbox at the end of its
        lifetime."""
        from reef.execution.sandbox.e2b import delete_sandbox

        self._sandbox = None
        try:
            await asyncio.wait_for(
                delete_sandbox(AgentEnvTemplates.from_environ(os.environ), sandbox_id), DELETE_WAIT_SECONDS
            )
        except (TimeoutError, aiohttp.ClientError) as error:
            logger.warning("could not delete the sandbox %s of a failed start: %s", sandbox_id, error)

    async def run_image_start(self, prepared: PreparedImage) -> None:
        """Run the start of ``prepared`` in the sandbox, as AgentENV's template runner ran it before the template's
        snapshot.

        Both commands run as the image's own user, in its working directory and with its variables, which the sandbox
        has from the image: the call gives them no user, directory or variable of the trial. ``prepared.start`` starts
        in the background and keeps running. The start is ready when ``prepared.ready`` exits with 0; it runs again
        ``READY_POLL_SECONDS`` after each other exit. Without a ready command, the start is ready
        ``DEFAULT_READY_SECONDS`` after it began, and an image without either command is ready at once.

        Raise ``ColdSandboxError`` when the start command ends with an error before the start is ready, and when the
        start is not ready ``READY_LIMIT_SECONDS`` after it began, or at the environment's start time limit when that
        comes first. A run of the ready command ends at that limit too."""
        # Keep this optional SDK import inside sandbox creation.
        from e2b.sandbox.commands.command_handle import CommandExitException

        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        if prepared.start is None and prepared.ready is None:
            return
        began = time.monotonic()
        limit_seconds = min(READY_LIMIT_SECONDS, max(self.start_seconds_left(), 0.0))
        deadline = began + limit_seconds
        # A timeout of 0 sets no limit on the start command.
        started = (
            None
            if prepared.start is None
            else await self._sandbox.commands.run(cmd=prepared.start, background=True, timeout=0)
        )
        # The end of the start command. An end with 0 is a start that leaves no process of its own.
        ended = None if started is None else asyncio.ensure_future(started.wait())

        def check_start() -> None:
            failure = ended.exception() if ended is not None and ended.done() else None
            if failure is not None:
                raise ColdSandboxError(
                    f"the start command of {prepared.image} ended before the start was ready: {failure}"
                ) from failure

        try:
            while True:
                check_start()
                seconds_left = deadline - time.monotonic()
                if seconds_left <= 0:
                    raise ColdSandboxError(f"the start of {prepared.image} was not ready in {limit_seconds:.0f} s")
                if prepared.ready is None:
                    if time.monotonic() - began >= DEFAULT_READY_SECONDS:
                        return
                else:
                    try:
                        await self._sandbox.commands.run(cmd=prepared.ready, timeout=math.ceil(seconds_left))
                        # The readiness probe and the entrypoint can finish in the same event loop turn.
                        if ended is not None:
                            await asyncio.sleep(0)
                            check_start()
                        return
                    except CommandExitException:
                        # The ready command failed: the start is not ready yet.
                        pass
                await asyncio.sleep(min(READY_POLL_SECONDS, seconds_left))
        finally:
            if ended is not None:
                if ended.done():
                    # The start command's error, if it had one, counts as read.
                    ended.exception()
                else:
                    ended.cancel()
            if started is not None:
                # The start command keeps running after the disconnect.
                await started.disconnect()


def task_environments(task: Task) -> list[tuple[Path, EnvironmentConfig]]:
    """The environments of a trial of ``task``, each as the directory and the config that Harbor starts it from: the
    task's environment, then each separate verifier environment of the task or of its steps. A verifier that runs in
    the task's environment has no environment of its own."""
    environments = [(task.paths.environment_dir, task.config.environment)]
    for step in task.config.steps or [None]:
        verifier = resolve_verifier_environment_definition(task.config, task.paths, step)
        if verifier is not None:
            environments.append((verifier.directory, verifier.config))
    return environments


async def build_task_template(task_path: Path) -> str:
    """Build each template of the ``e2b`` trials of the task directory ``task_path`` that does not exist, and return
    the alias of the template of the task's environment.

    The templates are the ones that a trial of the task builds: the template of the task's environment, which
    ``KillOnTimeoutE2BEnvironment.start`` builds before the agent, and the template of each separate verifier
    environment of the task or of its steps, which the trial builds after the agent. Each has the alias, the
    resources and the build time limit that it has in a trial. A trial after this call finds the templates and does
    not build them. So a caller that builds each task's templates before its trials keeps the builds out of the
    trials' environment start limits. With ``REEF_E2B_DOCKERFILE_BUILD`` ``aenv``, a build waits for a free AgentENV
    builder until its build time limit (``ensure_aenv_template``). Raise what a build raises, such as
    ``TimeoutError`` at that limit.

    A task environment that runs a Compose project has no template: the call builds the image of its ``main``
    service, on every call, and still returns the alias (``build_task_main_image``).
    """
    task = Task(task_path)
    aliases: list[str] = []
    with tempfile.TemporaryDirectory(prefix="reef-template-") as trial_directory:
        for environment_dir, environment_config in task_environments(task):
            environment = KillOnTimeoutE2BEnvironment(
                environment_dir=environment_dir,
                environment_name=task.short_name,
                session_id=f"{task.short_name}__template",
                trial_paths=TrialPaths(Path(trial_directory)),
                task_env_config=environment_config,
            )
            # A verifier environment that is the task's environment has the task environment's template.
            if environment._template_name in aliases:
                continue
            # Harbor's ``start`` without the sandbox: these protected methods are the template part of it.
            if not await environment._does_template_exist():
                await environment._create_template()
            aliases.append(environment._template_name)
    return aliases[0]


async def build_task_main_image(task_path: Path) -> str | None:
    """Build the image of the ``main`` service of the Compose project of the task directory ``task_path``, and return
    its reference. Return None, and build nothing, when a trial of the task runs no Compose project
    (``KillOnTimeoutE2BEnvironment.compose_ops`` is None) or runs the task's ``docker_image`` as ``main``.

    A trial of the task after this call finds the image, and does not build it. A caller that makes the task's
    templates without ``build_task_template`` calls this for the image. The build waits for a free AgentENV builder
    until the task's build time limit, and raises what the build raises.
    """
    task = Task(task_path)
    with tempfile.TemporaryDirectory(prefix="reef-template-") as trial_directory:
        environment = KillOnTimeoutE2BEnvironment(
            environment_dir=task.paths.environment_dir,
            environment_name=task.short_name,
            session_id=f"{task.short_name}__template",
            trial_paths=TrialPaths(Path(trial_directory)),
            task_env_config=task.config.environment,
        )
        if environment.compose_ops is None or environment.task_env_config.docker_image:
            return None
        await environment._create_template()
        return main_image(environment._template_name)[1]


__all__ = [
    "AENV_TEMPLATE_SUFFIX",
    "COPY_MODES_TEMPLATE_SUFFIX",
    "KillOnTimeoutE2BEnvironment",
    "TemplateNotPreparedError",
    "build_task_main_image",
    "build_task_template",
    "dockerfile_with_copy_modes",
    "task_environments",
]
