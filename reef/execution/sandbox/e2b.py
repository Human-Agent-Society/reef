"""Run task processes in an E2B cloud sandbox (:mod:`reef.execution.sandbox.remote`).

An E2B sandbox is a microVM of its own. :class:`E2BExecutor` is the deployment's
setting (the API key, the template, how long a sandbox may live);
:meth:`E2BExecutor.open` starts one sandbox and answers an :class:`E2BSession`.
The template is the harness's pinned binary on Node 22; one Reef builds on
first use (about a minute) when the alias does not exist yet. E2B grants a
sandbox at most an hour when it is created, so a session that runs longer
asks for the hour again before it ends (:class:`SandboxKeeper`).

``KillOnTimeoutE2BEnvironment`` reuses Harbor's E2B environment for task trials. It requests deletion at timeout,
so AgentENV deletes an abandoned sandbox instead of pausing it. It also keeps the file modes of a task's
``COPY --chmod``, which E2B's Dockerfile parser ignores (:func:`dockerfile_with_copy_modes`). On AgentENV, whose E2B
template builds take no files, ``REEF_E2B_DOCKERFILE_BUILD=aenv`` makes it build a task's Dockerfile with
``aenv build`` (:mod:`reef.execution.sandbox.agentenv_build`). There it also runs the Compose project of a task that
has one in one sandbox, so that the agent reaches the task's other services
(:mod:`reef.execution.sandbox.agentenv_compose`). The shared
``environment_config`` selects this class for Reef's Harbor rollouts and generator checks through ``import_path``.
A standalone Harbor trial that selects only ``e2b`` uses Harbor's own class.

A service can keep a sandbox's ports private and return a traffic access token for them. An :class:`E2BSession`
then sends that token with each request to the sandbox's relays (:meth:`E2BSession.tunnel_headers`).

This module can load without the optional E2B SDK. Opening a sandbox requires the SDK; the Harbor subclass also
requires Harbor 0.25, installed through ``reef-infra[e2b]``.
"""

from __future__ import annotations

import io
import logging
import math
import os
import posixpath
import re
import shlex
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Self, override

from harbor.environments.base import ExecResult
from harbor.environments.definition import effective_exec_cwd
from harbor.environments.e2b import E2BEnvironment
from harbor.models.task.config import NetworkMode
from harbor.models.task.task import Task
from harbor.models.task.verifier_mode import resolve_verifier_environment_definition
from harbor.models.trial.paths import TrialPaths
from tenacity import retry, retry_if_exception_type, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from reef.core.errors import InternalError
from reef.execution.sandbox.agentenv_build import (
    AgentEnvTemplates,
    DockerfileBuild,
    build_image,
    dockerfile_build,
    dockerfile_sets_start_command,
    ensure_aenv_template,
)
from reef.execution.sandbox.agentenv_compose import (
    COMPOSE_FILE,
    AgentEnvComposeOps,
    ComposeHost,
    ComposeSandbox,
    ImageNotFoundError,
    TransientStartError,
    compose_project,
    delete_sandbox,
    main_image,
    prepare_guest,
)
from reef.execution.sandbox.process import (
    ProcessOutcome,
    ProgramLaunchError,
    ProgramTimeoutError,
    SandboxProgram,
    SandboxUnavailableError,
    parse_env_from,
)
from reef.execution.sandbox.remote import (
    SANDBOX_MARGIN_S,
    TUNNEL_PORT_BASE,
    RemoteExecutor,
    RemoteSession,
    check_options,
    parse_forward_ports,
    probe_command,
    protected_command,
    remote_command,
    remote_root,
)

logger = logging.getLogger(__name__)

#: The longest life E2B grants a sandbox at creation; asking for more is refused outright, so a run that needs
#: more keeps its own sandbox alive instead (:class:`SandboxKeeper`).
SANDBOX_MAX_TIMEOUT_S = 3600
#: How long before that deadline the keeper asks for it again, so a slow extension still lands in time.
SANDBOX_RENEW_MARGIN_S = 300
#: The header that carries a sandbox's traffic access token to a port of a private sandbox.
TRAFFIC_ACCESS_TOKEN_HEADER = "e2b-traffic-access-token"
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


def template_alias(adapter: str, version: str) -> str:
    """The pinned agent template, versioned separately for its run isolation dependencies."""
    return f"reef-{adapter}-{version}-runs-v1".replace(".", "-").lower()


def ensure_template(alias: str, npm_package: str, api_key: str) -> None:
    """Build ``alias`` from Node 22 with ``npm_package`` installed globally, unless it exists already."""
    from e2b import Template

    if Template.alias_exists(alias, api_key=api_key):
        return
    logger.warning(
        "building the E2B template %s with %s; the first run takes about a minute longer", alias, npm_package
    )
    Template.build(
        Template().from_node_image("22").apt_install("bubblewrap").npm_install(npm_package, g=True),
        alias=alias,
        cpu_count=2,
        memory_mb=2048,
        api_key=api_key,
    )


@dataclass(frozen=True)
class E2BExecutor(RemoteExecutor):
    """A :class:`RemoteExecutor` on E2B.

    ``api_key`` is the E2B key; ``template`` is the sandbox's image by alias. With ``npm_package`` a missing
    alias is built first.
    """

    provider: ClassVar[str] = "e2b"
    provider_options: ClassVar[frozenset[str]] = frozenset({"e2b_api_key", "e2b_api_key_env", "e2b_template"})
    image_option: ClassVar[str] = "e2b_template"

    api_key: str = field(default="", repr=False)
    template: str = ""

    @classmethod
    def from_config(cls, section: Mapping[str, object], environ: Mapping[str, str]) -> Self:
        check_options("E2B", section, cls.provider_options)
        key_name = section.get("e2b_api_key_env", "E2B_API_KEY")
        if not isinstance(key_name, str) or not key_name.isidentifier():
            raise SandboxUnavailableError("e2b_api_key_env must name an environment variable")
        api_key = section.get("e2b_api_key") or environ.get(key_name, "")
        template = section.get("e2b_template", "")
        if not isinstance(api_key, str) or not isinstance(template, str):
            raise SandboxUnavailableError("e2b_api_key and e2b_template must be strings")
        return cls(
            api_key=api_key.strip(),
            template=template.strip(),
            forward_ports=parse_forward_ports(section),
            env=parse_env_from(section, environ),
        )

    def has_image(self) -> bool:
        return bool(self.template)

    def with_program(self, program: SandboxProgram, binary: str) -> Self:
        return replace(
            self,
            template=template_alias(program.name, program.version),
            npm_package=program.npm_package,
            binary=binary,
        )

    def reap(self) -> int:
        if not self.owner:
            return 0
        from e2b import Sandbox, SandboxQuery

        pages = Sandbox.list(query=SandboxQuery(metadata=self.labels()), api_key=self.api_key)
        stopped = 0
        while pages.has_next:
            for info in pages.next_items():
                if (info.metadata or {}).get("reef_owner") == self.owner and Sandbox.kill(
                    info.sandbox_id, api_key=self.api_key
                ):
                    stopped += 1
        return stopped

    def preflight(self) -> None:
        if not self.api_key:
            raise SandboxUnavailableError(
                "an E2B sandbox needs an API key: set E2B_API_KEY or the selected sandbox's e2b_api_key"
            )
        try:
            import e2b
        except ImportError as exc:
            raise SandboxUnavailableError(
                "an E2B sandbox needs the e2b package: pip install 'reef-infra[e2b]'"
            ) from exc
        if not self.template or self.validated:
            return
        # Probing must not reap a running deployment's sandboxes or require its HTTP server to be listening yet.
        session = None
        try:
            session = replace(self, owner="", forward_ports=(), timeout_s=60).open()
            session.sandbox.commands.run(probe_command(self.binary), timeout=30)
        except (e2b.SandboxException, ProgramLaunchError) as exc:
            raise SandboxUnavailableError(f"E2B startup check failed: {exc}") from exc
        finally:
            if session is not None:
                try:
                    session.close()
                except ProgramLaunchError as exc:
                    raise SandboxUnavailableError("E2B startup probe could not be cleaned up") from exc
        object.__setattr__(self, "validated", True)

    def open(self) -> E2BSession:
        if not self.template:
            raise ProgramLaunchError("an E2B sandbox needs a template")
        from e2b import Sandbox

        self.reap_once()
        try:
            if self.npm_package:
                ensure_template(self.template, self.npm_package, self.api_key)
            sandbox = Sandbox.create(
                template=self.template,
                # E2B refuses a longer life than its ceiling; the session extends it while the run still needs it.
                timeout=min(int(self.timeout_s + SANDBOX_MARGIN_S), SANDBOX_MAX_TIMEOUT_S),
                api_key=self.api_key,
                metadata=self.labels(),
            )
        except Exception as exc:
            raise ProgramLaunchError(f"the E2B sandbox did not start: {exc}") from exc
        session = E2BSession(sandbox, env=self.env)
        try:
            for index, port in enumerate(self.forward_ports):
                session.forward(port, TUNNEL_PORT_BASE + index)
        except BaseException:
            session.close()
            raise
        return session


class SandboxKeeper(threading.Thread):
    """Hold one sandbox open past E2B's creation ceiling, for as long as its session is.

    E2B grants a sandbox at most :data:`SANDBOX_MAX_TIMEOUT_S` when it is created, so an agent run allowed more
    than that would lose its sandbox mid-run. The keeper asks for the ceiling again before each one expires; it
    stops with the session, and a refusal ends it rather than repeating against a sandbox that is gone.
    """

    def __init__(self, sandbox: Any) -> None:
        super().__init__(name="reef-e2b-keeper", daemon=True)
        self.sandbox = sandbox
        self.stopped = threading.Event()

    def run(self) -> None:
        period = max(SANDBOX_MAX_TIMEOUT_S - SANDBOX_RENEW_MARGIN_S, 1)
        while not self.stopped.wait(period):
            try:
                self.sandbox.set_timeout(SANDBOX_MAX_TIMEOUT_S)
            except Exception as exc:
                logger.warning("the E2B sandbox's deadline could not be extended, it may end mid-run: %s", exc)
                return

    def stop(self) -> None:
        self.stopped.set()


class E2BSession(RemoteSession):
    """One running E2B sandbox (:class:`RemoteSession`), kept alive by its :class:`SandboxKeeper`."""

    provider: ClassVar[str] = "E2B"

    def __init__(self, sandbox: Any, *, env: Mapping[str, str] | None = None) -> None:
        super().__init__(env=env)
        self.sandbox = sandbox
        self.keeper = SandboxKeeper(sandbox)
        self.keeper.start()

    def write_file(self, path: str, data: bytes, *, as_root: bool = False) -> None:
        if as_root:
            self.sandbox.files.write(path, data, user="root")
        else:
            self.sandbox.files.write(path, data)

    def read_file(self, path: str) -> bytes:
        return bytes(self.sandbox.files.read(path, format="bytes"))

    def run(self, command: str, *, as_root: bool = False) -> None:
        if as_root:
            self.sandbox.commands.run(command, user="root")
        else:
            self.sandbox.commands.run(command)

    def start(self, command: str, *, as_root: bool = False) -> None:
        if as_root:
            self.sandbox.commands.run(command, background=True, user="root", timeout=0)
        else:
            self.sandbox.commands.run(command, background=True, timeout=0)

    def public_url(self, port: int) -> str:
        return f"https://{self.sandbox.get_host(port)}"

    def tunnel_headers(self) -> dict[str, str]:
        """The sandbox's traffic access token, when the service returned a non-empty one; a private sandbox's ports
        require it. The E2B API key is never one of these headers."""
        token = self.sandbox.traffic_access_token
        return {TRAFFIC_ACCESS_TOKEN_HEADER: token} if token else {}

    def stop(self) -> None:
        self.keeper.stop()
        self.sandbox.kill()

    def launch(
        self,
        argv: Sequence[str],
        *,
        root: Path,
        workspace: Path,
        env: Mapping[str, str],
        timeout: float,
        writable_paths: Sequence[Path] = (),
        readonly_paths: Sequence[Path] = (),
    ) -> ProcessOutcome:
        from connectrpc.code import Code
        from connectrpc.errors import ConnectError
        from e2b import CommandExitException, TimeoutException
        from e2b.sandbox_sync.commands.command_handle import CommandHandle

        command, envs = remote_command(argv, {**env, **self.env}, root)
        command = protected_command(command, envs, root, workspace, writable_paths, readonly_paths)
        cwd = f"{remote_root(root)}/{workspace.relative_to(root).as_posix()}"
        process: CommandHandle | None = None
        finished = False
        try:
            self.push(root)
            deadline = time.monotonic() + timeout
            process = self.sandbox.commands.run(command, envs=envs, cwd=cwd, timeout=timeout, background=True)
            process_id = process.pid
            for attempt in range(3):
                try:
                    if attempt:
                        remaining_seconds = deadline - time.monotonic()
                        if remaining_seconds <= 0:
                            raise TimeoutException("the command deadline expired while reconnecting")
                        process = self.sandbox.commands.connect(process_id, timeout=remaining_seconds)
                    result = process.wait()
                    finished = True
                    return ProcessOutcome(result.exit_code, result.stdout, result.stderr)
                except ConnectError as exc:
                    if exc.code not in (Code.INTERNAL, Code.UNKNOWN, Code.UNAVAILABLE) or attempt == 2:
                        raise
                    # Reattach to the same PID: replaying a command could repeat its side effects.
                    process.disconnect()
                    logger.warning("the E2B command connection dropped; reconnecting to process %s", process_id)
            raise ProgramLaunchError("the E2B command connection could not be restored")
        except CommandExitException as exc:
            finished = True
            return ProcessOutcome(exc.exit_code, exc.stdout, exc.stderr)
        except TimeoutException as exc:
            raise ProgramTimeoutError(f"the process ran past its {timeout:g} s limit in the E2B sandbox") from exc
        except Exception as exc:
            raise ProgramLaunchError(f"the E2B sandbox could not run the process: {exc}") from exc
        finally:
            if process is not None and not finished:
                try:
                    self.sandbox.commands.kill(process.pid)
                except Exception as exc:
                    logger.warning("could not stop the E2B process %s: %s", process.pid, exc)
            try:
                self.pull(root)
            except Exception as exc:
                if finished:
                    raise ProgramLaunchError(f"could not copy {root.name} back from the E2B sandbox") from exc
                logger.warning("could not copy %s back from the failed E2B process", root.name)


class TemplateNotPreparedError(InternalError):
    """A trial that builds no template (``KillOnTimeoutE2BEnvironment.builds_templates`` false) found its template
    missing, or not ready on AgentENV."""


class KillOnTimeoutE2BEnvironment(E2BEnvironment, ComposeHost):
    """Harbor's ``E2BEnvironment`` whose sandbox is created with ``lifecycle={"on_timeout": "kill"}``, and whose
    template keeps the modes of ``COPY --chmod``.

    Values: those of ``E2BEnvironment``. Harbor makes one from a trial's environment config, so the sandbox's lifetime
    (``sandbox_timeout_secs``), its metadata, its startup environment and its network options are the parent's, and
    ``type()`` is ``e2b``. ``dockerfile_build`` is the value of ``REEF_E2B_DOCKERFILE_BUILD`` when it was made. The
    template is the parent's, with two differences for a task that builds its image from a Dockerfile:

    - With ``dockerfile_build`` ``aenv``, ``aenv build`` builds the template on the AgentENV deployment
      (``reef.execution.sandbox.agentenv_build``), and its alias ends with ``AENV_TEMPLATE_SUFFIX``.
    - Else, when the Dockerfile has ``--chmod``, the template is built from ``dockerfile_with_copy_modes`` of it, and
      its alias ends with ``COPY_MODES_TEMPLATE_SUFFIX``.

    ``builds_templates`` is the bool of the environment config's ``kwargs.builds_templates``, True when the config
    has none. When it is false, the environment builds no template: ``_create_template`` raises
    ``TemplateNotPreparedError``, and on AgentENV ``_does_template_exist`` is true only for a ready template.

    ``compose_ops`` is not None if and only if ``dockerfile_build`` is ``aenv`` and the environment directory has
    ``docker-compose.yaml``. The environment then runs the task's Compose project in one AgentENV sandbox
    (``start_compose``), and it differs from the above in these points:

    - The sandbox is the guest of the project, and each command and file transfer of the trial goes to the ``main``
      service. The project's services reach one another by name.
    - The environment uses no template. ``aenv build --image`` builds the task's Dockerfile into the image of
      ``main`` (``main_image`` of the template alias); a task with a ``docker_image`` runs that image.
      ``_create_template`` builds the image, and ``_does_template_exist`` is false, because AgentENV does not tell
      whether it has an image. A start builds the image only after AgentENV refused the project for a missing image.
    - AgentENV pauses the sandbox at its timeout, and does not delete it: ``aenv compose up`` sets no other policy.
    - The start does not wait for systemd: the first process of ``main`` is the project's command.

    Purpose: Harbor's ``EnvironmentFactory`` makes one for an ``e2b`` trial whose environment config names this class
    as its ``import_path``, and one for each separate verifier environment of the trial, from the same config.
    ``start`` builds a missing template with ``_create_template``, creates the sandbox with ``_create_sandbox``, and
    ``stop`` deletes it, as before. Then ``start`` waits until the sandbox's systemd, if its first process is systemd,
    emptied ``/tmp`` (``SYSTEMD_TMP_CLEANUP_WAIT``). ``build_task_template`` makes one that builds templates.
    """

    def __init__(self, *args: Any, builds_templates: bool = True, **kwargs: Any) -> None:
        if not isinstance(builds_templates, bool):
            raise TypeError("builds_templates is a bool")
        super().__init__(*args, **kwargs)
        self.builds_templates = builds_templates
        self.dockerfile_build = dockerfile_build(os.environ)
        self.compose_ops: AgentEnvComposeOps | None = (
            AgentEnvComposeOps(self)
            if self.dockerfile_build == "aenv" and (self.environment_dir / COMPOSE_FILE).is_file()
            else None
        )
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
        if self.compose_ops is not None:
            await self.start_compose(force_build)
            return
        await super().start(force_build)
        # A task image can boot systemd, which empties /tmp some seconds after the start. The agent's install uploads
        # its archives there next, so it waits for that step. The wait is a best effort: its result is not checked.
        await self.exec(SYSTEMD_TMP_CLEANUP_WAIT, timeout_sec=SYSTEMD_TMP_CLEANUP_WAIT_SECONDS, user="root")

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

        templates = AgentEnvTemplates.from_environ(os.environ)
        image = self.task_env_config.docker_image or main_image(self._template_name)[1]
        cpus = self._effective_cpus
        sandbox = ComposeSandbox(
            project=compose_project((self.environment_dir / COMPOSE_FILE).read_text(), image, self._startup_env()),
            cpu_count=None if cpus is None else int(cpus),
            memory_mb=self._effective_memory_mb,
            lifetime_seconds=self._sandbox_timeout,
            startup_seconds=min(max(math.ceil(self.task_env_config.build_timeout_sec), 1), 86400),
        )
        if force_build:
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
            self._sandbox = await AsyncSandbox.connect(sandbox_id, timeout=self._sandbox_timeout)
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

    # Harbor 0.25.0's template build, with four changes: an environment that builds no template refuses; with a
    # Compose project, ``aenv build --image`` builds the image of the main service; with ``builds_with_aenv``,
    # ``aenv build`` builds the template; else the Dockerfile is ``dockerfile_with_copy_modes`` of the task's
    # Dockerfile. Harbor's retry policy applies to builds. A trial that cannot build fails immediately.
    # The parity test in tests/sandbox/test_task_environment.py compares the build with Harbor's for a Dockerfile
    # without ``--chmod``.
    @retry(
        retry=retry_if_not_exception_type(TemplateNotPreparedError),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    @override
    async def _create_template(self) -> None:
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

    # Harbor 0.25.0 names this method and the attributes that it reads; an override cannot rename them. The body is
    # Harbor's create call with one more argument, and the decorator is Harbor's retry policy, because Harbor's
    # decorator wraps its own method and not this one. When Harbor adds an argument to the call, add it here too: the
    # parity test in tests/sandbox/test_task_environment.py compares the two calls.
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    @override
    async def _create_sandbox(self) -> None:
        # Keep this optional SDK import inside sandbox creation.
        from e2b import AsyncSandbox

        self._sandbox = await AsyncSandbox.create(
            template=self._template_name,
            metadata={"environment_name": self.environment_name, "session_id": self.session_id},
            envs=self._startup_env(),
            timeout=self._sandbox_timeout,
            allow_internet_access=self.network_policy.network_mode != NetworkMode.NO_NETWORK,
            network=self._sandbox_create_network_options(),
            lifecycle={"on_timeout": "kill"},
        )


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
    environments = [(task.paths.environment_dir, task.config.environment)]
    for step in task.config.steps or [None]:
        verifier = resolve_verifier_environment_definition(task.config, task.paths, step)
        if verifier is not None:
            environments.append((verifier.directory, verifier.config))
    aliases: list[str] = []
    with tempfile.TemporaryDirectory(prefix="reef-template-") as trial_directory:
        for environment_dir, environment_config in environments:
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
    "TRAFFIC_ACCESS_TOKEN_HEADER",
    "E2BExecutor",
    "E2BSession",
    "KillOnTimeoutE2BEnvironment",
    "SandboxKeeper",
    "TemplateNotPreparedError",
    "build_task_main_image",
    "build_task_template",
    "dockerfile_with_copy_modes",
    "ensure_template",
    "template_alias",
]
