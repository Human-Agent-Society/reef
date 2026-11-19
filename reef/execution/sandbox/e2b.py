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
``aenv build`` (:mod:`reef.execution.sandbox.agentenv_build`). The shared
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

from harbor.environments.e2b import E2BEnvironment
from harbor.models.task.config import NetworkMode
from harbor.models.task.task import Task
from harbor.models.task.verifier_mode import resolve_verifier_environment_definition
from harbor.models.trial.paths import TrialPaths
from tenacity import retry, stop_after_attempt, wait_exponential

from reef.execution.sandbox.agentenv_build import (
    AgentEnvTemplates,
    DockerfileBuild,
    dockerfile_build,
    dockerfile_sets_start_command,
    ensure_aenv_template,
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


class KillOnTimeoutE2BEnvironment(E2BEnvironment):
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

    Purpose: Harbor's ``EnvironmentFactory`` makes one for an ``e2b`` trial whose environment config names this class
    as its ``import_path``. ``start`` builds a missing template with ``_create_template``, creates the sandbox with
    ``_create_sandbox``, and ``stop`` deletes it, as before.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dockerfile_build = dockerfile_build(os.environ)
        # Harbor builds the template from the task's image when it has one; then the Dockerfile is not read.
        if self.task_env_config.docker_image:
            return
        if self.dockerfile_build == "aenv":
            self._template_name += AENV_TEMPLATE_SUFFIX
            return
        dockerfile = self._environment_definition_path.read_text()
        if dockerfile_with_copy_modes(dockerfile, self.environment_dir) != dockerfile:
            self._template_name += COPY_MODES_TEMPLATE_SUFFIX

    @property
    def builds_with_aenv(self) -> bool:
        """Whether ``aenv build`` builds this trial's template: the task builds from a Dockerfile, and
        ``REEF_E2B_DOCKERFILE_BUILD`` is ``aenv``."""
        return self.dockerfile_build == "aenv" and not self.task_env_config.docker_image

    @override
    async def _does_template_exist(self) -> bool:
        # An AgentENV alias can name a template that is still building or that failed. ``ensure_aenv_template`` in
        # ``_create_template`` reads its status and returns at once for a ready template.
        if self.builds_with_aenv:
            return False
        return await super()._does_template_exist()

    # Harbor 0.25.0's template build, with two changes: with ``builds_with_aenv``, ``aenv build`` builds the template;
    # else the Dockerfile is ``dockerfile_with_copy_modes`` of the task's Dockerfile. The decorator is Harbor's retry
    # policy. The parity test in tests/sandbox/test_task_environment.py compares the build with Harbor's for a
    # Dockerfile without ``--chmod``.
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    @override
    async def _create_template(self) -> None:
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


__all__ = [
    "AENV_TEMPLATE_SUFFIX",
    "COPY_MODES_TEMPLATE_SUFFIX",
    "TRAFFIC_ACCESS_TOKEN_HEADER",
    "E2BExecutor",
    "E2BSession",
    "KillOnTimeoutE2BEnvironment",
    "SandboxKeeper",
    "build_task_template",
    "dockerfile_with_copy_modes",
    "ensure_template",
    "template_alias",
]
