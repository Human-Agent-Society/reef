"""Run task processes in an E2B cloud sandbox (:mod:`reef.execution.sandbox.remote`).

An E2B sandbox is a microVM of its own. :class:`E2BExecutor` is the deployment's
setting (the API key, the template, how long a sandbox may live);
:meth:`E2BExecutor.open` starts one sandbox and answers an :class:`E2BSession`.
The template is the harness's pinned binary on Node 22; one Reef builds on
first use (about a minute) when the alias does not exist yet. E2B grants a
sandbox at most an hour when it is created, so a session that runs longer
asks for the hour again before it ends (:class:`SandboxKeeper`).

``KillOnTimeoutE2BEnvironment`` reuses Harbor's E2B environment for task trials. It requests deletion at timeout,
so AgentENV deletes an abandoned sandbox instead of pausing it. The shared ``environment_config`` selects this
class for Reef's Harbor rollouts and generator checks through ``import_path``. A standalone Harbor trial that
selects only ``e2b`` uses Harbor's own class.

``REEF_E2B_PRIVATE_TRAFFIC=true`` (:func:`private_traffic_enabled`) makes each sandbox that both classes create
private: its ports answer only a request with the sandbox's traffic access token. Creation fails, and the sandbox
is killed, when the service returns no token. An :class:`E2BSession` sends the token to its relays
(:meth:`E2BSession.tunnel_headers`). The SDK's own command and file requests use the sandbox's envd token.

This module can load without the optional E2B SDK. Opening a sandbox requires the SDK; the Harbor subclass also
requires Harbor 0.25, installed through ``reef-infra[e2b]``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Self, override

from harbor.environments.e2b import E2BEnvironment
from harbor.models.task.config import NetworkMode
from tenacity import retry, stop_after_attempt, wait_exponential

from reef.execution.sandbox.process import (
    ProcessOutcome,
    ProgramLaunchError,
    ProgramTimeoutError,
    SandboxProgram,
    SandboxUnavailableError,
)
from reef.execution.sandbox.remote import (
    SANDBOX_MARGIN_S,
    TUNNEL_PORT_BASE,
    RemoteExecutor,
    RemoteSession,
    check_options,
    parse_env_from,
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
#: The process setting that makes each E2B sandbox Reef creates private: ``true`` or ``false``; unset is ``false``.
PRIVATE_TRAFFIC_ENV = "REEF_E2B_PRIVATE_TRAFFIC"
#: The header that carries a sandbox's traffic access token to a port of a private sandbox.
TRAFFIC_ACCESS_TOKEN_HEADER = "e2b-traffic-access-token"


def private_traffic_enabled() -> bool:
    """Whether ``REEF_E2B_PRIVATE_TRAFFIC`` asks for private sandboxes: ``true`` yes; ``false`` or unset no.

    A private sandbox is created with ``network.allow_public_traffic`` false. Raise ``ValueError`` for any other
    value, including an empty one, so that a mistyped setting never leaves a sandbox public. The message names the
    setting and its accepted values, not the value that is set.
    """
    value = os.environ.get(PRIVATE_TRAFFIC_ENV)
    if value is None or value == "false":
        return False
    if value == "true":
        return True
    raise ValueError(f"{PRIVATE_TRAFFIC_ENV} must be true or false")


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
    alias is built first. :meth:`open` creates a private sandbox when :func:`private_traffic_enabled`.
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
            private_traffic = private_traffic_enabled()
            if self.npm_package:
                ensure_template(self.template, self.npm_package, self.api_key)
            sandbox = Sandbox.create(
                template=self.template,
                # E2B refuses a longer life than its ceiling; the session extends it while the run still needs it.
                timeout=min(int(self.timeout_s + SANDBOX_MARGIN_S), SANDBOX_MAX_TIMEOUT_S),
                api_key=self.api_key,
                metadata=self.labels(),
                network={"allow_public_traffic": False} if private_traffic else None,
            )
        except Exception as exc:
            raise ProgramLaunchError(f"the E2B sandbox did not start: {exc}") from exc
        if private_traffic and not sandbox.traffic_access_token:
            try:
                sandbox.kill()
            except Exception as exc:
                logger.warning("could not stop the E2B sandbox that has no traffic access token: %s", exc)
            raise ProgramLaunchError("the E2B service returned no traffic access token for a private sandbox")
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
    """Harbor's ``E2BEnvironment`` whose sandbox is created with ``lifecycle={"on_timeout": "kill"}``.

    Values: those of ``E2BEnvironment``. Harbor makes one from a trial's environment config, so the template, the
    sandbox's lifetime (``sandbox_timeout_secs``), its metadata, its startup environment and its network options are
    the parent's, and ``type()`` is ``e2b``.
    Purpose: Harbor's ``EnvironmentFactory`` makes one for an ``e2b`` trial whose environment config names this class
    as its ``import_path``. ``start`` creates the sandbox with ``_create_sandbox``, and ``stop`` deletes it, as before.
    When :func:`private_traffic_enabled`, ``_create_sandbox`` adds ``allow_public_traffic`` false to Harbor's network
    options, and keeps their egress rules. It then requires the sandbox's traffic access token: without one, it kills
    the sandbox and raises ``RuntimeError``. An invalid setting raises ``ValueError``. Harbor's retry policy tries
    either failure once more.
    """

    # Harbor 0.25.0 names this method and the attributes that it reads; an override cannot rename them. The body is
    # Harbor's create call with one more argument, and the private traffic rule when the setting asks for it. The
    # decorator is Harbor's retry policy, because Harbor's decorator wraps its own method and not this one. When
    # Harbor adds an argument to the call, add it here too: the parity test in tests/sandbox/test_task_environment.py
    # compares the two calls.
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    @override
    async def _create_sandbox(self) -> None:
        # Keep this optional SDK import inside sandbox creation.
        from e2b import AsyncSandbox

        private_traffic = private_traffic_enabled()
        network = self._sandbox_create_network_options()
        if private_traffic:
            network = (
                {"allow_public_traffic": False} if network is None else {**network, "allow_public_traffic": False}
            )
        sandbox = await AsyncSandbox.create(
            template=self._template_name,
            metadata={"environment_name": self.environment_name, "session_id": self.session_id},
            envs=self._startup_env(),
            timeout=self._sandbox_timeout,
            allow_internet_access=self.network_policy.network_mode != NetworkMode.NO_NETWORK,
            network=network,
            lifecycle={"on_timeout": "kill"},
        )
        if private_traffic and not sandbox.traffic_access_token:
            try:
                await sandbox.kill()
            except Exception as exc:
                logger.warning("could not stop the E2B sandbox that has no traffic access token: %s", exc)
            raise RuntimeError("the E2B service returned no traffic access token for a private sandbox")
        self._sandbox = sandbox


__all__ = [
    "PRIVATE_TRAFFIC_ENV",
    "TRAFFIC_ACCESS_TOKEN_HEADER",
    "E2BExecutor",
    "E2BSession",
    "KillOnTimeoutE2BEnvironment",
    "SandboxKeeper",
    "ensure_template",
    "private_traffic_enabled",
    "template_alias",
]
