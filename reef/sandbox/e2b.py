"""Run rollout processes in an E2B cloud sandbox (:mod:`reef.sandbox.remote`).

An E2B sandbox is a microVM of its own. :class:`E2BExecutor` is the deployment's
setting (the API key, the template, how long a sandbox may live);
:meth:`E2BExecutor.open` starts one sandbox and answers an :class:`E2BSession`.
The template is the harness's pinned binary on Node 22; one Reef builds on
first use (about a minute) when the alias does not exist yet. E2B grants a
sandbox at most an hour when it is created, so a session that runs longer
asks for the hour again before it ends (:class:`SandboxKeeper`).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Self

from reef.sandbox.process import ProcessOutcome, RolloutLaunchError, RolloutTimeout, SandboxProgram, SandboxUnavailable
from reef.sandbox.remote import (
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


def template_alias(adapter: str, version: str) -> str:
    """The pinned agent template, versioned separately for its rollout isolation dependencies."""
    return f"reef-{adapter}-{version}-rollouts-v1".replace(".", "-").lower()


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
            raise SandboxUnavailable("e2b_api_key_env must name an environment variable")
        api_key = section.get("e2b_api_key") or environ.get(key_name, "")
        template = section.get("e2b_template", "")
        if not isinstance(api_key, str) or not isinstance(template, str):
            raise SandboxUnavailable("e2b_api_key and e2b_template must be strings")
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
            raise SandboxUnavailable(
                "an E2B sandbox needs an API key: set E2B_API_KEY or the selected sandbox's e2b_api_key"
            )
        try:
            import e2b
        except ImportError as exc:
            raise SandboxUnavailable("an E2B sandbox needs the e2b package: pip install 'reef-infra[e2b]'") from exc
        if not self.template or self.validated:
            return
        # Probing must not reap a running deployment's sandboxes or require its HTTP server to be listening yet.
        session = None
        try:
            session = replace(self, owner="", forward_ports=(), timeout_s=60).open()
            session.sandbox.commands.run(probe_command(self.binary), timeout=30)
        except (e2b.SandboxException, RolloutLaunchError) as exc:
            raise SandboxUnavailable(f"E2B startup check failed: {exc}") from exc
        finally:
            if session is not None:
                try:
                    session.close()
                except RolloutLaunchError as exc:
                    raise SandboxUnavailable("E2B startup probe could not be cleaned up") from exc
        object.__setattr__(self, "validated", True)

    def open(self) -> E2BSession:
        if not self.template:
            raise RolloutLaunchError("an E2B sandbox needs a template")
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
            raise RolloutLaunchError(f"the E2B sandbox did not start: {exc}") from exc
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
            raise RolloutLaunchError("the E2B command connection could not be restored")
        except CommandExitException as exc:
            finished = True
            return ProcessOutcome(exc.exit_code, exc.stdout, exc.stderr)
        except TimeoutException as exc:
            raise RolloutTimeout(f"the process ran past its {timeout:g} s limit in the E2B sandbox") from exc
        except Exception as exc:
            raise RolloutLaunchError(f"the E2B sandbox could not run the process: {exc}") from exc
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
                    raise RolloutLaunchError(f"could not copy {root.name} back from the E2B sandbox") from exc
                logger.warning("could not copy %s back from the failed E2B process", root.name)


__all__ = ["E2BExecutor", "E2BSession", "SandboxKeeper", "ensure_template", "template_alias"]
