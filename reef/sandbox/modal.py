"""Run rollout processes in a Modal Sandbox (:mod:`reef.sandbox.remote`).

A Modal Sandbox is a gVisor container that Modal runs; nested bubblewrap jails work in it. :class:`ModalExecutor` is
the deployment's setting (the Modal app the sandboxes belong to, the image, the resources, how long a sandbox may
live); :meth:`ModalExecutor.open` starts one sandbox and answers a :class:`ModalSession`. Modal's client takes its
credentials from ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET``, or from the profile in ``~/.modal.toml``.

Without ``modal_image``, the image is Node 22 with bubblewrap, Python 3 and the harness's pinned npm package; Modal
builds it on first use and caches it. A ``modal_image`` must hold the program's binary, ``bwrap``, ``python3`` and
coreutils' ``timeout``. Modal grants a sandbox at most a day, and a session cannot extend it.
"""

from __future__ import annotations

import logging
import math
import shlex
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
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

#: The longest life Modal grants a sandbox.
SANDBOX_MAX_TIMEOUT_S = 86400
#: The Modal app that a deployment's sandboxes belong to when ``modal_app`` is not set.
DEFAULT_APP = "reef-sandboxes"
#: The base image of a sandbox built for an npm program.
NODE_IMAGE = "node:22-bookworm-slim"
#: Seconds a process may outlive its limit before ``timeout`` kills it outright.
KILL_AFTER_S = 10
#: The exit status of coreutils' ``timeout`` when the limit ended the command, and when it had to kill it.
TIMEOUT_STATUSES = (124, 137)


@dataclass(frozen=True)
class ModalExecutor(RemoteExecutor):
    """A :class:`RemoteExecutor` on Modal.

    ``app_name`` is the non-empty name of the Modal app the sandboxes belong to; ``image`` is a registry image
    reference, or ``""`` for the image built from ``npm_package``; ``cpu`` (cores) and ``memory_mb`` are the
    sandbox's requests, or ``None`` for Modal's defaults.
    """

    provider: ClassVar[str] = "modal"
    provider_options: ClassVar[frozenset[str]] = frozenset(
        {"modal_app", "modal_image", "modal_cpu", "modal_memory_mb"}
    )
    image_option: ClassVar[str] = "modal_image"

    app_name: str = DEFAULT_APP
    image: str = ""
    cpu: float | None = None
    memory_mb: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.app_name, str) or not self.app_name:
            raise SandboxUnavailable("modal_app must be a non-empty string")
        if self.cpu is not None and (
            isinstance(self.cpu, bool) or not isinstance(self.cpu, int | float) or self.cpu <= 0
        ):
            raise SandboxUnavailable("modal_cpu must be a positive number")
        if self.memory_mb is not None and (
            isinstance(self.memory_mb, bool) or not isinstance(self.memory_mb, int) or self.memory_mb <= 0
        ):
            raise SandboxUnavailable("modal_memory_mb must be a positive int")

    @classmethod
    def from_config(cls, section: Mapping[str, object], environ: Mapping[str, str]) -> Self:
        check_options("Modal", section, cls.provider_options)
        app_name = section.get("modal_app", DEFAULT_APP)
        image = section.get("modal_image", "")
        if not isinstance(app_name, str) or not isinstance(image, str):
            raise SandboxUnavailable("modal_app and modal_image must be strings")
        cpu = section.get("modal_cpu")
        if cpu is not None and (isinstance(cpu, bool) or not isinstance(cpu, int | float)):
            raise SandboxUnavailable("modal_cpu must be a positive number")
        memory_mb = section.get("modal_memory_mb")
        if memory_mb is not None and (isinstance(memory_mb, bool) or not isinstance(memory_mb, int)):
            raise SandboxUnavailable("modal_memory_mb must be a positive int")
        return cls(
            app_name=app_name.strip(),
            image=image.strip(),
            cpu=None if cpu is None else float(cpu),
            memory_mb=memory_mb,
            forward_ports=parse_forward_ports(section),
            env=parse_env_from(section, environ),
        )

    def has_image(self) -> bool:
        return bool(self.image)

    def with_program(self, program: SandboxProgram, binary: str) -> Self:
        return replace(self, npm_package=program.npm_package, binary=binary)

    def sandbox_image(self) -> Any:
        """The Modal image the sandbox runs: the named one, or Node 22 with the program and the jail's tools."""
        import modal

        if self.image:
            return modal.Image.from_registry(self.image)
        if not self.npm_package:
            raise RolloutLaunchError("a Modal sandbox needs modal_image or a program with an npm package")
        return (
            modal.Image.from_registry(NODE_IMAGE)
            .apt_install("bubblewrap", "python3", "coreutils")
            .run_commands(f"npm install -g {shlex.quote(self.npm_package)}")
        )

    def app(self) -> Any:
        import modal

        return modal.App.lookup(self.app_name, create_if_missing=True)

    def reap(self) -> int:
        if not self.owner:
            return 0
        import modal

        stopped = 0
        for sandbox in modal.Sandbox.list(app_id=self.app().app_id, tags=self.labels()):
            if sandbox.get_tags().get("reef_owner") == self.owner:
                sandbox.terminate()
                stopped += 1
        return stopped

    def preflight(self) -> None:
        try:
            import modal
            from modal.config import config
        except ImportError as exc:
            raise SandboxUnavailable(
                "a Modal sandbox needs the modal package: pip install 'reef-infra[modal]'"
            ) from exc
        if not (config.get("token_id") and config.get("token_secret")):
            raise SandboxUnavailable(
                "a Modal sandbox needs credentials: set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET, or run `modal setup`"
            )
        if self.validated or not (self.image or self.npm_package):
            return
        # Probing must not reap a running deployment's sandboxes or require its HTTP server to be listening yet.
        session = None
        try:
            session = replace(self, owner="", forward_ports=(), timeout_s=60).open()
            session.run(probe_command(self.binary))
        except (modal.exception.Error, RolloutLaunchError) as exc:
            raise SandboxUnavailable(f"Modal startup check failed: {exc}") from exc
        finally:
            if session is not None:
                try:
                    session.close()
                except RolloutLaunchError as exc:
                    raise SandboxUnavailable("Modal startup probe could not be cleaned up") from exc
        object.__setattr__(self, "validated", True)

    def open(self) -> ModalSession:
        lifetime = math.ceil(self.timeout_s + SANDBOX_MARGIN_S)
        if lifetime > SANDBOX_MAX_TIMEOUT_S:
            raise RolloutLaunchError(
                f"a Modal sandbox lives at most {SANDBOX_MAX_TIMEOUT_S} s; this run needs {lifetime} s"
            )
        import modal

        self.reap_once()
        ports = [TUNNEL_PORT_BASE + index for index in range(len(self.forward_ports))]
        try:
            sandbox = modal.Sandbox.create(
                app=self.app(),
                image=self.sandbox_image(),
                timeout=lifetime,
                tags=self.labels(),
                encrypted_ports=ports,
                cpu=self.cpu,
                memory=self.memory_mb,
            )
        except Exception as exc:
            raise RolloutLaunchError(f"the Modal sandbox did not start: {exc}") from exc
        session = ModalSession(sandbox, env=self.env)
        try:
            for port, tunnel_port in zip(self.forward_ports, ports, strict=True):
                session.forward(port, tunnel_port)
        except BaseException:
            session.close()
            raise
        return session


class ModalSession(RemoteSession):
    """One running Modal Sandbox (:class:`RemoteSession`); its processes run as root inside the jail."""

    provider: ClassVar[str] = "Modal"

    def __init__(self, sandbox: Any, *, env: Mapping[str, str] | None = None) -> None:
        super().__init__(env=env)
        self.sandbox = sandbox
        #: The processes started in the background, such as the relays; they end with the sandbox.
        self.background: list[Any] = []

    def write_file(self, path: str, data: bytes, *, as_root: bool = False) -> None:
        self.sandbox.filesystem.write_bytes(data, path)

    def read_file(self, path: str) -> bytes:
        return bytes(self.sandbox.filesystem.read_bytes(path))

    def run(self, command: str, *, as_root: bool = False) -> None:
        process = self.sandbox.exec("/bin/sh", "-c", command)
        process.wait()
        if process.returncode != 0:
            raise RolloutLaunchError(
                f"the Modal sandbox command exited with {process.returncode}: {process.stderr.read().strip()}"
            )

    def start(self, command: str, *, as_root: bool = False) -> None:
        self.background.append(self.sandbox.exec("/bin/sh", "-c", command))

    def public_url(self, port: int) -> str:
        return str(self.sandbox.tunnels()[port].url)

    def stop(self) -> None:
        self.sandbox.terminate()

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
        """Run one process in the jail; ``timeout`` stops its whole tree, and Modal's own limit backs it up."""
        command, envs = remote_command(argv, {**env, **self.env}, root)
        command = protected_command(command, envs, root, workspace, writable_paths, readonly_paths)
        seconds = max(math.ceil(timeout), 1)
        limited = f"timeout --kill-after={KILL_AFTER_S} {seconds} /bin/sh -c {shlex.quote(command)}"
        cwd = f"{remote_root(root)}/{workspace.relative_to(root).as_posix()}"
        finished = False
        try:
            self.push(root)
            started = time.monotonic()
            process = self.sandbox.exec(
                "/bin/sh", "-c", limited, workdir=cwd, env=envs, timeout=seconds + KILL_AFTER_S + 30
            )
            process.wait()
            status = process.returncode
            if status in TIMEOUT_STATUSES and time.monotonic() - started >= timeout:
                raise RolloutTimeout(f"the process ran past its {timeout:g} s limit in the Modal sandbox")
            outcome = ProcessOutcome(status, process.stdout.read(), process.stderr.read())
            finished = True
            return outcome
        except RolloutTimeout:
            raise
        except Exception as exc:
            raise RolloutLaunchError(f"the Modal sandbox could not run the process: {exc}") from exc
        finally:
            try:
                self.pull(root)
            except Exception as exc:
                if finished:
                    raise RolloutLaunchError(f"could not copy {root.name} back from the Modal sandbox") from exc
                logger.warning("could not copy %s back from the failed Modal process", root.name)


__all__ = ["ModalExecutor", "ModalSession"]
