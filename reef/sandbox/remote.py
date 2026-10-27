"""Run episode processes in a cloud sandbox that reaches the Reef host only through a tunnel.

A cloud sandbox (an E2B microVM, a Modal Sandbox) has the internet but no route to the machine Reef runs on. A
:class:`RemoteExecutor` is the deployment's setting for one provider; :meth:`RemoteExecutor.open` starts one
sandbox and answers a :class:`RemoteSession`, which launches any number of processes in it until it is closed.
Each process's root is copied in before it starts and copied back once it ends, so the caller reads what it wrote
as a local run leaves it; :meth:`RemoteSession.pull` refreshes a directory while a process still runs. Every
process runs in a bubblewrap jail (:func:`protected_command`) that keeps the sandbox's filesystem read-only except
the directories the episode declares.

A port in ``forward_ports`` is reachable inside the sandbox at the same loopback address it has on the Reef host,
so no URL handed to a process changes. The sandbox runs :mod:`reef.sandbox.relay` on that port; a
:class:`TunnelPump` on the Reef host polls the relay over the provider's public address for each request a
process made, sends it to the local port, and streams the answer back. Only the forwarded ports are reachable
this way, and every credential stays on the Reef side of them.

A sandbox carries its deployment as ``owner``; a Reef that stopped mid run never closed its sandbox, so before a
process starts its first sandbox it stops the ones its deployment left (:meth:`RemoteExecutor.reap_once`), and
never another deployment's.

A provider module (:mod:`reef.sandbox.e2b`, :mod:`reef.sandbox.modal`) implements the sandbox's start, its file and
command primitives, and the launch of one process. :mod:`reef.sandbox.config` is the only place that names a
provider.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import secrets
import shlex
import shutil
import socket
import tarfile
import tempfile
import threading
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Self

from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError, ClientSession, ClientTimeout

from reef.sandbox.process import (
    ISOLATION_ENV,
    EpisodeExecutor,
    EpisodeLaunchError,
    ProcessOutcome,
    SandboxProgram,
    SandboxUnavailable,
)

logger = logging.getLogger(__name__)

RELAY_SCRIPT = Path(__file__).with_name("relay.py")
#: Where a process's root lands in the sandbox: one directory per root, named as the local one.
REMOTE_BASE = "/home/user/reef"
#: The first sandbox port a relay's tunnel listens on; the provider exposes it at a public address.
TUNNEL_PORT_BASE = 21080
#: Concurrent polls per forwarded port: a long request (a trial) must not hold up the calls it makes itself.
POLLERS = 8
#: How long a sandbox outlives the run it was opened for, so the copy back and the close still find it.
SANDBOX_MARGIN_S = 600
#: How long a new relay has to answer before the session gives up on it.
RELAY_READY_S = 60.0
#: The (provider, owner) pairs this process has reaped: each once, before its first sandbox.
REAPED_OWNERS: set[tuple[str, str]] = set()
REAP_LOCK = threading.Lock()
#: Headers that describe one connection and never cross the tunnel.
HOP_BY_HOP = frozenset(
    {"connection", "content-length", "host", "keep-alive", "te", "trailer", "transfer-encoding", "upgrade"}
)
#: The Harbor environment types that run a task's containers on a cloud provider, each with the environment
#: variables its credentials come from. A process that runs Harbor tasks in one of them needs these variables.
REMOTE_TASK_ENVIRONMENTS: Mapping[str, tuple[str, ...]] = {
    "e2b": ("E2B_API_KEY",),
    "modal": ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"),
}


def deployment_owner(anchor: Path) -> str:
    """Which Reef deployment a sandbox belongs to: this host and one of the deployment's own state directories."""
    return hashlib.sha256(f"{socket.gethostname()}:{anchor.resolve()}".encode()).hexdigest()[:16]


def remote_root(root: Path) -> str:
    return f"{REMOTE_BASE}/{root.name}"


def remote_command(argv: Sequence[str], env: Mapping[str, str], root: Path) -> tuple[str, dict[str, str]]:
    """The command line and environment for the sandbox: the root's local path becomes its sandbox path, the
    binary is resolved in the sandbox's image, and the host's ``PATH`` stays behind."""
    local, remote = str(root), remote_root(root)
    args = [arg.replace(local, remote) for arg in argv]
    envs = {name: value.replace(local, remote) for name, value in env.items() if name != "PATH"}
    return shlex.join(args), envs


def protected_command(
    command: str,
    env: Mapping[str, str],
    root: Path,
    workspace: Path,
    writable_paths: Sequence[Path],
    readonly_paths: Sequence[Path],
) -> str:
    """Keep the sandbox's filesystem read-only except the episode's declared output directories.

    The PID namespace retires tool children when the command is stopped. Bubblewrap
    also prevents privilege escalation, so sudo cannot undo the read-only mounts.
    """
    arguments = [
        "bwrap",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--die-with-parent",
        "--new-session",
        "--ro-bind",
        "/",
        "/",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--clearenv",
    ]
    remote = remote_root(root)
    for path in (workspace, *writable_paths):
        target = f"{remote}/{path.relative_to(root).as_posix()}"
        arguments.extend(("--bind", target, target))
    for path in readonly_paths:
        target = f"{remote}/{path.relative_to(root).as_posix()}"
        arguments.extend(("--ro-bind", target, target))
    for name, value in {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        **env,
        ISOLATION_ENV: "bwrap",
        "REEF_NATIVE_ENFORCE": "bwrap",
    }.items():
        arguments.extend(("--setenv", name, value))
    arguments.extend(
        ("--chdir", f"{remote}/{workspace.relative_to(root).as_posix()}", "--", "/bin/sh", "-c", f"exec {command}")
    )
    return f"exec {shlex.join(arguments)}"


def pack(directory: Path) -> bytes:
    """``directory``'s contents as a gzipped tar, paths relative to it."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(directory.iterdir()):
            archive.add(path, arcname=path.name)
    return buffer.getvalue()


def keep_safe(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
    """tarfile's ``data`` filter, skipping what it refuses instead of failing the whole copy: a process leaves
    such things in its home (pulseaudio's runtime link to an absolute path), and the rest must still come back."""
    try:
        return tarfile.data_filter(member, path)
    except tarfile.FilterError:
        logger.debug("left %s in the sandbox: it would land outside the copy", member.name)
        return None


def unpack(data: bytes, directory: Path) -> None:
    """Replace ``directory``'s contents with the archive's; a member that would land outside it, or link there,
    is left out."""
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reef-copy-", dir=directory.parent) as temporary:
        staged = Path(temporary) / "root"
        staged.mkdir()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            archive.extractall(staged, filter=keep_safe)
        if directory.exists():
            shutil.rmtree(directory)
        staged.replace(directory)


def probe_command(binary: str) -> str:
    """The startup check of a remote sandbox: a nested bubblewrap jail starts, and ``binary`` is on ``PATH``."""
    jail = "bwrap --unshare-user --unshare-pid --ro-bind / / --proc /proc --dev /dev --"
    command = f"{jail} {jail} /bin/true"
    if binary:
        command += f" && command -v {shlex.quote(binary)} >/dev/null"
    return command


def parse_forward_ports(section: Mapping[str, object]) -> tuple[int, ...]:
    """``sandbox.forward_ports``: distinct TCP ports, in order; ``SandboxUnavailable`` for another value."""
    ports = section.get("forward_ports", [])
    if not isinstance(ports, (list, tuple)) or any(
        isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535 for port in ports
    ):
        raise SandboxUnavailable("sandbox.forward_ports must list TCP ports between 1 and 65535")
    return tuple(dict.fromkeys(ports))


def parse_env_from(section: Mapping[str, object], environ: Mapping[str, str]) -> dict[str, str]:
    """``sandbox.env_from``: the named host variables and their values; each must be set and non-empty."""
    names = section.get("env_from", [])
    if not isinstance(names, (list, tuple)) or not all(
        isinstance(name, str) and name.isidentifier() for name in names
    ):
        raise SandboxUnavailable("sandbox.env_from must list environment variable names")
    env = {}
    for name in names:
        if not environ.get(name):
            raise SandboxUnavailable(f"sandbox.env_from requires environment variable {name!r}")
        env[name] = environ[name]
    return env


def check_options(provider: str, section: Mapping[str, object], provider_options: frozenset[str]) -> None:
    """Refuse an option no remote sandbox takes, and the local sandbox's ``egress_hosts`` and ``limits``."""
    unknown = set(section) - provider_options - {"forward_ports", "env_from", "egress_hosts", "limits"}
    if unknown:
        raise SandboxUnavailable(f"unknown {provider} sandbox options: {', '.join(sorted(unknown))}")
    if section.get("egress_hosts") or section.get("limits"):
        raise SandboxUnavailable(
            f"{provider} does not implement sandbox.egress_hosts or sandbox.limits; "
            "it uses the image's resources and permits outbound internet"
        )


@dataclass(frozen=True)
class RemoteExecutor(EpisodeExecutor):
    """Launch each process in a cloud sandbox of one provider; :meth:`open` keeps one sandbox for several.

    Values: ``npm_package`` (``<package>@<version>``) is the program a built image installs, or ``""`` for an
    image the deployment names; ``timeout_s`` (positive) bounds how long a sandbox lives past its margin;
    ``forward_ports`` are distinct Reef host loopback ports the sandbox reaches through the tunnel; ``owner``
    names the deployment on every sandbox it starts, for :meth:`reap`, or is ``""`` for none; ``binary`` is the
    program's binary in the image; ``env`` is the deployment's explicit environment for every process. A
    subclass adds its provider's credentials and image, and implements :meth:`open`, :meth:`reap`,
    :meth:`preflight`, :meth:`has_image` and :meth:`with_program`.

    Purpose: :mod:`reef.sandbox.config` builds one from a deployment's sandbox section; a caller binds it to a
    program with :meth:`for_program` and runs processes through :meth:`launch` or :meth:`open_session`.
    """

    #: The provider's name, as a deployment selects it.
    provider: ClassVar[str] = ""
    #: The sandbox section's options that only this provider reads.
    provider_options: ClassVar[frozenset[str]] = frozenset()
    #: The provider option that names an image holding the program, for an error message.
    image_option: ClassVar[str] = ""

    npm_package: str = ""
    timeout_s: float = 3600.0
    forward_ports: tuple[int, ...] = ()
    owner: str = ""
    binary: str = ""
    env: Mapping[str, str] = field(default_factory=dict, repr=False)
    validated: bool = field(default=False, init=False, compare=False, repr=False)

    @classmethod
    @abstractmethod
    def from_config(cls, section: Mapping[str, object], environ: Mapping[str, str]) -> Self:
        """Parse the deployment's sandbox section without copying the host environment into the sandbox."""

    @abstractmethod
    def has_image(self) -> bool:
        """Whether the sandbox's image is known: one the deployment names, or one built for a program."""

    @abstractmethod
    def with_program(self, program: SandboxProgram, binary: str) -> Self:
        """This executor with an image built for ``program`` (which has an ``npm_package``), launching ``binary``."""

    @abstractmethod
    def open(self) -> RemoteSession:
        """Start a sandbox from the image with a relay on each forwarded port."""

    @abstractmethod
    def reap(self) -> int:
        """Stop this deployment's running sandboxes and answer how many; without an ``owner`` none is touched."""

    def for_program(self, program: SandboxProgram, *, binary: str | None = None) -> tuple[EpisodeExecutor, str]:
        """Resolve the program in the sandbox's image, without installing it on the Reef host."""
        if program.self_isolating:
            raise SandboxUnavailable(
                f"{program.name!r} manages its own containers on the Reef host and cannot run in a "
                f"{self.provider} sandbox"
            )
        remote_binary = binary or self.binary or program.binary
        if self.has_image():
            bound = self if remote_binary == self.binary else replace(self, binary=remote_binary)
            return bound, remote_binary
        if not program.npm_package:
            raise SandboxUnavailable(
                f"running {program.name!r} in {self.provider} requires an explicit {self.image_option} containing "
                "its binary, bubblewrap and python3"
            )
        return self.with_program(program, remote_binary), remote_binary

    def open_session(self, *, forward_ports: tuple[int, ...] = (), timeout_s: float) -> EpisodeExecutor:
        """One sandbox for the run, each forwarded port reachable inside it through the tunnel.

        The sandbox lives at least as long as this executor's own ``timeout_s``: a run may launch several
        processes, each within ``timeout_s``.
        """
        ports = tuple(dict.fromkeys((*self.forward_ports, *forward_ports)))
        return replace(self, forward_ports=ports, timeout_s=max(self.timeout_s, timeout_s)).open()

    def labels(self) -> dict[str, str]:
        """The tags every sandbox this executor starts carries."""
        return {"reef": "episode", **({"reef_owner": self.owner} if self.owner else {})}

    def reap_once(self) -> None:
        """Stop this deployment's leftovers the first time this process opens a sandbox; a failed look is
        logged and tried again next time. Held under a lock, so no sandbox of this process starts meanwhile."""
        if not self.owner:
            return
        key = (self.provider, self.owner)
        with REAP_LOCK:
            if key in REAPED_OWNERS:
                return
            try:
                stopped = self.reap()
            except Exception as exc:
                logger.warning("could not look for %s sandboxes a previous run left: %s", self.provider, exc)
                return
            REAPED_OWNERS.add(key)
        if stopped:
            logger.warning(
                "stopped %d %s sandbox(es) a previous run of this deployment left running", stopped, self.provider
            )

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
        session = replace(self, timeout_s=max(self.timeout_s, timeout)).open()
        try:
            return session.launch(
                argv,
                root=root,
                workspace=workspace,
                env=env,
                timeout=timeout,
                writable_paths=writable_paths,
                readonly_paths=readonly_paths,
            )
        finally:
            session.close()


class RemoteSession(EpisodeExecutor):
    """One running cloud sandbox: launch processes in it, pull what they wrote, close it when done.

    ``env`` is the deployment's explicit environment; every process gets it over the caller's. A subclass
    implements the provider's primitives (:meth:`write_file`, :meth:`read_file`, :meth:`run`, :meth:`start`,
    :meth:`public_url`, :meth:`stop`) and :meth:`launch`; this class forwards ports, copies roots and closes.
    """

    #: The provider's name, for messages.
    provider: ClassVar[str] = ""

    def __init__(self, *, env: Mapping[str, str] | None = None) -> None:
        self.env = dict(env or {})
        self.closed = False
        self.pumps: list[TunnelPump] = []

    @abstractmethod
    def write_file(self, path: str, data: bytes, *, as_root: bool = False) -> None:
        """Write ``data`` to ``path`` in the sandbox."""

    @abstractmethod
    def read_file(self, path: str) -> bytes:
        """The bytes of ``path`` in the sandbox."""

    @abstractmethod
    def run(self, command: str, *, as_root: bool = False) -> None:
        """Run the shell ``command`` in the sandbox to its end; raise when it fails."""

    @abstractmethod
    def start(self, command: str, *, as_root: bool = False) -> None:
        """Start the shell ``command`` in the sandbox and return while it runs."""

    @abstractmethod
    def public_url(self, port: int) -> str:
        """The ``https`` address at which the provider exposes the sandbox's ``port``."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the sandbox."""

    def preflight(self) -> None:
        return None

    def open_session(self, *, forward_ports: tuple[int, ...] = (), timeout_s: float) -> EpisodeExecutor:
        """The session itself: it is already open, and its owner closes it."""
        return self

    def forward(self, port: int, tunnel_port: int) -> None:
        """Make the Reef host's ``127.0.0.1:port`` answer at the same address inside the sandbox."""
        secret = secrets.token_urlsafe(32)
        secret_file = f"/root/reef-relay-{port}.secret"
        try:
            self.write_file("/root/reef-relay.py", RELAY_SCRIPT.read_bytes(), as_root=True)
            self.write_file(secret_file, secret.encode(), as_root=True)
            self.run(f"chmod 600 {secret_file}", as_root=True)
            self.start(
                f"python3 /root/reef-relay.py --port {port} --tunnel-port {tunnel_port} --secret-file {secret_file}",
                as_root=True,
            )
            url = self.public_url(tunnel_port)
        except Exception as exc:
            raise EpisodeLaunchError(f"the {self.provider} relay for port {port} did not start: {exc}") from exc
        pump = TunnelPump(url, secret, port)
        pump.start()
        self.pumps.append(pump)

    def push(self, root: Path) -> None:
        """Copy ``root`` into the sandbox at its sandbox path."""
        remote = remote_root(root)
        archive = f"/tmp/reef-push-{root.name}.tgz"
        self.write_file(archive, pack(root))
        # The sandbox's user owns the copy: a root that kept the host's owner could not write it in the jail.
        self.run(
            f"mkdir -p {shlex.quote(remote)} && tar xzf {archive} --no-same-owner -C {shlex.quote(remote)}"
            f" && rm -f {archive}"
        )

    def pull(self, root: Path, relative: str = "") -> None:
        """Replace ``root / relative`` with what the sandbox holds there now."""
        remote = remote_root(root) + (f"/{relative}" if relative else "")
        archive = f"/tmp/reef-pull-{secrets.token_hex(6)}.tgz"
        self.run(f"tar czf {archive} -C {shlex.quote(remote)} .")
        try:
            data = self.read_file(archive)
        finally:
            self.run(f"rm -f {archive}")
        unpack(data, root / relative if relative else root)

    def close(self) -> None:
        if self.closed:
            return
        for pump in self.pumps:
            pump.stop()
        self.pumps.clear()
        try:
            self.stop()
        except Exception as exc:
            raise EpisodeLaunchError(
                f"could not stop the {self.provider} sandbox; its provider timeout still applies"
            ) from exc
        self.closed = True


class TunnelPump:
    """Answer the relay's requests from the Reef host: poll ``/next``, send each to the local port, stream the
    answer back as ``/reply`` pieces. Runs its own event loop on a thread."""

    def __init__(self, url: str, secret: str, port: int, *, pollers: int = POLLERS) -> None:
        self.url = url
        self.port = port
        self.pollers = pollers
        self.headers = {"x-relay-secret": secret}
        self.ready = threading.Event()
        self.failure: str | None = None
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_until_complete, args=(self.main(),), daemon=True)
        self.task: asyncio.Task[None] | None = None

    def start(self, timeout: float = RELAY_READY_S) -> None:
        self.thread.start()
        if not self.ready.wait(timeout) or self.failure is not None:
            self.stop()
            raise EpisodeLaunchError(f"the sandbox relay did not answer at {self.url}: {self.failure or 'timed out'}")

    def stop(self) -> None:
        task = self.task
        if task is not None and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(task.cancel)
        if self.thread.is_alive():
            self.thread.join(timeout=10)
        if not self.thread.is_alive() and not self.loop.is_closed():
            self.loop.close()

    async def main(self) -> None:
        self.task = asyncio.current_task()
        try:
            async with ClientSession(trust_env=True) as tunnel, ClientSession(auto_decompress=False) as local:
                if not await self.wait_ready(tunnel):
                    return
                self.ready.set()
                await asyncio.gather(*(self.poll(tunnel, local) for _ in range(self.pollers)))
        except asyncio.CancelledError:
            return
        finally:
            self.ready.set()

    async def wait_ready(self, tunnel: ClientSession) -> bool:
        deadline = self.loop.time() + RELAY_READY_S
        while self.loop.time() < deadline:
            try:
                async with tunnel.get(f"{self.url}/ready", headers=self.headers) as response:
                    if response.status == 200:
                        # A poll made before the relay listened failed; the tunnel is up now.
                        self.failure = None
                        return True
                    self.failure = f"HTTP {response.status}"
            except Exception as exc:
                self.failure = str(exc) or type(exc).__name__
            await asyncio.sleep(0.5)
        return False

    async def poll(self, tunnel: ClientSession, local: ClientSession) -> None:
        while True:
            try:
                async with tunnel.get(
                    f"{self.url}/next", headers=self.headers, timeout=ClientTimeout(total=60)
                ) as response:
                    if response.status == 204:
                        continue
                    if response.status != 200:
                        await asyncio.sleep(1)
                        continue
                    request = await response.json()
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(1)
                continue
            await self.serve(tunnel, local, request)

    async def serve(self, tunnel: ClientSession, local: ClientSession, request: Mapping[str, Any]) -> None:
        reply = f"{self.url}/reply/{request['id']}"
        sequence = 0

        async def send(piece: dict[str, Any]) -> None:
            nonlocal sequence
            for attempt in range(3):
                try:
                    async with tunnel.post(
                        reply,
                        json={**piece, "sequence": sequence},
                        headers=self.headers,
                        timeout=ClientTimeout(total=30),
                    ) as response:
                        # A final piece can reach the client before its acknowledgement is lost.
                        if response.status == 410 and piece.get("end") and attempt:
                            return
                        response.raise_for_status()
                        await response.read()
                    sequence += 1
                    return
                except ClientResponseError as exc:
                    if (exc.status < 500 and exc.status != 429) or attempt == 2:
                        raise
                except (ClientConnectionError, ClientPayloadError, TimeoutError):
                    if attempt == 2:
                        raise
                await asyncio.sleep(0.25 * (attempt + 1))

        headers = {
            name: value for name, value in (request.get("headers") or {}).items() if name.lower() not in HOP_BY_HOP
        }
        body = base64.b64decode(request.get("body") or "")
        head: dict[str, Any] | None = None
        try:
            async with local.request(
                str(request.get("method") or "GET"),
                f"http://127.0.0.1:{self.port}{request.get('path') or '/'}",
                headers=headers,
                data=body or None,
                timeout=ClientTimeout(total=None, sock_read=3600),
            ) as response:
                head = {
                    "status": response.status,
                    "headers": {n: v for n, v in response.headers.items() if n.lower() not in HOP_BY_HOP},
                }
                async for chunk in response.content.iter_any():
                    # Whatever arrived while the last piece crossed goes as one piece.
                    await send({**head, "data": base64.b64encode(chunk).decode()})
                    head = {}
            await send({**(head or {}), "data": "", "end": True})
        except asyncio.CancelledError:
            raise
        except (ClientConnectionError, ClientPayloadError, ClientResponseError, TimeoutError) as exc:
            logger.warning("the sandbox tunnel response was interrupted: %s", exc)
            try:
                if head is None:
                    error = base64.b64encode(f"the Reef host did not answer: {exc}".encode()).decode()
                    await send({"status": 502, "headers": {"Content-Type": "text/plain"}, "data": error, "end": True})
                else:
                    await send({"data": "", "end": True, "error": True})
            except (ClientConnectionError, ClientPayloadError, ClientResponseError, TimeoutError) as failure:
                logger.debug("the sandbox tunnel lost a reply: %s", failure)


__all__ = [
    "REMOTE_TASK_ENVIRONMENTS",
    "RemoteExecutor",
    "RemoteSession",
    "TunnelPump",
    "deployment_owner",
    "pack",
    "probe_command",
    "protected_command",
    "remote_command",
    "remote_root",
    "unpack",
]
