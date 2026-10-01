"""Run episode processes in an E2B cloud sandbox that reaches the Reef host only through a tunnel.

An E2B sandbox is a microVM of its own, with the internet but no route to the
machine Reef runs on. :class:`E2BExecutor` is the deployment's setting (the
API key, the template, how long a sandbox may live); :meth:`E2BExecutor.open`
starts one sandbox and answers an :class:`E2BSession`, which launches any
number of processes in it until it is closed. Each process's root is copied
in before it starts and copied back once it ends, so the caller reads what
it wrote as a local run leaves it; :meth:`E2BSession.pull` refreshes a
directory while a process still runs.

A port in ``forward_ports`` is reachable inside the sandbox at the same
loopback address it has on the Reef host, so no URL handed to a process
changes. The sandbox runs :mod:`reef.harness.episodes.e2b_relay` on that port;
a :class:`TunnelPump` on the Reef host polls the relay over E2B's public
address for each request a process made, sends it to the local port, and
streams the answer back. Only the forwarded ports are reachable this way, and
every credential stays on the Reef side of them.

The template is the harness's pinned binary on Node 22; one Reef builds on
first use (about a minute) when the alias does not exist yet.

A sandbox carries its deployment as ``owner``; a Reef that stopped mid run
never closed its sandbox, so before a process starts its first sandbox it
stops the ones its deployment left (:meth:`E2BExecutor.reap_once`), and never
another deployment's. It waits for that first sandbox rather than reaping at
start: a second start that cannot bind the port serves no request, so it
never reaches a peer's sandbox that is still in use.
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
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError, ClientSession, ClientTimeout

from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.episodes.executor import (
    ISOLATION_ENV,
    EpisodeExecutor,
    EpisodeLaunchError,
    EpisodeTimeout,
    ProcessOutcome,
    SandboxUnavailable,
)

logger = logging.getLogger(__name__)

RELAY_SCRIPT = Path(__file__).with_name("e2b_relay.py")
#: Where a process's root lands in the sandbox: one directory per root, named as the local one.
REMOTE_BASE = "/home/user/reef"
#: The first sandbox port a relay's tunnel listens on; E2B exposes it at a public address.
TUNNEL_PORT_BASE = 21080
#: Concurrent polls per forwarded port: a long request (a trial) must not hold up the calls it makes itself.
POLLERS = 8
#: How long a sandbox outlives the run it was opened for, so the copy back and the close still find it.
SANDBOX_MARGIN_S = 600
#: The longest life E2B grants a sandbox at creation; asking for more is refused outright, so a run that needs
#: more keeps its own sandbox alive instead (:class:`SandboxKeeper`).
SANDBOX_MAX_TIMEOUT_S = 3600
#: How long before that deadline the keeper asks for it again, so a slow extension still lands in time.
SANDBOX_RENEW_MARGIN_S = 300
#: How long a new relay has to answer before the session gives up on it.
RELAY_READY_S = 60.0
#: The deployments this process has reaped: each once, before its first sandbox.
REAPED_OWNERS: set[str] = set()
REAP_LOCK = threading.Lock()
#: Headers that describe one connection and never cross the tunnel.
HOP_BY_HOP = frozenset(
    {"connection", "content-length", "host", "keep-alive", "te", "trailer", "transfer-encoding", "upgrade"}
)


def template_alias(adapter: str, version: str) -> str:
    """The pinned agent template, versioned separately for its episode isolation dependencies."""
    return f"reef-{adapter}-{version}-episodes-v1".replace(".", "-").lower()


def deployment_owner(anchor: Path) -> str:
    """Which Reef deployment a sandbox belongs to: this host and one of the deployment's own state directories."""
    return hashlib.sha256(f"{socket.gethostname()}:{anchor.resolve()}".encode()).hexdigest()[:16]


def remote_root(root: Path) -> str:
    return f"{REMOTE_BASE}/{root.name}"


def remote_command(argv: Sequence[str], env: Mapping[str, str], root: Path) -> tuple[str, dict[str, str]]:
    """The command line and environment for the sandbox: the root's local path becomes its sandbox path, the
    binary is resolved in the template, and the host's ``PATH`` stays behind."""
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
    """Keep the VM filesystem read-only except the episode's declared output directories.

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
        logger.debug("left %s in the E2B sandbox: it would land outside the copy", member.name)
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
class E2BExecutor(EpisodeExecutor):
    """Launch each process in an E2B sandbox; :meth:`open` keeps one sandbox for several.

    ``template`` is the sandbox's image by alias; with ``npm_package``
    (``<package>@<version>``) a missing alias is built first. ``timeout_s``
    bounds how long a sandbox lives past its margin. ``forward_ports`` are the
    Reef host's loopback ports the sandbox reaches through the tunnel. ``owner``
    names the deployment on every sandbox it starts, for :meth:`reap`.
    """

    api_key: str = field(default="", repr=False)
    template: str = ""
    npm_package: str = ""
    timeout_s: float = 3600.0
    forward_ports: tuple[int, ...] = ()
    owner: str = ""
    binary: str = ""
    env: Mapping[str, str] = field(default_factory=dict, repr=False)
    validated: bool = field(default=False, init=False, compare=False, repr=False)

    @classmethod
    def from_config(cls, section: Mapping[str, object], environ: Mapping[str, str]) -> E2BExecutor:
        """Parse shared E2B settings without copying the host environment into the VM."""
        allowed = {
            "e2b_api_key",
            "e2b_api_key_env",
            "e2b_template",
            "forward_ports",
            "env_from",
            "egress_hosts",
            "limits",
        }
        unknown = set(section) - allowed
        if unknown:
            raise SandboxUnavailable(f"unknown E2B sandbox options: {', '.join(sorted(unknown))}")
        if section.get("egress_hosts") or section.get("limits"):
            raise SandboxUnavailable(
                "E2B does not implement sandbox.egress_hosts or sandbox.limits; "
                "it uses template resources and permits outbound internet"
            )
        key_name = section.get("e2b_api_key_env", "E2B_API_KEY")
        if not isinstance(key_name, str) or not key_name.isidentifier():
            raise SandboxUnavailable("e2b_api_key_env must name an environment variable")
        api_key = section.get("e2b_api_key") or environ.get(key_name, "")
        template = section.get("e2b_template", "")
        if not isinstance(api_key, str) or not isinstance(template, str):
            raise SandboxUnavailable("e2b_api_key and e2b_template must be strings")
        ports = section.get("forward_ports", [])
        if not isinstance(ports, (list, tuple)) or any(
            isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535 for port in ports
        ):
            raise SandboxUnavailable("sandbox.forward_ports must list TCP ports between 1 and 65535")
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
        return cls(
            api_key=api_key.strip(), template=template.strip(), forward_ports=tuple(dict.fromkeys(ports)), env=env
        )

    def for_adapter(self, descriptor: AdapterDescriptor, *, binary: str | None = None) -> E2BExecutor:
        """Resolve the agent in its remote template, without installing it on the Reef host."""
        if descriptor.self_isolating:
            raise SandboxUnavailable(
                f"adapter {descriptor.name!r} manages its own containers; use its hosted configuration"
            )
        remote_binary = binary or self.binary or descriptor.binary
        if self.template:
            if remote_binary == self.binary:
                return self
            return replace(self, binary=remote_binary)
        install = descriptor.install
        if install is None or install.kind != "npm":
            raise SandboxUnavailable(
                f"E2B adapter {descriptor.name!r} requires an explicit e2b_template containing its binary and bubblewrap"
            )
        return replace(
            self,
            template=template_alias(descriptor.name, install.version),
            npm_package=f"{install.package}@{install.version}",
            binary=remote_binary,
        )

    def labels(self) -> dict[str, str]:
        """The metadata every sandbox this executor starts carries."""
        return {"reef": "episode", **({"reef_owner": self.owner} if self.owner else {})}

    def reap_once(self) -> None:
        """Stop this deployment's leftovers the first time this process opens a sandbox; a failed look is
        logged and tried again next time. Held under a lock, so no sandbox of this process starts meanwhile."""
        if not self.owner:
            return
        with REAP_LOCK:
            if self.owner in REAPED_OWNERS:
                return
            try:
                stopped = self.reap()
            except Exception as exc:
                logger.warning("could not look for E2B sandboxes a previous run left: %s", exc)
                return
            REAPED_OWNERS.add(self.owner)
        if stopped:
            logger.warning("stopped %d E2B sandbox(es) a previous run of this deployment left running", stopped)

    def reap(self) -> int:
        """Stop this deployment's running sandboxes and answer how many; without an ``owner`` none is touched."""
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
            jail = "bwrap --unshare-user --unshare-pid --ro-bind / / --proc /proc --dev /dev --"
            command = f"{jail} {jail} /bin/true"
            if self.binary:
                command += f" && command -v {shlex.quote(self.binary)} >/dev/null"
            session.sandbox.commands.run(command, timeout=30)
        except (e2b.SandboxException, EpisodeLaunchError) as exc:
            raise SandboxUnavailable(f"E2B startup check failed: {exc}") from exc
        finally:
            if session is not None:
                try:
                    session.close()
                except EpisodeLaunchError as exc:
                    raise SandboxUnavailable("E2B startup probe could not be cleaned up") from exc
        object.__setattr__(self, "validated", True)

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
                env={**env, **self.env},
                timeout=timeout,
                writable_paths=writable_paths,
                readonly_paths=readonly_paths,
            )
        finally:
            session.close()

    def open(self) -> E2BSession:
        """Start a sandbox from the template with a relay on each forwarded port."""
        if not self.template:
            raise EpisodeLaunchError("an E2B sandbox needs a template")
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
            raise EpisodeLaunchError(f"the E2B sandbox did not start: {exc}") from exc
        session = E2BSession(sandbox, binary=self.binary)
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


class E2BSession(EpisodeExecutor):
    """One running sandbox: launch processes in it, pull what they wrote, close it when done."""

    def __init__(self, sandbox: Any, *, binary: str = "") -> None:
        self.sandbox = sandbox
        self.binary = binary
        self.closed = False
        self.pumps: list[TunnelPump] = []
        self.keeper = SandboxKeeper(sandbox)
        self.keeper.start()

    def preflight(self) -> None:
        return None

    def forward(self, port: int, tunnel_port: int) -> None:
        """Make the Reef host's ``127.0.0.1:port`` answer at the same address inside the sandbox."""
        secret = secrets.token_urlsafe(32)
        secret_file = f"/root/reef-relay-{port}.secret"
        try:
            self.sandbox.files.write("/root/reef-relay.py", RELAY_SCRIPT.read_text(encoding="utf-8"), user="root")
            self.sandbox.files.write(secret_file, secret, user="root")
            self.sandbox.commands.run(f"chmod 600 {secret_file}", user="root")
            self.sandbox.commands.run(
                f"python3 /root/reef-relay.py --port {port} --tunnel-port {tunnel_port} --secret-file {secret_file}",
                background=True,
                user="root",
                timeout=0,
            )
            url = f"https://{self.sandbox.get_host(tunnel_port)}"
        except Exception as exc:
            raise EpisodeLaunchError(f"the E2B relay for port {port} did not start: {exc}") from exc
        pump = TunnelPump(url, secret, port)
        pump.start()
        self.pumps.append(pump)

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

        command, envs = remote_command(argv, env, root)
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
            raise EpisodeLaunchError("the E2B command connection could not be restored")
        except CommandExitException as exc:
            finished = True
            return ProcessOutcome(exc.exit_code, exc.stdout, exc.stderr)
        except TimeoutException as exc:
            raise EpisodeTimeout(f"the process ran past its {timeout:g} s limit in the E2B sandbox") from exc
        except Exception as exc:
            raise EpisodeLaunchError(f"the E2B sandbox could not run the process: {exc}") from exc
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
                    raise EpisodeLaunchError(f"could not copy {root.name} back from the E2B sandbox") from exc
                logger.warning("could not copy %s back from the failed E2B process", root.name)

    def push(self, root: Path) -> None:
        """Copy ``root`` into the sandbox at its sandbox path."""
        remote = remote_root(root)
        archive = f"/tmp/reef-push-{root.name}.tgz"
        self.sandbox.files.write(archive, pack(root))
        self.sandbox.commands.run(
            f"mkdir -p {shlex.quote(remote)} && tar xzf {archive} -C {shlex.quote(remote)} && rm -f {archive}"
        )

    def pull(self, root: Path, relative: str = "") -> None:
        """Replace ``root / relative`` with what the sandbox holds there now."""
        remote = remote_root(root) + (f"/{relative}" if relative else "")
        archive = f"/tmp/reef-pull-{secrets.token_hex(6)}.tgz"
        self.sandbox.commands.run(f"tar czf {archive} -C {shlex.quote(remote)} .")
        try:
            data = self.sandbox.files.read(archive, format="bytes")
        finally:
            self.sandbox.commands.run(f"rm -f {archive}")
        unpack(bytes(data), root / relative if relative else root)

    def close(self) -> None:
        if self.closed:
            return
        self.keeper.stop()
        for pump in self.pumps:
            pump.stop()
        self.pumps.clear()
        try:
            self.sandbox.kill()
        except Exception as exc:
            raise EpisodeLaunchError("could not stop the E2B sandbox; its provider timeout still applies") from exc
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
            raise EpisodeLaunchError(f"the E2B relay did not answer at {self.url}: {self.failure or 'timed out'}")

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
            logger.warning("the E2B tunnel response was interrupted: %s", exc)
            try:
                if head is None:
                    error = base64.b64encode(f"the Reef host did not answer: {exc}".encode()).decode()
                    await send({"status": 502, "headers": {"Content-Type": "text/plain"}, "data": error, "end": True})
                else:
                    await send({"data": "", "end": True, "error": True})
            except (ClientConnectionError, ClientPayloadError, ClientResponseError, TimeoutError) as failure:
                logger.debug("the E2B tunnel lost a reply: %s", failure)


__all__ = [
    "E2BExecutor",
    "E2BSession",
    "TunnelPump",
    "deployment_owner",
    "ensure_template",
    "pack",
    "remote_command",
    "remote_root",
    "template_alias",
    "unpack",
]
