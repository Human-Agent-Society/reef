"""Run task processes on a boxd machine (:mod:`reef.execution.sandbox.remote`).

A `boxd <https://boxd.sh/?ref=reef>`__ machine is a KVM microVM with its own kernel and disk.
:class:`BoxdExecutor` is the deployment's setting (the API key, the organization, the region, the size, an optional
snapshot); :meth:`BoxdExecutor.open` creates one machine and answers a :class:`BoxdSession`, and closing the session
deletes the machine.

Every machine is isolated: it joins no private network of the organization and has no boxd credentials or connected
integrations. Environment variables and secrets scoped to the machine still reach commands that run on it outside the
jail, so a deployment must not scope a secret there that a task may not read. The API key stays on the Reef host.

A process runs in the jail as ``reef-run``, a user that this module creates on each machine. That user has no ``sudo``,
is not in the ``docker`` group, and cannot read the home directory or the user sockets of the machine's own user,
which preparing the machine makes private. On the machine's own addresses it can reach only the forwarded ports and
ports 1024 to 32767, except a port that was in use when the machine was prepared.

Without ``boxd_snapshot``, a new machine installs bubblewrap and the harness's pinned npm package when it starts. A
``boxd_snapshot`` must come from a boxd machine and hold the program's binary, ``bwrap``, ``python3`` and coreutils'
``timeout``.
"""

from __future__ import annotations

import logging
import math
import posixpath
import re
import secrets
import shlex
import socket
import ssl
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Self

from reef.execution.sandbox.process import (
    ProcessOutcome,
    ProgramLaunchError,
    ProgramTimeoutError,
    SandboxProgram,
    SandboxUnavailableError,
    parse_env_from,
)
from reef.execution.sandbox.remote import (
    REMOTE_BASE,
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
    unpack,
)

logger = logging.getLogger(__name__)

#: The machine sizes boxd offers: vCPUs, each with 4 GiB of memory.
MACHINE_VCPUS = (1, 2, 4, 8)
MEMORY_GIB_PER_VCPU = 4
#: The start of a machine's name when ``boxd_name_prefix`` is not set.
DEFAULT_NAME_PREFIX = "reef-"
#: A name prefix: a lowercase letter, then lowercase letters, digits and hyphens.
NAME_PREFIX = re.compile(r"[a-z][a-z0-9-]{0,39}")
#: The user that a task's processes, and the copies of their roots, run as.
RUN_USER = "reef-run"
#: The machine's own user, which file transfers run as.
MACHINE_OWNER = "boxd"
#: A machine-wide file of environment variables that some machine images hold.
MACHINE_ENV_FILE = "/etc/profile.d/boxd-env.sh"
#: The subdomain of the HTTPS route to a relay's tunnel port, before the port's index.
TUNNEL_ROUTE_PREFIX = "reef-relay-"
#: The file that holds an ID the kernel chooses at each boot.
BOOT_ID_FILE = "/proc/sys/kernel/random/boot_id"
#: The ports the kernel picks for a process that names none. They stay above the relays' tunnel ports, and inside the
#: local ports that a task may reach.
EPHEMERAL_PORTS = "22000 32767"
#: The firewall chain that holds the rules for ``RUN_USER``.
FIREWALL_CHAIN = "REEF_RUN"
#: Seconds a machine may take to answer its first command.
MACHINE_READY_S = 180.0
#: Seconds a new HTTPS route may take to present its certificate.
ROUTE_READY_S = 120.0
#: Seconds one setup command, such as a package install or a copy, may take.
COMMAND_TIMEOUT_S = 900.0
#: Seconds between two checks of whether a process ended.
POLL_INTERVAL_S = 0.5
#: Seconds a process may outlive its limit before ``timeout`` kills it outright.
KILL_AFTER_S = 10
#: Seconds past that kill before Reef stops waiting for the process's exit status.
EXIT_MARGIN_S = 30
#: The exit status of coreutils' ``timeout`` when the limit ended the command, and when it had to kill it.
TIMEOUT_STATUSES = (124, 137)


def firewall_command(forward_ports: Sequence[int]) -> str:
    """The root shell command that limits which of the machine's own ports ``RUN_USER`` can reach.

    It waits for the boot to end, then lets ``RUN_USER`` reach each of ``forward_ports`` and refuses each port that is
    in use and every port outside 1024 to 32767. It exits non-zero, with the rules so far in place, when a tool fails.
    """
    match = "-m addrtype --dst-type LOCAL"
    allowed = "".join(f'add_rule "{match} -p tcp --dport {port} -j ACCEPT"; ' for port in forward_ports)
    entry = f"OUTPUT -m owner --uid-owner {RUN_USER} -j {FIREWALL_CHAIN}"
    return (
        "( add_rule() { for table in iptables ip6tables; do "
        f"$table -A {FIREWALL_CHAIN} $1 || exit 1; done; }}; "
        "timeout 120 systemctl is-system-running --wait >/dev/null 2>&1; "
        f"sysctl -q -w net.ipv4.ip_local_port_range='{EPHEMERAL_PORTS}' || exit 1; "
        "tcp_listeners=$(ss -ltnH) || exit 1; udp_listeners=$(ss -lunH) || exit 1; "
        # A machine from a snapshot holds the rules of the machine it was taken from, with other forwarded ports.
        "for table in iptables ip6tables; do "
        f"$table -N {FIREWALL_CHAIN} 2>/dev/null; $table -F {FIREWALL_CHAIN} || exit 1; "
        f"$table -C {entry} 2>/dev/null || $table -I {entry.replace('OUTPUT', 'OUTPUT 1', 1)} || exit 1; done; "
        f"{allowed}"
        "for protocol in tcp udp; do "
        'if [ "$protocol" = tcp ]; then listeners=$tcp_listeners; else listeners=$udp_listeners; fi; '
        "for port in $(printf '%s\\n' \"$listeners\" | awk 'NF {print $4}' | sed 's/.*://' | sort -un); do "
        f'add_rule "{match} -p $protocol --dport $port -j REJECT"; done; '
        f'add_rule "{match} -p $protocol --dport 0:1023 -j REJECT"; '
        f'add_rule "{match} -p $protocol --dport 32768:65535 -j REJECT"; '
        "done )"
    )


@dataclass(frozen=True)
class BoxdExecutor(RemoteExecutor):
    """A :class:`RemoteExecutor` on boxd.

    Values: ``api_key`` is the boxd API key, or ``""`` when none is configured, which :meth:`preflight` refuses.
    ``org`` and ``region`` name the organization and region, or are ``""`` for the key's defaults. ``snapshot`` names
    the snapshot a machine starts from, or is ``""`` for the base image. ``vcpu`` is a member of ``MACHINE_VCPUS``, or
    ``None`` for the organization's default size. ``name_prefix`` matches ``NAME_PREFIX``. ``extra_labels`` maps
    label names to values and holds none of ``reef``, ``reef_owner`` and ``reef_machine``. ``auto_destroy`` is True if
    and only if each machine gets a delete timer for the end of its lifetime, counted from its creation.

    Purpose: :mod:`reef.execution.sandbox.config` builds one from a deployment's sandbox section. :meth:`open` makes
    one isolated machine for each value it is called on.
    """

    provider: ClassVar[str] = "boxd"
    provider_options: ClassVar[frozenset[str]] = frozenset(
        {
            "boxd_api_key",
            "boxd_api_key_env",
            "boxd_org",
            "boxd_region",
            "boxd_snapshot",
            "boxd_vcpu",
            "boxd_name_prefix",
            "boxd_labels",
            "boxd_auto_destroy",
        }
    )
    image_option: ClassVar[str] = "boxd_snapshot"

    api_key: str = field(default="", repr=False)
    org: str = ""
    region: str = ""
    snapshot: str = ""
    vcpu: int | None = None
    name_prefix: str = DEFAULT_NAME_PREFIX
    extra_labels: Mapping[str, str] = field(default_factory=dict)
    auto_destroy: bool = True

    def __post_init__(self) -> None:
        if self.vcpu is not None and (type(self.vcpu) is not int or self.vcpu not in MACHINE_VCPUS):
            raise SandboxUnavailableError(f"boxd_vcpu must be one of {', '.join(map(str, MACHINE_VCPUS))}")
        if not isinstance(self.name_prefix, str) or not NAME_PREFIX.fullmatch(self.name_prefix):
            raise SandboxUnavailableError(
                "boxd_name_prefix must start with a lowercase letter and hold at most 40 lowercase letters, "
                "digits and hyphens"
            )
        if not isinstance(self.extra_labels, Mapping) or not all(
            isinstance(name, str) and isinstance(value, str) for name, value in self.extra_labels.items()
        ):
            raise SandboxUnavailableError("boxd_labels must map label names to strings")
        if {"reef", "reef_owner", "reef_machine"} & set(self.extra_labels):
            raise SandboxUnavailableError("boxd_labels cannot set the reef, reef_owner or reef_machine label")
        if not isinstance(self.auto_destroy, bool):
            raise SandboxUnavailableError("boxd_auto_destroy must be true or false")

    @classmethod
    def from_config(cls, section: Mapping[str, object], environ: Mapping[str, str]) -> Self:
        check_options("boxd", section, cls.provider_options)
        key_name = section.get("boxd_api_key_env", "BOXD_API_KEY")
        if not isinstance(key_name, str) or not key_name.isidentifier():
            raise SandboxUnavailableError("boxd_api_key_env must name an environment variable")
        api_key = section.get("boxd_api_key") or environ.get(key_name, "")
        org = section.get("boxd_org") or environ.get("BOXD_ORG", "")
        region = section.get("boxd_region", "")
        snapshot = section.get("boxd_snapshot", "")
        if not all(isinstance(value, str) for value in (api_key, org, region, snapshot)):
            raise SandboxUnavailableError("boxd_api_key, boxd_org, boxd_region and boxd_snapshot must be strings")
        vcpu = section.get("boxd_vcpu")
        if vcpu is not None and type(vcpu) is not int:
            raise SandboxUnavailableError(f"boxd_vcpu must be one of {', '.join(map(str, MACHINE_VCPUS))}")
        name_prefix = section.get("boxd_name_prefix", DEFAULT_NAME_PREFIX)
        if not isinstance(name_prefix, str):
            raise SandboxUnavailableError("boxd_name_prefix must be a string")
        labels = section.get("boxd_labels", {})
        if not isinstance(labels, Mapping):
            raise SandboxUnavailableError("boxd_labels must map label names to strings")
        auto_destroy = section.get("boxd_auto_destroy", True)
        if not isinstance(auto_destroy, bool):
            raise SandboxUnavailableError("boxd_auto_destroy must be true or false")
        return cls(
            api_key=str(api_key).strip(),
            org=str(org).strip(),
            region=str(region).strip(),
            snapshot=str(snapshot).strip(),
            vcpu=vcpu,
            name_prefix=name_prefix,
            extra_labels=dict(labels),
            auto_destroy=auto_destroy,
            forward_ports=parse_forward_ports(section),
            env=parse_env_from(section, environ),
        )

    def has_image(self) -> bool:
        return bool(self.snapshot)

    def with_program(self, program: SandboxProgram, binary: str) -> Self:
        return replace(self, npm_package=program.npm_package, binary=binary)

    def client(self) -> Any:
        """A boxd client with this deployment's key and region."""
        from boxd import Boxd

        return Boxd(api_key=self.api_key, region=self.region or None)

    def reap(self) -> int:
        """Delete this deployment's machines. A fork or restore inherits the labels under another name, so
        ``reef_machine`` must equal the machine's own name."""
        if not self.owner:
            return 0
        from boxd.errors import NotFoundError

        client = self.client()
        stopped = 0
        for machine in client.machines.list(org=self.org or None, labels=self.labels()):
            if machine.labels.get("reef_owner") == self.owner and machine.labels.get("reef_machine") == machine.name:
                try:
                    client.machines.delete(machine.id)
                except NotFoundError:
                    continue
                stopped += 1
        return stopped

    def delete_unanswered(self, client: Any, name: str) -> None:
        """Delete the machine ``name`` if a create call that got no answer made it; it carries its own name."""
        from boxd.errors import BoxdError, NotFoundError

        try:
            machine = client.machines.get(name)
            if machine.labels.get("reef_machine") == name:
                client.machines.delete(machine.id)
        except NotFoundError:
            return
        except BoxdError as exc:
            logger.warning("could not check for a boxd machine %s after a failed create: %s", name, exc)

    def preflight(self) -> None:
        if not self.api_key:
            raise SandboxUnavailableError(
                "a boxd sandbox needs an API key: set BOXD_API_KEY or the selected sandbox's boxd_api_key"
            )
        try:
            from boxd.errors import BoxdError
        except ImportError as exc:
            raise SandboxUnavailableError(
                "a boxd sandbox needs the boxd package: pip install 'reef-infra[boxd]'"
            ) from exc
        if self.validated or not (self.snapshot or self.npm_package):
            return
        # Probing must not reap a running deployment's machines or require its HTTP server to be listening yet.
        session = None
        try:
            session = replace(self, owner="", forward_ports=(), timeout_s=60).open()
            session.run(probe_command(self.binary))
        except (BoxdError, ProgramLaunchError) as exc:
            raise SandboxUnavailableError(f"boxd startup check failed: {exc}") from exc
        finally:
            if session is not None:
                try:
                    session.close()
                except ProgramLaunchError as exc:
                    raise SandboxUnavailableError("boxd startup probe could not be cleaned up") from exc
        object.__setattr__(self, "validated", True)

    def open(self) -> BoxdSession:
        from boxd.models import ProxyEntry

        self.reap_once()
        # The timer runs from the create call, so it also covers the machine's setup.
        lifetime_seconds = math.ceil(self.timeout_s + SANDBOX_MARGIN_S + MACHINE_READY_S + COMMAND_TIMEOUT_S)
        client = self.client()
        name = f"{self.name_prefix}{secrets.token_hex(4)}"
        routes = [
            ProxyEntry(name=f"{TUNNEL_ROUTE_PREFIX}{index}", port=TUNNEL_PORT_BASE + index)
            for index in range(len(self.forward_ports))
        ]
        try:
            machine = client.machines.create(
                name,
                org=self.org or None,
                isolated=True,
                labels={**self.extra_labels, **self.labels(), "reef_machine": name},
                from_snapshot=self.snapshot or None,
                vcpu=self.vcpu,
                memory=None if self.vcpu is None else f"{self.vcpu * MEMORY_GIB_PER_VCPU}G",
                # The idle timers watch network traffic, and a task can compute for a long time without any.
                auto_suspend_timeout=0,
                delete_after=lifetime_seconds if self.auto_destroy else None,
                proxies=routes or None,
            )
        except BaseException as exc:
            self.delete_unanswered(client, name)
            if isinstance(exc, Exception):
                raise ProgramLaunchError(f"the boxd machine did not start: {exc}") from exc
            raise
        session = BoxdSession(client, machine.id, env=self.env)
        try:
            client.machines.wait_until_ready(machine.id, timeout=MACHINE_READY_S)
            client.machines.set_auto_hibernate_timeout(machine.id, 0)
            session.prepare("" if self.snapshot else self.npm_package, self.forward_ports)
            for index, port in enumerate(self.forward_ports):
                session.forward(port, TUNNEL_PORT_BASE + index)
        except BaseException as exc:
            try:
                session.close()
            except ProgramLaunchError as close_error:
                logger.warning("could not delete the boxd machine %s: %s", machine.id, close_error)
            if isinstance(exc, Exception) and not isinstance(exc, ProgramLaunchError):
                raise ProgramLaunchError(f"the boxd machine did not become ready: {exc}") from exc
            raise
        return session


class BoxdSession(RemoteSession):
    """One running boxd machine (:class:`RemoteSession`); its processes run as ``RUN_USER`` inside the jail.

    Values: ``client`` is a boxd client and ``machine_id`` the non-empty ID of a machine that ``client`` created.
    Purpose: :meth:`BoxdExecutor.open` makes one; :meth:`stop` deletes exactly the machine ``machine_id``.
    """

    provider: ClassVar[str] = "boxd"

    def __init__(self, client: Any, machine_id: str, *, env: Mapping[str, str] | None = None) -> None:
        if not isinstance(machine_id, str) or not machine_id:
            raise ValueError("machine_id must be a non-empty machine ID")
        super().__init__(env=env)
        self.client = client
        self.machine_id = machine_id
        #: The machine's boot ID when :meth:`prepare` finished, or ``""`` before that.
        self.boot_id = ""

    def execute(self, command: str, *, as_root: bool = False, timeout: float = COMMAND_TIMEOUT_S) -> Any:
        """Run the shell ``command`` as root or as ``RUN_USER`` to its end; answer its result, whatever its status."""
        user = "root" if as_root else RUN_USER
        shell = ["sudo", "-n", "-u", user, "/bin/sh", "-c", command]
        return self.client.machines.exec(self.machine_id, shell, cwd="/", timeout=timeout)

    def prepare(self, npm_package: str, forward_ports: Sequence[int] = ()) -> None:
        """Make ``RUN_USER`` and its directories, install bubblewrap and ``npm_package`` when one is named, and limit
        the machine's own ports that ``RUN_USER`` can reach (:func:`firewall_command`).

        The machine's Node lives in its own user's home, which ``RUN_USER`` cannot read, so ``/usr/local/bin/node``
        becomes a copy of that binary.
        """
        parent = posixpath.dirname(REMOTE_BASE)
        steps = [
            (
                f"{{ id -u {RUN_USER} >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin "
                f"{RUN_USER}; }}"
            ),
            f"install -d -o {RUN_USER} -g {RUN_USER} {shlex.quote(parent)} {shlex.quote(REMOTE_BASE)}",
            # A snapshot may come from a machine whose home or environment file other users can read.
            (
                f'owner_home=$(getent passwd {MACHINE_OWNER} | cut -d: -f6) && [ -n "$owner_home" ] && '
                'chmod o-rwx "$owner_home"'
            ),
            (
                f"if [ -e {MACHINE_ENV_FILE} ]; then chown root:{MACHINE_OWNER} {MACHINE_ENV_FILE} && "
                f"chmod 640 {MACHINE_ENV_FILE}; fi"
            ),
        ]
        if npm_package:
            steps.extend(
                (
                    (
                        "command -v bwrap >/dev/null || { apt-get update -qq && "
                        "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq bubblewrap; }"
                    ),
                    # The jail's PATH holds /usr/local/bin, and not the prefix of the machine's own Node install.
                    f"npm install -g --prefix /usr/local {shlex.quote(npm_package)}",
                )
            )
        steps.append(
            "if [ -L /usr/local/bin/node ]; then node_binary=$(readlink -f /usr/local/bin/node) && "
            'rm /usr/local/bin/node && install -m 755 "$node_binary" /usr/local/bin/node; fi'
        )
        steps.append(firewall_command(forward_ports))
        self.run(" && ".join(steps), as_root=True)
        self.boot_id = str(self.execute(f"cat {BOOT_ID_FILE}").stdout).strip()
        if not self.boot_id:
            raise ProgramLaunchError("the boxd machine did not report its boot ID")

    def write_file(self, path: str, data: bytes, *, as_root: bool = False) -> None:
        # A private directory holds the file until its owner has it: an upload is readable by every user.
        staging = str(
            self.client.machines.exec(
                self.machine_id, ["mktemp", "-d", "/tmp/reef-upload-XXXXXXXX"], timeout=COMMAND_TIMEOUT_S
            ).stdout
        )
        staging = staging.strip()
        if not staging:
            raise ProgramLaunchError("the boxd machine could not make a directory for the upload")
        staged = f"{staging}/file"
        self.client.machines.files.upload(self.machine_id, staged, data)
        owner = "root" if as_root else RUN_USER
        self.run(
            f"install -m 600 -o {owner} -g {owner} {shlex.quote(staged)} {shlex.quote(path)}; status=$?; "
            f"rm -rf {shlex.quote(staging)}; exit $status",
            as_root=True,
        )

    def read_file(self, path: str) -> bytes:
        """The bytes of ``path``; root copies it to a private file that the download can read."""
        staging = str(
            self.client.machines.exec(
                self.machine_id, ["mktemp", "-d", "/tmp/reef-download-XXXXXXXX"], timeout=COMMAND_TIMEOUT_S
            ).stdout
        )
        staging = staging.strip()
        if not staging:
            raise ProgramLaunchError("the boxd machine could not make a directory for the download")
        staged = f"{staging}/file"
        try:
            self.run(f"install -m 600 -o {MACHINE_OWNER} {shlex.quote(path)} {shlex.quote(staged)}", as_root=True)
            return bytes(self.client.machines.files.download(self.machine_id, staged))
        finally:
            try:
                self.client.machines.exec(self.machine_id, ["rm", "-rf", staging], timeout=COMMAND_TIMEOUT_S)
            except Exception as exc:
                logger.warning("could not remove %s from the boxd machine: %s", staging, exc)

    def pull(self, root: Path, relative: str = "") -> None:
        """Replace ``root / relative`` with what the machine holds there now; a temporary file left behind is logged."""
        remote = remote_root(root) + (f"/{relative}" if relative else "")
        archive = f"/tmp/reef-pull-{secrets.token_hex(6)}.tgz"
        self.run(f"tar czf {archive} -C {shlex.quote(remote)} .")
        try:
            data = self.read_file(archive)
        finally:
            try:
                self.run(f"rm -f {archive}")
            except Exception as exc:
                logger.warning("could not remove %s from the boxd machine: %s", archive, exc)
        unpack(data, root / relative if relative else root)

    def run(self, command: str, *, as_root: bool = False) -> None:
        result = self.execute(command, as_root=as_root)
        if result.exit_code != 0:
            raise ProgramLaunchError(
                f"the boxd machine command exited with {result.exit_code}: {str(result.stderr).strip()}"
            )

    def start(self, command: str, *, as_root: bool = False) -> None:
        detached = f"nohup setsid /bin/sh -c {shlex.quote(command)} >/dev/null 2>&1 </dev/null &"
        self.run(detached, as_root=as_root)

    def public_url(self, port: int) -> str:
        """The HTTPS address of ``port``, once its route presents a valid certificate."""
        route_name = f"{TUNNEL_ROUTE_PREFIX}{port - TUNNEL_PORT_BASE}"
        routes = self.client.machines.proxies.list(self.machine_id)
        domain = next((route.domain for route in routes if route.name == route_name), None)
        if domain is None:
            raise ProgramLaunchError(f"the boxd machine has no route {route_name}")
        context = ssl.create_default_context()
        deadline = time.monotonic() + ROUTE_READY_S
        while True:
            try:
                with (
                    socket.create_connection((domain, 443), timeout=5) as connection,
                    context.wrap_socket(connection, server_hostname=domain),
                ):
                    return f"https://{domain}"
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise ProgramLaunchError(f"the boxd route {domain} did not become ready: {exc}") from exc
                time.sleep(1)

    def stop(self) -> None:
        """Delete the machine; a machine that is already gone counts as deleted."""
        from boxd.errors import NotFoundError

        try:
            self.client.machines.delete(self.machine_id)
        except NotFoundError:
            logger.info("the boxd machine %s was already deleted", self.machine_id)

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
        """Run one process in the jail, with an empty standard input.

        The process runs detached and writes its output and exit status to files, so a dropped connection to the
        machine does not end it. ``timeout`` stops its whole tree, and a kill of its session backs that up.
        """
        from boxd.errors import BoxdError, ConflictError, NotFoundError

        command, envs = remote_command(argv, {**env, **self.env}, root)
        command = protected_command(command, envs, root, workspace, writable_paths, readonly_paths)
        seconds = max(math.ceil(timeout), 1)
        cwd = f"{remote_root(root)}/{workspace.relative_to(root).as_posix()}"
        directory = f"/tmp/reef-process-{secrets.token_hex(6)}"
        limited = (
            f"echo $$ > {directory}/session.partial; mv {directory}/session.partial {directory}/session; "
            f"timeout --kill-after={KILL_AFTER_S} {seconds} /bin/sh -c {shlex.quote(command)} "
            f"< /dev/null > {directory}/stdout 2> {directory}/stderr; "
            f"echo $? > {directory}/status.partial; mv {directory}/status.partial {directory}/status"
        )
        restarted = (
            f'elif [ "$(cat {BOOT_ID_FILE})" != {shlex.quote(self.boot_id)} ]; then echo restarted; '
            if self.boot_id
            else ""
        )
        status_command = (
            f"if [ -f {directory}/status ]; then cat {directory}/status; {restarted}"
            f'elif [ -f {directory}/session ] && ! kill -0 "$(cat {directory}/session)" 2>/dev/null; then '
            f"cat {directory}/status 2>/dev/null || echo ended; fi"
        )
        finished = False
        started_process = False
        try:
            # A restart ends the relays and the rules that keep a task from the machine's services.
            if self.boot_id and str(self.execute(f"cat {BOOT_ID_FILE}").stdout).strip() != self.boot_id:
                raise ProgramLaunchError("the boxd machine restarted, so this session cannot run another process")
            self.push(root)
            self.run(f"mkdir -m 700 {directory}")
            started = time.monotonic()
            # A dropped call may still have started the process, so from here on the cleanup stops it.
            started_process = True
            self.run(
                f"cd {shlex.quote(cwd)} || exit 1; "
                f"nohup setsid /bin/sh -c {shlex.quote(limited)} >/dev/null 2>&1 </dev/null &"
            )
            deadline = started + seconds + KILL_AFTER_S + EXIT_MARGIN_S
            status_text = ""
            while not status_text:
                if time.monotonic() >= deadline:
                    raise ProgramTimeoutError(f"the process ran past its {timeout:g} s limit on the boxd machine")
                time.sleep(POLL_INTERVAL_S)
                try:
                    status_text = str(self.execute(status_command, timeout=30).stdout).strip()
                except (NotFoundError, ConflictError):
                    raise
                except BoxdError as exc:
                    logger.warning("could not read the boxd process's status; trying again: %s", exc)
            if status_text == "restarted":
                raise ProgramLaunchError("the boxd machine restarted while the process ran")
            if status_text == "ended":
                raise ProgramLaunchError("the process on the boxd machine ended without an exit status")
            status = int(status_text)
            if status in TIMEOUT_STATUSES and time.monotonic() - started >= timeout:
                raise ProgramTimeoutError(f"the process ran past its {timeout:g} s limit on the boxd machine")
            outcome = ProcessOutcome(
                status,
                self.read_file(f"{directory}/stdout").decode(errors="replace"),
                self.read_file(f"{directory}/stderr").decode(errors="replace"),
            )
            finished = True
            return outcome
        except (ProgramTimeoutError, ProgramLaunchError):
            raise
        except Exception as exc:
            raise ProgramLaunchError(f"the boxd machine could not run the process: {exc}") from exc
        finally:
            if started_process and not finished:
                try:
                    self.execute(f'pkill -KILL -s "$(cat {directory}/session)"', as_root=True, timeout=30)
                except BoxdError as exc:
                    logger.warning("could not stop the boxd process: %s", exc)
            # Before the process starts, the machine's copy is not newer than the local root and must not replace it.
            if started_process:
                try:
                    self.pull(root)
                except Exception as exc:
                    if finished:
                        raise ProgramLaunchError(f"could not copy {root.name} back from the boxd machine") from exc
                    logger.warning("could not copy %s back from the failed boxd process", root.name)
            try:
                self.execute(f"rm -rf {directory}", timeout=30)
            except BoxdError as exc:
                logger.warning("could not remove the boxd process's output: %s", exc)


__all__ = ["BoxdExecutor", "BoxdSession"]
