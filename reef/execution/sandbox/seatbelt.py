"""The macOS sandbox executor: a Seatbelt profile around one program, launched with ``sandbox-exec``."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from reef.execution.sandbox.process import (
    ISOLATION_ENV,
    ProcessOutcome,
    ProgramExecutor,
    SandboxLimits,
    SandboxProgram,
    SandboxUnavailableError,
    inherited_environment,
    run_process,
)

#: The rules every profile starts with. Reads and writes are denied except where a later rule allows them; the last
#: matching rule wins. Path traversal needs metadata reads everywhere, so ``stat`` of a path outside the allowed
#: directories succeeds, but its contents stay unreadable. Signals reach only processes in the same sandbox.
PROFILE_HEAD = """(version 1)
(allow default)
(deny file-read*)
(allow file-read-metadata)
(allow file-read* (literal "/"))
(deny file-write*)
(allow file-write* (literal "/dev/null") (literal "/dev/zero") (literal "/dev/tty") (literal "/dev/dtracehelper"))
(deny signal)
(allow signal (target same-sandbox))
"""


@dataclass(frozen=True)
class SeatbeltExecutor(ProgramExecutor):
    """Run the binary under a Seatbelt profile, the macOS counterpart of ``SandboxExecutor``.

    Values: ``egress_hosts``, ``network``, ``forward_ports``, ``limits`` and ``env`` have the meaning they have on
    ``SandboxExecutor``, except that ``limits.memory_bytes`` must be zero: macOS refuses ``RLIMIT_AS``. ``base_paths`` are the host directories a run may read so its binary and runtime resolve.
    The profile lets a run read the base paths and the run root, and write only the workspace, the declared
    writable paths and a private temporary directory. Read-only inputs inside a writable path stay read-only. The
    environment holds only the inherited minimum, the run's variables and ``env``.

    The network follows ``network``. ``default`` denies all network access unless ``egress_hosts`` is set, in which
    case the network is open, as on Linux. ``isolated`` opens outbound internet and denies every loopback port except
    ``forward_ports``; unlike pasta, it cannot hide the host's LAN addresses.

    Seatbelt profiles do not nest: a process under a profile cannot apply another one. A native loop in this sandbox
    therefore runs its tool calls under the run's profile (``REEF_NATIVE_ENFORCE=none``), not a narrower profile per
    tool. ``SandboxExecutor`` narrows each call on Linux.
    """

    egress_hosts: tuple[str, ...] = ()
    network: str = "default"
    forward_ports: tuple[int, ...] = ()
    limits: SandboxLimits = field(default_factory=SandboxLimits)
    base_paths: tuple[str, ...] = (
        "/usr",
        "/bin",
        "/sbin",
        "/System",
        "/Library/Apple",
        "/Library/Developer/CommandLineTools",
        "/Library/Preferences",
        "/private/etc",
        "/private/var/db/dyld",
        "/private/var/db/timezone",
        "/dev",
        "/opt/homebrew",
        "/usr/local",
    )
    env: Mapping[str, str] = field(default_factory=dict, repr=False)

    def preflight(self) -> None:
        if shutil.which("sandbox-exec") is None:
            raise SandboxUnavailableError(
                "the macOS sandbox executor requires sandbox-exec on PATH; set evolution.executor: local for "
                "development"
            )
        if self.network not in ("default", "isolated"):
            raise SandboxUnavailableError(f"unknown sandbox network {self.network!r}; use 'default' or 'isolated'")
        if self.limits.memory_bytes:
            # setrlimit(RLIMIT_AS) fails on macOS, so the limit would fail every launch instead of bounding it.
            raise SandboxUnavailableError(
                "macOS cannot limit a run's address space; remove evolution.sandbox.limits.memory_bytes"
            )
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            try:
                done = subprocess.run(
                    self.seatbelt_argv(["/usr/bin/true"], root=workspace, workspace=workspace, temporary=workspace),
                    env={"PATH": os.environ.get("PATH", "")},
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise SandboxUnavailableError(f"sandbox-exec cannot start a run on this host: {exc}") from exc
        if done.returncode != 0:
            raise SandboxUnavailableError(
                f"sandbox-exec cannot start a run on this host: {done.stderr.strip()[-600:]}"
            )

    def for_program(self, program: SandboxProgram, *, binary: str | None = None) -> tuple[ProgramExecutor, str]:
        """Allow reads of the whole install of a binary Reef installed, as ``SandboxExecutor`` binds it."""
        if binary is not None:
            return self, binary
        resolved = program.host_binary()
        install_dir = program.host_install_dir()
        if install_dir is None:
            return self, resolved
        return replace(self, base_paths=(*self.base_paths, str(install_dir))), resolved

    def open_session(self, *, forward_ports: tuple[int, ...] = (), timeout_s: float) -> ProgramExecutor:
        """An isolated network allows the run's ports; the default network already reaches the host."""
        ports = tuple(dict.fromkeys((*self.forward_ports, *forward_ports)))
        if self.network != "isolated" or ports == self.forward_ports:
            return self
        return replace(self, forward_ports=ports)

    def profile(
        self,
        *,
        writable_count: int,
        readonly_count: int,
        readable_count: int,
    ) -> str:
        """The Seatbelt profile text. Paths arrive as ``-D`` parameters, so no path is quoted into the text.

        ``READ_i`` names a readable directory, ``WRITE_i`` a writable one and ``READONLY_i`` an input inside a
        writable directory that a run may read but not change.
        """
        rules = [PROFILE_HEAD]
        rules += [f'(allow file-read* (subpath (param "READ_{index}")))' for index in range(readable_count)]
        rules += [
            f'(allow file-read* file-write* (subpath (param "WRITE_{index}")))' for index in range(writable_count)
        ]
        rules += [f'(deny file-write* (subpath (param "READONLY_{index}")))' for index in range(readonly_count)]
        if self.network == "isolated":
            rules.append('(deny network-outbound (remote ip "localhost:*"))')
            rules += [f'(allow network-outbound (remote ip "localhost:{port}"))' for port in self.forward_ports]
        elif not self.egress_hosts:
            rules.append("(deny network*)")
            # Local sockets are files, not network access; bubblewrap's private network keeps them too.
            rules.append("(allow network* (local unix-socket) (remote unix-socket))")
        return "\n".join(rules) + "\n"

    def seatbelt_argv(
        self,
        argv: Sequence[str],
        *,
        root: Path,
        workspace: Path,
        temporary: Path,
        writable_paths: Sequence[Path] = (),
        readonly_paths: Sequence[Path] = (),
    ) -> list[str]:
        """The ``sandbox-exec`` command that runs ``argv`` under this executor's profile.

        Seatbelt matches resolved paths, so every path is resolved first: ``/tmp`` is ``/private/tmp`` on macOS.
        """
        readable = [Path(base).resolve() for base in self.base_paths if Path(base).exists()]
        readable.append(root.resolve())
        writable = [path.resolve() for path in (workspace, temporary, *writable_paths)]
        readonly = [
            path.resolve()
            for path in readonly_paths
            if any(path.resolve() == opened or opened in path.resolve().parents for opened in writable)
        ]
        parameters = [f"-DREAD_{index}={path}" for index, path in enumerate(readable)]
        parameters += [f"-DWRITE_{index}={path}" for index, path in enumerate(writable)]
        parameters += [f"-DREADONLY_{index}={path}" for index, path in enumerate(readonly)]
        text = self.profile(writable_count=len(writable), readonly_count=len(readonly), readable_count=len(readable))
        return ["sandbox-exec", "-p", text, *parameters, *argv]

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
        self.preflight()
        # bubblewrap mounts a fresh /tmp; here each run gets its own directory, removed when the run ends.
        with tempfile.TemporaryDirectory(prefix="reef-run-") as directory:
            temporary = Path(directory)
            full = self.seatbelt_argv(
                argv,
                root=root,
                workspace=workspace,
                temporary=temporary,
                writable_paths=writable_paths,
                readonly_paths=readonly_paths,
            )
            run_env = {
                **inherited_environment(),
                **env,
                **self.env,
                "TMPDIR": str(temporary.resolve()),
                "REEF_NATIVE_ENFORCE": "none",
                ISOLATION_ENV: "seatbelt",
            }
            return run_process(full, cwd=workspace, env=run_env, timeout=timeout, limits=self.limits)
