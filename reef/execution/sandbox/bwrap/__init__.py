"""The Linux sandbox executor: a bubblewrap jail around one program."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from reef.execution.sandbox.process import (
    ISOLATION_ENV,
    ProcessOutcome,
    SandboxExecutor,
    SandboxUnavailableError,
    inherited_environment,
    run_process,
)

#: What every jail unshares and mounts before its binds; the run jail and the nesting probe share it.
ISOLATION: tuple[str, ...] = (
    "--unshare-user",
    "--unshare-ipc",
    "--unshare-pid",
    "--unshare-uts",
    "--unshare-cgroup-try",
    "--die-with-parent",
    "--new-session",
    "--proc",
    "/proc",
    "--dev",
    "/dev",
    "--tmpfs",
    "/tmp",
    "--clearenv",
)

#: Nested tool jails bind the run's proc filesystem instead of mounting another one.
TOOL_ISOLATION: tuple[str, ...] = (
    "--unshare-user",
    "--unshare-ipc",
    "--unshare-pid",
    "--unshare-uts",
    "--unshare-cgroup-try",
    "--die-with-parent",
    "--new-session",
    "--ro-bind",
    "/proc",
    "/proc",
    "--dev",
    "/dev",
    "--tmpfs",
    "/tmp",
    "--clearenv",
)


@dataclass(frozen=True)
class BwrapExecutor(SandboxExecutor):
    """Run the binary inside a bubblewrap jail.

    bubblewrap (``bwrap``) is an unprivileged, daemonless sandbox: it builds a
    new mount/pid/user namespace, binds a read-only base filesystem, exposes
    only the run root (workspace writable), clears the environment, and
    dies with its parent. Network is unshared (no egress) unless
    ``egress_hosts`` lists the model endpoint the run must reach, in which
    case the network namespace is shared so the endpoint is reachable; a
    single-host firewall is the follow-up (issue #18). An ``isolated`` network
    is a network namespace of its own that pasta connects to the internet: no
    host address is reachable (``--no-map-gw``) except the loopback ports in
    ``forward_ports``, which pasta forwards to the host. The jail also carries
    ``REEF_NATIVE_ENFORCE=bwrap``, so the native loop runs each tool call in
    a nested jail derived from the tool's declared capabilities; preflight
    proves the host can nest one.
    """

    kind: ClassVar[str] = "bwrap"
    #: Base directories bound read-only so the binary and its runtime resolve.
    base_paths: tuple[str, ...] = ("/usr", "/bin", "/lib", "/lib64", "/etc/alternatives", "/etc/ssl", "/opt")

    def preflight(self) -> None:
        if shutil.which("bwrap") is None:
            raise SandboxUnavailableError(
                "the sandbox executor requires bubblewrap (bwrap) on PATH; install it or set "
                "evolution.executor: local for development"
            )
        if self.network not in ("default", "isolated"):
            raise SandboxUnavailableError(f"unknown sandbox network {self.network!r}; use 'default' or 'isolated'")
        if self.network == "isolated":
            self.preflight_pasta()
        # The native loop nests one jail per tool call inside the run's, so a host that refuses a user namespace
        # inside another fails here instead of ending every call in SANDBOX_FAILED and tying every pairing.
        try:
            done = subprocess.run(
                self.nested_probe_argv(),
                env={"PATH": os.environ.get("PATH", "")},
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailableError(
                f"the sandbox executor cannot nest a bubblewrap jail on this host: {exc}"
            ) from exc
        if done.returncode != 0:
            raise SandboxUnavailableError(
                "the sandbox executor cannot nest a bubblewrap jail on this host, which every sandboxed native "
                f"tool call needs: {done.stderr.strip()[-600:]}"
            )

    def preflight_pasta(self) -> None:
        """An isolated network is one pasta sets up: prove it can on this host, as this user.

        pasta cannot map users under root (it drops to nobody), and a container may deny it the namespaces it
        needs; either way the jail would never start, so the host is refused here with pasta's own reason.
        """
        if shutil.which("pasta") is None:
            raise SandboxUnavailableError(
                "an isolated sandbox network requires pasta (the passt package) on PATH; install it, or run "
                "without isolation where that is allowed"
            )
        if os.geteuid() == 0:
            raise SandboxUnavailableError(
                "pasta cannot map users when started as root; run the Reef service as a non-root user with user "
                "namespaces allowed to isolate the network"
            )
        probe = ["pasta", "--config-net", "--no-map-gw", "--quiet", "-t", "none", "-u", "none", "-T", "none"]
        try:
            done = subprocess.run(
                [*probe, "-U", "none", "--", "/bin/true"],
                env={"PATH": os.environ.get("PATH", "")},
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailableError(f"pasta cannot set up an isolated network on this host: {exc}") from exc
        if done.returncode != 0:
            raise SandboxUnavailableError(
                "pasta cannot set up an isolated network on this host (run the Reef service as a non-root user "
                f"with user namespaces allowed): {(done.stderr or done.stdout).strip()[-600:]}"
            )

    def nested_probe_argv(self) -> list[str]:
        """A run jail with a tool jail inside it around ``true``: the shape every sandboxed tool call takes."""
        bwrap = shutil.which("bwrap") or "bwrap"
        binds = [token for base in self.base_paths if Path(base).exists() for token in ("--ro-bind", base, base)]
        task_run = [bwrap, *ISOLATION, "--unshare-net", *binds]
        tool = [bwrap, *TOOL_ISOLATION, "--unshare-net", *binds]
        return [*task_run, "--", *tool, "--", "/bin/true"]

    def bwrap_argv(
        self,
        argv: Sequence[str],
        *,
        root: Path,
        workspace: Path,
        env: Mapping[str, str],
        writable_paths: Sequence[Path] = (),
        readonly_paths: Sequence[Path] = (),
    ) -> list[str]:
        cmd = ["bwrap", *ISOLATION]
        isolated = self.network == "isolated"
        if isolated:
            # pasta made the network namespace; bwrap stays in it rather than unsharing a disconnected one.
            ports = ",".join(str(port) for port in self.forward_ports) or "none"
            # Every forward is explicit: pasta's default ("auto") would expose each port the host listens on.
            forwards = ["-t", "none", "-u", "none", "-T", ports, "-U", "none"]
            cmd = ["pasta", "--config-net", "--no-map-gw", "--quiet", *forwards, "--", *cmd]
        elif not self.egress_hosts:
            cmd.append("--unshare-net")
        for base in self.base_paths:
            if Path(base).exists():
                cmd += ["--ro-bind", base, base]
        if self.egress_hosts or isolated:
            for dns_path in ("/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf"):
                if Path(dns_path).exists():
                    cmd += ["--ro-bind", dns_path, dns_path]
        # Mount the root read-only, then open only the workspace and the
        # declared runtime-state directories. Rendered inputs inside those
        # directories are rebound read-only so a run cannot mutate the
        # admitted composition, as a remote sandbox's command does.
        cmd += ["--ro-bind", str(root), str(root)]
        opened = (workspace, *writable_paths)
        for path in opened:
            cmd += ["--bind", str(path), str(path)]
        for path in readonly_paths:
            if any(path == writable or writable in path.parents for writable in opened):
                cmd += ["--ro-bind", str(path), str(path)]
        cmd += ["--chdir", str(workspace)]
        for key, value in {
            **inherited_environment(),
            **env,
            **self.env,
            "REEF_NATIVE_ENFORCE": "bwrap",
            ISOLATION_ENV: "bwrap",
        }.items():
            cmd += ["--setenv", key, value]
        cmd += ["--", *argv]
        return cmd

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
        full = self.bwrap_argv(
            argv,
            root=root,
            workspace=workspace,
            env=env,
            writable_paths=writable_paths,
            readonly_paths=readonly_paths,
        )
        return run_process(
            full,
            cwd=workspace,
            env={"PATH": os.environ.get("PATH", "")},
            timeout=timeout,
            limits=self.limits,
        )
