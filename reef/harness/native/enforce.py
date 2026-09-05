"""Capability enforcement for native tool calls: the enforcer ``REEF_NATIVE_ENFORCE`` names runs each ``run`` and every ``tool/result`` names the mode and the declaration's complement."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

ENFORCE_ENV = "REEF_NATIVE_ENFORCE"
#: What a profile can withhold; ``read`` is not here because every tool sees the workspace, read only at least.
DENIABLE: tuple[str, ...] = ("write", "exec", "network")
#: What an interpreter needs to start and to resolve names; no bin or sbin directory is among them, so without
#: ``exec`` no shell exists in the jail, though an executable installed under a library directory stays reachable.
LIBRARY_PATHS: tuple[str, ...] = (
    "/lib",
    "/lib64",
    "/usr/lib",
    "/usr/lib64",
    "/usr/share",
    "/etc/alternatives",
    "/etc/ssl",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
    "/etc/localtime",
    "/etc/resolv.conf",
    "/etc/hosts",
    "/etc/nsswitch.conf",
    "/etc/passwd",
    "/etc/group",
)
#: What the tool jail unshares and mounts before its binds; it binds the episode's ``/proc`` read only rather than
#: mounting a fresh one, which a jail inside the episode's cannot do, and which the interpreter needs to resolve
#: ``$ORIGIN`` in its RPATH and so find its own ``libpython``.
ISOLATION: tuple[str, ...] = (
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
#: What ``exec`` adds: the base directories the sandbox executor binds for a whole episode.
EXEC_PATHS: tuple[str, ...] = ("/usr", "/bin", "/lib", "/lib64", "/etc/alternatives", "/etc/ssl", "/opt")
#: The child side of a sandboxed call; it imports nothing from reef, so only this file needs binding.
CHILD = Path(__file__).with_name("sandboxed.py")


class ToolFailed(Exception):
    """The tool's ``run`` raised in the child; the message is already the child's type and text."""


class SandboxFailed(Exception):
    """The enforcer could not run the call at all, so the failure is the sandbox's and not the tool's."""


class Tool(Protocol):
    """What an enforcer needs of a tool module: its name, its declaration, its file, and its ``run``."""

    name: str
    capabilities: tuple[str, ...]
    path: Path | None

    def run(self, args: dict[str, Any], workdir: str, /) -> Any: ...


class Enforcer(Protocol):
    """One way to run a tool call: a mode name for the log, what it withholds per tool, and the run itself."""

    mode: str

    def describe(self, tool: Tool | None) -> dict[str, Any]: ...

    def run(self, tool: Tool, arguments: dict[str, Any], workdir: Path) -> Any: ...


def denied(capabilities: Sequence[str]) -> list[str]:
    """What a bwrap profile withholds from a tool declaring ``capabilities``, in the closed order."""
    return [item for item in DENIABLE if item not in capabilities]


def bwrap_argv(
    capabilities: Sequence[str],
    *,
    workdir: Path,
    tools_dir: Path,
    interpreter: Path,
    prefix: Path,
    base_prefix: Path,
    path: str = "",
) -> list[str]:
    """The bubblewrap command one call runs under, derived from its declaration; the adapter guide states the profile."""
    granted = set(capabilities)
    cmd = ["bwrap", *ISOLATION]
    if "network" not in granted:
        cmd.append("--unshare-net")
    candidates = [Path(base) for base in ((*EXEC_PATHS, *LIBRARY_PATHS) if "exec" in granted else LIBRARY_PATHS)]
    candidates += [interpreter, interpreter.resolve()]
    for root in dict.fromkeys((base_prefix, prefix)):
        candidates += [root / "lib", root / "lib64", root / "pyvenv.cfg"]
    candidates += [CHILD, tools_dir]
    bound: list[Path] = []
    for candidate in candidates:
        if candidate.exists() and not any(candidate == seen or seen in candidate.parents for seen in bound):
            bound.append(candidate)
            cmd += ["--ro-bind", str(candidate), str(candidate)]
    cmd += ["--bind" if "write" in granted else "--ro-bind", str(workdir), str(workdir)]
    cmd += ["--chdir", str(workdir), "--setenv", "HOME", str(workdir)]
    if "exec" in granted and path:
        cmd += ["--setenv", "PATH", path]
    cmd += ["--", str(interpreter), "-I", "-X", "utf8", str(CHILD)]
    return cmd


class InProcessEnforcer:
    """The default: ``run`` is a plain call in the loop's process and nothing is enforced."""

    mode = "none"

    def describe(self, tool: Tool | None) -> dict[str, Any]:
        return {"mode": self.mode, "denied": []}

    def run(self, tool: Tool, arguments: dict[str, Any], workdir: Path) -> Any:
        return tool.run(arguments, str(workdir))


class BwrapEnforcer:
    """Each call imports the tool's module afresh in a child under the profile its declaration derives."""

    mode = "bwrap"

    def describe(self, tool: Tool | None) -> dict[str, Any]:
        return {"mode": self.mode, "denied": denied(tool.capabilities) if tool is not None else []}

    def run(self, tool: Tool, arguments: dict[str, Any], workdir: Path) -> Any:
        if tool.path is None:
            raise SandboxFailed(f"tool {tool.name!r} has no module file to run in a child process")
        argv = bwrap_argv(
            tool.capabilities,
            workdir=workdir,
            tools_dir=tool.path.parent,
            interpreter=Path(sys.executable),
            prefix=Path(sys.prefix),
            base_prefix=Path(sys.base_prefix),
            path=os.environ.get("PATH", ""),
        )
        request = json.dumps({"path": str(tool.path), "arguments": arguments, "workdir": str(workdir)}, default=str)
        try:
            done = subprocess.run(
                argv,
                input=request,
                capture_output=True,
                text=True,
                env={"PATH": os.environ.get("PATH", "")},
                check=False,
            )
        except OSError as exc:
            raise SandboxFailed(f"bwrap could not start: {exc}") from exc
        reply: Any = None
        if done.returncode == 0:
            try:
                reply = json.loads(done.stdout)
            except json.JSONDecodeError:
                reply = None
        if not isinstance(reply, dict):
            raise SandboxFailed(f"tool process exited {done.returncode}: {done.stderr.strip()[-600:]}")
        if not reply.get("ok"):
            raise ToolFailed(str(reply.get("error") or "the tool failed"))
        return reply.get("text", "")


def select_enforcer(environ: Mapping[str, str]) -> Enforcer:
    """The enforcer ``REEF_NATIVE_ENFORCE`` names: unset or ``none`` runs tools in process, ``bwrap`` needs bwrap."""
    mode = environ.get(ENFORCE_ENV) or "none"
    if mode == "none":
        return InProcessEnforcer()
    if mode == "bwrap":
        if shutil.which("bwrap") is None:
            raise ValueError(f"{ENFORCE_ENV}=bwrap but bubblewrap (bwrap) is not on PATH")
        return BwrapEnforcer()
    raise ValueError(f"{ENFORCE_ENV}={mode!r} names no enforcer; use none or bwrap")
