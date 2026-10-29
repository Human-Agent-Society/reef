"""Run lifecycle: render, launch headless, collect, clean up exactly.

One run is one process of the harness binary inside a throwaway root:
the rendered files land under the root, the descriptor's relocation
environment points the binary's whole configuration at it, the binary runs
headless in an empty ``workspace/`` working directory, and the trajectory is
read back before the root is removed. The bundled harnesses re-read their
whole configuration at process start and accept environment variables that
relocate their config and session roots, so the inverse of a run is
exactly "delete the root" - no shared state survives.

``launch_agent`` is the launch itself: write the rendered files, prepare the
writable directories and run the process. A task runs it in a throwaway
root; harness evolution runs the deployment's coding agent with it in a
step's workspace. ``read_session`` reads back the agent's session log.

Cleanup is audited, not assumed: every file found under the root that the
run did not write, is not under ``workspace/`` (the task's own output
area), and does not match the descriptor's cleanup whitelist is reported as
``residue``. The whitelist is where adapter traps live - opencode's boot
mutates its own config directory (npm install, ``.gitignore``), and both
harnesses write session storage; those are declared, everything else is a
finding.
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from reef.core.errors import ReefError
from reef.execution.sandbox.environment import execute
from reef.execution.sandbox.process import (
    LocalExecutor,
    ProcessOutcome,
    ProgramExecutor,
    ProgramLaunchError,
    ProgramTimeout,
)
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.tasks.trajectory import reader_for


class TaskRunError(ReefError):
    """The run could not be launched or torn down."""


class TrajectoryKeepError(ReefError):
    """The run completed but its trajectory could not be kept, so the step has no record of it."""


@dataclass(frozen=True)
class TaskResult:
    """One headless invocation's observable outcome."""

    exit_code: int
    stdout: str
    stderr: str
    trajectory: tuple[dict[str, Any], ...]
    residue: tuple[str, ...]


def _remove_run_root(root: Path) -> None:
    """Remove a root, repairing ordinary permission damage once."""

    if not root.exists() and not root.is_symlink():
        return

    def repair(path: Path) -> None:
        if path.is_symlink():
            return
        if path.is_dir():
            path.chmod(path.stat().st_mode | stat.S_IRWXU)
            for child in path.iterdir():
                repair(child)
        else:
            path.chmod(path.stat().st_mode | stat.S_IRUSR | stat.S_IWUSR)

    try:
        shutil.rmtree(root)
    except OSError:
        if not root.exists() and not root.is_symlink():
            return
        try:
            repair(root)
            shutil.rmtree(root)
        except OSError as exc:
            raise TaskRunError(f"cannot remove run root {root}: {exc}") from exc


def _keep_trajectory(source: Path, target: Path) -> None:
    """Copy the trajectory directory out of the root; a crash before any event leaves nothing to keep."""
    if not source.is_dir():
        return
    try:
        # Executors may create the writable session directory before the harness starts.
        if not any(source.iterdir()):
            return
        # Links are copied as links so an evolved tool cannot pull an outside file into the record,
        # and an existing target is refused so a kept record is never merged over.
        shutil.copytree(source, target, symlinks=True)
    except OSError as exc:
        raise TrajectoryKeepError(f"cannot keep run trajectory under {target}: {exc}") from exc


def _whitelisted(path: str, patterns: tuple[str, ...]) -> bool:
    pure = PurePosixPath(path)
    for pattern in patterns:
        # "dir/**" whitelists a whole subtree; other patterns are plain
        # glob matches against the root-relative path.
        if pattern.endswith("/**"):
            prefix = pattern[:-3]
            if path == prefix or path.startswith(prefix + "/"):
                return True
        elif pure.match(pattern):
            return True
    return False


def headless_env(descriptor: AdapterDescriptor, root: Path) -> dict[str, str]:
    """The descriptor's relocation environment pointed at ``root``, and ``root`` as ``HOME`` unless the descriptor
    relocates it itself: config discovery that ignores the relocation variables still lands inside ``root``, not in
    the operator's real home."""
    env = {key: value.replace("{root}", str(root)) for key, value in descriptor.env.items()}
    env.setdefault("HOME", str(root))
    return env


def headless_argv(descriptor: AdapterDescriptor, binary: str, task: str, root: Path) -> list[str]:
    """The descriptor's headless invocation of ``binary``, with ``task`` in its ``{task}`` slots and ``root``,
    the run root, in its ``{root}`` slots, such as the path of the plugin a flag loads."""
    return [binary, *(token.replace("{root}", str(root)).replace("{task}", task) for token in descriptor.argv)]


def write_rendered(descriptor: AdapterDescriptor, files: Mapping[str, str], directory: Path, root: str) -> None:
    """Write the rendered ``files`` under ``directory``, with ``{root}`` in each config file as ``root``.

    ``ProgramLaunchError`` for a path that leaves ``directory`` or a file that cannot be written.
    """
    for relative, text in descriptor.place_root(files, root).items():
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ProgramLaunchError(f"render path {relative!r} escapes the run root")
        target = directory / relative_path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        except (OSError, UnicodeEncodeError) as exc:
            raise ProgramLaunchError(f"cannot write render path {relative!r}: {exc}") from exc


def launch_agent(
    descriptor: AdapterDescriptor,
    executor: ProgramExecutor,
    argv: Sequence[str],
    files: Mapping[str, str],
    env: Mapping[str, str],
    *,
    root: Path,
    home: Path,
    cwd: Path,
    timeout: float,
    writable: Sequence[Path] = (),
    readonly: Sequence[Path] = (),
) -> ProcessOutcome:
    """Run ``argv`` as ``descriptor``'s agent over the rendered ``files``, and wait for it.

    ``root`` holds everything the process may see, and ``cwd``, its working
    directory, is writable. ``files`` are written under ``home``, with
    ``{root}`` as ``home``, and stay read-only; the descriptor's state
    directories under ``home`` (``writable_paths``) and ``writable`` stay
    writable, and ``readonly`` names more paths the process only reads.
    Raises ``ProgramTimeout`` as the executor does, and ``ProgramLaunchError``
    when a file cannot be written or the process cannot start.
    """
    write_rendered(descriptor, files, home, str(home))
    state = tuple(home / PurePosixPath(relative) for relative in descriptor.writable_paths)
    try:
        cwd.mkdir(parents=True, exist_ok=True)
        for path in state:
            path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ProgramLaunchError(f"cannot prepare writable run state: {exc}") from exc
    return execute(
        executor,
        list(argv),
        root=root,
        workspace=cwd,
        env=dict(env),
        timeout=timeout,
        writable_paths=(*state, *writable),
        readonly_paths=(*(home / PurePosixPath(relative) for relative in files), *readonly),
    )


def read_session(descriptor: AdapterDescriptor, home: Path) -> list[Mapping[str, Any]]:
    """The session log of the agent whose home is ``home``, read by the descriptor's trajectory reader."""
    return list(reader_for(descriptor.trajectory_format)(home / descriptor.trajectory_path))


def run_task(
    descriptor: AdapterDescriptor,
    files: Mapping[str, str],
    task: str,
    *,
    binary: str | None = None,
    timeout: float = 600.0,
    executor: ProgramExecutor | None = None,
    keep_dir: Path | None = None,
) -> TaskResult:
    """Run one headless run of ``descriptor``'s harness over ``files``.

    ``binary`` overrides the descriptor's binary name (the seam hermetic
    tests drive a fake harness through); ``task``, the instruction or the task directory the adapter takes,
    substitutes into the
    descriptor's argv template. ``executor`` decides how the process runs
    (the default local subprocess, or a sandbox); the rendered files,
    relocation environment, trajectory read, and residue collection are the
    same for every executor. Only an executor that runs on the Reef host
    keeps the descriptor's ``host_env`` from the service and, on macOS, makes
    the root of an adapter with ``is_root_bind_mounted`` under
    ``~/.reef/runs`` rather than the temp directory. The run root is removed before this returns, success
    or failure; ``keep_dir`` receives a copy of the trajectory directory
    first, so a step record can hold what the root held, and a copy that
    fails raises ``TrajectoryKeepError`` rather than a ``TaskRunError``.
    """
    executor = executor or LocalExecutor()
    # Adapters can validate a conditional boundary (for example, remote task
    # containers with a sandboxed runner). Otherwise keep the default refusal
    # to nest an adapter's local container inside bubblewrap.
    if descriptor.execution_validator is not None:
        try:
            descriptor.execution_validator(files, executor)
        except ProgramLaunchError as exc:
            raise TaskRunError(str(exc)) from exc
    elif descriptor.self_isolating and not executor.runs_on_host:
        raise TaskRunError(
            f"adapter {descriptor.name!r} isolates runs in its own container, which runs only on the Reef "
            "host; use evolution.executor: local and let the adapter's container be the boundary"
        )
    reader = reader_for(descriptor.trajectory_format)  # fail before any disk work
    # Docker on macOS runs in a VM that shares the home directory with the host, and colima does not share the temp
    # directory, so a root the binary bind-mounts from lives under the home there. Linux Docker shares every path,
    # and an executor off the Reef host runs no local container.
    if descriptor.is_root_bind_mounted and executor.runs_on_host and sys.platform == "darwin":
        parent = Path.home() / ".reef" / "runs"
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise TaskRunError(f"cannot create the run root directory {parent}: {exc}") from exc
    else:
        parent = None
    root = Path(tempfile.mkdtemp(prefix=f"reef-run-{descriptor.name}-", dir=parent))
    try:
        env = headless_env(descriptor, root)
        if executor.runs_on_host:
            # Host tools the relocated HOME would hide keep the service's own settings; a sandbox forwards only
            # its explicit env_from.
            for key, default in descriptor.host_env.items():
                value = os.environ.get(key) or default.replace("{home}", str(Path.home()))
                if value:
                    env[key] = value
        argv = headless_argv(descriptor, binary or descriptor.binary, task, root)
        try:
            outcome = launch_agent(
                descriptor, executor, argv, files, env, root=root, home=root, cwd=root / "workspace", timeout=timeout
            )
        except (ProgramLaunchError, ProgramTimeout) as exc:
            raise TaskRunError(str(exc)) from exc
        trajectory = reader(root / descriptor.trajectory_path)
        written = {str(PurePosixPath(relative)) for relative in files}
        residue = tuple(
            sorted(
                relative
                for relative in (str(path.relative_to(root).as_posix()) for path in root.rglob("*") if path.is_file())
                if relative not in written
                and not relative.startswith("workspace/")
                and not _whitelisted(relative, descriptor.cleanup_whitelist)
            )
        )
        return TaskResult(
            exit_code=outcome.exit_code,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            trajectory=trajectory,
            residue=residue,
        )
    finally:
        try:
            if keep_dir is not None:
                _keep_trajectory(root / descriptor.trajectory_path, keep_dir)
        finally:
            _remove_run_root(root)
