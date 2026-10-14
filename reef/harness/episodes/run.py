"""Episode lifecycle: render, launch headless, collect, clean up exactly.

One episode is one process of the harness binary inside a throwaway root:
the rendered files land under the root, the descriptor's relocation
environment points the binary's entire composition at it, the binary runs
headless in an empty ``workspace/`` working directory, and the trajectory is
read back before the root is removed. Both surveyed harnesses re-read their
whole composition at process start and accept environment variables that
relocate their config and session roots, so the inverse of an episode is
exactly "delete the root" - no shared state survives.

Cleanup is audited, not assumed: every file found under the root that the
episode did not write, is not under ``workspace/`` (the task's own output
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
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from reef.core.errors import ReefError
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.episodes.executor import (
    EpisodeExecutor,
    EpisodeLaunchError,
    EpisodeTimeout,
    LocalExecutor,
    SandboxExecutor,
)
from reef.harness.episodes.trajectory import reader_for


class EpisodeError(ReefError):
    """The episode could not be launched or torn down."""


class TrajectoryKeepError(ReefError):
    """The episode ran but its trajectory could not be kept, so the step has no record of it."""


@dataclass(frozen=True)
class EpisodeResult:
    """One headless invocation's observable outcome."""

    exit_code: int
    stdout: str
    stderr: str
    trajectory: tuple[dict[str, Any], ...]
    residue: tuple[str, ...]


def _remove_episode_root(root: Path) -> None:
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
            raise EpisodeError(f"cannot remove episode root {root}: {exc}") from exc


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
        raise TrajectoryKeepError(f"cannot keep episode trajectory under {target}: {exc}") from exc


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


def run_episode(
    descriptor: AdapterDescriptor,
    files: Mapping[str, str],
    prompt: str,
    *,
    binary: str | None = None,
    timeout: float = 600.0,
    executor: EpisodeExecutor | None = None,
    keep_dir: Path | None = None,
    workspace_files: Mapping[str, str] | None = None,
    initialize_git: bool = False,
    keep_workspace: bool = False,
    online: bool = False,
    task_environment: Mapping[str, str] | None = None,
) -> EpisodeResult:
    """Run one headless episode of ``descriptor``'s harness over ``files``.

    ``binary`` overrides the descriptor's binary name (the seam hermetic
    tests drive a fake harness through); ``prompt`` substitutes into the
    descriptor's argv template. ``executor`` decides how the process runs
    (the default local subprocess, or a sandbox); the rendered files,
    relocation environment, trajectory read, and residue collection are the
    same for every executor. Only the local executor keeps the descriptor's
    ``host_env`` from the service and, on macOS, makes the root of an adapter
    with ``is_root_bind_mounted`` under ``~/.reef/episodes`` rather than the
    temp directory. The episode root is removed before this returns, success
    or failure; ``keep_dir`` receives a copy of the trajectory directory
    first, so a step record can hold what the root held, and a copy that
    fails raises ``TrajectoryKeepError`` rather than an ``EpisodeError``.
    ``workspace_files`` seeds a fresh application workspace; ``initialize_git``
    commits that fixture for diff-based tasks. ``keep_workspace`` retains the
    resulting regular files under ``keep_dir/workspace`` with size/count limits.
    ``online`` enables model-backed pi extensions and makes the configured
    binary available to child agents; the default keeps existing offline episodes.
    ``task_environment`` explicitly forwards operator-selected variables,
    such as network proxies, without replacing the harness relocation environment.
    """
    executor = executor or LocalExecutor()
    # Adapters can validate a conditional boundary (for example, remote task
    # containers with a sandboxed runner). Otherwise keep the default refusal
    # to nest an adapter's local container inside bubblewrap.
    if descriptor.validate_execution is not None:
        try:
            descriptor.validate_execution(files, executor)
        except EpisodeLaunchError as exc:
            raise EpisodeError(str(exc)) from exc
    elif descriptor.self_isolating and isinstance(executor, SandboxExecutor):
        raise EpisodeError(
            f"adapter {descriptor.name!r} isolates episodes in its own container and cannot run under "
            "evolution.executor: sandbox; use 'local' and let the adapter's container be the boundary"
        )
    reader = reader_for(descriptor.trajectory_format)  # fail before any disk work
    # Docker on macOS runs in a VM that shares the home directory with the host, and colima does not share the temp
    # directory, so a root the binary bind-mounts from lives under the home there. Linux Docker shares every path,
    # and a sandbox or a remote executor runs no local container.
    if descriptor.is_root_bind_mounted and isinstance(executor, LocalExecutor) and sys.platform == "darwin":
        parent = Path.home() / ".reef" / "episodes"
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise EpisodeError(f"cannot create the episode root directory {parent}: {exc}") from exc
    else:
        parent = None
    root = Path(tempfile.mkdtemp(prefix=f"reef-episode-{descriptor.name}-", dir=parent))
    try:
        written = set()
        for relative, text in files.items():
            relative_path = PurePosixPath(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise EpisodeError(f"render path {relative!r} escapes the episode root")
            target = root / relative_path
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
            except (OSError, UnicodeEncodeError) as exc:
                raise EpisodeError(f"cannot write render path {relative!r}: {exc}") from exc
            written.add(str(relative_path))
        workspace = root / "workspace"
        workspace.mkdir()
        for relative, text in (workspace_files or {}).items():
            relative_path = PurePosixPath(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts or not relative_path.parts:
                raise EpisodeError(f"workspace path {relative!r} escapes the workspace")
            target = workspace / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        if initialize_git:
            for command in (
                ["git", "init", "-q"],
                ["git", "add", "."],
                [
                    "git",
                    "-c",
                    "user.name=Reef evaluation",
                    "-c",
                    "user.email=reef@example.invalid",
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "-qm",
                    "Evaluation fixture",
                    "--allow-empty",
                ],
            ):
                try:
                    subprocess.run(command, cwd=workspace, check=True, capture_output=True, timeout=30)
                except (OSError, subprocess.SubprocessError) as exc:
                    raise EpisodeError(f"cannot initialize fixture repository: {exc}") from exc
        writable_paths = tuple(root / PurePosixPath(relative) for relative in descriptor.writable_paths)
        try:
            for path in writable_paths:
                path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise EpisodeError(f"cannot prepare writable episode state: {exc}") from exc
        env = {key: value.replace("{root}", str(root)) for key, value in descriptor.env.items()}
        # Point HOME at the root unless the descriptor relocates it itself:
        # config discovery that ignores the relocation vars still lands inside
        # the episode instead of in the operator's real home.
        env.setdefault("HOME", str(root))
        for name, value in (task_environment or {}).items():
            if not isinstance(name, str) or not isinstance(value, str) or not name or name in env:
                raise EpisodeError("task environment must contain string variables outside the harness environment")
            env[name] = value
        if online:
            env.pop("PI_OFFLINE", None)
            installed_binary = binary or shutil.which(descriptor.binary)
            if installed_binary is not None:
                binary_directory = str(Path(installed_binary).absolute().parent)
                env["PATH"] = os.pathsep.join([binary_directory, os.environ.get("PATH", "")])
        if isinstance(executor, LocalExecutor):
            # Host tools the relocated HOME would hide keep the service's own settings; a sandbox forwards only
            # its explicit env_from.
            for key, default in descriptor.host_env.items():
                value = os.environ.get(key) or default.replace("{home}", str(Path.home()))
                if value:
                    env[key] = value
        argv = [binary or descriptor.binary, *(token.replace("{prompt}", prompt) for token in descriptor.argv)]
        try:
            outcome = executor.launch(
                argv,
                root=root,
                workspace=workspace,
                env=env,
                timeout=timeout,
                writable_paths=writable_paths,
                readonly_paths=tuple(root / PurePosixPath(relative) for relative in written),
            )
        except EpisodeLaunchError as exc:
            raise EpisodeError(str(exc)) from exc
        except EpisodeTimeout as exc:
            raise EpisodeError(str(exc)) from exc
        trajectory = reader(root / descriptor.trajectory_path)
        residue = tuple(
            sorted(
                relative
                for relative in (str(path.relative_to(root).as_posix()) for path in root.rglob("*") if path.is_file())
                if relative not in written
                and not relative.startswith("workspace/")
                and not _whitelisted(relative, descriptor.cleanup_whitelist)
            )
        )
        return EpisodeResult(
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
                if keep_workspace and (root / "workspace").is_dir():
                    keep_workspace_files(root / "workspace", keep_dir / "workspace")
        finally:
            _remove_episode_root(root)


def keep_workspace_files(source: Path, target: Path) -> None:
    """Retain bounded regular output files; never follow agent-created symlinks."""
    total_bytes = 0
    file_count = 0
    target.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if ".git" in relative.parts or path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            continue
        if not path.is_file():
            continue
        file_count += 1
        total_bytes += path.stat().st_size
        if file_count > 1000 or total_bytes > 50 * 1024 * 1024:
            raise TrajectoryKeepError("workspace output exceeds 1000 files or 50 MiB")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
