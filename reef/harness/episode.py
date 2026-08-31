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

import contextlib
import os
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from reef.core.errors import ReefError
from reef.harness.descriptor import AdapterDescriptor
from reef.harness.trajectory import reader_for


class EpisodeError(ReefError):
    """The episode could not be launched or torn down."""


@dataclass(frozen=True)
class EpisodeResult:
    """One headless invocation's observable outcome."""

    exit_code: int
    stdout: str
    stderr: str
    trajectory: tuple[dict[str, Any], ...]
    residue: tuple[str, ...]


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
) -> EpisodeResult:
    """Run one headless episode of ``descriptor``'s harness over ``files``.

    ``binary`` overrides the descriptor's binary name (the seam hermetic
    tests drive a fake harness through); ``prompt`` substitutes into the
    descriptor's argv template. The episode root is removed before this
    returns, success or failure.
    """
    reader = reader_for(descriptor.trajectory_format)  # fail before any disk work
    root = Path(tempfile.mkdtemp(prefix=f"reef-episode-{descriptor.name}-"))
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
            except OSError as exc:
                raise EpisodeError(f"cannot write render path {relative!r}: {exc}") from exc
            written.add(str(relative_path))
        workspace = root / "workspace"
        workspace.mkdir()
        env = {key: value.replace("{root}", str(root)) for key, value in descriptor.env.items()}
        # Point HOME at the root unless the descriptor relocates it itself:
        # config discovery that ignores the relocation vars still lands inside
        # the episode instead of in the operator's real home.
        env.setdefault("HOME", str(root))
        argv = [binary or descriptor.binary, *(token.replace("{prompt}", prompt) for token in descriptor.argv)]
        try:
            # Own session so a timeout can kill the binary's whole process
            # group: coding agents spawn tool children as a matter of course,
            # and killing only the direct child would orphan them.
            process = subprocess.Popen(
                argv,
                cwd=workspace,
                env={**_inherited_env(), **env},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise EpisodeError(f"harness binary {argv[0]!r} not found") from exc
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise EpisodeError(f"episode timed out after {timeout}s: {argv}") from exc
        completed = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
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
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            trajectory=trajectory,
            residue=residue,
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _inherited_env() -> dict[str, str]:
    """The minimal parent environment an episode keeps: how to find binaries."""
    import os

    return {key: value for key, value in os.environ.items() if key in ("PATH", "SYSTEMROOT", "TMPDIR")}
