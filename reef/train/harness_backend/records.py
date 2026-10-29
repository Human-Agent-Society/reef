"""The step records: each harness step's workspace, kept after the step for debugging and audit.

Each step claims a fresh directory ``<steps>/<step>`` (a retried step
``<step>-<attempt>``) under the scenario's state directory. The directory is
the step's workspace (``reef.train.harness_backend.workspace``) without its
``scratch/``, plus ``calls.json``, every model call of the step. What the
method says about its step is in the workspace's ``output/``. The backend
redacts credential-shaped literals from ``calls.json`` and from the summary
and notes in ``output/``, and clips them, since they hold model traffic that
the tree's checks have not seen.
``material/`` keeps the traces as the clients sent them, as the record store
does, and ``harness/`` the tree as the method left it. ``read_step_records``
redacts every text it serves, so the release pages show no credential of
either.
``StepRecord.read`` serves bounded raw reads of a step's files to the
release pages.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from reef.harness.tree.checks import redact_secret_shaped

CALLS_FILE = "calls.json"


class StepRecord:
    """The directory that holds one scenario's step records.

    Values: ``root`` is the directory, created at construction so an
    unwritable path refuses to start.
    Purpose: the backend claims one directory per step here, and the release
    pages read the records through ``read``.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"the step record directory {root} cannot be created: {exc}") from exc

    def claim(self, step: int) -> Path:
        """A fresh directory for ``step``; a retried step gets ``<step>-<attempt>``, so no record is overwritten."""
        attempt = 1
        step_dir = self.root / str(step)
        while True:
            try:
                step_dir.mkdir(parents=True)
            except FileExistsError:
                attempt += 1
                step_dir = self.root / f"{step}-{attempt}"
                continue
            return step_dir

    @staticmethod
    def write(step_dir: Path, name: str, payload: Any) -> None:
        """One record file of a step as JSON, created exclusively: a record file is never replaced."""
        with open(step_dir / name, "x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, default=str) + "\n")

    def read(self, directory: str, relative: str | None) -> dict[str, Any]:
        """The listing of a step's retained JSON files, or one file's text, within the inspection limits."""
        return read_step_records(self.root, directory, relative)


MAX_RECORD_BYTES = 4 * 1024 * 1024
MAX_RECORD_FILES = 1000


def read_step_records(root: Path, directory: str, relative: str | None) -> dict[str, Any]:
    """Bounded raw reads of one step's record directory under ``root``, never interpreting trajectories."""
    base = root.resolve()
    step = Path(directory)
    # A recorded step is one immediate child of this backend's scenario directory.
    if step.is_symlink() or step.resolve().parent != base:
        raise ValueError("step record is outside this scenario's record directory")
    if not step.is_dir():
        return {"status": "missing", "files": []}
    step = step.resolve()
    if relative is not None:
        parts = Path(relative).parts
        if not parts or Path(relative).is_absolute() or ".." in parts:
            raise ValueError("path must name a relative record file")
        target = step
        for part in parts:
            target = target / part
            if target.is_symlink():
                raise ValueError("record symlinks cannot be read")
        if target.suffix not in {".json", ".jsonl"} or not target.is_file():
            raise FileNotFoundError("record file is not retained")
        with target.open("rb") as handle:
            content = handle.read(MAX_RECORD_BYTES + 1)
        if len(content) > MAX_RECORD_BYTES:
            raise ValueError("record exceeds the 4 MiB inspection limit")
        # A step record holds what the method wrote and what clients sent; the route serves no credential of either.
        text = redact_secret_shaped(content.decode("utf-8", errors="replace"))
        return {"status": "retained", "path": relative, "text": text}
    files = []
    visited = 0
    for current, dirs, names in os.walk(step, followlinks=False):
        dirs[:] = sorted(name for name in dirs if not (Path(current) / name).is_symlink())
        visited += len(dirs) + len(names)
        if visited > MAX_RECORD_FILES:
            raise ValueError("step exceeds the 1000 entry inspection limit")
        for name in sorted(names):
            target = Path(current) / name
            if target.suffix in {".json", ".jsonl"} and not target.is_symlink() and target.is_file():
                files.append({"path": target.relative_to(step).as_posix(), "bytes": target.stat().st_size})
    return {"status": "retained", "files": sorted(files, key=lambda item: item["path"])}
