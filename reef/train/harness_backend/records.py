"""The step record: the raw material behind each harness step, for debugging and audit.

With ``evolution.step_record_dir`` set, each step claims a fresh directory
``<scenario>/<step>`` (a retried step ``<step>-<attempt>``) and writes
``proposer.json`` (every model call the proposer made), ``patch.json`` (the
proposed patch, written before validation) and ``rollouts/`` (each evaluation
rollout's trajectory files). Every text in a record is redacted of
credential-shaped literals and clipped, since the record holds traffic the
tree's checks have not seen. ``StepRecord.read`` serves bounded raw reads of
a committed step's files to the release pages.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reef.harness.tree.checks import clip_record_text

PROPOSER_FILE = "proposer.json"
PATCH_FILE = "patch.json"


def bounded(value: Any) -> Any:
    """A copy of ``value`` that JSON can write, with every string redacted and clipped at the record cap."""
    if isinstance(value, str):
        return clip_record_text(value)
    if isinstance(value, Mapping):
        return {str(key): bounded(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return [bounded(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return clip_record_text(str(value))


class StepRecord:
    """The record directory of one scenario's steps.

    ``root`` is the scenario's directory under ``evolution.step_record_dir``,
    created at construction so an unwritable path refuses to boot; ``None``
    turns the record off, and then every write does nothing.
    """

    def __init__(self, root: Path | None) -> None:
        self.root = root
        if root is not None:
            try:
                root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ValueError(f"step_record_dir {root} cannot be created: {exc}") from exc

    def claim(self, step: int) -> Path | None:
        """A fresh directory for ``step``; a retried step gets ``<step>-<attempt>``, so no record is overwritten."""
        if self.root is None:
            return None
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
    def write(step_dir: Path | None, name: str, payload: Any) -> None:
        """One record file of a step as JSON, created exclusively: a record file is never replaced."""
        if step_dir is None:
            return
        with open(step_dir / name, "x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, default=str) + "\n")

    def read(self, directory: str, relative: str | None) -> dict[str, Any]:
        """The listing of a committed step's retained files, or one file's text, within the inspection limits."""
        return read_step_records(self.root, directory, relative)


MAX_RECORD_BYTES = 4 * 1024 * 1024
MAX_RECORD_FILES = 1000


def read_step_records(root: Path | None, directory: str, relative: str | None) -> dict[str, Any]:
    """Bounded raw reads of one step's record directory under ``root``, never interpreting trajectories."""
    if root is None:
        return {"status": "disabled", "files": []}
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
        return {"status": "retained", "path": relative, "text": content.decode("utf-8", errors="replace")}
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
