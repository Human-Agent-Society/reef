"""Bounded raw reads of a committed step's retained files, without interpreting trajectories."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

MAX_RECORD_BYTES = 4 * 1024 * 1024
MAX_RECORD_FILES = 1000


def read_step_records(root: Path | None, directory: str, relative: str | None) -> dict[str, Any]:
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
