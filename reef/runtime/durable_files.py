"""Durable files: JSON objects written atomically, and directories created durably.

A writer replaces a file in one step and fsyncs it and its directory, so a reader or a restart sees the old file or
the new one, never a torn one. Training components keep their job records with these helpers.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    """The JSON object in ``path``, or None when the file is missing; ``ValueError`` for another JSON value."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write ``value`` atomically: a reader sees the old file or the new one, never a torn one."""
    mkdir_durable(path.parent)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name.lstrip('.')}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            json.dump(value, file, allow_nan=False, separators=(",", ":"), sort_keys=True)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        fsync_dir(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def mkdir_durable(path: Path) -> None:
    """Create ``path`` and its parents, refusing symlinks, and fsync each new entry."""
    missing = []
    current = path
    while not current.exists():
        if current.is_symlink():
            raise OSError(f"refusing symlinked directory: {current}")
        missing.append(current)
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise OSError(f"unsafe directory: {current}")
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            if directory.is_symlink() or not directory.is_dir():
                raise OSError(f"unsafe directory: {directory}") from None
        fsync_dir(directory.parent)


def fsync_dir(path: Path) -> None:
    """Flush the directory ``path`` itself, so the entries created or renamed in it survive a crash."""
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
