"""Crash-safe JSON and directory primitives for the training-job marker.

Invariant: after any crash, a reader sees either the previous complete file
or the new complete file, never a partial write. ``write_json`` writes to a
same-directory temporary file, fsyncs it, renames it over the target, and
fsyncs the directory; ``mkdir_durable`` refuses symlinked path components and
fsyncs each created directory's parent so the entries themselves survive.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
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
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
