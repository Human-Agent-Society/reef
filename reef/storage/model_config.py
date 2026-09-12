"""Read, replace, and archive private per-scenario model JSON files.

Registry operations serialize access and validate model values before writing.
These functions preserve the existing filenames and owner-only permissions;
they hold no cache or runtime state and do not select a record backend.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path


def _model_path(directory: Path | None, scenario: str) -> Path | None:
    if directory is None:
        return None
    key = hashlib.sha256(scenario.encode("utf-8")).hexdigest()
    return directory / f"{key}-model.json"


def read_model_config(directory: Path | None, scenario: str) -> object:
    """Read the stored value, or null if no override has been saved."""
    path = _model_path(directory, scenario)
    return None if path is None or not path.exists() else json.loads(path.read_text())


def write_model_config(directory: Path | None, scenario: str, value: object) -> None:
    """Atomically replace a validated value; directory=None selects memory-only use."""
    path = _model_path(directory, scenario)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".model-")
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(value, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def archive_model_config(directory: Path | None, scenario: str) -> tuple[str, ...]:
    """Move a removed scenario's settings out of the active namespace."""
    path = _model_path(directory, scenario)
    if path is None or not path.exists():
        return ()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    key = hashlib.sha256(scenario.encode("utf-8")).hexdigest()
    destination = path.parent / "archived" / f"{key}-{stamp}-{uuid.uuid4().hex}"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / path.name
    shutil.move(str(path), str(target))
    return (str(target),)
