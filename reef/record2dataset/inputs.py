"""Text snapshots and record documents supplied to the task designer."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from reef.core.artifact_ref import encode_artifact_ref
from reef.core.records_types import AgentRecord

MAX_ASSET_FILES = 128
MAX_ASSET_BYTES = 256 * 1024


def record_document(record: AgentRecord) -> dict[str, object]:
    """Keep the source record's payload, identity, references, and artifact version."""
    return {
        "agent_record_id": record.agent_record_id,
        "scenario": record.scenario,
        "request_type": record.request_type.value,
        "payload": dict(record.payload),
        "created_at": record.created_at,
        "references": list(record.references),
        "artifact_ref": encode_artifact_ref(record.artifact_ref) if record.artifact_ref is not None else None,
    }


def read_asset_files(paths: Sequence[Path]) -> dict[str, str]:
    """Read selected UTF-8 files or directories on the caller's machine.

    Names are relative to each selected directory, prefixed with ``asset-N/``.
    The reader rejects symlinks and special files. Missing, unreadable, binary,
    or oversized inputs raise ValueError. The reader does not omit content.
    """
    files: dict[str, str] = {}
    byte_count = 0

    def _read_file(path: Path, name: str) -> None:
        nonlocal byte_count
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"asset must be a regular file, not a symlink or special file: {path}")
        if len(files) >= MAX_ASSET_FILES:
            raise ValueError(f"assets exceed {MAX_ASSET_FILES} files; select a smaller snapshot")
        with path.open("rb") as stream:
            content = stream.read(MAX_ASSET_BYTES - byte_count + 1)
        byte_count += len(content)
        if byte_count > MAX_ASSET_BYTES:
            raise ValueError(f"assets exceed {MAX_ASSET_BYTES} bytes; select a smaller snapshot")
        text = content.decode("utf-8")
        if "\x00" in text:
            raise ValueError(f"asset must be UTF-8 text without NUL bytes: {path}")
        files[name] = text

    def _raise_walk_error(error: OSError) -> None:
        raise error

    try:
        for index, path in enumerate(paths):
            if path.is_symlink():
                raise ValueError(f"asset must not be a symlink: {path}")
            prefix = f"asset-{index}"
            if path.is_dir():
                for directory, directories, names in os.walk(path, onerror=_raise_walk_error, followlinks=False):
                    directories.sort()
                    for name in directories:
                        if (Path(directory) / name).is_symlink():
                            raise ValueError(f"asset directory contains a symlink: {Path(directory) / name}")
                    for name in sorted(names):
                        child = Path(directory) / name
                        _read_file(child, f"{prefix}/{child.relative_to(path).as_posix()}")
            else:
                _read_file(path, f"{prefix}/{path.name}")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read assets as UTF-8 text: {exc}") from exc
    return files
