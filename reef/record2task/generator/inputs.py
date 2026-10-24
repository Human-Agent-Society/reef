"""The request to a task generator, and the text snapshots and record documents supplied to the task designer."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from reef.core.artifact_ref import encode_artifact_ref
from reef.core.records_types import AgentRecord

MAX_ASSET_FILES = 128
MAX_ASSET_BYTES = 256 * 1024


@dataclass(frozen=True)
class TaskGenerationRequest:
    """Source records, requirements, and materials for one call to a task generator.

    Assets name files or directories on the caller's machine. Alternatively,
    asset_files maps relative file names to UTF-8 contents. Supply paths or
    contents, not both. Construction validates fields but performs no I/O.
    Consumers own file reads, transport limits, and generation settings.
    Description-only generation uses an empty source_records tuple.
    ``generation`` numbers the request among the requests of one recipe; the
    recipe sends the results of the request's tasks back under this number.
    """

    source_records: tuple[AgentRecord, ...]
    description: str
    assets: tuple[Path, ...] = ()
    asset_files: Mapping[str, str] = field(default_factory=dict)
    generation: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if not isinstance(self.source_records, tuple):
            raise ValueError("source_records must be a tuple of AgentRecord values")
        if any(not isinstance(record, AgentRecord) for record in self.source_records):
            raise TypeError("source_records must contain AgentRecord values")
        if len({record.scenario for record in self.source_records}) > 1:
            raise ValueError("source_records must belong to one scenario")
        record_ids = [record.agent_record_id for record in self.source_records]
        if any(not isinstance(record_id, str) or not record_id for record_id in record_ids) or len(
            set(record_ids)
        ) != len(record_ids):
            raise ValueError("source_records must have distinct non-empty record ids")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be non-empty text")
        if not isinstance(self.assets, tuple) or any(not isinstance(path, Path) for path in self.assets):
            raise TypeError("assets must be a tuple of pathlib.Path values")
        if not isinstance(self.asset_files, Mapping) or any(
            not isinstance(name, str) or not isinstance(text, str) for name, text in self.asset_files.items()
        ):
            raise ValueError("asset_files must map relative file names to UTF-8 text")
        if self.assets and self.asset_files:
            raise ValueError("supply assets or asset_files, not both")
        for name, text in self.asset_files.items():
            path = PurePosixPath(name)
            if not name or name == "." or path.is_absolute() or ".." in path.parts or "\\" in name or "\x00" in name:
                raise ValueError("asset_files must use relative file names without parent traversal")
            if "\x00" in text:
                raise ValueError("asset_files must contain UTF-8 text without NUL bytes")
            try:
                text.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("asset_files must contain UTF-8 text") from exc


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
