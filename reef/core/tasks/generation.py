"""Task generation inputs shared by processors and generation services.

This module holds values and field validation. Consumers own file reads,
transport limits, generation, and execution.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from reef.core.records_types import AgentRecord


@dataclass(frozen=True)
class TaskGenerationRequest:
    """Source records, requirements, and materials for one generated task.

    Assets name files or directories on the caller's machine. Alternatively,
    asset_files maps relative file names to UTF-8 contents. Supply paths or
    contents, not both. Construction validates fields but performs no I/O.
    Consumers own file reads, transport limits, and generation settings.
    Description-only generation uses an empty source_records tuple.
    """

    source_records: tuple[AgentRecord, ...]
    description: str
    assets: tuple[Path, ...] = ()
    asset_files: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
