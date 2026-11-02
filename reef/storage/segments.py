"""Segment files: JSON Lines files that keep the large content of data items outside the database.

A segment is one immutable file of JSON Lines, in one partition of a ``SegmentStore``. The data store writes a
batch of contents as one segment, and keeps the place of each line, ``SegmentLine``, in its database. A reader reads
only the bytes of the lines that it needs. A deployment selects the store in ``storage.segment_backend``: ``local``
selects ``LocalSegmentStore``, and a plugin reference selects another subclass of ``SegmentStore``.
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from reef.core.errors import ReefError
from reef.core.plugins import NoSettings, Plugin


class SegmentNotFound(ReefError):
    """A segment that the store does not hold, or a range past its end."""


@dataclass(frozen=True)
class SegmentLine:
    """The place of one line in a segment.

    Values: ``segment_id`` is a non-empty string. ``offset`` is a non-negative int, the first byte of the line.
    ``length`` is a positive int, the byte count of the line without its newline.

    Purpose: ``segment_lines`` makes it for each line that it writes. The data store keeps it with an item or a
    trajectory node, and gives it to ``SegmentStore.read``.
    """

    segment_id: str
    offset: int
    length: int

    def __post_init__(self) -> None:
        if not isinstance(self.segment_id, str) or not self.segment_id:
            raise ValueError("a segment line's segment_id is a non-empty string")
        if isinstance(self.offset, bool) or not isinstance(self.offset, int) or self.offset < 0:
            raise ValueError("a segment line's offset is a non-negative int")
        if isinstance(self.length, bool) or not isinstance(self.length, int) or self.length <= 0:
            raise ValueError("a segment line's length is a positive int")


def segment_lines(segment_id: str, lines: Sequence[str]) -> tuple[bytes, tuple[SegmentLine, ...]]:
    """The bytes of the segment ``segment_id`` with ``lines``, and the place of each line, in order.

    Each line is UTF-8 and ends with a newline. A line holds no newline, so that the file is JSON Lines.
    """
    data = bytearray()
    places: list[SegmentLine] = []
    for line in lines:
        if not line or "\n" in line:
            raise ValueError("a segment line is a non-empty string without a newline")
        encoded = line.encode("utf-8")
        places.append(SegmentLine(segment_id, len(data), len(encoded)))
        data += encoded + b"\n"
    return bytes(data), tuple(places)


class SegmentStore(Plugin, ABC):
    """Keeps segments, each an immutable byte string in one partition.

    Values: the state of a store is a finite map from ``(partition, segment_id)`` to bytes. Both keys are non-empty
    strings.

    Operations:

    * ``put(partition, segment_id, data)`` stores ``data``. The segment is durable when the method returns. The data
      store gives each segment a new ID, so it never puts one ID twice.
    * ``read(partition, lines)`` returns the bytes of each ``SegmentLine`` of ``lines``, in order, without the
      newline. A missing segment or a line past its end is a ``SegmentNotFound``.
    * ``delete(partition, segment_id)`` removes the segment. A missing segment is not an error.

    Purpose: a deployment names a subclass in ``storage.segment_backend``. The service builds it at startup, gives it
    to the data store, and closes it at shutdown.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping[str, Any], *, directory: Path) -> Self:
        """Build the store that a deployment names.

        ``config`` is the configuration of the plugin. ``directory`` is ``<storage.agent_record_dir>/segments``. A
        local store keeps the segments there. A remote store can use it as a local cache.
        """

    @abstractmethod
    def put(self, partition: str, segment_id: str, data: bytes) -> None: ...

    @abstractmethod
    def read(self, partition: str, lines: Sequence[SegmentLine]) -> tuple[bytes, ...]: ...

    @abstractmethod
    def delete(self, partition: str, segment_id: str) -> None: ...

    def close(self) -> None:
        """Release the resources of the store. The default holds none."""


@dataclass(frozen=True)
class SegmentPartition:
    """One partition of a segment store.

    Values: ``store`` is an open ``SegmentStore``. ``partition`` is a non-empty string.

    Purpose: a data store makes one per owner, and reads and writes that owner's segments through it.
    """

    store: SegmentStore
    partition: str

    def __post_init__(self) -> None:
        if not isinstance(self.partition, str) or not self.partition:
            raise ValueError("a segment partition is a non-empty string")

    def read_text(self, lines: Sequence[SegmentLine]) -> tuple[str, ...]:
        """The text of each line of ``lines``, in order."""
        return tuple(data.decode("utf-8") for data in self.store.read(self.partition, lines))


class LocalSegmentStore(SegmentStore):
    """Keeps the segments in a local directory, as ``<directory>/<sha256 of partition>/<segment_id>.jsonl``.

    The store writes a file under a temporary name, syncs it, and renames it into place. Only the processes of one
    host can share it.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, directory: Path) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls(directory)

    def path(self, partition: str, segment_id: str) -> Path:
        """The file of the segment ``segment_id`` of ``partition``."""
        if not partition or not segment_id or "/" in segment_id or segment_id.startswith("."):
            raise ValueError("a segment's partition and ID are non-empty, and the ID is a plain file name")
        return self.directory / hashlib.sha256(partition.encode("utf-8")).hexdigest() / f"{segment_id}.jsonl"

    def put(self, partition: str, segment_id: str, data: bytes) -> None:
        target = self.path(partition, segment_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{segment_id}.partial")
        with staging.open("wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        staging.replace(target)

    def read(self, partition: str, lines: Sequence[SegmentLine]) -> tuple[bytes, ...]:
        found: list[bytes] = []
        for line in lines:
            try:
                with self.path(partition, line.segment_id).open("rb") as file:
                    file.seek(line.offset)
                    data = file.read(line.length)
            except FileNotFoundError:
                raise SegmentNotFound(f"no segment {line.segment_id!r} in partition {partition!r}") from None
            if len(data) != line.length:
                raise SegmentNotFound(f"segment {line.segment_id!r} ends before byte {line.offset + line.length}")
            found.append(data)
        return tuple(found)

    def delete(self, partition: str, segment_id: str) -> None:
        self.path(partition, segment_id).unlink(missing_ok=True)


__all__ = [
    "LocalSegmentStore",
    "SegmentLine",
    "SegmentNotFound",
    "SegmentPartition",
    "SegmentStore",
    "segment_lines",
]
