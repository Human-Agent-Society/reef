"""The files of an artifact directory, read where they are stored (RFC #995).

An import checks a directory on the service's file system, or a Hugging Face revision that Reef keeps by reference,
with one layout predicate. ``ArtifactFiles`` is what that predicate reads.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath


def path_order(path: str) -> list[str]:
    """The sort key of a relative POSIX path: its components, the order of ``pathlib`` paths."""
    return path.split("/")


class ArtifactFiles(ABC):
    """The files of one artifact directory ``D``: a finite map from relative POSIX paths to bytes.

    Values: ``paths()`` is the set of ``D``'s paths without duplicates, possibly empty, sorted by ``path_order``.
    For each path ``p`` in it, ``sha256(p)`` is the lowercase hex SHA-256 of ``D[p]`` and ``read_bytes(p)`` is
    ``D[p]``. Both raise ``FileNotFoundError`` for a path that is not in ``paths()``. ``content_id()`` is the
    SHA-256 of the pairs ``(p, sha256(p))`` in that order, so equal files at equal paths give equal IDs wherever
    they are stored.

    Purpose: an artifact import makes one for what it imports; the component's ``ArtifactLayout`` reads it, and
    the import names its content with ``content_id``.
    """

    @abstractmethod
    def paths(self) -> tuple[str, ...]: ...

    @abstractmethod
    def sha256(self, path: str) -> str: ...

    @abstractmethod
    def read_bytes(self, path: str) -> bytes: ...

    def content_id(self) -> str:
        files = [(path, self.sha256(path)) for path in self.paths()]
        return f"content:{hashlib.sha256(json.dumps(files).encode()).hexdigest()}"


class DirectoryFiles(ArtifactFiles):
    """The regular files under ``directory``, at their paths relative to it."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def paths(self) -> tuple[str, ...]:
        found = (path.relative_to(self.directory).as_posix() for path in self.directory.rglob("*") if path.is_file())
        return tuple(sorted(found, key=path_order))

    def sha256(self, path: str) -> str:
        with self.file(path).open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()

    def read_bytes(self, path: str) -> bytes:
        return self.file(path).read_bytes()

    def file(self, path: str) -> Path:
        """The file at ``path``; ``FileNotFoundError`` unless ``path`` is one of ``paths()``."""
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts or not (self.directory / relative).is_file():
            raise FileNotFoundError(f"{self.directory} holds no file {path}")
        return self.directory / relative


def directory_content_id(directory: Path) -> str:
    """The content ID of the files under ``directory`` (``ArtifactFiles.content_id``)."""
    return DirectoryFiles(directory).content_id()
