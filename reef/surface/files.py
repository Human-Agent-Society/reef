"""The file tree that serves an artifact's text files as stored."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from reef.artifact.artifact import Artifact
from reef.surface.base import FileTree

REPOSITORY_FILES = frozenset({"reef-artifact.json", ".gitattributes"})
#: The file of every harness artifact that holds its composition entries, at the tree's root. A client's pull
#: leaves it out: it describes the tree, and is no part of the harness.
ENTRIES_FILE = "reef-entries.json"


class TextFileTree(FileTree):
    """Read every UTF-8 text file in an artifact, byte-faithfully."""

    def read_files(self, artifact: Artifact) -> Mapping[str, str] | None:
        local_path = artifact.materialize().local_path
        if local_path is None:
            return None
        files: dict[str, str] = {}
        for path in sorted(Path(local_path).rglob("*")):
            if not path.is_file():
                continue
            relative = str(path.relative_to(local_path))
            if relative in REPOSITORY_FILES or relative == ENTRIES_FILE:
                continue
            try:
                # Bytes-then-decode avoids universal-newline rewriting. The
                # pulled tree must equal the tree the validator checked.
                files[relative] = path.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                continue
        return files or None


__all__ = ["ENTRIES_FILE", "REPOSITORY_FILES", "TextFileTree"]
