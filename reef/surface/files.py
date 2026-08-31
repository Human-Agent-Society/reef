"""File trees shared by harness and skill surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from reef.artifact.artifact import Artifact

REPOSITORY_FILES = frozenset({"reef-artifact.json", ".gitattributes"})


class TextFileTree:
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
            if relative in REPOSITORY_FILES:
                continue
            try:
                # Bytes-then-decode avoids universal-newline rewriting. The
                # pulled tree must equal the tree measured by the gate.
                files[relative] = path.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                continue
        return files or None


__all__ = ["REPOSITORY_FILES", "TextFileTree"]
