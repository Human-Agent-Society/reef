"""The file tree that serves an artifact's text files as stored."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.artifact.files import ArtifactFiles
from reef.surface.base import FileTree

REPOSITORY_FILES = frozenset({"reef-artifact.json", ".gitattributes"})
#: The file of every harness artifact that holds its composition entries, at the tree's root. A client's pull
#: leaves it out: it describes the tree, and is no part of the harness.
ENTRIES_FILE = "reef-entries.json"


def entries_file(entries: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """``entries`` as the ``ENTRIES_FILE`` of a harness artifact.

    The keys keep their order, so ``read_entries`` gives back entries that render to the same bytes: a native
    module's header writes its parameters in their order.
    """
    return {ENTRIES_FILE: json.dumps([dict(entry) for entry in entries], indent=2) + "\n"}


def read_entries(files: ArtifactFiles) -> tuple[dict[str, Any], ...]:
    """The entries that the harness tree ``files`` holds in its ``ENTRIES_FILE``.

    Raise ``ValueError`` when the tree has no such file, or the file is not a JSON list of objects.
    """
    try:
        listed = json.loads(files.read_bytes(ENTRIES_FILE))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"the harness tree has no readable {ENTRIES_FILE}: {exc}") from exc
    if not isinstance(listed, list) or not all(isinstance(entry, Mapping) for entry in listed):
        raise ValueError(f"{ENTRIES_FILE} is not a list of objects")
    return tuple(dict(entry) for entry in listed)


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


__all__ = ["ENTRIES_FILE", "REPOSITORY_FILES", "TextFileTree", "entries_file", "read_entries"]
