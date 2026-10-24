"""The harness source tree and the patches that change it.

A harness is one directory of text files. ``HarnessTree`` is that directory as
a value: a release stores one, the harness backend reads the head tree from
git, and ``render_tree`` turns one into the files an adapter's binary reads.
``TreePatch`` is a change to a tree: a proposer returns one, the proposal
inbox stores one, and ``HarnessTree.apply`` applies it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType


def normalized_path(path: str) -> str:
    """``path`` as a tree key, or ValueError when it cannot be one.

    A tree key is a relative POSIX path of one or more segments. No segment
    is empty, ``.`` or ``..``, and the path holds no backslash or NUL. So
    ``skills/review/SKILL.md`` is a key, and ``/etc/passwd``, ``a//b``,
    ``./a`` and ``a/../b`` are not.
    """
    if not isinstance(path, str) or not path:
        raise ValueError(f"tree path must be a non-empty string, got {path!r}")
    if "\\" in path or "\x00" in path:
        raise ValueError(f"tree path {path!r} must be a POSIX path without NUL")
    segments = path.split("/")
    if path.startswith("/") or any(segment in ("", ".", "..") for segment in segments):
        raise ValueError(f"tree path {path!r} must be relative, without empty, '.' or '..' segments")
    return path


def require_text(path: str, text: object) -> str:
    """``text`` when it is a ``str`` that encodes as UTF-8; ValueError naming ``path`` otherwise."""
    if not isinstance(text, str):
        raise ValueError(f"tree file {path!r} must hold text, got {type(text).__name__}")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"tree file {path!r} must be UTF-8 encodable text: {exc}") from exc
    return text


@dataclass(frozen=True)
class TreePatch:
    """A change to a ``HarnessTree``.

    Values: ``changes`` maps tree keys (see ``normalized_path``) to the new
    text of that file, or to ``None``, which removes the file. A key that
    ``changes`` does not name keeps its text. The empty patch changes nothing.
    The mapping is read-only, and its order has no meaning.
    Purpose: a proposer returns a patch against the head tree, the proposal
    inbox stores one from a client, and ``HarnessTree.apply`` applies it.
    """

    changes: Mapping[str, str | None]

    def __post_init__(self) -> None:
        if not isinstance(self.changes, Mapping):
            raise TypeError(f"TreePatch.changes must be a mapping, got {type(self.changes).__name__}")
        checked = {
            normalized_path(path): None if text is None else require_text(path, text)
            for path, text in self.changes.items()
        }
        object.__setattr__(self, "changes", MappingProxyType(dict(sorted(checked.items()))))

    @classmethod
    def from_json(cls, value: object) -> TreePatch:
        """The patch a JSON object encodes: path to text, or path to ``null`` for a removal."""
        if not isinstance(value, Mapping):
            raise ValueError(f"a tree patch must be a JSON object of path to text or null, got {type(value).__name__}")
        return cls(dict(value))

    def to_json(self) -> dict[str, str | None]:
        """The JSON object ``from_json`` reads back as this patch."""
        return dict(self.changes)


@dataclass(frozen=True)
class HarnessTree:
    """A harness source tree: file paths and the text each file holds.

    Values: ``files`` maps tree keys (see ``normalized_path``) to text that
    encodes as UTF-8. No key is a directory of another key, so ``a`` and
    ``a/b`` never occur together. The mapping is read-only, and its order has
    no meaning. The empty tree is a value.
    Purpose: the one representation of a harness. A release stores one; the
    harness backend reads the head tree from git and evaluates
    ``head.apply(patch)``; ``render_tree`` writes a tree out for an adapter.
    """

    files: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.files, Mapping):
            raise TypeError(f"HarnessTree.files must be a mapping, got {type(self.files).__name__}")
        checked = {normalized_path(path): require_text(path, text) for path, text in self.files.items()}
        for path in checked:
            for parent in PurePosixPath(path).parents:
                if str(parent) in checked:
                    raise ValueError(f"tree path {str(parent)!r} is a file and also the directory of {path!r}")
        object.__setattr__(self, "files", MappingProxyType(dict(sorted(checked.items()))))

    @classmethod
    def read(cls, directory: Path) -> HarnessTree:
        """Every regular file under ``directory`` as a tree.

        Raises ValueError for a symbolic link (it could point outside the
        tree) and for a file that is not UTF-8 text.
        """
        root = Path(directory)
        if not root.is_dir():
            raise ValueError(f"harness tree directory {root} does not exist")
        files: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ValueError(f"harness tree {root} holds a symbolic link at {relative!r}")
            if not path.is_file():
                continue
            try:
                files[relative] = path.read_bytes().decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"harness tree file {relative!r} is not UTF-8 text: {exc}") from exc
        return cls(files)

    def write(self, directory: Path) -> None:
        """Write every file of the tree under ``directory``, which must be absent or empty."""
        root = Path(directory)
        if root.exists() and any(root.iterdir()):
            raise ValueError(f"cannot write a harness tree into {root}: the directory is not empty")
        for relative, text in self.files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode("utf-8"))

    def apply(self, patch: TreePatch) -> HarnessTree:
        """The tree ``patch`` makes from this one; this tree does not change.

        The result holds every file of this tree that ``patch`` does not name,
        and every file that ``patch`` maps to text. Removing an absent path
        changes nothing.
        """
        kept = {path: text for path, text in self.files.items() if path not in patch.changes}
        written = {path: text for path, text in patch.changes.items() if text is not None}
        return HarnessTree({**kept, **written})

    def diff(self, other: HarnessTree) -> TreePatch:
        """The smallest patch ``p`` with ``self.apply(p) == other``."""
        removed: dict[str, str | None] = {path: None for path in self.files if path not in other.files}
        written: dict[str, str | None] = {
            path: text for path, text in other.files.items() if self.files.get(path) != text
        }
        return TreePatch({**removed, **written})

    def checksum(self) -> str:
        """A SHA-256 hex digest of the paths and texts, length-prefixed so no two trees encode alike.

        Equal trees have equal checksums; a commit record keeps it so recovery can tell whether the release it
        reads is the tree the step committed.
        """
        digest = hashlib.sha256()
        for path, text in self.files.items():
            for part in (path, text):
                encoded = part.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
        return digest.hexdigest()


__all__ = ["HarnessTree", "TreePatch", "normalized_path", "require_text"]
