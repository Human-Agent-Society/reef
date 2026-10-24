"""An editable copy of a harness tree for an agent session.

An agent proposer gets the head tree as plain files in a directory, edits them
with its own tools, and stops. ``TreeWorkspace.read`` then turns the
directory back into a patch against the tree it started from, and lists the
files the agent wrote off the layout, so the proposer can report them instead
of losing them silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reef.harness.tree.layout import locate
from reef.harness.tree.tree import HarnessTree, TreePatch


@dataclass(frozen=True)
class WorkspaceResult:
    """What an agent left in a workspace, relative to the tree the workspace started from.

    Values: ``patch`` changes exactly the layout paths whose text differs from
    the base tree, and removes the base files the agent deleted. ``outside``
    is the sorted tuple, without duplicates, of the other paths the directory
    holds: files off the layout, symbolic links, and files that are not
    UTF-8 text. ``patch`` never names a path in ``outside``.
    Purpose: ``TreeWorkspace.read`` makes it; the agent proposer returns
    ``patch`` and tells the agent about ``outside``.
    """

    patch: TreePatch
    outside: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.outside))) != self.outside:
            raise ValueError("WorkspaceResult.outside must be sorted and without duplicates")
        if set(self.outside) & set(self.patch.changes):
            raise ValueError("WorkspaceResult.patch must not name a path outside the layout")


@dataclass(frozen=True)
class TreeWorkspace:
    """A directory holding one editable copy of ``base``.

    Values: ``root`` is a directory that ``create`` filled with ``base``'s
    files; the agent may change it in any way afterwards.
    Purpose: an agent proposer creates one per session, runs the agent in
    ``root``, and reads the result back with ``read``.
    """

    root: Path
    base: HarnessTree

    @classmethod
    def create(cls, base: HarnessTree, root: Path) -> TreeWorkspace:
        """Write ``base`` into ``root``, which must be absent or empty."""
        base.write(root)
        return cls(Path(root), base)

    def read(self) -> WorkspaceResult:
        """The patch from ``base`` to the layout files now in ``root``, and the paths left outside the layout."""
        files: dict[str, str] = {}
        outside: list[str] = []
        for path in sorted(self.root.rglob("*")):
            relative = path.relative_to(self.root).as_posix()
            if path.is_symlink():
                outside.append(relative)
                continue
            if not path.is_file():
                continue
            if locate(relative) is None:
                outside.append(relative)
                continue
            try:
                files[relative] = path.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                outside.append(relative)
        return WorkspaceResult(self.base.diff(HarnessTree(files)), tuple(outside))


__all__ = ["TreeWorkspace", "WorkspaceResult"]
