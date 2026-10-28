"""Move a harness between a person's machine and Reef, with one adapter mapping in both directions.

An adapter's ``tree.yaml`` says where each file of a tree lands when the tree
renders: ``rules.md`` at ``pi-agent/AGENTS.md``, under the render root
``pi-agent``. The descriptor's ``roots`` say where each render root lives on a
person's machine (``~/.pi/agent``). The client copies files and knows no
adapter; this module does the mapping on the server.

Reef to client
    ``HarnessFiles`` serves a release as the adapter's binary reads it: its
    source tree rendered without a model binding. ``client_files`` renders it
    with a binding at Reef's scenario route instead, so the harness that a
    client installs sends its model calls to Reef.
Client to Reef
    ``import_files`` reads the files at their render paths back into a tree:
    each file that a part renders becomes the part's tree file. Config files
    are not read back. They hold the endpoints and credentials that the model
    binding owns, so the import keeps the config parts of the release the
    client installed. ``merge_trees`` then puts the client's changes on the
    head, which may have moved on since that release.

For a tree whose files are each in the form their render writes,
``import_files(render_tree(tree))`` is the tree again.

The merge is per file, with ``b``, ``h`` and ``c`` a path's text in the base,
the head and the client's tree, absent counting as a value:

==========  ==========  ===========================
``c == b``  ``h == b``  result
==========  ==========  ===========================
yes         any         ``h``: the client kept it
no          yes         ``c``: only the client changed it
no          no          ``c`` when ``c == h``, else a conflict
==========  ==========  ===========================
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from reef.artifact.artifact import Artifact
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.adapters.parts import Format
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.tree import HarnessTree, render_tree
from reef.surface.base import ComponentSurface, FileTree, Surface
from reef.surface.files import TextFileTree

#: The component name a harness surface binds.
HARNESS_COMPONENT = "harness"
#: The text a client puts its Reef token in place of, in the files ``client_files`` renders.
TOKEN_PLACEHOLDER = "REEF_TOKEN_PLACEHOLDER"


class HarnessFiles(FileTree):
    """A release's files as the adapter's binary reads them: its source tree rendered without a model binding.

    A release stores the source tree; the client gets the adapter's files,
    and binds its own model endpoint.
    """

    def __init__(self, descriptor: AdapterDescriptor) -> None:
        self.descriptor = descriptor
        self.source = TextFileTree()

    def read_files(self, artifact: Artifact) -> Mapping[str, str] | None:
        files = self.source.read_files(artifact)
        if files is None:
            return None
        return render_tree(HarnessTree(files), self.descriptor)


def create_harness_surface(descriptor: AdapterDescriptor) -> Surface:
    """A surface that serves each release of the ``harness`` component as ``descriptor``'s rendered files."""
    return Surface(components={HARNESS_COMPONENT: ComponentSurface(files=HarnessFiles(descriptor))})


def client_files(
    tree: HarnessTree, descriptor: AdapterDescriptor, served: ModelBinding, scenario_url: str, root: str
) -> dict[str, str]:
    """``tree`` rendered for a client that installs it at ``root``: its model calls go to ``scenario_url``, Reef's
    route of one scenario.

    The binding asks for ``served``'s model in ``served``'s dialect, with ``TOKEN_PLACEHOLDER`` as its key: the
    client writes its own token there, so Reef never sends a credential back.
    """
    binding = replace(served, base_url=scenario_url, api_key=TOKEN_PLACEHOLDER)
    return descriptor.place_root(render_tree(tree, descriptor, binding.config_overlays(descriptor)), root)


@dataclass(frozen=True)
class ImportedTree:
    """A person's harness files read back into a tree.

    Values: ``tree`` holds the config files of the release the client
    installed, and one tree file for each imported file that a part renders to.
    ``ignored`` holds the imported paths that no part renders to, sorted.
    Purpose: ``import_files`` makes it; the import route publishes ``tree``
    and tells the client what it ``ignored``.
    """

    tree: HarnessTree
    ignored: tuple[str, ...]


def import_files(files: Mapping[str, str], base: HarnessTree, descriptor: AdapterDescriptor) -> ImportedTree:
    """``files``, by render path, as ``descriptor``'s tree: ``base``'s config files and a tree file per part file.

    ``base`` is the release the client installed. A file that no part renders to is ignored. The imported files
    replace every other file of ``base``, so a file the person removed is removed from the tree too.
    """
    kept = {}
    for path, text in base.files.items():
        location = descriptor.layout.locate(path)
        if location is not None and location.part.format is Format.CONFIG:
            kept[path] = text
    imported = {}
    ignored = []
    for path, text in files.items():
        location = descriptor.layout.locate_render(path)
        if location is None:
            ignored.append(path)
        else:
            imported[location.part.tree_path(location.name)] = text
    return ImportedTree(HarnessTree({**kept, **imported}), tuple(sorted(ignored)))


@dataclass(frozen=True)
class TreeMerge:
    """A client's tree merged onto the head, by the table of this module.

    Values: ``tree`` is the merged tree when ``conflicts`` is empty, else
    ``None``. ``conflicts`` holds the tree paths that the head and the client
    both changed to different texts, sorted.
    Purpose: ``merge_trees`` makes it; the import route publishes ``tree``,
    or refuses with the ``conflicts``.
    """

    tree: HarnessTree | None
    conflicts: tuple[str, ...]

    def __post_init__(self) -> None:
        if (self.tree is None) == (not self.conflicts):
            raise ValueError("a merge has a tree exactly when it has no conflict")


def merge_trees(base: HarnessTree, head: HarnessTree, client: HarnessTree) -> TreeMerge:
    """``client``'s changes since ``base`` put on ``head``, file by file, by the table of this module."""
    merged: dict[str, str] = {}
    conflicts = []
    for path in sorted(base.files.keys() | head.files.keys() | client.files.keys()):
        before, theirs, mine = base.files.get(path), head.files.get(path), client.files.get(path)
        if mine in (before, theirs):
            text = theirs
        elif theirs == before:
            text = mine
        else:
            conflicts.append(path)
            continue
        if text is not None:
            merged[path] = text
    if conflicts:
        return TreeMerge(None, tuple(conflicts))
    return TreeMerge(HarnessTree(merged), ())


__all__ = [
    "HARNESS_COMPONENT",
    "TOKEN_PLACEHOLDER",
    "HarnessFiles",
    "ImportedTree",
    "TreeMerge",
    "client_files",
    "create_harness_surface",
    "import_files",
    "merge_trees",
]
