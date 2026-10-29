"""Move a harness between a person's machine and Reef, with one adapter mapping in both directions.

The ``evolve`` section of an adapter's ``adapter.yaml`` says where each file
of a tree lands when the tree renders, such as ``rules.md`` at
``claude/CLAUDE.md``; the kind of each file says how it merges. Its
``run.layer`` says how a person's machine runs the agent with a release as a
layer over the person's own configuration (``AdapterLayer``). The client copies files and runs one command; it knows no
adapter, and this module does the mapping on the server.

Reef to client
    ``HarnessFiles`` serves a release as the adapter's binary reads it: its
    source tree rendered without a model binding. ``layer_install`` renders
    it as a layer for one client install instead: the files, bound to Reef's
    route of the scenario, and the command that runs the agent on them.
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
the head and the client's tree, absent counting as a value. Where both sides
changed a file, its part's ``merge`` decides: ``file`` refuses, and ``lines``
merges the lines each side changed, refusing only lines both changed
(``merge_lines``).

==========  ==========  =================================================
``c == b``  ``h == b``  result
==========  ==========  =================================================
yes         any         ``h``: the client kept it
no          yes         ``c``: only the client changed it
no          no          ``c`` when ``c == h``, else the part's ``merge``
==========  ==========  =================================================
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from reef.artifact.artifact import Artifact
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.adapters.parts import Format, TreeLayout
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.tree import HarnessTree, render_tree
from reef.surface.base import ComponentSurface, FileTree, Surface
from reef.surface.files import TextFileTree

#: The component name a harness surface binds.
HARNESS_COMPONENT = "harness"
#: The text a client puts its Reef token in place of, in the files and variables ``layer_install`` makes.
TOKEN_PLACEHOLDER = "REEF_TOKEN_PLACEHOLDER"
FILE_TEMPLATE = re.compile(r"\{file:([^}]+)\}")


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


@dataclass(frozen=True)
class LayerInstall:
    """A release as a layer for one client install, and the command that runs the agent on it.

    Values: ``files`` maps render paths under the install to text, with
    ``TOKEN_PLACEHOLDER`` where the client's token goes. ``args`` are added to
    the adapter's binary, and ``env`` is set for it; ``env`` may hold the
    placeholder, ``args`` never do. ``synced`` holds the render templates of
    the parts a client sends back, ``{name}`` standing for one path segment.
    Purpose: ``layer_install`` makes it; the client-form manifest serves it,
    and the client installs ``files`` and runs the command.
    """

    files: Mapping[str, str]
    args: tuple[str, ...]
    env: Mapping[str, str]
    synced: tuple[str, ...]


def layer_install(
    tree: HarnessTree, descriptor: AdapterDescriptor, served: ModelBinding, scenario_url: str, root: str
) -> LayerInstall:
    """``tree`` as ``descriptor``'s layer for a client that installs it at ``root``.

    The binding asks for ``served``'s model at ``scenario_url``, Reef's route of one scenario, with
    ``TOKEN_PLACEHOLDER`` as its key: the client writes its own token there, so Reef never sends a credential back.
    The layer's templates are filled in as ``AdapterLayer`` defines. ``ValueError`` when the adapter has no layer.
    """
    layer = descriptor.layer
    if layer is None:
        raise ValueError(f"adapter {descriptor.name!r} does not run on a person's machine: it declares no layer")
    binding = replace(served, base_url=scenario_url, api_key=TOKEN_PLACEHOLDER)
    rendered = render_tree(tree, descriptor, binding.config_overlays(descriptor), layer=True)
    files = descriptor.place_root(rendered, root)
    values = {"{root}": root, "{model}": binding.model, "{base_url}": scenario_url, "{token}": TOKEN_PLACEHOLDER}

    def filled(template: str) -> str:
        for placeholder, value in values.items():
            template = template.replace(placeholder, value)
        return template

    args: list[str] = []
    for group in layer.args:
        named = [match.group(1) for arg in group for match in FILE_TEMPLATE.finditer(arg)]
        if any(not files.get(path, "").strip() for path in named):
            continue
        args.extend(FILE_TEMPLATE.sub(lambda match: json.dumps(files[match.group(1)]), filled(arg)) for arg in group)
    synced = tuple(part.renders for part in descriptor.layout.parts.values() if part.renders is not None)
    return LayerInstall(files, tuple(args), {name: filled(value) for name, value in layer.env.items()}, synced)


@dataclass(frozen=True)
class ImportedTree:
    """A person's harness files read back into a tree.

    Values: ``tree`` holds the config files of the release the client
    installed, and one tree file for each imported file that a part renders
    to and that holds more than whitespace. ``ignored`` holds the imported
    paths that no part renders to, sorted.
    Purpose: ``import_files`` makes it; the import route publishes ``tree``
    and tells the client what it ``ignored``.
    """

    tree: HarnessTree
    ignored: tuple[str, ...]


def import_files(files: Mapping[str, str], base: HarnessTree, descriptor: AdapterDescriptor) -> ImportedTree:
    """``files``, by render path, as ``descriptor``'s tree: ``base``'s config files and a tree file per part file.

    ``base`` is the release the client installed. A file that no part renders to is ignored, and an empty one is
    no file: a layer writes an empty rules file for a tree that has none. The imported files replace every other
    file of ``base``, so a file the person removed is removed from the tree too.
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
        elif text.strip():
            imported[location.part.tree_path(location.name)] = text
    return ImportedTree(HarnessTree({**kept, **imported}), tuple(sorted(ignored)))


@dataclass(frozen=True)
class TreeMerge:
    """A client's tree merged onto the head, by the table of this module.

    Values: ``tree`` is the merged tree when ``conflicts`` is empty, else
    ``None``. ``conflicts`` holds the tree paths that the head and the client
    both changed in a way their part's ``merge`` refuses, sorted.
    Purpose: ``merge_trees`` makes it; the import route publishes ``tree``,
    or refuses with the ``conflicts``.
    """

    tree: HarnessTree | None
    conflicts: tuple[str, ...]

    def __post_init__(self) -> None:
        if (self.tree is None) == (not self.conflicts):
            raise ValueError("a merge has a tree exactly when it has no conflict")


def merge_trees(base: HarnessTree, head: HarnessTree, client: HarnessTree, layout: TreeLayout) -> TreeMerge:
    """``client``'s changes since ``base`` put on ``head``, file by file, by the table of this module.

    ``layout`` gives each path its part's ``merge``; a path no part takes merges as ``file``.
    """
    merged: dict[str, str] = {}
    conflicts = []
    for path in sorted(base.files.keys() | head.files.keys() | client.files.keys()):
        before, theirs, mine = base.files.get(path), head.files.get(path), client.files.get(path)
        location = layout.locate(path)
        if mine in (before, theirs):
            text = theirs
        elif theirs == before:
            text = mine
        elif location is not None and location.part.merge == "lines" and None not in (before, theirs, mine):
            text = merge_lines(str(before), str(theirs), str(mine))
            if text is None:
                conflicts.append(path)
                continue
        else:
            conflicts.append(path)
            continue
        if text is not None:
            merged[path] = text
    if conflicts:
        return TreeMerge(None, tuple(conflicts))
    return TreeMerge(HarnessTree(merged), ())


def merge_lines(base: str, head: str, client: str) -> str | None:
    """``client``'s line changes since ``base`` put on ``head``; ``None`` when both changed the same lines.

    Each side's changes are runs of ``base`` lines it replaced, possibly none (an insertion). Runs that share a
    line, or insertions at one place, form a group. A group one side made alone applies, so does one both made
    alike, and two different insertions at one place keep both, the head's first; any other group is a conflict.
    Changes that only touch apply in ``base`` order.
    """
    lines = base.splitlines(keepends=True)
    edits: list[tuple[int, int, int, list[str]]] = []
    for side, text in enumerate((head, client)):
        other = text.splitlines(keepends=True)
        opcodes = SequenceMatcher(None, lines, other, autojunk=False).get_opcodes()
        edits.extend((start, end, side, other[low:high]) for tag, start, end, low, high in opcodes if tag != "equal")
    edits.sort(key=lambda edit: (edit[0], edit[1], edit[2]))
    out: list[str] = []
    cursor = index = 0
    while index < len(edits):
        group = [edits[index]]
        stop = edits[index][1]
        index += 1
        while index < len(edits) and (edits[index][0] < stop or edits[index][:2] == group[0][:2] == (stop, stop)):
            group.append(edits[index])
            stop = max(stop, edits[index][1])
            index += 1
        theirs = [(start, end, text) for start, end, side, text in group if side == 0]
        mine = [(start, end, text) for start, end, side, text in group if side == 1]
        insertions = all(start == end for start, end, _ in theirs + mine)
        if theirs and mine and theirs != mine and not insertions:
            return None
        applied = theirs if theirs == mine or not mine else mine if not theirs else theirs + mine
        for start, end, replacement in applied:
            out.extend(lines[cursor:start])
            out.extend(replacement)
            cursor = end
    out.extend(lines[cursor:])
    return "".join(out)


__all__ = [
    "HARNESS_COMPONENT",
    "TOKEN_PLACEHOLDER",
    "HarnessFiles",
    "ImportedTree",
    "LayerInstall",
    "TreeMerge",
    "create_harness_surface",
    "import_files",
    "layer_install",
    "merge_lines",
    "merge_trees",
]
