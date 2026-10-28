"""Serve a harness release to a client: the release's source tree, rendered for the scenario's adapter.

The surface contracts are ``reef.surface``'s; this module binds them to the
harness tree and its render, so the surface package stays free of harness
code.
"""

from __future__ import annotations

from collections.abc import Mapping

from reef.artifact.artifact import Artifact
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.tree import HarnessTree, render_tree
from reef.surface.base import ComponentSurface, FileTree, Surface
from reef.surface.files import TextFileTree

#: The component name a harness surface binds.
HARNESS_COMPONENT = "harness"


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


__all__ = ["HARNESS_COMPONENT", "HarnessFiles", "create_harness_surface"]
