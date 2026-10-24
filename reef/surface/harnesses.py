"""Serve a harness release to a client: the release's source tree, rendered for the scenario's adapter."""

from __future__ import annotations

from collections.abc import Mapping

from reef.artifact.artifact import Artifact
from reef.core.model_metadata import ModelMetadata
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.tree import HarnessTree, render_tree
from reef.surface.base import ComponentSurface, FileTree, HarnessInfo, Surface
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


def create_harness_surface(
    descriptor: AdapterDescriptor,
    *,
    served_model: str | None = None,
    client_models: tuple[str, ...] = (),
    served_api: str = "openai",
    model_metadata: Mapping[str, ModelMetadata] | None = None,
    served_metadata: ModelMetadata | None = None,
) -> Surface:
    """A surface that serves each release of the ``harness`` component as ``descriptor``'s rendered files.

    ``harness`` carries the model the recipe serves, its API dialect, its
    metadata, and the further models a client may switch to, for the routes
    that tell a client how to reach Reef.
    """
    return Surface(
        components={HARNESS_COMPONENT: ComponentSurface(files=HarnessFiles(descriptor))},
        harness=HarnessInfo(
            served_model=served_model,
            served_api=served_api,
            model_metadata=dict(model_metadata or {}),
            served_metadata=served_metadata,
            client_models=tuple(client_models),
        ),
    )


__all__ = ["HARNESS_COMPONENT", "HarnessFiles", "create_harness_surface"]
