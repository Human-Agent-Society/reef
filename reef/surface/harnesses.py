"""Serve an agent harness file tree by client pull."""

from __future__ import annotations

from collections.abc import Mapping

from reef.core.model_metadata import ModelMetadata

from reef.surface.base import ComponentSurface, HarnessInfo, Surface
from reef.surface.files import TextFileTree

#: The component name a harness surface binds.
HARNESS_COMPONENT = "harness"


def create_harness_surface(
    served_model: str | None = None,
    client_models: tuple[str, ...] = (),
    served_api: str = "openai",
    model_metadata: Mapping[str, ModelMetadata] | None = None,
    served_metadata: ModelMetadata | None = None,
) -> Surface:
    """Build a surface exposing every text file in a harness tree as the ``harness`` component.

    The tree is adapter-specific: its paths are whatever the adapter
    descriptor's render engine produced (config files, rules, agent
    commands, code extensions). The surface does not validate or
    transform; the client pulls the raw tree and applies it. ``harness``
    carries the model the recipe serves and its API dialect, which the install
    route binds an installed tree with when a release has no gate of its own,
    and the further models an installed client may switch to. The entries of
    the base harness travel with the base artifact, not with the surface.
    """
    return Surface(
        components={HARNESS_COMPONENT: ComponentSurface(files=TextFileTree(), kind="harness_plugin")},
        harness=HarnessInfo(
            served_model=served_model,
            served_api=served_api,
            model_metadata=dict(model_metadata or {}),
            served_metadata=served_metadata,
            client_models=tuple(client_models),
        ),
    )


__all__ = ["HARNESS_COMPONENT", "create_harness_surface"]
