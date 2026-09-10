"""Serve an agent harness file tree by client pull."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.surface.base import HarnessInfo, Surface
from reef.surface.files import TextFileTree


def create_harness_surface(
    seed_entries: tuple[Mapping[str, Any], ...] = (),
    served_model: str | None = None,
    client_models: tuple[str, ...] = (),
) -> Surface:
    """Build a surface exposing every text file in a harness tree.

    The tree is adapter-specific: its paths are whatever the adapter
    descriptor's render engine produced (config files, rules, agent
    commands, code extensions). The surface does not validate or
    transform; the client pulls the raw tree and applies it. ``harness``
    carries the recipe's seed, the composition behind the base release no
    step published, the model the recipe serves, which the install
    route binds an installed tree with when a release has no gate of its own,
    and the further models an installed client may switch to.
    """
    return Surface(
        files=TextFileTree(),
        harness=HarnessInfo(
            seed_entries=tuple(seed_entries), served_model=served_model, client_models=tuple(client_models)
        ),
    )


__all__ = ["create_harness_surface"]
