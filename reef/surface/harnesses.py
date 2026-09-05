"""Serve an agent harness file tree by client pull."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.surface.base import Surface
from reef.surface.files import TextFileTree


@dataclass(frozen=True)
class HarnessSurface(Surface):
    """The harness tree plus what the harness routes need that no artifact carries.

    ``seed_entries`` are the recipe's seed, the composition behind the base
    release no step published; ``served_model`` is the model the recipe
    serves and gates against. The install route binds an installed tree
    with them when a release has no training record of its own.
    """

    seed_entries: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    served_model: str | None = None


def create_harness_surface(
    seed_entries: tuple[Mapping[str, Any], ...] = (), served_model: str | None = None
) -> HarnessSurface:
    """Build a surface exposing every text file in a harness tree.

    The tree is adapter-specific: its paths are whatever the adapter
    descriptor's render engine produced (config files, rules, agent
    commands, code extensions). The surface does not validate or
    transform; the client pulls the raw tree and applies it.
    """
    return HarnessSurface(files=TextFileTree(), seed_entries=tuple(seed_entries), served_model=served_model)


__all__ = ["HarnessSurface", "create_harness_surface"]
