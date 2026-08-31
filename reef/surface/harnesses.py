"""Serve an agent harness file tree by client pull."""

from __future__ import annotations

from reef.surface.base import Surface
from reef.surface.files import TextFileTree


def create_harness_surface() -> Surface:
    """Build a surface exposing every text file in a harness tree.

    The tree is adapter-specific: its paths are whatever the adapter
    descriptor's render engine produced (config files, rules, agent
    commands, code extensions). The surface does not validate or
    transform; the client pulls the raw tree and applies it.
    """
    return Surface(files=TextFileTree())


__all__ = ["create_harness_surface"]
