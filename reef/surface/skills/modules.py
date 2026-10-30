"""Composable layers in a skill artifact."""

from __future__ import annotations

from collections.abc import Mapping


class SkillLayer:
    """One top-level directory and its artifact validator."""

    layer: str

    def validate(self, files: Mapping[str, str]) -> None:
        """Validate files keyed relative to this layer. Default: accept."""


__all__ = ["SkillLayer"]
