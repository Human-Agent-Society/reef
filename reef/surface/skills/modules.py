"""Composable layers in a skill artifact."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class SkillLayer:
    """One top-level directory and its artifact validator."""

    layer: str

    def validate(self, files: Mapping[str, str]) -> None:
        """Validate files keyed relative to this layer. Default: accept."""


class RequestSkillLayer(SkillLayer, ABC):
    """A skill layer that also injects content into inference requests."""

    layer: str

    @abstractmethod
    def prepare_request(self, files: Mapping[str, str], path: str, request: dict[str, Any]) -> dict[str, Any]: ...


__all__ = ["RequestSkillLayer", "SkillLayer"]
