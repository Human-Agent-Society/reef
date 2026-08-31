"""Serve a layered skill tree by injection and client pull."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.errors import ReefError
from reef.surface.base import Surface
from reef.surface.files import TextFileTree
from reef.surface.skills.modules import RequestSkillLayer, SkillLayer


def validate_tree(files: Mapping[str, str], layers: Sequence[SkillLayer]) -> None:
    """Validate every file and hand each top-level directory to its layer."""
    if not files:
        raise ReefError("skill artifact has no files")
    owners = {item.layer: item for item in layers}
    by_layer: dict[str, dict[str, str]] = {layer: {} for layer in owners}
    for path, text in files.items():
        if not text.strip():
            raise ReefError(f"{path} must not be empty")
        layer, _, name = path.partition("/")
        if not name or layer not in owners:
            raise ReefError(f"{path} is not under a skill layer ({', '.join(sorted(owners))})")
        by_layer[layer][name] = text
    for layer, item in owners.items():
        item.validate(by_layer[layer])


@dataclass(frozen=True)
class SkillValidator:
    """Validate the shape and contents of a layered skill artifact."""

    layers: tuple[SkillLayer, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", _skill_layers(self.layers))

    def validate(self, artifact: Artifact) -> None:
        files = TextFileTree().read_files(artifact)
        validate_tree(files or {}, self.layers)


@dataclass(frozen=True)
class SkillInferenceHooks:
    """Inject request-aware skill layers in declaration order."""

    layers: tuple[RequestSkillLayer, ...]

    def prepare_request(self, artifact: Artifact, path: str, request: dict[str, Any]) -> dict[str, Any]:
        local_path = artifact.materialize().local_path
        if local_path is None:
            return request
        for layer in self.layers:
            request = layer.prepare_request(_layer_files(local_path, layer.layer), path, request)
        return request

    def verify_response(self, artifact: Artifact, path: str, response: Mapping[str, Any]) -> None:
        return None


def create_skill_surface(layers: Sequence[SkillLayer]) -> Surface:
    """Build optional request injection and client-pulled skill files."""
    layer_tuple = _skill_layers(layers)
    request_layers = tuple(item for item in layer_tuple if isinstance(item, RequestSkillLayer))
    return Surface(
        inference=SkillInferenceHooks(request_layers) if request_layers else None,
        files=TextFileTree(),
    )


def _skill_layers(layers: Sequence[SkillLayer]) -> tuple[SkillLayer, ...]:
    if not layers:
        raise ValueError("a skill surface requires at least one layer")
    layer_tuple = tuple(layers)
    names = [item.layer for item in layer_tuple]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate skill layers: {sorted(names)}")
    return layer_tuple


def _layer_files(local_path: Path, layer: str) -> dict[str, str]:
    layer_path = local_path / layer
    if not layer_path.is_dir():
        return {}
    files = {}
    for path in sorted(layer_path.rglob("*")):
        if not path.is_file():
            continue
        try:
            files[str(path.relative_to(layer_path))] = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue
    return files


__all__ = ["SkillInferenceHooks", "SkillValidator", "create_skill_surface", "validate_tree"]
