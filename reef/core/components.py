"""The named components one release binds, stored as a manifest in release metadata.

A release's content is a set of named components, such as ``weights``,
``harness``, or ``config``. A release with one component keeps the flat
layout: its files sit at the artifact root and its ``content_id`` is the
component's. A release with several components keeps each one in a directory
named after it, and its ``content_id`` is derived from the component content
ids, so two releases binding the same combination share one ``content_id``.

The manifest travels in artifact metadata under ``COMPONENTS_METADATA_KEY``.
A release without a manifest, published before components existed, is read as
one component at the root.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

COMPONENTS_METADATA_KEY = "components"

#: The component a scenario's only trainer is bound to when its surface serves no component at all.
RECORDS_COMPONENT = "records"

#: The prefix of a derived multi-component ``content_id``.
COMPOSITE_CONTENT_PREFIX = "composite:"


def validate_component_name(name: object) -> str:
    """A component name doubles as its directory name in a multi-component release."""
    if not isinstance(name, str) or not name:
        raise ValueError("component name must be a non-empty string")
    if name in (".", "..") or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"component name {name!r} must be a plain directory name")
    return name


@dataclass(frozen=True)
class ComponentEntry:
    """One component's selected content and its component-scoped metadata."""

    content_id: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.content_id, str) or not self.content_id:
            raise ValueError("component content_id must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("component metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ReleaseComponents:
    """The manifest of one release: which named components it binds, in declaration order."""

    entries: Mapping[str, ComponentEntry]

    def __post_init__(self) -> None:
        if not isinstance(self.entries, Mapping) or not self.entries:
            raise ValueError("a release binds at least one component")
        validated: dict[str, ComponentEntry] = {}
        for name, entry in self.entries.items():
            if not isinstance(entry, ComponentEntry):
                raise ValueError(f"component {name!r} must be a ComponentEntry")
            validated[validate_component_name(name)] = entry
        object.__setattr__(self, "entries", MappingProxyType(validated))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.entries)

    @property
    def content_ids(self) -> dict[str, str]:
        """The content id of each component, in declaration order."""
        return {name: entry.content_id for name, entry in self.entries.items()}

    @property
    def single(self) -> bool:
        """True when the release keeps the flat, one-component layout."""
        return len(self.entries) == 1

    def relative_path(self, name: str) -> Path:
        """Where ``name``'s files live under the release root."""
        if name not in self.entries:
            raise KeyError(f"release binds no component {name!r}")
        return Path(".") if self.single else Path(name)

    @property
    def content_id(self) -> str:
        """The release content id: the sole component's, or one derived from every component's."""
        if self.single:
            return next(iter(self.entries.values())).content_id
        digest = hashlib.sha256(
            json.dumps(
                sorted((name, entry.content_id) for name, entry in self.entries.items()),
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return f"{COMPOSITE_CONTENT_PREFIX}{digest}"

    def to_dict(self) -> dict[str, Any]:
        return {
            name: {"content_id": entry.content_id, "metadata": dict(entry.metadata)}
            for name, entry in self.entries.items()
        }

    @classmethod
    def from_dict(cls, value: object) -> ReleaseComponents:
        if not isinstance(value, Mapping) or not value:
            raise ValueError("components manifest must be a non-empty object")
        entries: dict[str, ComponentEntry] = {}
        for name, raw in value.items():
            if not isinstance(raw, Mapping):
                raise ValueError(f"components manifest entry {name!r} must be an object")
            metadata = raw.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise ValueError(f"components manifest entry {name!r} metadata must be an object")
            entries[str(name)] = ComponentEntry(content_id=raw.get("content_id", ""), metadata=metadata)
        return cls(entries)


def release_components(metadata: Mapping[str, object] | None) -> ReleaseComponents | None:
    """The manifest carried by release metadata, or ``None`` for a release published without one."""
    if metadata is None:
        return None
    raw = metadata.get(COMPONENTS_METADATA_KEY)
    if raw is None:
        return None
    return ReleaseComponents.from_dict(raw)


__all__ = [
    "COMPONENTS_METADATA_KEY",
    "COMPOSITE_CONTENT_PREFIX",
    "RECORDS_COMPONENT",
    "ComponentEntry",
    "ReleaseComponents",
    "release_components",
    "validate_component_name",
]
