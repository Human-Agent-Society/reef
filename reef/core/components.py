"""The named components one snapshot binds, stored as a manifest in release metadata.

A snapshot with one component keeps its files at the root. A snapshot with
several components keeps each one in a directory named after it. The mapping
has no content identity of its own.

Each component is one artifact. New components receive independent artifact
IDs before publication. A snapshot records those IDs explicitly, and carrying
a component into another snapshot preserves its ID.

The manifest travels in artifact metadata under ``COMPONENTS_METADATA_KEY``.
A snapshot without a manifest is read as one component at the root, which the
snapshot made.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

COMPONENTS_METADATA_KEY = "components"

#: The component a scenario's only trainer is bound to when the scenario serves no component at all.
RECORDS_COMPONENT = "records"


def validate_component_name(name: object) -> str:
    """A component name doubles as its directory name in a multi-component release."""
    if not isinstance(name, str) or not name:
        raise ValueError("component name must be a non-empty string")
    if name in (".", "..") or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"component name {name!r} must be a plain directory name")
    return name


@dataclass(frozen=True)
class ComponentEntry:
    """One component's artifact in a snapshot manifest: its content, its identity and its metadata.

    Values: ``artifact_id`` is a non-empty string. ``metadata`` is a mapping of that component's metadata, copied
    and frozen at construction.

    Purpose: the committer and the scenario factory make entries when a
    snapshot is published or a scenario is created; readers resolve a
    component's artifact and content from them.
    """

    artifact_id: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not self.artifact_id:
            raise ValueError("component artifact_id must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("component metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class SnapshotComponents:
    """The manifest of one snapshot: which named components it binds, in declaration order.

    Values: ``entries`` is a non-empty mapping from valid component names to
    ``ComponentEntry`` values. No two entries have the same explicit
    ``artifact_id``.
    """

    entries: Mapping[str, ComponentEntry]

    def __post_init__(self) -> None:
        if not isinstance(self.entries, Mapping) or not self.entries:
            raise ValueError("a snapshot binds at least one component")
        validated: dict[str, ComponentEntry] = {}
        for name, entry in self.entries.items():
            if not isinstance(entry, ComponentEntry):
                raise ValueError(f"component {name!r} must be a ComponentEntry")
            validated[validate_component_name(name)] = entry
        explicit = [entry.artifact_id for entry in validated.values() if entry.artifact_id is not None]
        if len(set(explicit)) != len(explicit):
            raise ValueError("one artifact ID cannot name several components of a snapshot")
        object.__setattr__(self, "entries", MappingProxyType(validated))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.entries)

    def artifact_id(self, name: str, revision: str) -> str:
        """The artifact ID of component ``name`` in the snapshot ``revision``.

        Raise ``KeyError`` for an unknown ``name``.
        """
        return self.entries[name].artifact_id

    def artifact_ids(self, revision: str) -> dict[str, str]:
        """The artifact ID of each component in the snapshot ``revision`` (``artifact_id``), in declaration order."""
        return {name: self.artifact_id(name, revision) for name in self.entries}

    def carried_from(self, revision: str) -> SnapshotComponents:
        """This manifest as a later snapshot carries it forward: every entry names its artifact explicitly."""
        identities = self.artifact_ids(revision)
        return SnapshotComponents(
            {name: replace(entry, artifact_id=identities[name]) for name, entry in self.entries.items()}
        )

    @property
    def single(self) -> bool:
        """True when the snapshot keeps the flat, one-component layout."""
        return len(self.entries) == 1

    def relative_path(self, name: str) -> Path:
        """Where ``name``'s files live under the snapshot root."""
        if name not in self.entries:
            raise KeyError(f"release binds no component {name!r}")
        return Path(".") if self.single else Path(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: {
                "metadata": dict(entry.metadata),
                "artifact_id": entry.artifact_id,
            }
            for name, entry in self.entries.items()
        }

    @classmethod
    def from_dict(cls, value: object) -> SnapshotComponents:
        if not isinstance(value, Mapping) or not value:
            raise ValueError("components manifest must be a non-empty object")
        entries: dict[str, ComponentEntry] = {}
        for name, raw in value.items():
            if not isinstance(raw, Mapping):
                raise ValueError(f"components manifest entry {name!r} must be an object")
            metadata = raw.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise ValueError(f"components manifest entry {name!r} metadata must be an object")
            artifact_id = raw.get("artifact_id")
            if not isinstance(artifact_id, str):
                raise ValueError(f"components manifest entry {name!r} requires artifact_id")
            entries[str(name)] = ComponentEntry(metadata=metadata, artifact_id=artifact_id)
        return cls(entries)


def snapshot_components(metadata: Mapping[str, object] | None) -> SnapshotComponents | None:
    """The manifest carried by release metadata, or ``None`` for a snapshot published without one."""
    if metadata is None:
        return None
    raw = metadata.get(COMPONENTS_METADATA_KEY)
    if raw is None:
        return None
    return SnapshotComponents.from_dict(raw)


__all__ = [
    "COMPONENTS_METADATA_KEY",
    "RECORDS_COMPONENT",
    "ComponentEntry",
    "SnapshotComponents",
    "snapshot_components",
    "validate_component_name",
]
