"""The named components one release binds, stored as a manifest in release metadata.

A snapshot with one component keeps its files at the root. A snapshot with
several components keeps each one in a directory named after it. The mapping
has no content identity of its own.

Each component is one artifact. New components receive independent artifact
IDs before publication. A snapshot records those IDs explicitly, and carrying
a component into another snapshot preserves its ID. Legacy entries without an
explicit ID retain their original revision-based identity.

The manifest travels in artifact metadata under ``COMPONENTS_METADATA_KEY``.
A release without a manifest is read as one component at the root, which the
release made.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

COMPONENTS_METADATA_KEY = "components"

#: The component a scenario's only trainer is bound to when its surface serves no component at all.
RECORDS_COMPONENT = "records"


def validate_component_name(name: object) -> str:
    """A component name doubles as its directory name in a multi-component release."""
    if not isinstance(name, str) or not name:
        raise ValueError("component name must be a non-empty string")
    if name in (".", "..") or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"component name {name!r} must be a plain directory name")
    return name


def legacy_artifact_id(revision: str, component: str, *, single: bool) -> str:
    """Recover an old component ID from its snapshot revision when the manifest has no explicit ID.

    Legacy flat artifacts used the revision itself. Legacy multi-component
    snapshots used ``<revision>:<component>``. New artifacts store independent IDs.
    """
    return revision if single else f"{revision}:{component}"


@dataclass(frozen=True)
class ComponentEntry:
    """One component's artifact in a release manifest: its content, its identity and its metadata.

    Values: ``artifact_id`` is a non-empty string for a current entry, or
    None for a legacy entry whose snapshot revision determines its identity.
    ``metadata`` is a mapping of that component's metadata, copied and frozen
    at construction. The legacy identity is the snapshot revision for one
    component, or ``<revision>:<component>`` for a multi-component snapshot.

    Purpose: the committer and the scenario factory make entries when a
    release is published or a scenario is created; readers resolve a
    component's artifact and content from them.
    """

    artifact_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.artifact_id is not None and (not isinstance(self.artifact_id, str) or not self.artifact_id):
            raise ValueError("component artifact_id must be a non-empty string or null")
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

    def artifact_ids(self, revision: str) -> dict[str, str]:
        """Return explicit component IDs, with revision-based IDs only for legacy entries lacking an ID."""
        return {
            name: entry.artifact_id or legacy_artifact_id(revision, name, single=self.single)
            for name, entry in self.entries.items()
        }

    def carried_from(self, revision: str) -> ReleaseComponents:
        """This manifest as a later release carries it forward: every entry names its artifact explicitly."""
        identities = self.artifact_ids(revision)
        return ReleaseComponents(
            {name: replace(entry, artifact_id=identities[name]) for name, entry in self.entries.items()}
        )

    @property
    def single(self) -> bool:
        """True when the release keeps the flat, one-component layout."""
        return len(self.entries) == 1

    def relative_path(self, name: str) -> Path:
        """Where ``name``'s files live under the release root."""
        if name not in self.entries:
            raise KeyError(f"release binds no component {name!r}")
        return Path(".") if self.single else Path(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: {
                "metadata": dict(entry.metadata),
                **({} if entry.artifact_id is None else {"artifact_id": entry.artifact_id}),
            }
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
            entries[str(name)] = ComponentEntry(metadata=metadata, artifact_id=raw.get("artifact_id"))
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
    "RECORDS_COMPONENT",
    "ComponentEntry",
    "ReleaseComponents",
    "legacy_artifact_id",
    "release_components",
    "validate_component_name",
]
