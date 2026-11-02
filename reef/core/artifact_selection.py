"""Scenario component selections, separate from immutable artifact storage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from reef.core.components import validate_component_name
from reef.core.snapshot import SnapshotRef, decode_snapshot_ref


@dataclass(frozen=True)
class ArtifactSelection:
    """A nonempty immutable mapping from component names to nonempty artifact IDs.

    Every key is a valid component name. The mapping describes scenario state,
    with no identity of its own. Repositories materialize its artifacts;
    operation records persist it for recovery.
    """

    components: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.components, Mapping) or not self.components:
            raise ValueError("an artifact selection requires components")
        for name, artifact_id in self.components.items():
            validate_component_name(name)
            if not isinstance(artifact_id, str) or not artifact_id:
                raise ValueError("an artifact selection maps components to nonempty artifact IDs")
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))

    @property
    def artifact_ids(self) -> dict[str, str]:
        """Return a mutable copy of the selected component IDs."""
        return dict(self.components)

    def to_dict(self) -> dict[str, object]:
        """Encode component IDs without storage locators."""
        return dict(self.components)

    @classmethod
    def from_dict(cls, value: object) -> ArtifactSelection:
        """Decode a stored selection; an older entry ``{"artifact_id", "ref"}`` gives its ``artifact_id``."""
        if not isinstance(value, Mapping):
            raise ValueError("an artifact selection must be an object")
        components: dict[str, str] = {}
        for name, entry in value.items():
            if not isinstance(name, str):
                raise ValueError("an artifact selection requires component names")
            artifact_id = entry.get("artifact_id") if isinstance(entry, Mapping) else entry
            if not isinstance(artifact_id, str):
                raise ValueError("an artifact selection maps component names to artifact IDs")
            components[name] = artifact_id
        return cls(components)


def legacy_selection_locators(value: object) -> dict[str, SnapshotRef]:
    """The durable storage snapshot that each entry of an older stored selection ``value`` recorded, by component.

    The current format records no locators, so the result is empty for it. Recovery indexes these snapshots, so
    their artifacts resolve by ID. Raise ``ValueError`` for a malformed entry or a locator of a local view.
    """
    if not isinstance(value, Mapping):
        raise ValueError("an artifact selection must be an object")
    locators: dict[str, SnapshotRef] = {}
    for name, entry in value.items():
        if not isinstance(entry, Mapping):
            continue
        ref = entry.get("ref")
        if not isinstance(ref, Mapping):
            raise ValueError("a legacy selected artifact requires artifact_id and ref")
        locator = decode_snapshot_ref(ref)
        if not locator.durable:
            raise ValueError("a legacy selected artifact requires durable storage")
        locators[str(name)] = locator
    return locators
