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
        """Decode current selections or retain legacy locators for index migration."""
        if not isinstance(value, Mapping):
            raise ValueError("an artifact selection must be an object")
        components: dict[str, str] = {}
        locators: dict[str, SnapshotRef] = {}
        for name, entry in value.items():
            if not isinstance(name, str):
                raise ValueError("an artifact selection requires component names")
            if isinstance(entry, str):
                components[name] = entry
            elif isinstance(entry, Mapping):
                artifact_id, ref = entry.get("artifact_id"), entry.get("ref")
                if not isinstance(artifact_id, str) or not isinstance(ref, Mapping):
                    raise ValueError("a legacy selected artifact requires artifact_id and ref")
                components[name] = artifact_id
                locators[name] = decode_snapshot_ref(ref)
            else:
                raise ValueError("an artifact selection requires artifact IDs")
        if locators:
            return LegacyArtifactSelection(components, locators)
        return cls(components)


@dataclass(frozen=True)
class LegacyArtifactSelection(ArtifactSelection):
    """A decoded selection with durable locators for some or all components.

    ``locators`` is a nonempty mapping whose keys occur in ``components``.
    Each value names durable storage, never a local or composed view.
    Recovery uses these locators to index artifacts from older records.
    Serialization emits only the inherited component IDs.
    """

    locators: Mapping[str, SnapshotRef]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.locators, Mapping) or not self.locators:
            raise ValueError("a legacy selection requires storage locators")
        for name, ref in self.locators.items():
            if name not in self.components or not isinstance(ref, SnapshotRef):
                raise ValueError("legacy locators must name selected components")
            if ref.revision.startswith(("local:", "selection:")):
                raise ValueError("a legacy selected artifact requires durable storage")
        object.__setattr__(self, "locators", MappingProxyType(dict(self.locators)))
