"""Scenario component selections, separate from immutable artifact storage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from reef.core.artifact_ref import ArtifactRef, decode_artifact_ref, encode_artifact_ref
from reef.core.components import validate_component_name


@dataclass(frozen=True)
class SelectedArtifact:
    """An existing artifact and the storage reference containing its component files.

    ``artifact_id`` is a nonempty opaque artifact identity. ``ref`` is an ``ArtifactRef`` whose locator starts with
    neither ``local:`` nor ``selection:``. It names storage containing the enclosing selection's component;
    repository materialization checks that the stored component has ``artifact_id``. Scenario operations write
    these values; recovery reads them.
    """

    artifact_id: str
    ref: ArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not self.artifact_id:
            raise ValueError("a selected artifact requires a nonempty artifact_id")
        if not isinstance(self.ref, ArtifactRef):
            raise TypeError("a selected artifact requires an ArtifactRef")
        if self.ref.release_id.startswith(("local:", "selection:")):
            raise ValueError("a selected artifact requires durable storage")


@dataclass(frozen=True)
class ArtifactSelection:
    """A nonempty immutable mapping from component names to existing artifacts.

    Every key is a valid component name and every value a ``SelectedArtifact``. The mapping describes scenario
    state, with no identity of its own. Repositories materialize it; operation records persist it for recovery.
    """

    components: Mapping[str, SelectedArtifact]

    def __post_init__(self) -> None:
        if not isinstance(self.components, Mapping) or not self.components:
            raise ValueError("an artifact selection requires components")
        for name, selected in self.components.items():
            validate_component_name(name)
            if not isinstance(selected, SelectedArtifact):
                raise TypeError("an artifact selection maps components to SelectedArtifact values")
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))

    @property
    def artifact_ids(self) -> dict[str, str]:
        """Return the selected artifact identity for each component."""
        return {name: selected.artifact_id for name, selected in self.components.items()}

    def to_dict(self) -> dict[str, object]:
        """Encode the component mapping, including the private storage locators needed for recovery."""
        return {
            name: {"artifact_id": selected.artifact_id, "ref": encode_artifact_ref(selected.ref)}
            for name, selected in self.components.items()
        }

    @classmethod
    def from_dict(cls, value: object) -> ArtifactSelection:
        """Decode a component mapping, raising ``ValueError`` for malformed stored selections."""
        if not isinstance(value, Mapping):
            raise ValueError("an artifact selection must be an object")
        components: dict[str, SelectedArtifact] = {}
        for name, entry in value.items():
            if not isinstance(name, str) or not isinstance(entry, Mapping):
                raise ValueError("an artifact selection maps component names to objects")
            artifact_id, ref = entry.get("artifact_id"), entry.get("ref")
            if not isinstance(artifact_id, str) or not isinstance(ref, Mapping):
                raise ValueError("a selected artifact requires artifact_id and ref")
            components[name] = SelectedArtifact(artifact_id, decode_artifact_ref(ref))
        return cls(components)
