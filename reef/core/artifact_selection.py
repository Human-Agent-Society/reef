"""Scenario component selections, separate from immutable artifact storage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from reef.core.components import validate_component_name


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
        """Decode a stored selection."""
        if not isinstance(value, Mapping):
            raise ValueError("an artifact selection must be an object")
        components: dict[str, str] = {}
        for name, entry in value.items():
            if not isinstance(name, str):
                raise ValueError("an artifact selection requires component names")
            if not isinstance(entry, str):
                raise ValueError("an artifact selection maps component names to artifact IDs")
            components[name] = entry
        return cls(components)
