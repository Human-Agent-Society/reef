"""The release and content identity of an artifact, and where a scenario's root release came from."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ArtifactRef:
    """Immutable binding from one release to its selected contents."""

    content_id: str
    release_id: str
    parent_release_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.content_id, str) or not self.content_id:
            raise ValueError("content_id must be a non-empty string")
        if not isinstance(self.release_id, str) or not self.release_id:
            raise ValueError("release_id must be a non-empty string")
        if self.parent_release_id is not None and not isinstance(self.parent_release_id, str):
            raise ValueError("parent_release_id must be a string or null")


def encode_artifact_ref(ref: ArtifactRef) -> dict[str, Any]:
    """Encode a ref in the release/content wire format."""
    return {
        "kind": "artifact",
        "content_id": ref.content_id,
        "release_id": ref.release_id,
        "parent_release_id": ref.parent_release_id,
    }


def decode_artifact_ref(value: Mapping[str, Any]) -> ArtifactRef:
    """Decode the release/content ref wire format."""
    kind = value.get("kind")
    content_id = value.get("content_id")
    release_id = value.get("release_id")
    parent_release_id = value.get("parent_release_id")
    if kind != "artifact":
        raise ValueError(f"artifact ref kind must be 'artifact', got {kind!r}")
    if not isinstance(content_id, str) or not content_id:
        raise ValueError("artifact ref requires content_id")
    if not isinstance(release_id, str) or not release_id:
        raise ValueError("artifact ref requires release_id")
    if parent_release_id is not None and not isinstance(parent_release_id, str):
        raise ValueError("artifact ref parent_release_id must be a string or null")
    if "runtime_load_id" in value:
        raise ValueError("artifact ref must not carry runtime_load_id")
    return ArtifactRef(content_id, release_id, parent_release_id)


@dataclass(frozen=True)
class RootSource:
    """Where the content of a scenario's root release came from.

    Values: ``kind`` is ``"default"`` or ``"item"``.

    - ``"default"``: the root holds the deployment's ``DefaultStart``. ``inputs`` is its ``inputs``; ``item``,
      ``uri`` and ``version`` are ``None``.
    - ``"item"``: the root holds an artifact item in one component. ``item`` is the item's non-empty ID, ``uri``
      the non-empty URI that its files were read from, and ``version`` what the source served there (``""`` when
      the source names none), as an import records them. ``inputs`` is empty.

    Purpose: ``ScenarioFactory`` makes the value when it creates a scenario, and the scenario metadata stores it as
    ``to_dict``. A trainer gives it to ``TrainingComponent.root_start``, and the release list shows it on the
    creation row.
    """

    kind: Literal["default", "item"]
    inputs: Mapping[str, str] = field(default_factory=dict)
    item: str | None = None
    uri: str | None = None
    version: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, Mapping):
            raise ValueError("a root source's inputs are a mapping")
        if self.kind == "default":
            if (self.item, self.uri, self.version) != (None, None, None):
                raise ValueError("a default root source names no item, uri or version")
            for name, value in self.inputs.items():
                if not isinstance(name, str) or not name or not isinstance(value, str) or not value:
                    raise ValueError(f"root source input {name!r} must map a non-empty name to a non-empty string")
        elif self.kind == "item":
            if self.inputs:
                raise ValueError("an item root source has no inputs")
            if not isinstance(self.item, str) or not self.item or not isinstance(self.uri, str) or not self.uri:
                raise ValueError("an item root source names a non-empty item and uri")
            if not isinstance(self.version, str):
                raise ValueError("an item root source's version is a string")
        else:
            raise ValueError(f"a root source's kind is 'default' or 'item', not {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        """``{kind: "default", inputs}`` or ``{kind: "item", item, uri, version}``; ``from_dict`` reads it back."""
        if self.kind == "default":
            return {"kind": "default", "inputs": dict(self.inputs)}
        return {"kind": "item", "item": self.item, "uri": self.uri, "version": self.version}

    @classmethod
    def from_dict(cls, value: object) -> RootSource:
        """Read ``to_dict``'s object; ``ValueError`` for any other value."""
        if not isinstance(value, Mapping):
            raise ValueError(f"a root source is an object, not {value!r}")
        if (
            value.get("kind") == "default"
            and set(value) == {"kind", "inputs"}
            and isinstance(value["inputs"], Mapping)
        ):
            return cls("default", inputs=dict(value["inputs"]))
        if value.get("kind") == "item" and set(value) == {"kind", "item", "uri", "version"}:
            return cls("item", item=value["item"], uri=value["uri"], version=value["version"])
        raise ValueError(f"a root source is {{kind, inputs}} or {{kind, item, uri, version}}, not {value!r}")


__all__ = [
    "ArtifactRef",
    "RootSource",
    "decode_artifact_ref",
    "encode_artifact_ref",
]
