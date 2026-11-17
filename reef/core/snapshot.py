"""Private artifact storage locators and the source of a scenario's initial artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

#: Revision prefix of a process-local snapshot: staged or unpublished, with no durable bytes.
LOCAL_SNAPSHOT_PREFIX = "local:"
#: Revision prefix of a local view composed from an ``ArtifactSelection``, with no durable bytes.
SELECTION_VIEW_PREFIX = "selection:"


@dataclass(frozen=True)
class SnapshotRef:
    """An immutable storage snapshot locator, separate from component artifact identity.

    ``revision`` is a nonempty backend revision or process-local view key.
    ``parent_revision`` is None for roots, otherwise a nonempty backend revision.
    Repository adapters construct these values for publication and recovery.
    Component selections use artifact IDs, never these storage keys.
    A revision that starts with ``LOCAL_SNAPSHOT_PREFIX`` or
    ``SELECTION_VIEW_PREFIX`` names a process-local view; every other revision
    names durable storage (``durable``).
    """

    revision: str
    parent_revision: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.revision, str) or not self.revision:
            raise ValueError("snapshot revision must be a nonempty string")
        if self.parent_revision is not None and (
            not isinstance(self.parent_revision, str) or not self.parent_revision
        ):
            raise ValueError("snapshot parent_revision must be a nonempty string or null")

    @property
    def local(self) -> bool:
        """True for a process-local snapshot: staged or unpublished, with no durable bytes."""
        return self.revision.startswith(LOCAL_SNAPSHOT_PREFIX)

    @property
    def selection_view(self) -> bool:
        """True for a local view composed from an ``ArtifactSelection``."""
        return self.revision.startswith(SELECTION_VIEW_PREFIX)

    @property
    def durable(self) -> bool:
        """True when ``revision`` names backend storage: neither ``local`` nor ``selection_view``."""
        return not self.local and not self.selection_view


def encode_snapshot_ref(ref: SnapshotRef) -> dict[str, object]:
    """Encode a backend locator without claiming a component artifact identity."""
    return {"kind": "snapshot", "revision": ref.revision, "parent_revision": ref.parent_revision}


def decode_snapshot_ref(value: Mapping[str, object]) -> SnapshotRef:
    """Decode a locator that ``encode_snapshot_ref`` wrote."""
    if value.get("kind") != "snapshot":
        raise ValueError("snapshot ref kind must be 'snapshot'")
    revision, parent = value.get("revision"), value.get("parent_revision")
    if not isinstance(revision, str) or (parent is not None and not isinstance(parent, str)):
        raise ValueError("snapshot ref revision and parent_revision must be strings")
    if "runtime_load_id" in value:
        raise ValueError("snapshot ref must not carry runtime_load_id")
    return SnapshotRef(revision, parent)


@dataclass(frozen=True)
class RootSource:
    """Where the content of a scenario's root snapshot came from.

    Values: ``kind`` is ``"default"`` or ``"item"``.

    - ``"default"``: the root holds the deployment's ``DefaultStart``. ``inputs`` is its ``inputs``; ``item``,
      ``uri`` and ``version`` are ``None``.
    - ``"item"``: the root holds an artifact item in one component. ``item`` is the item's non-empty ID, ``uri``
      the non-empty URI that its files were read from, and ``version`` what the source served there (``""`` when
      the source names none), as an import records them. ``inputs`` is empty.

    Purpose: ``ScenarioFactory`` makes the value when it creates a scenario, and the scenario metadata stores it as
    ``to_dict``. A trainer gives it to ``TrainingComponent.root_start``, and the snapshot list shows it on the
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


@dataclass(frozen=True)
class ItemIdentity:
    """What an artifact made from an artifact item takes from the item, besides its files.

    Values: ``item_id`` is the item's non-empty ID. ``artifact_id`` is the item's artifact ID, which the new artifact
    keeps (RFC #1218), or None for a new one. Construction checks these types.

    Purpose: ``reef.service.artifact_imports`` makes it from a stored item; a scenario's root (``ItemStart``) and an
    import (``ScenarioCommitter.import_artifact``) read it.
    """

    item_id: str
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id:
            raise ValueError("an item identity names a non-empty item_id")
        if self.artifact_id is not None and (not isinstance(self.artifact_id, str) or not self.artifact_id):
            raise ValueError("an item identity's artifact_id is a non-empty string or None")


__all__ = [
    "LOCAL_SNAPSHOT_PREFIX",
    "SELECTION_VIEW_PREFIX",
    "ItemIdentity",
    "RootSource",
    "SnapshotRef",
    "decode_snapshot_ref",
    "encode_snapshot_ref",
]
