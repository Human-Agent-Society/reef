"""The release and content identity of an artifact, shared across Reef."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


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


__all__ = [
    "ArtifactRef",
    "decode_artifact_ref",
    "encode_artifact_ref",
]
