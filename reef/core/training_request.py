"""A user's training instruction and the session and release it came from."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingRequest:
    """A training instruction, independent of inference batches and feedback.

    ``id`` is filled from the enclosing AgentRecord when it becomes a batch.
    Session and release are provenance; they do not select an inference batch.
    """

    text: str
    session: str
    release_id: str
    id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-empty string")
        if len(self.text) > 4000:
            raise ValueError("text must not exceed 4000 characters")
        if not isinstance(self.session, str) or not isinstance(self.release_id, str):
            raise ValueError("session and release_id must be strings")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrainingRequest:
        fields: dict[str, str] = {}
        for key in ("text", "session", "release_id"):
            value = payload.get(key)
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            fields[key] = value
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "session": self.session, "release_id": self.release_id}
