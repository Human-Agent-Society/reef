"""Canonical identity for one concrete serving-runtime weight load."""

from __future__ import annotations

import uuid
from dataclasses import dataclass


def new_runtime_load_id_incarnation() -> str:
    """Return a training-group-lifetime namespace for runtime loads."""
    return uuid.uuid4().hex


@dataclass(frozen=True)
class RuntimeLoadId:
    """Globally unique identity for one serving-runtime weight load."""

    incarnation: str
    sequence: int

    def __post_init__(self) -> None:
        if not self.incarnation or ":" in self.incarnation:
            raise ValueError("runtime-load incarnation must be non-empty and contain no ':'")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence < 0:
            raise ValueError("runtime-load sequence must be a non-negative integer")

    def __str__(self) -> str:
        return f"{self.incarnation}:{self.sequence}"

    @classmethod
    def parse(cls, value: str) -> RuntimeLoadId:
        if not isinstance(value, str):
            raise TypeError("runtime load id must be a string")
        incarnation, separator, sequence = value.rpartition(":")
        if not separator or not sequence.isascii() or not sequence.isdecimal():
            raise ValueError("runtime load id must use '<incarnation>:<sequence>'")
        return cls(incarnation, int(sequence))


__all__ = ["RuntimeLoadId", "new_runtime_load_id_incarnation"]
