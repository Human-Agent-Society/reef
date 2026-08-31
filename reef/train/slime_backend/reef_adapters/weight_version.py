"""Canonical identity for one serving-weight publication."""

from __future__ import annotations

import uuid
from dataclasses import dataclass


def new_weight_version_incarnation() -> str:
    """Return a training-group-lifetime namespace for serving updates."""
    return uuid.uuid4().hex


@dataclass(frozen=True)
class WeightVersion:
    """Globally unique identity for one serving-weight update."""

    incarnation: str
    sequence: int

    def __post_init__(self) -> None:
        if not self.incarnation or ":" in self.incarnation:
            raise ValueError("weight-version incarnation must be non-empty and contain no ':'")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence < 0:
            raise ValueError("weight-version sequence must be a non-negative integer")

    def __str__(self) -> str:
        return f"{self.incarnation}:{self.sequence}"

    @classmethod
    def parse(cls, value: str) -> WeightVersion:
        if not isinstance(value, str):
            raise TypeError("weight version must be a string")
        incarnation, separator, sequence = value.rpartition(":")
        if not separator or not sequence.isascii() or not sequence.isdecimal():
            raise ValueError("weight version must use '<incarnation>:<sequence>'")
        return cls(incarnation, int(sequence))


__all__ = ["WeightVersion", "new_weight_version_incarnation"]
