"""Checks of plain values that several modules read from stored or sent data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: JSON-compatible values: what ``json.loads`` gives, with the keys of every object strings.
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def require_text(value: object, name: str) -> str:
    """``value`` when it is a non-empty string without surrounding spaces; else ``ValueError`` naming ``name``."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding spaces")
    return value


def require_integer(value: object, name: str, *, minimum: int = 0) -> None:
    """Raise ``ValueError`` naming ``name`` unless ``value`` is an integer from ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} is an integer from {minimum}")


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """``base`` with ``overlay`` merged in: nested objects merge key by key, every other value of ``overlay`` wins."""
    merged = dict(base)
    for key, value in overlay.items():
        below = merged.get(key)
        merged[key] = deep_merge(below, value) if isinstance(below, Mapping) and isinstance(value, Mapping) else value
    return merged


__all__ = ["JsonValue", "deep_merge", "require_integer", "require_text"]
