"""Checks of plain values that several modules read from stored or sent data."""

from __future__ import annotations

#: JSON-compatible values: what ``json.loads`` gives, with the keys of every object strings.
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def require_text(value: object, name: str) -> str:
    """``value`` when it is a non-empty string without surrounding spaces; else ``ValueError`` naming ``name``."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding spaces")
    return value


__all__ = ["JsonValue", "require_text"]
