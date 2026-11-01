"""Checks of plain values that several modules read from stored or sent data."""

from __future__ import annotations


def require_text(value: object, name: str) -> str:
    """``value`` when it is a non-empty string without surrounding spaces; else ``ValueError`` naming ``name``."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding spaces")
    return value


__all__ = ["require_text"]
