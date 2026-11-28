"""Checks of plain values that several modules read from stored or sent data."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

#: JSON-compatible values: what ``json.loads`` gives, with the keys of every object strings.
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def canonical_json(value: object) -> str:
    """Encode a JSON value with sorted keys, compact separators and unescaped Unicode."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def require_text(value: object, name: str) -> str:
    """``value`` when it is a non-empty string without surrounding spaces; else ``ValueError`` naming ``name``."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding spaces")
    return value


def require_integer(value: object, name: str, *, minimum: int = 0) -> None:
    """Raise ``ValueError`` naming ``name`` unless ``value`` is an integer from ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} is an integer from {minimum}")


def strict_json_object(data: bytes) -> dict[str, object]:
    """Decode one JSON object, refusing repeated keys and non-finite numbers.

    Raises:
        ValueError: The data is not such an object, is not valid text, or nests too deeply to decode.
    """

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("JSON object repeats a key")
        return result

    def finite_number(text: str) -> float:
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("JSON number is not finite")
        return value

    def no_constant(name: str) -> float:
        raise ValueError(f"JSON constant {name} is not allowed")

    try:
        value = json.loads(
            data, object_pairs_hook=unique_object, parse_float=finite_number, parse_constant=no_constant
        )
    except RecursionError as exc:
        raise ValueError("JSON nests too deeply") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON value is not an object")
    return value


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """``base`` with ``overlay`` merged in: nested objects merge key by key, every other value of ``overlay`` wins."""
    merged = dict(base)
    for key, value in overlay.items():
        below = merged.get(key)
        merged[key] = deep_merge(below, value) if isinstance(below, Mapping) and isinstance(value, Mapping) else value
    return merged


__all__ = ["JsonValue", "canonical_json", "deep_merge", "require_integer", "require_text", "strict_json_object"]
