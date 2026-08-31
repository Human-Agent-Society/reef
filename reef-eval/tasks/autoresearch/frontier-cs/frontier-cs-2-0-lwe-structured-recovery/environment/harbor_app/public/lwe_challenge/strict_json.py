from __future__ import annotations

import json
import math
from typing import Any


class JsonContractError(ValueError):
    """The input violates a bounded, unambiguous JSON contract."""


def _reject_unicode_surrogates(value: str) -> None:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise JsonContractError("Unicode surrogate in JSON string")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        _reject_unicode_surrogates(key)
        if key in out:
            raise JsonContractError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _constant(value: str) -> None:
    raise JsonContractError(f"non-finite JSON number: {value}")


def _float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise JsonContractError(f"non-finite JSON number: {value}")
    return parsed


def _check_shape(value: Any, *, max_depth: int, max_nodes: int) -> None:
    nodes = 0

    def visit(node: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            raise JsonContractError(f"JSON exceeds {max_nodes} nodes")
        if depth > max_depth:
            raise JsonContractError(f"JSON exceeds depth {max_depth}")
        if isinstance(node, str):
            _reject_unicode_surrogates(node)
        elif isinstance(node, dict):
            for key, child in node.items():
                visit(key, depth + 1)
                visit(child, depth + 1)
        elif isinstance(node, list):
            for child in node:
                visit(child, depth + 1)

    visit(value, 0)


def loads_object(
    data: bytes,
    *,
    max_bytes: int,
    max_depth: int = 8,
    max_nodes: int = 10_000,
) -> dict[str, Any]:
    if len(data) > max_bytes:
        raise JsonContractError(f"JSON exceeds {max_bytes} bytes")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
            parse_float=_float,
        )
    except JsonContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise JsonContractError("invalid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise JsonContractError("top-level JSON value must be an object")
    _check_shape(value, max_depth=max_depth, max_nodes=max_nodes)
    return value
