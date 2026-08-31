"""The engine-side naming contract for scenario-owned adapter revisions."""

from __future__ import annotations

import base64

ADAPTER_NAME_PREFIX = "reef-adapter-"
_SEPARATOR = "."


def adapter_name(scenario: str, runtime_load_id: str) -> str:
    """The engine-side adapter name for one scenario's concrete runtime load.

    Naming is the entire routing contract: whoever registers an adapter with
    the engine uses this name, and ``prepare_request`` selects by the same
    rule, so a recorded ``lora_path`` names exactly one (scenario, revision).
    The scenario is base64url-encoded without padding and the runtime load ID kept
    verbatim, joined by ``.`` — a character outside the base64url alphabet —
    so the name is lossless and two scenarios that reuse one human-readable
    revision label can never collide. :func:`parse_adapter_name` inverts it.
    """
    if not scenario:
        raise ValueError("adapter name requires a non-empty scenario")
    if not runtime_load_id:
        raise ValueError("adapter name requires a non-empty runtime_load_id")
    encoded = base64.urlsafe_b64encode(scenario.encode("utf-8")).decode("ascii").rstrip("=")
    return f"{ADAPTER_NAME_PREFIX}{encoded}{_SEPARATOR}{runtime_load_id}"


def parse_adapter_name(name: str) -> tuple[str, str]:
    """Recover ``(scenario, runtime_load_id)`` from a name produced by :func:`adapter_name`."""
    if not name.startswith(ADAPTER_NAME_PREFIX):
        raise ValueError(f"{name!r} is not a Reef adapter name")
    encoded, separator, runtime_load_id = name[len(ADAPTER_NAME_PREFIX) :].partition(_SEPARATOR)
    if not separator or not encoded or not runtime_load_id:
        raise ValueError(f"{name!r} is not a Reef adapter name")
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        scenario = base64.urlsafe_b64decode(padded).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"{name!r} does not carry a decodable scenario") from exc
    return scenario, runtime_load_id


__all__ = [
    "ADAPTER_NAME_PREFIX",
    "adapter_name",
    "parse_adapter_name",
]
