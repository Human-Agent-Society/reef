from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_SENSITIVE_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "bearer_token",
    "secret",
    "token",
}


def sanitize_verifier_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return persistable verifier settings with credential values removed recursively."""

    def sanitize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): sanitize(item)
                for key, item in value.items()
                if str(key).strip().lower().replace("-", "_") not in _SENSITIVE_KEYS
            }
        if isinstance(value, (list, tuple)):
            return [sanitize(item) for item in value]
        return value

    return sanitize(config)
