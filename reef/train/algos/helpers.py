"""Optional helpers for implementing backend-neutral step preparers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def next_steps(state: Mapping[str, Any]) -> int:
    """Validate and increment the conventional algorithm step counter."""
    steps = state.get("steps", 0)
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("algorithm state 'steps' must be a non-negative integer")
    return steps + 1
