"""REEF Eval/Harbor harness for the OpenClaw-RL GSM8K stream.

``HermesStreamAgent`` is exported lazily so importing the package does not
require the harbor runtime; REEF Eval loads it when resolving
``--agent harness:HermesStreamAgent``.
"""

from typing import Any

__all__ = ["HermesStreamAgent"]


def __getattr__(name: str) -> Any:
    if name == "HermesStreamAgent":
        from .agent import HermesStreamAgent

        return HermesStreamAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
