"""The skill stream's Harbor agent: one trial runs one stage's training stream in the task container.

The export is lazy: importing this package must not require Harbor.
"""

__all__ = ["HarborAgent"]


def __getattr__(name: str):
    if name == "HarborAgent":
        from .agent import HarborAgent

        return HarborAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
