"""A Harbor agent harness that runs the SDPO grid runner in the task's container.

The export is lazy: importing this package must not require Harbor.
"""

__all__ = ["HarborAgent"]


def __getattr__(name: str):
    if name == "HarborAgent":
        from .agent import HarborAgent

        return HarborAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
