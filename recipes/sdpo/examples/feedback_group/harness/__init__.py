"""Harbor agent for the SDPO feedback-group example."""

__all__ = ["HarborAgent"]


def __getattr__(name: str):
    if name == "HarborAgent":
        from .agent import HarborAgent

        return HarborAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
