"""Harbor agent harness for CEO-Bench on Reef.

``harness.agent`` runs one benchmark episode with the agent role served by
Reef; ``harness.report`` posts the verifier's reward against every model call
once Harbor ends the trial. The export is lazy: importing this package must
not require Harbor.
"""

__all__ = ["HarborAgent"]


def __getattr__(name: str):
    if name == "HarborAgent":
        from .agent import HarborAgent

        return HarborAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
