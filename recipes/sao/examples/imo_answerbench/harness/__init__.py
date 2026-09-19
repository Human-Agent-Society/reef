"""Harbor agent harness for the SAO functional smoke.

``harness.agent`` drives the scored rollouts through Reef; ``harness.report``
posts the verifier reward back once Harbor ends the trial. The export is lazy:
importing this package must not require Harbor.
"""

__all__ = ["HarborAgent"]


def __getattr__(name: str):
    if name == "HarborAgent":
        from .agent import HarborAgent

        return HarborAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
