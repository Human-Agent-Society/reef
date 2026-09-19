"""CEO-Bench's bash agent, served by Reef.

``harness.agent`` is the benchmark's agent loop and ``harness.tools`` its
tools; ``harness.harbor_agent`` plays it as one Harbor trial with its model
calls served by Reef, and ``harness.report`` credits each finished week to
its decision turns and posts the reports. The export is lazy: importing this
package must not require Harbor.
"""

__all__ = ["HarborAgent"]


def __getattr__(name: str):
    if name == "HarborAgent":
        from .harbor_agent import HarborAgent

        return HarborAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
