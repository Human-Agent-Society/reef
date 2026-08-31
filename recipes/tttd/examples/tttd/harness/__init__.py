"""Harness-side building blocks for Test-Time Training to Discover.

``harness.agent`` contains the Reef-specific harness subclass.
``harness.run_controller`` pairs each search step with Reef's durable training
commit. ``harness.search`` implements the PUCT archive, the search algorithm,
and generic prompt construction. ``harness.scorer`` provides ``JudgeScorer``
(HTTP to a judge server) and ``ProgramScorer`` (direct subprocess sandbox).
``harness.sandbox`` provides the isolated subprocess executor.

The harness is task-agnostic: it scores every generated solution via a
``Scorer`` callable. The task instruction comes from the Harbor task's
``instruction.md``.
"""

from .search import TTTDChatRequestBuilder, openai_action

__all__ = ["HarborAgent", "TTTDChatRequestBuilder", "openai_action"]


def __getattr__(name: str):
    if name == "HarborAgent":
        from .harbor_agent import HarborAgent

        return HarborAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
