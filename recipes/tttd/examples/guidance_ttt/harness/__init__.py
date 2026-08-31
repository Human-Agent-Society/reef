"""Harness-side building blocks for summary-only Guidance-TTT.

``harness.agent`` is the Reef adapter: one trainable guidance call per rollout,
one frozen external execution, and one verifier reward reported against the
exact guidance receipt. ``harness.harbor_agent`` wires that loop to Harbor and
is the only module that imports the external ``harbor`` package, so everything
else stays importable standalone.

The harness is task-agnostic: the problem statement arrives as the Harbor
task's ``instruction.md`` and the rest of its vocabulary as that task's
``contract.json``, so a new discovery problem is a new Harbor task directory,
not a harness change.
"""

from .agent import GuidanceRolloutResult, ReefGuidanceTTTHarness, prepare_library
from .contract import TaskContract
from .execution import (
    ExecutionBackend,
    OpenAICompatibleExecutionClient,
    gpt_oss_120b_backend,
    openrouter_glm_5_2_backend,
)
from .library import GuidanceLibrary
from .scorer import JudgeScorer, JudgeUnavailableError, Scorer, extract_solution_code
from .search import guidance_chat_request, openai_action

__all__ = [
    "ExecutionBackend",
    "GuidanceLibrary",
    "GuidanceRolloutResult",
    "HarborAgent",
    "JudgeScorer",
    "JudgeUnavailableError",
    "OpenAICompatibleExecutionClient",
    "ReefGuidanceTTTHarness",
    "Scorer",
    "TaskContract",
    "extract_solution_code",
    "gpt_oss_120b_backend",
    "guidance_chat_request",
    "openai_action",
    "openrouter_glm_5_2_backend",
    "prepare_library",
]


def __getattr__(name: str):
    if name == "HarborAgent":
        from .harbor_agent import HarborAgent

        return HarborAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
