"""Summary-only Guidance-TTT on Reef."""

from .execution import (
    ExecutionBackend,
    OpenAICompatibleExecutionClient,
    gpt_oss_120b_backend,
    openrouter_glm_5_2_backend,
)
from .bootstrap import create_verified_baseline_seed
from .harness import GuidanceRolloutResult, ReefGuidanceTTTHarness, prepare_library
from .library import GuidanceLibrary
from .tasks import POLYOMINO_TASK, TRIMUL_TASK, VLIW_TASK, TaskSpec, get_task_spec, task_ids

__all__ = [
    "POLYOMINO_TASK",
    "TRIMUL_TASK",
    "VLIW_TASK",
    "ExecutionBackend",
    "GuidanceLibrary",
    "GuidanceRolloutResult",
    "OpenAICompatibleExecutionClient",
    "ReefGuidanceTTTHarness",
    "TaskSpec",
    "create_verified_baseline_seed",
    "get_task_spec",
    "gpt_oss_120b_backend",
    "openrouter_glm_5_2_backend",
    "prepare_library",
    "task_ids",
]
