"""Interpret Harbor trial outcomes without importing the evaluation runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def trial_outcome(result: Mapping[str, Any]) -> dict[str, Any]:
    """Separate a measured task failure from a failed evaluation.

    Harbor may verify successfully after its agent timeout. Its exception and
    reward populations therefore overlap; neither cost nor exception count is
    an execution-status flag. Unknown usage remains unknown.
    """
    exception = result.get("exception_info") or {}
    agent = result.get("agent_result") or {}
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    measured_reward = (
        float(reward)
        if isinstance(reward, (int, float)) and not isinstance(reward, bool) and math.isfinite(reward)
        else None
    )
    kind = exception.get("exception_type")
    executed = bool(result.get("agent_execution"))
    if not executed:
        phase = "setup_failure"
    elif kind not in (None, "AgentTimeoutError"):
        phase = "verifier_failure" if kind == "VerifierTimeoutError" else "execution_failure"
    elif measured_reward is None:
        phase = "verifier_failure"
    else:
        phase = "agent_timeout_verified" if kind == "AgentTimeoutError" else "verified"
    valid = phase in ("verified", "agent_timeout_verified")
    cost = agent.get("cost_usd")
    if not isinstance(cost, (int, float)) or isinstance(cost, bool) or not math.isfinite(cost) or cost < 0:
        # A setup failure is known not to have invoked the agent, rather than
        # an unreported model bill that happens to look like zero.
        cost = 0.0 if not executed and not agent else None
    return {
        "valid": valid,
        "phase": phase,
        "reward": measured_reward,
        "cost_usd": cost,
        "agent_execution_started": executed,
        "exception_type": kind,
        "error": str(exception.get("exception_message") or ""),
        "started_at": result.get("started_at"),
        "finished_at": result.get("finished_at"),
        "timing": {
            key: result.get(key) for key in ("environment_setup", "agent_setup", "agent_execution", "verifier")
        },
        "usage": {key: agent.get(key) for key in ("n_input_tokens", "n_output_tokens", "n_cache_tokens")},
        "task_config": (result.get("config") or {}).get("task"),
        "task_id": result.get("task_id"),
    }
