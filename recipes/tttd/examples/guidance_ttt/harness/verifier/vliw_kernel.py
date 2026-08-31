from __future__ import annotations

from typing import Any

from ..state import VerificationResult
from .edgebench_adapter import (
    EdgeBenchEnvironmentError,
    evaluate_vliw_solution,
    vliw_training_reward,
)
from .python_code import extract_python_solution_code


def extract_vliw_solution_code(text: str) -> str | None:
    return extract_python_solution_code(text)


def verify_vliw_solution_text(
    text: str,
    *,
    timeout_s: int,
    config: dict[str, Any] | None = None,
) -> VerificationResult:
    code = extract_vliw_solution_code(text)
    if code is None:
        return VerificationResult(
            reward=0.0,
            raw_score=None,
            valid=False,
            status="parse_error",
            message="No complete Python code block found in <solution>",
            artifacts={},
        )
    if "class KernelBuilder" not in code or "def build_kernel" not in code:
        return VerificationResult(
            reward=0.0,
            raw_score=None,
            valid=False,
            status="parse_error",
            message="solution.py must define class KernelBuilder and build_kernel",
            artifacts={"code": code},
        )
    try:
        result = evaluate_vliw_solution(code, timeout_s=timeout_s, config=config or {})
    except EdgeBenchEnvironmentError as exc:
        return VerificationResult(
            reward=0.0,
            raw_score=None,
            valid=False,
            status="environment_error",
            message=str(exc),
            artifacts={"code": code},
        )
    artifacts = {"code": code, **result.artifacts}
    if not result.valid:
        return VerificationResult(
            reward=0.0,
            raw_score=None,
            valid=False,
            status="invalid",
            message=result.message,
            artifacts=artifacts,
        )
    reward_mode = str((config or {}).get("reward_mode", "inverse_cycles"))
    reward_scale = float((config or {}).get("reward_scale", 1_000_000.0))
    try:
        training_reward = vliw_training_reward(
            result.cycles,
            mode=reward_mode,
            scale=reward_scale,
        )
    except ValueError as exc:
        return VerificationResult(
            reward=0.0,
            raw_score=None,
            valid=False,
            status="environment_error",
            message=str(exc),
            artifacts=artifacts,
        )
    artifacts.update(
        {
            "official_normalized_score": result.normalized_score,
            "training_reward_mode": reward_mode,
            "training_reward_scale": reward_scale,
        }
    )
    return VerificationResult(
        reward=training_reward,
        raw_score=result.cycles,
        valid=True,
        status="valid",
        message=f"{result.message}; training reward={training_reward:.6f} ({reward_mode})",
        artifacts=artifacts,
    )
