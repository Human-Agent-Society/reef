from __future__ import annotations

from typing import Any

from ..state import VerificationResult
from .python_code import extract_python_solution_code
from .trimul_adapter import (
    TriMulEnvironmentError,
    evaluate_trimul_solution,
)

TRIMUL_REWARD_SCALE = 1500.0


def extract_trimul_solution_code(text: str) -> str | None:
    return extract_python_solution_code(text)


def verify_trimul_solution_text(
    text: str,
    *,
    timeout_s: int,
    config: dict[str, Any] | None = None,
) -> VerificationResult:
    code = extract_trimul_solution_code(text)
    if code is None:
        return VerificationResult(
            reward=0.0,
            raw_score=None,
            valid=False,
            status="parse_error",
            message="No complete Python code block found in <solution>",
            artifacts={},
        )
    if "def custom_kernel" not in code:
        return VerificationResult(
            reward=0.0,
            raw_score=None,
            valid=False,
            status="parse_error",
            message="TriMul submission must define custom_kernel(data)",
            artifacts={"code": code},
        )
    if "@triton.jit" not in code:
        return VerificationResult(
            reward=0.0,
            raw_score=None,
            valid=False,
            status="parse_error",
            message="Code must contain @triton.jit, matching the TTT-Discover evaluator precheck",
            artifacts={"code": code},
        )
    if "identity" in code:
        return VerificationResult(
            reward=0.0,
            raw_score=None,
            valid=False,
            status="parse_error",
            message="Identity kernels are not allowed for TriMul",
            artifacts={"code": code},
        )
    try:
        result = evaluate_trimul_solution(code, timeout_s=timeout_s, config=config or {})
    except TriMulEnvironmentError as exc:
        return VerificationResult(
            reward=0.0,
            raw_score=None,
            valid=False,
            status="environment_error",
            message=str(exc),
            artifacts={"code": code},
        )
    artifacts = {"code": code, **result.artifacts}
    if not result.valid or result.score_us is None:
        return VerificationResult(
            reward=0.0,
            raw_score=None,
            valid=False,
            status="invalid",
            message=result.message,
            artifacts=artifacts,
        )
    reward_scale = float((config or {}).get("reward_scale", TRIMUL_REWARD_SCALE))
    if reward_scale <= 0:
        return VerificationResult(
            reward=0.0,
            raw_score=None,
            valid=False,
            status="environment_error",
            message=f"reward_scale must be positive, got {reward_scale!r}",
            artifacts=artifacts,
        )
    reward = reward_scale / result.score_us
    artifacts["reward_scale"] = reward_scale
    artifacts["reward_formula"] = "reward_scale / geometric_mean_runtime_us"
    return VerificationResult(
        reward=reward,
        raw_score=result.score_us,
        valid=True,
        status="valid",
        message=f"{result.message}; reward={reward:.9f}",
        artifacts=artifacts,
    )
