"""Task registry for Reef's summary-only Guidance-TTT example."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from ..state import LibraryNode, VerificationResult
from ..verifier.polyomino import extract_cpp_solution_code, verify_polyomino_solution_text
from ..verifier.trimul import extract_trimul_solution_code, verify_trimul_solution_text
from ..verifier.vliw_kernel import extract_vliw_solution_code, verify_vliw_solution_text
from .polyomino import POLYOMINO_PROBLEM_PROMPT, create_root_node as create_polyomino_root_node
from .trimul import TRIMUL_PROBLEM_PROMPT, create_root_node as create_trimul_root_node
from .vliw_kernel import (
    VLIW_BASELINE_SOLUTION,
    VLIW_BASELINE_SUMMARY,
    VLIW_KERNEL_PROBLEM_PROMPT,
    create_root_node as create_vliw_root_node,
)

ScoreDirection = Literal["min", "max"]
GuidanceObjective = Callable[[float | None], str]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    problem_prompt: str
    solution_language: str
    execution_solution_contract: str
    guidance_mechanism_constraint: str
    score_direction: ScoreDirection
    raw_score_label: str
    create_root_node: Callable[..., LibraryNode]
    verifier: Callable[..., VerificationResult]
    solution_extractor: Callable[[str], str | None]
    guidance_objective: GuidanceObjective
    bootstrap_solution: str | None = None
    bootstrap_summary: str | None = None

    def verify_execution_text(
        self,
        text: str,
        *,
        timeout_s: int,
        config: dict[str, Any] | None = None,
    ) -> VerificationResult:
        return self.verifier(text, timeout_s=timeout_s, config=config or {})


def _verify_polyomino(
    text: str,
    *,
    timeout_s: int,
    config: dict[str, Any] | None = None,
) -> VerificationResult:
    frontiercs_config = dict(config or {})
    problem_id = str(frontiercs_config.pop("problem_id", "0"))
    return verify_polyomino_solution_text(
        text,
        problem_id=problem_id,
        timeout_s=timeout_s,
        config=frontiercs_config,
    )


def _polyomino_objective(target: float | None) -> str:
    _ = target
    return (
        "Provide the next evolutionary guidance that can improve the FrontierCS score. "
        "Higher FrontierCS score is better."
    )


def _vliw_objective(target: float | None) -> str:
    _ = target
    return (
        "Provide the next evolutionary guidance that preserves correctness while reducing the selected "
        "candidate's simulator cycles. Lower raw cycles and higher inverse-cycle reward are better."
    )


def _trimul_objective(target: float | None) -> str:
    _ = target
    return (
        "Provide the next evolutionary guidance that preserves numerical correctness while reducing the selected "
        "TriMul candidate's geometric-mean H100 runtime. Lower runtime in microseconds and higher inverse-runtime "
        "reward are better."
    )


POLYOMINO_TASK = TaskSpec(
    task_id="polyomino_packing",
    problem_prompt=POLYOMINO_PROBLEM_PROMPT,
    solution_language="cpp",
    execution_solution_contract=(
        "The <solution> block must contain one complete C++17 program in a ```cpp fenced block. "
        "It must read the Polyomino Packing instance from stdin and write the placement to stdout."
    ),
    guidance_mechanism_constraint=(
        "Every proposed mechanism must be implementable inside one self-contained C++17 program using only "
        "the current input instance. Do not rely on offline training data, benchmark access, external models, "
        "APIs, learned weights, or unavailable precomputation."
    ),
    score_direction="max",
    raw_score_label="FrontierCS score",
    create_root_node=create_polyomino_root_node,
    verifier=_verify_polyomino,
    solution_extractor=extract_cpp_solution_code,
    guidance_objective=_polyomino_objective,
)

VLIW_TASK = TaskSpec(
    task_id="vliw_kernel_optimization",
    problem_prompt=VLIW_KERNEL_PROBLEM_PROMPT,
    solution_language="python",
    execution_solution_contract=(
        "The <solution> block must contain one complete replacement solution.py in a ```python fenced block. "
        "It must preserve the required KernelBuilder interface and may modify no other file."
    ),
    guidance_mechanism_constraint=(
        "Every proposal must be implementable entirely in solution.py against the supplied public VLIW/SIMD "
        "ISA. Guidance may specify compiler-level mechanisms such as dependency-aware bundle packing, SIMD "
        "batching, software pipelining, unrolling, scratch allocation, and memory/compute overlap, but must not "
        "write Python code or concrete instruction arrays. Do not rely on external models, APIs, network access, "
        "hidden cases, seed-specific constants, or changes to protected benchmark files."
    ),
    score_direction="min",
    raw_score_label="Simulator cycles",
    create_root_node=create_vliw_root_node,
    verifier=verify_vliw_solution_text,
    solution_extractor=extract_vliw_solution_code,
    guidance_objective=_vliw_objective,
    bootstrap_solution=VLIW_BASELINE_SOLUTION,
    bootstrap_summary=VLIW_BASELINE_SUMMARY,
)

TRIMUL_TASK = TaskSpec(
    task_id="trimul",
    problem_prompt=TRIMUL_PROBLEM_PROMPT,
    solution_language="python",
    execution_solution_contract=(
        "The <solution> block must contain one complete replacement submission.py in a ```python fenced block. "
        "It must define custom_kernel(data), contain at least one @triton.jit kernel, support every official "
        "shape and mask/distribution case, and return a float32 output tensor."
    ),
    guidance_mechanism_constraint=(
        "Every proposal must be implementable in one self-contained submission.py using PyTorch and Triton "
        "3.3.1 on an H100. Guidance may propose fusion boundaries, tensor layouts, precision choices, tiling, "
        "persistent scheduling, contraction decomposition, launch reduction, or shape-specialized dispatch, "
        "but must remain concrete enough for the executor to implement. Scope each attempt to one primary "
        "optimization mechanism, or one tightly coupled change set, and preserve every unrelated stage of the "
        "verified parent. Do not bundle independent speculative rewrites across multiple pipeline stages. Do "
        "not rely on offline training, external models or APIs, hidden-case access, generated binary artifacts, "
        "or hardware other than the stated H100 environment."
    ),
    score_direction="min",
    raw_score_label="H100 geometric-mean runtime (us)",
    create_root_node=create_trimul_root_node,
    verifier=verify_trimul_solution_text,
    solution_extractor=extract_trimul_solution_code,
    guidance_objective=_trimul_objective,
)

_TASKS = {task.task_id: task for task in (POLYOMINO_TASK, VLIW_TASK, TRIMUL_TASK)}


def get_task_spec(task_id: str = "polyomino_packing") -> TaskSpec:
    try:
        return _TASKS[task_id]
    except KeyError as exc:
        known = ", ".join(sorted(_TASKS))
        raise KeyError(f"Unknown Guidance-TTT task: {task_id!r}; known tasks: {known}") from exc


def task_ids() -> tuple[str, ...]:
    return tuple(sorted(_TASKS))
