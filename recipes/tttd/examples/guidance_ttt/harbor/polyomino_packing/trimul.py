from __future__ import annotations

from importlib.resources import files

from ..state import LibraryNode, make_root_node
from .trimul_prompt import TRIMUL_PROMPT

_ASSET_DIR = files(__package__).joinpath("assets", "trimul")
TRIMUL_BASELINE_SOLUTION = _ASSET_DIR.joinpath("baseline_solution.py").read_text(encoding="utf-8")
TRIMUL_BASELINE_SUMMARY = _ASSET_DIR.joinpath("baseline_summary.md").read_text(encoding="utf-8").strip()


# The task statement and rules are the original TTT-Discover prompt. The
# framework transports the selected parent separately through <parent_code>, so
# the dynamic State.to_prompt code block is intentionally not duplicated here.
TRIMUL_PROBLEM_PROMPT = f"""{TRIMUL_PROMPT.rstrip()}

Rules:
- The tensors arguments passed in will be already on your cuda device.
- Define all of your code in one final ```python ``` block.
- We will test the correctness of your kernel on multiple input shapes, make sure to support different potential test cases.
- You are allowed to use mixed precision computations, but make sure your final output is in float32.
- You must use trition 3.3.1 and these kernels will be run on an H100.
- You do not have to implement everything in triton, you may choose to have some of the operations done in pytorch. However, you must implement at least part of the operations in a kernel.
- Include a short docstring at the top summarizing your algorithm.
"""


def create_root_node(*, raw_score: float | None = None, reward: float = 0.0) -> LibraryNode:
    node = make_root_node(problem_id="trimul", raw_score=raw_score, reward=reward)
    node.metadata.update(
        {
            "task": "trimul",
            "metric": "geometric_mean_runtime_us",
            "score_direction": "min",
            "evaluation_gpu": "H100",
            "triton_version": "3.3.1",
        }
    )
    return node
