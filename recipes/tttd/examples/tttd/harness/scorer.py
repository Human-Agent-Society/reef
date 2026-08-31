"""Scoring abstractions for TTT-Discover.

A ``Scorer`` is a callable that takes a solution string and returns a
``ScoredSolution``. Two implementations are provided:

- :class:`JudgeScorer` — POSTs the solution to a judge HTTP endpoint
  (the standard Harbor task infrastructure). No task-specific Python imports.
- :class:`ProgramScorer` — runs the solution in a subprocess sandbox on the
  host and applies a reward function. Used by the direct experiment runner
  where no judge container is running.
"""

from __future__ import annotations

import json
import math
import re
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .sandbox import ProgramExecutionError, execute_program
from .search import ScoredSolution

Scorer = Callable[[str], ScoredSolution]

_CODEBLOCK_RE = re.compile(r"```python\s+([\s\S]*?)\s*```")


def _codeblock_body(codeblock: str) -> str:
    match = _CODEBLOCK_RE.search(codeblock)
    if match is None:
        raise ValueError("cannot extract Python code")
    return match.group(1).strip()


class JudgeScorer:
    """Score solutions by POSTing to a judge HTTP endpoint.

    The judge runs the program, verifies the result, and returns
    ``{"score": float, "reason": str}``. This is the standard Harbor
    pattern — the task is fully declarative (instruction.md + score.py +
    judge_server.py) and no task-specific Python code is imported.
    """

    def __init__(self, judge_url: str, *, timeout_s: float = 7200) -> None:
        self._submit_url = judge_url.rstrip("/") + "/submit"
        self._timeout_s = timeout_s

    def __call__(self, solution: str) -> ScoredSolution:
        data = _codeblock_body(solution).encode("utf-8")
        req = urllib.request.Request(
            self._submit_url,
            data=data,
            headers={"Content-Type": "text/plain"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            payload = json.loads(resp.read())
        reward = float(payload["score"])
        reason = payload.get("reason", "")
        return ScoredSolution(
            solution=solution,
            reward=reward,
            value=reward,
            metrics={},
            output=reason,
        )


class ProgramScorer:
    """Score solutions by running them in a subprocess sandbox.

    Extracts the Python code block, runs it via :func:`execute_program`,
    and applies ``reward_fn`` to the return value. The reward function
    maps the program's result to ``(reward, value, metrics)``.

    Used by the direct experiment runner (``runner.py``) where no judge
    container is running and lower-latency host-side execution is needed.
    """

    def __init__(
        self,
        reward_fn: Callable[[Any], tuple[float, float, dict[str, Any]]],
        *,
        eval_timeout_s: float = 1_100,
        num_cpus: int = 1,
        work_dir: str | Path | None = None,
        program_budget_s: float | None = None,
        executor: Callable[..., Any] | None = None,
        prelude: str = "",
    ) -> None:
        if eval_timeout_s <= 0:
            raise ValueError("eval_timeout_s must be positive")
        if num_cpus < 1:
            raise ValueError("num_cpus must be positive")
        self._reward_fn = reward_fn
        self._eval_timeout_s = float(eval_timeout_s)
        self._num_cpus = num_cpus
        self._work_dir = None if work_dir is None else Path(work_dir)
        self._program_budget_s = program_budget_s
        self._executor = executor or execute_program
        self._prelude = prelude

    def __call__(self, solution: str) -> ScoredSolution:
        code = _codeblock_body(solution)
        source = f"{self._prelude}\n\n{code}\n" if self._prelude else f"{code}\n"
        try:
            execution = self._executor(
                source,
                entrypoint="run",
                timeout_s=self._eval_timeout_s,
                max_cpus=self._num_cpus,
                work_dir=self._work_dir,
                entrypoint_kwargs=(
                    {} if self._program_budget_s is None else {"seed": 42, "budget_s": self._program_budget_s}
                ),
            )
        except ProgramExecutionError as exc:
            raise ValueError(str(exc)) from exc

        reward, value, metrics = self._reward_fn(execution.result)
        reward = float(reward)
        value = float(value)
        if not math.isfinite(reward) or not math.isfinite(value):
            raise ValueError("reward function returned a non-finite value")
        return ScoredSolution(
            solution=solution,
            reward=reward,
            value=value,
            metrics=metrics,
            output=execution.stdout,
        )
