"""Deterministic grader for the demo task: score = fraction of checks passed.

A real CORAL ``TaskGrader``: the grader daemon instantiates it inside the
run's private grader venv and calls :meth:`evaluate` against an isolated
checkout of the attempt's commit (``self.codebase_path``). Partial credit —
one point per check — gives sibling attempts the score contrast the
relative-reward training group needs.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from coral.grader import TaskGrader


class _Tie:
    """Orderable by key only; identity reveals which input a tie came from."""

    __slots__ = ("key", "origin")

    def __init__(self, key: int, origin: str) -> None:
        self.key = key
        self.origin = origin

    def __lt__(self, other: _Tie) -> bool:
        return self.key < other.key

    def __le__(self, other: _Tie) -> bool:
        return self.key <= other.key

    def __gt__(self, other: _Tie) -> bool:
        return self.key > other.key

    def __ge__(self, other: _Tie) -> bool:
        return self.key >= other.key

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Tie) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)


class Grader(TaskGrader):
    """Score ``merge_sorted`` from the attempt's ``solution.py``."""

    def evaluate(self) -> float:
        program = Path(self.codebase_path) / self.args.get("program_file", "solution.py")
        checks = _run_checks(program)
        if not checks:
            return 0.0
        return sum(1.0 for passed in checks if passed) / len(checks)


def _load_merge(program: Path):
    spec = importlib.util.spec_from_file_location("attempt_solution", program)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {program}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.merge_sorted


def _uses_builtin_sort(program: Path) -> bool:
    """True when the source calls ``sorted()`` or ``list.sort()``."""
    tree = ast.parse(program.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "sorted":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "sort":
                return True
    return False


def _run_checks(program: Path) -> list[bool]:
    """One boolean per check. An unloadable/broken solution scores zero."""
    try:
        no_builtin_sort = not _uses_builtin_sort(program)
        merge = _load_merge(program)
    except Exception:
        return [False] * 10

    def check(a, b, expected) -> bool:
        try:
            return merge(list(a), list(b)) == expected
        except Exception:
            return False

    results = [
        check([], [], []),
        check([1, 2, 3], [], [1, 2, 3]),
        check([], [4, 5], [4, 5]),
        check([1, 3, 5], [2, 4, 6], [1, 2, 3, 4, 5, 6]),
        check([1, 1, 2], [1, 3], [1, 1, 1, 2, 3]),
        check([-5, 0, 7], [-9, 8], [-9, -5, 0, 7, 8]),
        check(list(range(0, 200, 2)), list(range(1, 200, 2)), list(range(200))),
    ]

    # Stability: on equal keys, a's element first, original order preserved.
    a = [_Tie(1, "a0"), _Tie(2, "a1")]
    b = [_Tie(1, "b0"), _Tie(2, "b1")]
    try:
        merged = merge(list(a), list(b))
        results.append([t.origin for t in merged] == ["a0", "b0", "a1", "b1"])
    except Exception:
        results.append(False)

    # No mutation of the inputs.
    left_sorted, right_sorted = [1, 3], [2]
    try:
        merge(left_sorted, right_sorted)
        results.append(left_sorted == [1, 3] and right_sorted == [2])
    except Exception:
        results.append(False)

    # The write-the-merge-yourself rule scores only alongside functional
    # progress; a stub that merely avoids sorted() has done nothing yet.
    results.append(no_builtin_sort and any(results))

    return results
