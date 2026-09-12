"""The demo task's grader math, tested without CORAL installed.

The grader package is a real CORAL ``TaskGrader`` and imports ``coral.grader``
at module load; the repo CI has no CORAL, so a minimal stub satisfies the
import. Everything scored — the check set, partial credit, the
no-builtin-sort rule — is exercised for real.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

GRADER_PATH = (
    Path(__file__).resolve().parent.parent
    / "recipes"
    / "beta"
    / "coral"
    / "examples"
    / "coral_demo"
    / "task"
    / "grader"
    / "src"
    / "coral_demo_grader"
    / "grader.py"
)


@pytest.fixture()
def grader_module(monkeypatch):
    if "coral.grader" not in sys.modules:
        coral_pkg = types.ModuleType("coral")
        grader_pkg = types.ModuleType("coral.grader")

        class TaskGrader:  # minimal stand-in for the ABC
            def __init__(self, config):
                self.config = config

        grader_pkg.TaskGrader = TaskGrader
        coral_pkg.grader = grader_pkg
        monkeypatch.setitem(sys.modules, "coral", coral_pkg)
        monkeypatch.setitem(sys.modules, "coral.grader", grader_pkg)
    spec = importlib.util.spec_from_file_location("coral_demo_grader_under_test", GRADER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORRECT = """
def merge_sorted(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if b[j] < a[i]:
            out.append(b[j]); j += 1
        else:
            out.append(a[i]); i += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out
"""

USES_SORTED = """
def merge_sorted(a, b):
    return sorted(a + b)
"""

WRONG_ON_TIES = """
def merge_sorted(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if b[j] <= a[i]:   # ties taken from b: unstable
            out.append(b[j]); j += 1
        else:
            out.append(a[i]); i += 1
    out.extend(a[i:]); out.extend(b[j:])
    return out
"""


def _score(grader_module, tmp_path, source):
    program = tmp_path / "solution.py"
    program.write_text(source, encoding="utf-8")
    checks = grader_module._run_checks(program)
    return sum(1.0 for c in checks if c) / len(checks)


def test_correct_solution_scores_full(grader_module, tmp_path):
    assert _score(grader_module, tmp_path, CORRECT) == 1.0


def test_seed_stub_scores_zero_without_crashing(grader_module, tmp_path):
    seed = GRADER_PATH.parents[3] / "seed" / "solution.py"
    checks = grader_module._run_checks(seed)
    assert sum(checks) == 0  # NotImplementedError on every call check
    assert len(checks) == 10


def test_builtin_sort_costs_exactly_the_sort_check(grader_module, tmp_path):
    # sorted(a+b) merges correctly but breaks the no-builtin-sort rule and
    # loses the identity-based stability check (sorted is stable on keys but
    # a's elements no longer precede b's after concatenation... they do —
    # so only the sort-rule point drops).
    score = _score(grader_module, tmp_path, USES_SORTED)
    assert score == pytest.approx(0.9)


def test_partial_credit_orders_solutions(grader_module, tmp_path):
    # broken < unstable < correct: the contrast sibling groups train on.
    broken = _score(grader_module, tmp_path, "def merge_sorted(a, b):\n    return None\n")
    unstable = _score(grader_module, tmp_path, WRONG_ON_TIES)
    correct = _score(grader_module, tmp_path, CORRECT)
    assert broken < unstable < correct
