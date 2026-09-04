"""The reproduction driver's iteration loop.

The loop stands in for Reef's scenario: it has to commit each iteration's
state before the next one begins, because ``settle_step`` returns speculative
state and aborts the staged population. A driver that skips that runs one
iteration and then dies on 'algorithm state differs from the last applied
commit', which is what happened on the first scaled run.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from recipes.meta_harness.examples.terminal_bench.budget import ObservedCostLedger
from recipes.meta_harness.examples.terminal_bench.run import search


class _Prepared:
    def __init__(self, outcome: str, state: dict, candidate: Any = "candidate") -> None:
        self.outcome = outcome
        self.state = state
        self.candidate = candidate
        self.metrics: dict = {}


class _Result:
    def __init__(self, state: dict) -> None:
        self.state = state


class _Decision:
    selected = False
    metrics: dict = {"candidate_mean": 0.0}


class _Backend:
    """Records the order of the calls the loop makes."""

    def __init__(self, outcome: str = "propose") -> None:
        self.calls: list[str] = []
        self.outcome = outcome
        self.step = 0

    def prepare_step(self, batch, state, iteration):
        self.calls.append(f"prepare:{iteration}")
        self.step = iteration
        return _Prepared(self.outcome, {"step": iteration})

    def settle_step(self, prepared, decision):
        self.calls.append(f"settle:{self.step}")
        return _Result({"step": self.step, "settled": True})

    def commit_applied(self, state):
        self.calls.append(f"commit:{state['step']}")

    def abort_step(self, prepared):
        self.calls.append("abort")


class _Evaluator:
    def evaluate(self, candidate):
        return "evaluation"

    def decide(self, candidate, evaluation):
        return _Decision()


@pytest.mark.unit
def test_each_iteration_is_committed_before_the_next_one_begins(tmp_path: Path) -> None:
    backend = _Backend()
    ledger = ObservedCostLedger(tmp_path / "spend.json", 10.0)
    search(backend, _Evaluator(), ledger, {}, iterations=3, log=io.StringIO())
    assert backend.calls == [
        "prepare:1",
        "settle:1",
        "commit:1",
        "prepare:2",
        "settle:2",
        "commit:2",
        "prepare:3",
        "settle:3",
        "commit:3",
    ]


@pytest.mark.unit
def test_the_loop_carries_the_committed_state_forward(tmp_path: Path) -> None:
    backend = _Backend()
    ledger = ObservedCostLedger(tmp_path / "spend.json", 10.0)
    state: dict = {}
    search(backend, _Evaluator(), ledger, state, iterations=2, log=io.StringIO())
    assert state == {"step": 2, "settled": True}


@pytest.mark.unit
def test_a_skipped_iteration_neither_settles_nor_commits(tmp_path: Path) -> None:
    backend = _Backend(outcome="skip")
    ledger = ObservedCostLedger(tmp_path / "spend.json", 10.0)
    search(backend, _Evaluator(), ledger, {}, iterations=2, log=io.StringIO())
    assert backend.calls == ["prepare:1", "prepare:2"]


@pytest.mark.unit
def test_the_spend_cap_stops_the_loop_before_the_next_iteration(tmp_path: Path) -> None:
    backend = _Backend()
    ledger = ObservedCostLedger(tmp_path / "spend.json", 1.0)
    ledger.record_trial("prior", 1.5)
    search(backend, _Evaluator(), ledger, {}, iterations=3, log=io.StringIO())
    assert backend.calls == []


@pytest.mark.unit
def test_a_failed_evaluation_aborts_rather_than_committing(tmp_path: Path) -> None:
    class _Exploding(_Evaluator):
        def evaluate(self, candidate):
            raise RuntimeError("evaluation failed")

    backend = _Backend()
    ledger = ObservedCostLedger(tmp_path / "spend.json", 10.0)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        search(backend, _Exploding(), ledger, {}, iterations=2, log=io.StringIO())
    assert backend.calls == ["prepare:1", "abort"]
