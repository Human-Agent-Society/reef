"""Reliability-panel selection from a seed evaluation."""

from __future__ import annotations

import pytest

from recipes.meta_harness.examples.terminal_bench.select_panel import partition, per_task_scores


@pytest.mark.unit
def test_scores_split_back_into_tasks_in_pairing_order() -> None:
    # Reef emits task-major with repeats inside each task.
    assert per_task_scores([1.0, 0.0, 0.0, 0.0], ["a", "b"]) == {"a": [1.0, 0.0], "b": [0.0, 0.0]}


@pytest.mark.unit
def test_a_vector_that_does_not_divide_is_refused() -> None:
    with pytest.raises(ValueError, match="do not divide"):
        per_task_scores([1.0, 0.0, 1.0], ["a", "b"])


@pytest.mark.unit
def test_only_tasks_the_seed_sometimes_passes_are_in_the_panel() -> None:
    # A task the seed always passes and one it always fails each add the same
    # constant to every mean, so neither can move a comparison.
    results = {"always": [1.0, 1.0], "never": [0.0, 0.0], "sometimes": [1.0, 0.0]}
    always_pass, always_fail, panel = partition(results)
    assert always_pass == ["always"]
    assert always_fail == ["never"]
    assert panel == ["sometimes"]


@pytest.mark.unit
def test_a_seed_that_decides_every_task_leaves_an_empty_panel() -> None:
    _, _, panel = partition({"a": [1.0, 1.0], "b": [0.0, 0.0]})
    assert panel == []
