"""The task-list parser both Terminal-Bench entry points share."""

from __future__ import annotations

from pathlib import Path

import pytest

from recipes.meta_harness.examples.terminal_bench.tasks import parse_tasks, read_tasks


@pytest.mark.unit
def test_commas_and_newlines_both_separate_tasks() -> None:
    assert parse_tasks("a,b\nc, d\n") == ("a", "b", "c", "d")


@pytest.mark.unit
def test_comments_and_blank_lines_are_dropped() -> None:
    text = "# provenance\n\nterminal-bench/one\nterminal-bench/two  # trailing\n"
    assert parse_tasks(text) == ("terminal-bench/one", "terminal-bench/two")


@pytest.mark.unit
def test_an_empty_list_is_reported_as_empty_not_as_one_blank_task() -> None:
    assert parse_tasks("# only a comment\n\n") == ()


@pytest.mark.unit
def test_the_shipped_task_list_carries_the_full_official_set() -> None:
    tasks = read_tasks(Path(__file__).parents[2] / "recipes/meta_harness/examples/terminal_bench/tasks-89.txt")
    assert len(tasks) == 89
    assert len(set(tasks)) == 89
    assert all(task.startswith("terminal-bench/") for task in tasks)
