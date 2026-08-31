"""reef_eval.tasks: the CLI's target resolution, available to scripts."""

from pathlib import Path

import pytest

from reef_eval import tasks

TASKS_ROOT = Path(__file__).parent.parent / "tasks"


def test_a_benchmark_name_gives_every_task_in_it():
    found = tasks("autoresearch/first-party", tasks_dir=TASKS_ROOT)
    assert [Path(t).name for t in found] == [
        "bin-packing",
        "circle-packing",
        "function-minimization",
        "string-compression",
        "symbolic-regression",
        "tsp-tour",
    ]


def test_the_bare_benchmark_name_works_too():
    """Benchmarks sit one level down, so `tasks("cl-bench")` resolves the
    same way `reef-eval stream cl-bench` does."""
    assert tasks("first-party", tasks_dir=TASKS_ROOT) == tasks(
        "autoresearch/first-party", tasks_dir=TASKS_ROOT
    )


def test_targets_concatenate_in_the_order_given():
    a, b = "autoresearch/first-party", "continual-learning/cl-bench"
    assert tasks(a, b, tasks_dir=TASKS_ROOT) == tasks(a, tasks_dir=TASKS_ROOT) + tasks(
        b, tasks_dir=TASKS_ROOT
    )


def test_the_result_is_an_ordinary_list_to_reorder_and_repeat():
    found = tasks("autoresearch/first-party", tasks_dir=TASKS_ROOT)
    picked = [found[2], found[0], found[2]]
    assert isinstance(found, list) and all(isinstance(t, str) for t in found)
    assert len(picked) == 3 and picked[0] == picked[2]


def test_a_task_directory_resolves_to_itself():
    task = str(TASKS_ROOT / "autoresearch" / "first-party" / "tsp-tour")
    assert tasks(task) == [task]


def test_an_unknown_target_raises_valueerror():
    """The CLI turns this into a SystemExit; a script gets an exception it
    can catch."""
    with pytest.raises(ValueError, match="not a task directory"):
        tasks("definitely-not-a-real-task", tasks_dir=TASKS_ROOT)
