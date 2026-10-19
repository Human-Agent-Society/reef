"""Tasks split by the records they came from: shared sources stay in one split, the seed fixes the draw."""

from __future__ import annotations

from pathlib import Path

import pytest

from reef.core.tasks import TaskSplit, TaskSplitError, manifest_task_paths, read_split_manifest, write_split_manifest
from reef.core.tasks.split import TaskSplitConflict, extend_split_by_source

SPLIT = TaskSplit(train=("t1", "t2", "t6"), eval=("t3", "t4", "t5"), seed=11, eval_fraction=0.5)


def first_split(sources: dict[str, object], eval_fraction: float = 0.5) -> TaskSplit:
    """Split new tasks when no task has a split yet."""
    previous = TaskSplit(train=(), eval=(), seed=0, eval_fraction=eval_fraction)
    return extend_split_by_source(previous=previous, previous_sources={}, sources=sources)


@pytest.mark.unit
def test_incremental_splits_keep_existing_placements_and_follow_shared_sources() -> None:
    previous = TaskSplit(train=("train-task",), eval=("eval-task",), seed=7, eval_fraction=0.5)
    previous_sources = {"train-task": ("train-source",), "eval-task": ("eval-source",)}
    extended = extend_split_by_source(
        previous=previous,
        previous_sources=previous_sources,
        sources={"new-task": ("train-source", "new-source")},
    )
    assert extended.train == ("new-task", "train-task")
    assert extended.eval == previous.eval


@pytest.mark.unit
def test_incremental_split_refuses_a_train_eval_bridge_without_changing_placements() -> None:
    previous = TaskSplit(train=("train-task",), eval=("eval-task",), seed=7, eval_fraction=0.5)
    previous_sources = {"train-task": ("train-source",), "eval-task": ("eval-source",)}
    with pytest.raises(TaskSplitConflict, match="split conflict"):
        extend_split_by_source(
            previous=previous,
            previous_sources=previous_sources,
            sources={"bridge": ("train-source", "eval-source")},
        )
    assert previous.train == ("train-task",)
    assert previous.eval == ("eval-task",)
    assert set(previous_sources) == {"train-task", "eval-task"}


@pytest.mark.unit
def test_incremental_split_detects_a_transitive_bridge() -> None:
    previous = TaskSplit(train=("train-task",), eval=("eval-task",), seed=7, eval_fraction=0.5)
    with pytest.raises(TaskSplitConflict, match="split conflict"):
        extend_split_by_source(
            previous=previous,
            previous_sources={"train-task": ("train-source",), "eval-task": ("eval-source",)},
            sources={"first": ("train-source", "shared"), "second": ("shared", "eval-source")},
        )


@pytest.mark.unit
@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_incremental_split_is_repeatable_and_respects_the_fraction_extremes(fraction: float) -> None:
    previous = TaskSplit(train=(), eval=(), seed=7, eval_fraction=fraction)
    sources = {"a": ("one",), "b": ("one",), "c": ("two",)}
    first = extend_split_by_source(previous=previous, previous_sources={}, sources=sources)
    assert first == extend_split_by_source(
        previous=previous, previous_sources={}, sources=dict(reversed(list(sources.items())))
    )
    assert ("a" in first.eval) == ("b" in first.eval)
    if fraction == 0.0:
        assert first.eval == ()
    if fraction == 1.0:
        assert first.train == ()


@pytest.mark.parametrize(
    ("sources", "message"),
    [
        ({"": ["r"]}, "non-empty task names"),
        ({"t": "r1"}, "sequence of non-empty record ids"),
        ({"t": None}, "sequence of non-empty record ids"),
        ({"t": 3}, "sequence of non-empty record ids"),
        ({"t": [""]}, "sequence of non-empty record ids"),
        ({"t": [1]}, "sequence of non-empty record ids"),
    ],
)
def test_sources_that_are_not_task_names_and_record_ids_are_refused(sources: dict[str, object], message: str) -> None:
    with pytest.raises(TaskSplitError, match=message):
        first_split(sources)


def test_tasks_that_share_a_record_in_any_iterable_get_one_split() -> None:
    linked = {"a": iter(["r"]), "b": map(str, ["r"]), "c": {"r"}, "d": {"r": 1}.keys()}
    split = first_split(linked)
    assert split.eval in ((), ("a", "b", "c", "d"))
    assert sorted(split.train + split.eval) == ["a", "b", "c", "d"]


def test_a_task_in_both_splits_is_refused() -> None:
    with pytest.raises(TaskSplitError, match="both the train split and the eval split"):
        TaskSplit(("a",), ("a",), 0, 0.5)


def test_a_split_normalizes_and_checks_itself() -> None:
    split = TaskSplit(["b", "a"], ["c"], 3, 1)  # type: ignore[arg-type]
    assert split.train == ("a", "b") and split.eval == ("c",) and split.eval_fraction == 1.0
    for train, eval_, seed, fraction, message in (
        (("a", "a"), (), 0, 0.5, "twice"),
        (("",), (), 0, 0.5, "sequence of task names"),
        ("ab", (), 0, 0.5, "sequence of task names"),
        ((), (), "0", 0.5, "seed must be"),
        ((), (), 0, 2, "between 0 and 1"),
        ((), (), 0, float("nan"), "between 0 and 1"),
    ):
        with pytest.raises(TaskSplitError, match=message):
            TaskSplit(train, eval_, seed, fraction)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------------------- manifest


def test_the_manifest_round_trips(tmp_path: Path) -> None:
    write_split_manifest(tmp_path / "split.json", SPLIT)
    assert read_split_manifest(tmp_path / "split.json") == SPLIT
    text = (tmp_path / "split.json").read_text()
    assert text.endswith("\n")
    assert '"version": 1' in text


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not json", "cannot read"),
        ('{"version": 2}', "not a version 1"),
        ('{"version": 1, "seed": 0, "eval_fraction": 0.5, "train": "a", "eval": []}', "list of task names"),
        ('{"version": 1, "seed": 0, "eval_fraction": 0.5, "train": ["a", "a"], "eval": []}', "twice"),
        ('{"version": 1, "seed": "0", "eval_fraction": 0.5, "train": [], "eval": []}', "seed must be"),
        ('{"version": 1, "seed": 0, "eval_fraction": 2, "train": [], "eval": []}', "eval_fraction"),
        ('{"version": 1, "seed": 0, "eval_fraction": 0.5, "train": ["a"], "eval": ["a"]}', "both the train split"),
    ],
)
def test_a_bad_manifest_is_refused(tmp_path: Path, text: str, message: str) -> None:
    (tmp_path / "split.json").write_text(text)
    with pytest.raises(TaskSplitError, match=message):
        read_split_manifest(tmp_path / "split.json")


def test_a_missing_manifest_is_refused(tmp_path: Path) -> None:
    with pytest.raises(TaskSplitError, match="cannot read"):
        read_split_manifest(tmp_path / "missing.json")


# ----------------------------------------------------------------------------------------------- round two


def test_a_split_built_from_one_shot_iterators_keeps_every_name() -> None:
    split = TaskSplit(iter(["b", "a"]), (x for x in ["c"]), 0, 0.5)  # type: ignore[arg-type]
    assert split.train == ("a", "b") and split.eval == ("c",)


def test_a_split_side_that_is_not_iterable_is_refused() -> None:
    with pytest.raises(TaskSplitError, match="sequence of task names"):
        TaskSplit(3, (), 0, 0.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("version", ["1.0", "true", '"1"'])
def test_a_manifest_version_that_is_not_the_integer_is_refused(tmp_path: Path, version: str) -> None:
    (tmp_path / "split.json").write_text(
        f'{{"version": {version}, "seed": 0, "eval_fraction": 0.5, "train": [], "eval": []}}'
    )
    with pytest.raises(TaskSplitError, match="not a version 1"):
        read_split_manifest(tmp_path / "split.json")


def test_a_manifest_with_keys_reef_did_not_write_is_refused(tmp_path: Path) -> None:
    (tmp_path / "split.json").write_text(
        '{"version": 1, "seed": 0, "eval_fraction": 0.5, "train": [], "eval": [], "holdout": ["a"]}'
    )
    with pytest.raises(TaskSplitError, match="keys reef did not write: holdout"):
        read_split_manifest(tmp_path / "split.json")


def test_a_manifest_write_that_fails_keeps_the_old_manifest_and_raises_a_split_error(tmp_path: Path) -> None:
    first = SPLIT
    write_split_manifest(tmp_path / "split.json", first)
    write_split_manifest(tmp_path / "missing" / "split.json", first)
    assert read_split_manifest(tmp_path / "missing" / "split.json") == first
    (tmp_path / "blocked").mkdir()
    with pytest.raises(TaskSplitError, match="cannot write split manifest"):
        write_split_manifest(tmp_path / "blocked", first)
    with pytest.raises(TaskSplitError, match="names no file"):
        write_split_manifest(Path("/"), first)
    assert read_split_manifest(tmp_path / "split.json") == first
    assert sorted(p.name for p in tmp_path.iterdir()) == [".staging", "blocked", "missing", "split.json"]
    assert list((tmp_path / ".staging").iterdir()) == []


def test_a_manifest_keeps_its_mode_and_its_symlink_when_replaced(tmp_path: Path) -> None:
    first = SPLIT
    second = TaskSplit(train=SPLIT.eval, eval=SPLIT.train, seed=2, eval_fraction=0.5)
    real = tmp_path / "runs" / "42" / "split.json"
    real.parent.mkdir(parents=True)
    write_split_manifest(real, first)
    real.chmod(0o444)
    link = tmp_path / "current.json"
    link.symlink_to(real)
    write_split_manifest(link, second)
    assert link.is_symlink() and read_split_manifest(real) == second
    assert real.stat().st_mode & 0o777 == 0o444


# ----------------------------------------------------------------------------------------------- manifest to gate


def written_tasks(root: Path, names: tuple[str, ...]) -> None:
    from reef.core.tasks import HarborTask, write_harbor_task

    for name in names:
        write_harbor_task(
            HarborTask(
                name=name,
                instruction=f"task {name}",
                tests={"test.sh": "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"},
                environment={"Dockerfile": "FROM python:3.12-slim\n"},
                source_agent_record_ids=(f"rec-{name}",),
            ),
            root,
        )


def test_the_eval_side_of_a_manifest_becomes_task_directory_paths(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    written_tasks(root, ("t1", "t2", "t3"))
    split = TaskSplit(("t1",), ("t2", "t3"), 0, 0.5)
    write_split_manifest(tmp_path / "split.json", split)
    assert manifest_task_paths(tmp_path / "split.json", root, "eval") == (root / "t2", root / "t3")
    assert manifest_task_paths(tmp_path / "split.json", root, "train") == (root / "t1",)


def test_a_manifest_task_that_is_missing_or_edited_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    written_tasks(root, ("t1",))
    write_split_manifest(tmp_path / "split.json", TaskSplit((), ("t1", "t2"), 0, 1))
    with pytest.raises(TaskSplitError, match=r"eval task 't2'.*not a task directory"):
        manifest_task_paths(tmp_path / "split.json", root, "eval")
    write_split_manifest(tmp_path / "split.json", TaskSplit((), ("t1",), 0, 1))
    (root / "t1" / "instruction.md").write_text("changed")
    with pytest.raises(TaskSplitError, match=r"eval task 't1'.*does not match its digest"):
        manifest_task_paths(tmp_path / "split.json", root, "eval")


def test_a_split_that_is_not_train_or_eval_is_refused(tmp_path: Path) -> None:
    write_split_manifest(tmp_path / "split.json", TaskSplit((), (), 0, 0))
    with pytest.raises(TaskSplitError, match="split must be"):
        manifest_task_paths(tmp_path / "split.json", tmp_path, "test")


@pytest.mark.parametrize("name", ["../x", "/abs/x", "./t1", "T1", "a b", "t1/"])
def test_a_side_name_that_is_not_a_task_directory_name_is_refused(name: str) -> None:
    with pytest.raises(TaskSplitError, match="task names"):
        TaskSplit((), (name,), 0, 1)
