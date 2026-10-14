"""Tasks split by the records they came from: shared sources stay in one split, the seed fixes the draw."""

from __future__ import annotations

from pathlib import Path

import pytest

from reef.core.tasks import (
    TaskSplit,
    TaskSplitError,
    manifest_task_paths,
    read_split_manifest,
    split_by_source,
    write_split_manifest,
)

SOURCES = {
    "t1": ["r1", "r2"],
    "t2": ["r2"],
    "t3": ["r3"],
    "t4": ["r4", "r5"],
    "t5": ["r5"],
    "t6": [],
    "t7": ["r7"],
    "t8": ["r8"],
}


def test_tasks_sharing_a_record_land_on_the_same_side() -> None:
    for seed in range(20):
        split = split_by_source(SOURCES, eval_fraction=0.5, seed=seed)
        for pair in (("t1", "t2"), ("t4", "t5")):
            in_eval = {name in split.eval for name in pair}
            assert len(in_eval) == 1, (seed, pair, split)


def test_every_task_lands_on_exactly_one_side() -> None:
    split = split_by_source(SOURCES, eval_fraction=0.4, seed=7)
    assert sorted(split.train + split.eval) == sorted(SOURCES)
    assert not set(split.train) & set(split.eval)


def test_the_seed_fixes_the_draw_and_a_new_seed_changes_it() -> None:
    a = split_by_source(SOURCES, eval_fraction=0.5, seed=1)
    b = split_by_source(SOURCES, eval_fraction=0.5, seed=1)
    assert a == b
    draws = {split_by_source(SOURCES, eval_fraction=0.5, seed=seed).eval for seed in range(30)}
    assert len(draws) > 1


def test_the_order_of_the_input_does_not_change_the_draw() -> None:
    reversed_sources = dict(reversed(list(SOURCES.items())))
    assert split_by_source(SOURCES, eval_fraction=0.5, seed=3) == split_by_source(
        reversed_sources, eval_fraction=0.5, seed=3
    )


def test_the_eval_side_holds_at_least_the_fraction() -> None:
    for seed in range(10):
        split = split_by_source(SOURCES, eval_fraction=0.25, seed=seed)
        assert len(split.eval) >= 2
        # One whole group can overshoot the target, never more than the largest group.
        assert len(split.eval) <= 2 + 1


def test_zero_and_one_are_the_two_trivial_splits() -> None:
    nothing = split_by_source(SOURCES, eval_fraction=0, seed=0)
    assert nothing.eval == () and nothing.train == tuple(sorted(SOURCES))
    everything = split_by_source(SOURCES, eval_fraction=1, seed=0)
    assert everything.train == () and everything.eval == tuple(sorted(SOURCES))


def test_no_tasks_is_an_empty_split() -> None:
    assert split_by_source({}, eval_fraction=0.5, seed=0) == TaskSplit((), (), 0, 0.5)


def test_one_group_goes_whole() -> None:
    linked = {"a": ["r"], "b": ["r"], "c": ["r"]}
    assert split_by_source(linked, eval_fraction=0.1, seed=5).eval == ("a", "b", "c")


@pytest.mark.parametrize(
    ("sources", "fraction", "seed", "message"),
    [
        (SOURCES, 1.5, 0, "between 0 and 1"),
        (SOURCES, -0.1, 0, "between 0 and 1"),
        (SOURCES, True, 0, "between 0 and 1"),
        (SOURCES, 0.5, "0", "seed must be an integer"),
        (SOURCES, 0.5, True, "seed must be an integer"),
        ({"": ["r"]}, 0.5, 0, "non-empty task names"),
        ({"t": "r1"}, 0.5, 0, "sequence of non-empty record ids"),
        ({"t": [""]}, 0.5, 0, "sequence of non-empty record ids"),
    ],
)
def test_a_bad_request_is_refused(sources: dict[str, list[str]], fraction: object, seed: object, message: str) -> None:
    with pytest.raises(TaskSplitError, match=message):
        split_by_source(sources, eval_fraction=fraction, seed=seed)  # type: ignore[arg-type]


def test_a_task_in_both_splits_is_refused() -> None:
    with pytest.raises(TaskSplitError, match="both the train split and the eval split"):
        TaskSplit(("a",), ("a",), 0, 0.5)


def test_one_shot_source_iterables_still_group() -> None:
    linked = {"a": iter(["r"]), "b": map(str, ["r"]), "c": (x for x in ["r"])}
    split = split_by_source(linked, eval_fraction=0.34, seed=0)  # one shot iterables, materialized once
    assert split.eval in ((), ("a", "b", "c"))
    assert split.eval == ("a", "b", "c")


@pytest.mark.parametrize("value", [None, 3, "r1", [""], [1]])
def test_a_source_value_that_is_not_a_sequence_of_ids_is_refused(value: object) -> None:
    with pytest.raises(TaskSplitError, match="sequence of non-empty record ids"):
        split_by_source({"a": value, "b": ["r"]}, eval_fraction=0.5, seed=0)  # type: ignore[dict-item]


def test_a_tiny_positive_fraction_still_holds_one_group() -> None:
    split = split_by_source(SOURCES, eval_fraction=1e-12, seed=0)
    assert len(split.eval) >= 1


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
    split = split_by_source(SOURCES, eval_fraction=0.5, seed=11)
    write_split_manifest(tmp_path / "split.json", split)
    assert read_split_manifest(tmp_path / "split.json") == split
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


def test_sets_and_dict_views_are_accepted_as_sources() -> None:
    split = split_by_source({"a": {"r"}, "b": {"r": 1}.keys()}, eval_fraction=0.5, seed=0)
    assert split.eval == ("a", "b")


def test_a_manifest_write_that_fails_keeps_the_old_manifest_and_raises_a_split_error(tmp_path: Path) -> None:
    first = split_by_source(SOURCES, eval_fraction=0.5, seed=1)
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
    first = split_by_source(SOURCES, eval_fraction=0.5, seed=1)
    second = split_by_source(SOURCES, eval_fraction=0.5, seed=2)
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
