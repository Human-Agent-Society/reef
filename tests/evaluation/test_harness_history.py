"""The history of harness evolution: what it keeps, what the cache reuses, and how it survives a crash."""

from __future__ import annotations

from pathlib import Path

from reef.core.evaluation import RolloutScore
from reef.evaluation.harness import DecisionRecord, FailureObservation, History, RunRecord
from reef.evaluation.harness.history import RUNS_FILE
from reef.harness.tree import HarnessTree

TREE = HarnessTree({"rules.md": "Answer briefly.\n"})


def task_run(history: History, score: float | None, *, repeat: int = 0) -> RunRecord:
    identifier, _ = history.new_run()
    failure = None if score is not None else FailureObservation("task", "launch", "the binary is missing")
    value = RolloutScore.failed() if score is None else RolloutScore.scored(score)
    return RunRecord(identifier, TREE.checksum(), "task", "key", repeat, "context", 1, value, failure)


def test_the_cache_reuses_scored_runs_only_and_survives_a_reopen(tmp_path: Path) -> None:
    history = History.open(tmp_path)
    history.add_candidate(TREE, step=1, origin="seed")
    failed, scored = task_run(history, None), task_run(history, 0.5)
    history.add_run(failed)
    assert history.cached(TREE.checksum(), "key", 0, "context") is None
    history.add_run(scored)
    reopened = History(tmp_path)
    assert reopened.cached(TREE.checksum(), "key", 0, "context") == scored
    assert reopened.cached(TREE.checksum(), "key", 0, "another context") is None
    assert reopened.runs == [failed, scored] and reopened.tree(TREE.checksum()) == TREE


def test_the_scores_of_a_tree_are_the_latest_run_of_each_task_and_repeat(tmp_path: Path) -> None:
    history = History.open(tmp_path)
    history.add_candidate(TREE, step=1, origin="seed")
    for score, repeat in ((0.5, 0), (1.0, 0), (0.25, 1), (None, 1)):
        history.add_run(task_run(history, score, repeat=repeat))
    # Repeat 0 takes its latest score; the latest run of repeat 1 could not run; no run completed "other".
    assert history.scores(TREE.checksum(), ("task", "other"), repeats=2) == (1.0, None, None, None)


def test_the_decision_on_a_step_is_its_latest(tmp_path: Path) -> None:
    history = History.open(tmp_path)
    rejected = DecisionRecord(1, "tree-a", "head", False, "compare", "lost")
    retried = DecisionRecord(1, "tree-b", "head", True, "compare", "won")
    history.add_decision(rejected)
    history.add_decision(retried)
    assert History(tmp_path).decision(1) == retried and history.decision(2) is None


def test_a_torn_last_line_is_cut_and_new_runs_skip_left_over_directories(tmp_path: Path) -> None:
    history = History.open(tmp_path)
    history.add_candidate(TREE, step=1, origin="seed")
    first = task_run(history, 1.0)
    history.add_run(first)
    with (tmp_path / RUNS_FILE).open("a", encoding="utf-8") as handle:
        handle.write('{"run": "00000002", "tre')
    (tmp_path / "runs" / "00000007").mkdir(parents=True)
    # A reader skips the torn line; the writer's open cuts it, so the appends after it start lines of their own.
    assert History(tmp_path).runs == [first]
    reopened = History.open(tmp_path)
    second = task_run(reopened, 0.5)
    assert second.run_id == "00000008"
    reopened.add_run(second)
    again = History.open(tmp_path)
    third = task_run(again, None)
    again.add_run(third)
    assert History.open(tmp_path).runs == [first, second, third]


def test_a_step_that_proposes_a_tree_its_method_scored_first_adds_its_summary_and_notes(tmp_path: Path) -> None:
    history = History.open(tmp_path)
    scored = history.add_candidate(TREE, step=1, origin="method", parent="head")
    proposed = history.add_candidate(TREE, step=1, origin="method", parent="head", summary="why", notes={"n": 1})
    assert (scored.summary, proposed.summary, dict(proposed.notes)) == ("", "why", {"n": 1})
    # A later step keeps the first description, and a reopen reads the latest line.
    assert history.add_candidate(TREE, step=2, origin="method", summary="other") == proposed
    assert History(tmp_path).candidates[TREE.checksum()] == proposed


def test_the_empty_tree_is_a_candidate(tmp_path: Path) -> None:
    history = History.open(tmp_path)
    record = history.add_candidate(HarnessTree({}), step=1, origin="seed")
    assert History(tmp_path).tree(record.tree) == HarnessTree({})
