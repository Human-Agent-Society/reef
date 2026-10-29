"""The history of harness evolution: what it keeps, what the cache reuses, and how it survives a crash."""

from __future__ import annotations

from pathlib import Path

from reef.core.evaluation import RolloutScore
from reef.evaluation.harness import DecisionRecord, FailureObservation, History, RolloutRecord
from reef.evaluation.harness.history import ROLLOUTS_FILE
from reef.harness.tree import HarnessTree

TREE = HarnessTree({"rules.md": "Answer briefly.\n"})


def rollout(history: History, score: float | None, *, repeat: int = 0) -> RolloutRecord:
    identifier, _ = history.new_rollout()
    failure = None if score is not None else FailureObservation("task", "launch", "the binary is missing")
    value = RolloutScore.failed() if score is None else RolloutScore.scored(score)
    return RolloutRecord(identifier, TREE.checksum(), "task", "key", repeat, "context", 1, value, failure)


def test_the_cache_reuses_scored_rollouts_only_and_survives_a_reopen(tmp_path: Path) -> None:
    history = History.open(tmp_path)
    history.add_candidate(TREE, step=1, origin="seed")
    failed, scored = rollout(history, None), rollout(history, 0.5)
    history.add_rollout(failed)
    assert history.cached(TREE.checksum(), "key", 0, "context") is None
    history.add_rollout(scored)
    reopened = History(tmp_path)
    assert reopened.cached(TREE.checksum(), "key", 0, "context") == scored
    assert reopened.cached(TREE.checksum(), "key", 0, "another context") is None
    assert reopened.rollouts == [failed, scored] and reopened.tree(TREE.checksum()) == TREE


def test_the_scores_of_a_tree_are_the_latest_rollout_of_each_task_and_repeat(tmp_path: Path) -> None:
    history = History.open(tmp_path)
    history.add_candidate(TREE, step=1, origin="seed")
    for score, repeat in ((0.5, 0), (1.0, 0), (0.25, 1), (None, 1)):
        history.add_rollout(rollout(history, score, repeat=repeat))
    # Repeat 0 takes its latest score; the latest rollout of repeat 1 could not run; no rollout ran "other".
    assert history.scores(TREE.checksum(), ("task", "other"), repeats=2) == (1.0, None, None, None)


def test_the_decision_on_a_step_is_its_latest(tmp_path: Path) -> None:
    history = History.open(tmp_path)
    rejected = DecisionRecord(1, "tree-a", "head", False, "compare", "lost")
    retried = DecisionRecord(1, "tree-b", "head", True, "compare", "won")
    history.add_decision(rejected)
    history.add_decision(retried)
    assert History(tmp_path).decision(1) == retried and history.decision(2) is None


def test_a_torn_last_line_is_cut_and_new_rollouts_skip_left_over_directories(tmp_path: Path) -> None:
    history = History.open(tmp_path)
    history.add_candidate(TREE, step=1, origin="seed")
    first = rollout(history, 1.0)
    history.add_rollout(first)
    with (tmp_path / ROLLOUTS_FILE).open("a", encoding="utf-8") as handle:
        handle.write('{"rollout": "00000002", "tre')
    (tmp_path / "rollouts" / "00000007").mkdir(parents=True)
    # A reader skips the torn line; the writer's open cuts it, so the appends after it start lines of their own.
    assert History(tmp_path).rollouts == [first]
    reopened = History.open(tmp_path)
    second = rollout(reopened, 0.5)
    assert second.rollout == "00000008"
    reopened.add_rollout(second)
    again = History.open(tmp_path)
    third = rollout(again, None)
    again.add_rollout(third)
    assert History.open(tmp_path).rollouts == [first, second, third]


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
