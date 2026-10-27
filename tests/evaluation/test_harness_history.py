"""The history of harness evolution: what it keeps, what the cache reuses, and how it survives a crash."""

from __future__ import annotations

from pathlib import Path

from reef.evaluation.harness import DecisionRecord, EpisodeRecord, FailureObservation, History
from reef.evaluation.harness.history import EPISODES_FILE
from reef.harness.tree import HarnessTree

TREE = HarnessTree({"rules.md": "Answer briefly.\n"})


def episode(history: History, score: float | None, *, repeat: int = 0) -> EpisodeRecord:
    identifier, _ = history.new_episode()
    failure = None if score is not None else FailureObservation("task", "launch", "the binary is missing")
    return EpisodeRecord(identifier, TREE.checksum(), "task", "key", repeat, "context", 1, score, failure)


def test_the_cache_reuses_scored_episodes_only_and_survives_a_reopen(tmp_path: Path) -> None:
    history = History.open(tmp_path)
    history.add_candidate(TREE, step=1, origin="seed")
    failed, scored = episode(history, None), episode(history, 0.5)
    history.add_episode(failed)
    assert history.cached(TREE.checksum(), "key", 0, "context") is None
    history.add_episode(scored)
    reopened = History(tmp_path)
    assert reopened.cached(TREE.checksum(), "key", 0, "context") == scored
    assert reopened.cached(TREE.checksum(), "key", 0, "another context") is None
    assert reopened.episodes == [failed, scored] and reopened.tree(TREE.checksum()) == TREE


def test_the_scores_of_a_tree_are_the_latest_episode_of_each_task_and_repeat(tmp_path: Path) -> None:
    history = History.open(tmp_path)
    history.add_candidate(TREE, step=1, origin="seed")
    for score, repeat in ((0.5, 0), (1.0, 0), (0.25, 1), (None, 1)):
        history.add_episode(episode(history, score, repeat=repeat))
    # Repeat 0 takes its latest score; the latest episode of repeat 1 could not run; no episode ran "other".
    assert history.scores(TREE.checksum(), ("task", "other"), repeats=2) == (1.0, None, None, None)


def test_the_decision_on_a_step_is_its_latest(tmp_path: Path) -> None:
    history = History.open(tmp_path)
    rejected = DecisionRecord(1, "tree-a", "head", False, "compare", "lost")
    retried = DecisionRecord(1, "tree-b", "head", True, "compare", "won")
    history.add_decision(rejected)
    history.add_decision(retried)
    assert History(tmp_path).decision(1) == retried and history.decision(2) is None


def test_a_torn_last_line_is_cut_and_new_episodes_skip_left_over_directories(tmp_path: Path) -> None:
    history = History.open(tmp_path)
    history.add_candidate(TREE, step=1, origin="seed")
    first = episode(history, 1.0)
    history.add_episode(first)
    with (tmp_path / EPISODES_FILE).open("a", encoding="utf-8") as handle:
        handle.write('{"episode": "00000002", "tre')
    (tmp_path / "episodes" / "00000007").mkdir(parents=True)
    # A reader skips the torn line; the writer's open cuts it, so the appends after it start lines of their own.
    assert History(tmp_path).episodes == [first]
    reopened = History.open(tmp_path)
    second = episode(reopened, 0.5)
    assert second.episode == "00000008"
    reopened.add_episode(second)
    again = History.open(tmp_path)
    third = episode(again, None)
    again.add_episode(third)
    assert History.open(tmp_path).episodes == [first, second, third]


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
