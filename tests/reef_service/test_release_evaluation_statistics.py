"""Comparison units are tasks, not repeated samples from the same task."""

import json

import pytest
from tutorials.release_evaluation.run import final_answer

from reef.scenario.evaluation import EvaluationOutcome, episode_record, paired_summary


def row(release, task, repeat, score, status="scored"):
    return EvaluationOutcome(repeat, release, task, repeat, status, score, 0.0)


def test_task_clustering_prevents_repeat_count_from_changing_task_weights():
    outcomes = []
    for repeat in range(10):
        outcomes += [row("base", "A", repeat, 0), row("new", "A", repeat, 1)]
    outcomes += [row("base", "B", 0, 1), row("new", "B", 0, 0)]
    summary = paired_summary(outcomes, "new", "base", ["A", "B"])
    assert summary.paired_tasks == 2
    assert summary.paired_runs == 11
    assert summary.mean_delta == 0  # Not 9 / 11 from treating repetitions as independent tasks.
    assert summary.task_bootstrap_95_percent_interval == (-1, 1)
    assert (summary.improved_tasks, summary.regressed_tasks) == (1, 1)


def test_failed_and_unrun_pairs_do_not_become_zero_or_narrow_the_interval():
    outcomes = [row("base", task, 0, 1) for task in ("valid", "error", "unrun")]
    outcomes += [
        row("new", "valid", 0, 0),
        row("new", "error", 0, None, "execution_error"),
        row("new", "unrun", 0, None, "unrun"),
    ]
    summary = paired_summary(outcomes, "new", "base", ["valid", "error", "unrun"])
    assert (summary.paired_tasks, summary.planned_tasks, summary.mean_delta) == (1, 3, -1)
    assert summary.task_bootstrap_95_percent_interval is None


def test_missing_native_usage_is_unknown_not_zero(tmp_path):
    session = tmp_path / "session.jsonl"
    complete = {"type": "assistant/message", "data": {"usage": {"input_tokens": 12, "output_tokens": 3}}}
    session.write_text(json.dumps(complete) + "\n")
    _, usage = episode_record(tmp_path, "native-jsonl")
    assert usage == {"input_tokens": 12, "output_tokens": 3, "model_responses": 1}
    with session.open("a") as stream:
        stream.write(json.dumps({"type": "assistant/message", "data": {"content": "unknown usage"}}) + "\n")
    _, usage = episode_record(tmp_path, "native-jsonl")
    assert usage == {"input_tokens": None, "output_tokens": None, "model_responses": 2}
    _, unsupported = episode_record(tmp_path, "pi-session-jsonl")
    assert unsupported == {"input_tokens": None, "output_tokens": None, "model_responses": None}


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("Option (A) is tempting, but incorrect.\nFINAL: (B)", "(B)"),
        ("I considered (B).", None),
        ("FINAL: (B) or (C)", None),
        ("FINAL: (B)\nActually, let me revise.\nFINAL: (C)", "(C)"),
    ],
)
def test_benchmark_scorer_requires_explicit_final_answer(answer, expected):
    assert final_answer(answer) == expected
