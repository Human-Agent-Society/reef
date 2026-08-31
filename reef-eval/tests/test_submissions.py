"""Submission-log parsing: the judge's history becomes trace points."""

from reef_eval.submissions import (
    AGENT_WRITABLE,
    SUBMISSIONS_LOG,
    find_submissions_log,
    load_trace,
    parse_submissions,
)


def test_parse_sorts_and_keeps_extras():
    text = "\n".join(
        [
            '{"t": 5.0, "score": 0.6, "n": 1}',
            '{"t": 1.0, "score": 0.2, "n": 0}',
        ]
    )
    points = parse_submissions(text)
    assert [(p.t, p.score) for p in points] == [(1.0, 0.2), (5.0, 0.6)]
    assert points[0].data == {"n": 0}


def test_parse_is_lenient_about_bad_lines():
    text = "\n".join(
        [
            "not json",
            '{"t": "oops", "score": 1}',
            '{"score": 0.4}',
            '{"t": 2.0, "score": 0.4}',
            "",
        ]
    )
    points = parse_submissions(text)
    assert [(p.t, p.score) for p in points] == [(2.0, 0.4)]


def test_verifier_log_outranks_an_agent_written_one(tmp_path):
    """The trusted log lives under verifier/. A file the agent left behind
    sorts earlier by name, and must not be picked up in its place."""
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / SUBMISSIONS_LOG).write_text('{"t": 1, "score": 9.9}\n')
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier" / SUBMISSIONS_LOG).write_text('{"t": 1, "score": 0.5}\n')

    assert find_submissions_log(tmp_path) == tmp_path / "verifier" / SUBMISSIONS_LOG
    assert [p.score for p in load_trace(tmp_path)] == [0.5]


def test_falls_back_when_there_is_no_verifier_dir(tmp_path):
    nested = tmp_path / "logs"
    nested.mkdir()
    (nested / SUBMISSIONS_LOG).write_text('{"t": 1, "score": 0.5}\n')
    assert find_submissions_log(tmp_path) == nested / SUBMISSIONS_LOG


def test_an_agent_written_log_is_never_the_trace(tmp_path):
    """With no verifier directory the search falls back to the whole trial,
    and the agent's own output is inside it. Accepting a log the agent wrote
    would let it hand in its own anytime curve."""
    for directory in AGENT_WRITABLE:
        (tmp_path / directory).mkdir()
        (tmp_path / directory / SUBMISSIONS_LOG).write_text('{"t": 1, "score": 9.9}\n')

    assert find_submissions_log(tmp_path) is None
    assert load_trace(tmp_path) == []
