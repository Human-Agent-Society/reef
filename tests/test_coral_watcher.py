"""The attempt watcher: CORAL's on-disk attempt records -> one reef report each.

Covers the real-runtime integration surface — attempt JSON layouts (single- and
multi-island), terminal-state filtering, budget classes, reference claiming at
the (agent, parent) coordinate, exactly-once reporting across restarts, and
transport-failure retry vs. loud rejection.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from recipes.coral.journal import CallJournal, CallRecord, commit_matches
from recipes.coral.watcher import AttemptWatcher, read_finalized_attempts

PARENT_A = "a" * 40
PARENT_B = "b" * 40
COMMIT_1 = "1" * 40
COMMIT_2 = "2" * 40
COMMIT_3 = "3" * 40


def _write_attempt(
    coral_dir,
    commit,
    *,
    agent="agent-1",
    score=0.5,
    status="improved",
    parent=PARENT_A,
    timestamp="2026-01-01T00:00:00+00:00",
    metadata=None,
    island=None,
):
    attempts_dir = coral_dir / "islands" / island / "attempts" if island else coral_dir / "public" / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "commit_hash": commit,
        "agent_id": agent,
        "title": "t",
        "score": score,
        "status": status,
        "parent_hash": parent,
        "timestamp": timestamp,
        "feedback": "",
    }
    if metadata:
        record["metadata"] = metadata
    (attempts_dir / f"{commit}.json").write_text(json.dumps(record), encoding="utf-8")


def _journal_with(tmp_path, *entries):
    """entries: (agent, commit12, record_id) tuples."""
    journal = CallJournal(tmp_path / "journal.jsonl")
    for i, (agent, commit, record_id) in enumerate(entries):
        journal.append(
            CallRecord(
                request_id=f"req-{i}",
                timestamp=f"2026-01-01T00:00:{i:02d}+00:00",
                scenario="s",
                agent_id=agent,
                commit_hash=commit,
                path="/v1/chat/completions",
                status_code=200,
                agent_record_id=record_id,
            )
        )
    return journal


class FakeReef:
    """Capture reports; optionally fail with a transport or HTTP error."""

    def __init__(self):
        self.reports = []
        self.fail_with = None

    def __call__(self, reef_url, report, *, token=None, timeout=30.0):
        if self.fail_with is not None:
            error, self.fail_with = self.fail_with, None
            raise error
        self.reports.append(report)
        return {"agent_record_id": f"ack-{report.commit_hash[:8]}"}


@pytest.fixture()
def fake_reef(monkeypatch):
    fake = FakeReef()
    monkeypatch.setattr("recipes.coral.watcher.report_attempt", fake)
    return fake


def _watcher(tmp_path, journal, **kwargs):
    return AttemptWatcher(
        coral_dir=tmp_path / ".coral",
        journal=journal,
        reef_url="http://reef",
        scenario="s",
        run_id="run-1",
        token="tok",
        state_path=tmp_path / "reported.json",
        **kwargs,
    )


def test_commit_matches_prefixes_but_never_placeholders():
    assert commit_matches(PARENT_A[:12], PARENT_A)
    assert commit_matches(PARENT_A, PARENT_A[:12])
    assert not commit_matches(PARENT_A[:12], PARENT_B)
    assert not commit_matches("unknown", PARENT_A)
    assert not commit_matches("", PARENT_A)


def test_reads_terminal_attempts_across_island_layouts(tmp_path):
    coral_dir = tmp_path / ".coral"
    _write_attempt(coral_dir, COMMIT_1, timestamp="2026-01-01T00:00:02+00:00")
    _write_attempt(coral_dir, COMMIT_2, island="0", timestamp="2026-01-01T00:00:01+00:00")
    _write_attempt(coral_dir, COMMIT_3, status="pending", score=None)
    (coral_dir / "public" / "attempts" / "torn.json").write_text("{not json", encoding="utf-8")

    attempts = read_finalized_attempts(coral_dir)
    # pending and torn skipped; sorted oldest-first across layouts
    assert [a.commit_hash for a in attempts] == [COMMIT_2, COMMIT_1]


def test_archived_attempts_are_ignored(tmp_path):
    coral_dir = tmp_path / ".coral"
    _write_attempt(coral_dir, COMMIT_1, metadata={"archived": True})
    assert read_finalized_attempts(coral_dir) == []


def test_reports_each_finalized_attempt_once_with_its_references(tmp_path, fake_reef):
    journal = _journal_with(
        tmp_path,
        ("agent-1", PARENT_A[:12], "rec-1"),
        ("agent-1", PARENT_A[:12], "rec-2"),
        ("agent-2", PARENT_A[:12], "rec-other-agent"),
    )
    _write_attempt(tmp_path / ".coral", COMMIT_1, agent="agent-1", parent=PARENT_A)

    watcher = _watcher(tmp_path, journal)
    posted = watcher.poll_once()

    assert len(posted) == 1
    report = posted[0]
    assert report.commit_hash == COMMIT_1
    assert report.references == ("rec-1", "rec-2")  # the other agent's record untouched
    assert report.run_id == "run-1"
    # second poll: nothing new
    assert watcher.poll_once() == []
    assert len(fake_reef.reports) == 1


def test_consecutive_attempts_from_the_same_parent_do_not_share_references(tmp_path, fake_reef):
    # A revert: two attempts by one agent, both from PARENT_A. The first
    # claims the records journaled so far; the second gets only the later ones.
    journal = _journal_with(tmp_path, ("agent-1", PARENT_A[:12], "rec-1"))
    _write_attempt(tmp_path / ".coral", COMMIT_1, parent=PARENT_A, timestamp="2026-01-01T00:00:01+00:00")
    watcher = _watcher(tmp_path, journal)
    (first,) = watcher.poll_once()
    assert first.references == ("rec-1",)

    journal.append(
        CallRecord(
            request_id="req-9",
            timestamp="2026-01-01T00:00:09+00:00",
            scenario="s",
            agent_id="agent-1",
            commit_hash=PARENT_A[:12],
            path="/v1/chat/completions",
            status_code=200,
            agent_record_id="rec-2",
        )
    )
    _write_attempt(tmp_path / ".coral", COMMIT_2, parent=PARENT_A, timestamp="2026-01-01T00:00:10+00:00")
    (second,) = watcher.poll_once()
    assert second.references == ("rec-2",)


def test_grader_error_and_tune_attempts_are_skipped_terminally(tmp_path, fake_reef):
    journal = _journal_with(tmp_path)
    _write_attempt(
        tmp_path / ".coral",
        COMMIT_1,
        status="crashed",
        score=None,
        metadata={"budget_class": "grader_error"},
    )
    _write_attempt(tmp_path / ".coral", COMMIT_2, metadata={"budget_class": "tune"})
    watcher = _watcher(tmp_path, journal)
    assert watcher.poll_once() == []
    assert fake_reef.reports == []
    # skips are terminal: still nothing on the next poll
    assert watcher.poll_once() == []


def test_state_survives_restart_no_duplicate_reports(tmp_path, fake_reef):
    journal = _journal_with(tmp_path, ("agent-1", PARENT_A[:12], "rec-1"))
    _write_attempt(tmp_path / ".coral", COMMIT_1, parent=PARENT_A)
    _watcher(tmp_path, journal).poll_once()
    assert len(fake_reef.reports) == 1

    # a fresh watcher (crash + restart) sees the persisted state
    restarted = _watcher(tmp_path, journal)
    assert restarted.poll_once() == []
    assert len(fake_reef.reports) == 1


def test_transport_error_is_retried_next_poll(tmp_path, fake_reef):
    journal = _journal_with(tmp_path, ("agent-1", PARENT_A[:12], "rec-1"))
    _write_attempt(tmp_path / ".coral", COMMIT_1, parent=PARENT_A)
    watcher = _watcher(tmp_path, journal)

    fake_reef.fail_with = OSError("connection refused")
    assert watcher.poll_once() == []
    assert fake_reef.reports == []

    (report,) = watcher.poll_once()  # retried, references intact
    assert report.references == ("rec-1",)


def test_http_rejection_raises_loudly(tmp_path, fake_reef):
    journal = _journal_with(tmp_path)
    _write_attempt(tmp_path / ".coral", COMMIT_1)
    watcher = _watcher(tmp_path, journal)
    fake_reef.fail_with = urllib.error.HTTPError("http://reef", 409, "conflict", None, None)
    with pytest.raises(urllib.error.HTTPError):
        watcher.poll_once()


def test_unscored_real_attempt_is_still_reported(tmp_path, fake_reef):
    # A real attempt that crashed carries no score; the report records the
    # outcome (the processor decides what trains).
    journal = _journal_with(tmp_path)
    _write_attempt(tmp_path / ".coral", COMMIT_1, status="crashed", score=None)
    (report,) = _watcher(tmp_path, journal).poll_once()
    assert report.score is None and report.status == "crashed"


def test_run_drains_once_after_stop(tmp_path, fake_reef):
    import threading

    journal = _journal_with(tmp_path, ("agent-1", PARENT_A[:12], "rec-1"))
    watcher = _watcher(tmp_path, journal)
    stop = threading.Event()
    stop.set()  # loop body never runs; the final drain still must
    _write_attempt(tmp_path / ".coral", COMMIT_1, parent=PARENT_A)
    watcher.run(stop, interval_s=0.01)
    assert len(fake_reef.reports) == 1
