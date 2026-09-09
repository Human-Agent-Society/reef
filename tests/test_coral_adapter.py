"""Adapter and correlation tests for the CORAL example (issue #3 acceptance list).

Covers: parallel agents, retries, late grading, duplicate reports, and
missing inference references — plus the header-stamping contract itself.
The fake downstream app plays LiteLLM+provider+reef in one: it asserts the
reef headers arrived and answers with a configurable receipt shape.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from recipes.coral.journal import CallJournal, CallRecord, deterministic_report_id
from recipes.coral.middleware import ReefGatewayMiddleware
from recipes.coral.reporter import AttemptReport


def _scope(path="/v1/chat/completions", headers=None):
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": list(headers or []),
    }


def _coral_headers(agent="agent-1", commit="abc123def456"):
    return [
        (b"authorization", b"Bearer sk-coral-agent-1-deadbeef"),
        (b"x-coral-agent-id", agent.encode()),
        (b"x-coral-session-id", commit.encode()),
    ]


class FakeDownstream:
    """The app behind the adapter: records what it saw, replies with a receipt."""

    def __init__(self, record_id="rec-001", mode="header", status=200):
        self.record_id = record_id
        self.mode = mode  # "header" | "sse" | "json-body" | "none"
        self.status = status
        self.seen_scopes = []

    async def __call__(self, scope, receive, send):
        self.seen_scopes.append(scope)
        headers = [(b"content-type", b"application/json")]
        body = b'{"choices": []}'
        if self.mode == "header":
            headers.append((b"x-reef-agent-record-id", self.record_id.encode()))
        elif self.mode == "sse":
            headers = [(b"content-type", b"text/event-stream")]
            receipt = json.dumps(
                {"object": "chat.completion.chunk", "choices": [], "reef": {"agent_record_id": self.record_id}}
            )
            body = f"data: {json.dumps({'choices': [{'delta': {'content': 'hi'}}]})}\n\ndata: {receipt}\n\ndata: [DONE]\n\n".encode()
        elif self.mode == "json-body":
            body = json.dumps({"choices": [], "reef": {"agent_record_id": self.record_id}}).encode()
        await send({"type": "http.response.start", "status": self.status, "headers": headers})
        await send({"type": "http.response.body", "body": body})


async def _call(mw, scope):
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message):
        sent.append(message)

    await mw(scope, receive, send)
    return sent


def _mw(tmp_path, downstream, scenario="coral-demo", extra_tags=None):
    journal = CallJournal(tmp_path / "calls.jsonl")
    return (
        ReefGatewayMiddleware(downstream, scenario=scenario, journal=journal, extra_tags=extra_tags),
        journal,
    )


# --------------------------------------------------------------------------
# Header stamping: reef sees scenario+tags, provider body untouched
# --------------------------------------------------------------------------


def test_stamps_scenario_and_tags_without_touching_body(tmp_path):
    downstream = FakeDownstream()
    mw, journal = _mw(tmp_path, downstream, extra_tags={"coral-run": "run-7"})
    asyncio.run(_call(mw, _scope(headers=_coral_headers())))

    (scope,) = downstream.seen_scopes
    headers = {bytes(k).decode(): bytes(v).decode() for k, v in scope["headers"]}
    assert headers["x-reef-scenario"] == "coral-demo"
    assert headers["x-reef-tag-coral-agent"] == "agent-1"
    assert headers["x-reef-tag-coral-commit"] == "abc123def456"
    assert headers["x-reef-tag-coral-run"] == "run-7"
    # provider-native parts pass through untouched
    assert headers["authorization"] == "Bearer sk-coral-agent-1-deadbeef"

    (record,) = journal.records()
    assert record.agent_record_id == "rec-001"
    assert record.agent_id == "agent-1"
    assert record.commit_hash == "abc123def456"


def test_client_cannot_smuggle_reef_headers(tmp_path):
    downstream = FakeDownstream()
    mw, _ = _mw(tmp_path, downstream)
    smuggled = [
        *_coral_headers(),
        (b"x-reef-scenario", b"evil-scenario"),
        (b"x-reef-tag-coral-agent", b"spoofed"),
    ]
    asyncio.run(_call(mw, _scope(headers=smuggled)))
    (scope,) = downstream.seen_scopes
    values = [(bytes(k).decode(), bytes(v).decode()) for k, v in scope["headers"]]
    scenarios = [v for k, v in values if k == "x-reef-scenario"]
    agents = [v for k, v in values if k == "x-reef-tag-coral-agent"]
    assert scenarios == ["coral-demo"]
    assert agents == ["agent-1"]


def test_non_api_paths_pass_through_unjournaled(tmp_path):
    downstream = FakeDownstream()
    mw, journal = _mw(tmp_path, downstream)
    asyncio.run(_call(mw, _scope(path="/health", headers=_coral_headers())))
    assert journal.records() == []


# --------------------------------------------------------------------------
# Receipt capture across response shapes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["header", "sse", "json-body"])
def test_receipt_capture(tmp_path, mode):
    downstream = FakeDownstream(record_id=f"rec-{mode}", mode=mode)
    mw, journal = _mw(tmp_path, downstream)
    asyncio.run(_call(mw, _scope(headers=_coral_headers())))
    (record,) = journal.records()
    assert record.agent_record_id == f"rec-{mode}"


def test_missing_receipt_still_journals_with_tags(tmp_path):
    """Missing inference reference: degraded mode, tags remain the key."""
    downstream = FakeDownstream(mode="none")
    mw, journal = _mw(tmp_path, downstream)
    asyncio.run(_call(mw, _scope(headers=_coral_headers())))
    (record,) = journal.records()
    assert record.agent_record_id is None
    assert record.tags["coral-agent"] == "agent-1"
    assert record.tags["coral-commit"] == "abc123def456"


# --------------------------------------------------------------------------
# Parallel agents and retries
# --------------------------------------------------------------------------


def test_parallel_agents_attribute_to_themselves(tmp_path):
    """Concurrent calls from different agents never cross-attribute."""
    journal = CallJournal(tmp_path / "calls.jsonl")

    async def run_all():
        tasks = []
        for i in range(8):
            agent = f"agent-{i % 4}"
            downstream = FakeDownstream(record_id=f"rec-{agent}-{i}")
            mw = ReefGatewayMiddleware(downstream, scenario="coral-demo", journal=journal)
            tasks.append(_call(mw, _scope(headers=_coral_headers(agent=agent, commit=f"c-{agent}"))))
        await asyncio.gather(*tasks)

    asyncio.run(run_all())
    records = journal.records()
    assert len(records) == 8
    for record in records:
        assert record.agent_record_id.startswith(f"rec-{record.agent_id}-")
    # per-attempt view: each agent's two calls, nothing from the others
    ids = journal.record_ids_since(0, "agent-2", "c-agent-2")
    assert len(ids) == 2
    assert all("agent-2" in record_id for record_id in ids)


def test_retry_produces_two_references_dedup_only_collapses_same_receipt(tmp_path):
    journal = CallJournal(tmp_path / "calls.jsonl")

    async def run():
        for record_id in ("rec-try1", "rec-try2", "rec-try2"):  # retry + duplicate receipt
            downstream = FakeDownstream(record_id=record_id)
            mw = ReefGatewayMiddleware(downstream, scenario="s", journal=journal)
            await _call(mw, _scope(headers=_coral_headers()))

    asyncio.run(run())
    ids = journal.record_ids_since(0, "agent-1", "abc123def456")
    assert ids == ["rec-try1", "rec-try2"]


def test_journal_concurrent_appends_from_threads(tmp_path):
    """The gateway serves agents from multiple worker threads."""
    journal = CallJournal(tmp_path / "calls.jsonl")

    def append_many(agent):
        for i in range(50):
            journal.append(
                CallRecord(
                    request_id=f"{agent}-{i}",
                    timestamp="t",
                    scenario="s",
                    agent_id=agent,
                    commit_hash="c",
                    path="/v1/chat/completions",
                    status_code=200,
                    agent_record_id=f"rec-{agent}-{i}",
                )
            )

    threads = [threading.Thread(target=append_many, args=(f"a{n}",)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    records = journal.records()
    assert len(records) == 200  # no torn/interleaved lines


# --------------------------------------------------------------------------
# Reporting: late grading, duplicates, missing references
# --------------------------------------------------------------------------


def test_late_grading_builds_report_from_journal(tmp_path):
    """The grader finalizes long after inference; the journal still resolves."""
    journal = CallJournal(tmp_path / "calls.jsonl")

    async def run():
        downstream = FakeDownstream(record_id="rec-late")
        mw = ReefGatewayMiddleware(downstream, scenario="coral-demo", journal=journal)
        await _call(mw, _scope(headers=_coral_headers(commit="late-commit")))

    asyncio.run(run())

    report = AttemptReport(
        scenario="coral-demo",
        agent_id="agent-1",
        commit_hash="late-commit",
        score=0.75,
        status="improved",
        parent_hash="parent-1",
        run_id="run-7",
        references=tuple(journal.record_ids_since(0, "agent-1", "late-commit")),
    )
    payload = report.payload()
    assert payload["score"] == 0.75
    assert payload["references"] == ["rec-late"]
    assert payload["metadata"]["coral"]["parent_hash"] == "parent-1"
    assert payload["metadata"]["coral"]["run_id"] == "run-7"


def test_duplicate_reports_share_a_deterministic_id():
    kwargs = {"scenario": "s", "agent_id": "a", "commit_hash": "c", "score": 1.0, "status": "improved"}
    first = AttemptReport(**kwargs).payload()
    second = AttemptReport(**kwargs).payload()
    assert first["agent_record_id"] == second["agent_record_id"]
    assert first == second  # identical resend -> reef dedups instead of double-counting
    # and the id is attempt-scoped, not global
    other = deterministic_report_id("s", "a", "c2")
    assert other != first["agent_record_id"]


def test_report_with_missing_references_still_carries_parent_links():
    report = AttemptReport(  # no references: receipts lost
        scenario="s",
        agent_id="a",
        commit_hash="c",
        score=None,
        status="crashed",
    )
    payload = report.payload()
    assert "references" not in payload
    assert "score" not in payload
    assert payload["metadata"]["coral"]["status"] == "crashed"


def test_torn_journal_line_is_skipped_not_fatal(tmp_path):
    journal = CallJournal(tmp_path / "calls.jsonl")
    journal.append(
        CallRecord(
            request_id="r1",
            timestamp="t",
            scenario="s",
            agent_id="a",
            commit_hash="c",
            path="/v1/chat/completions",
            status_code=200,
            agent_record_id="rec-1",
        )
    )
    with open(journal.path, "a", encoding="utf-8") as f:
        f.write('{"request_id": "r2", "timest')  # crash mid-write
    assert [r.request_id for r in journal.records()] == ["r1"]


def test_cursor_separates_sibling_attempts_at_the_same_base(tmp_path):
    """Two siblings run from one worktree state; the cursor gives each
    report exactly its own calls."""
    journal = CallJournal(tmp_path / "j.jsonl")

    def call(record_id):
        journal.append(
            CallRecord(
                request_id=record_id,
                timestamp="t",
                scenario="s",
                agent_id="a",
                commit_hash="base",
                path="/v1/chat/completions",
                status_code=200,
                agent_record_id=record_id,
            )
        )

    cursor_0 = journal.size()
    call("rec-sib0")
    refs_0 = journal.record_ids_since(cursor_0, "a", "base")
    cursor_1 = journal.size()
    call("rec-sib1")
    refs_1 = journal.record_ids_since(cursor_1, "a", "base")
    assert refs_0 == ["rec-sib0"]
    assert refs_1 == ["rec-sib1"]
    # cursor 0 gives the coordinate-wide view
    assert journal.record_ids_since(0, "a", "base") == ["rec-sib0", "rec-sib1"]


def test_record_ids_for_attempt_matches_truncated_hashes(tmp_path):
    """The coordinate query: 12-char journal hashes match the full 40-char
    parent, and the placeholder coordinate never matches anything."""
    parent = "d" * 40
    journal = CallJournal(tmp_path / "j.jsonl")
    for record_id, commit in (("rec-1", parent[:12]), ("rec-2", parent[:12]), ("rec-x", "unknown")):
        journal.append(
            CallRecord(
                request_id=record_id,
                timestamp="t",
                scenario="s",
                agent_id="a",
                commit_hash=commit,
                path="/v1/chat/completions",
                status_code=200,
                agent_record_id=record_id,
            )
        )
    assert journal.record_ids_for_attempt("a", parent) == ["rec-1", "rec-2"]
    assert journal.record_ids_for_attempt("a", "unknown") == ["rec-x"]  # exact only
    assert journal.record_ids_for_attempt("b", parent) == []
