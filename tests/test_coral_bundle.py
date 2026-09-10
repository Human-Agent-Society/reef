"""Result bundle and release/usage capture (issue #3: result bundle Done-when)."""

from __future__ import annotations

import asyncio
import json

from recipes.coral.bundle import build_result_bundle
from recipes.coral.journal import CallJournal, CallRecord
from recipes.coral.middleware import ReefGatewayMiddleware
from recipes.coral.reporter import AttemptReport


def _record(agent, commit, record_id, *, release=None, prompt=0, completion=0, run="run-1"):
    return CallRecord(
        request_id=f"{agent}-{commit}-{record_id}",
        timestamp="t",
        scenario="s",
        agent_id=agent,
        commit_hash=commit,
        path="/v1/chat/completions",
        status_code=200,
        agent_record_id=record_id,
        release_id=release,
        prompt_tokens=prompt,
        completion_tokens=completion,
        tags={"coral-run": run, "coral-agent": agent, "coral-commit": commit},
    )


def _report(agent, commit, score, *, status="improved", parent=None, references=()):
    return AttemptReport(
        scenario="s",
        agent_id=agent,
        commit_hash=commit,
        score=score,
        status=status,
        parent_hash=parent,
        references=tuple(references),
    )


def test_bundle_scores_best_revisions_and_tokens(tmp_path):
    # The gateway journals each call under the worktree commit it was made
    # from — the attempt's parent. Reports carry the exact record ids the
    # watcher resolved; the bundle correlates on those.
    journal = CallJournal(tmp_path / "journal.jsonl")
    journal.append(_record("a1", "seed0000000c", "r1", release="rel-0", prompt=100, completion=50))
    journal.append(_record("a1", "seed0000000c", "r2", release="rel-0", prompt=10, completion=5))
    journal.append(_record("a1", "c1", "r3", release="rel-1", prompt=20, completion=9))
    reports = [
        _report("a1", "c1", 0.4, parent="seed0000000c", references=("r1", "r2")),
        _report("a1", "c2", 0.9, parent="c1", references=("r3",)),
    ]

    bundle = build_result_bundle(journal, reports, run_id="run-1")

    assert [s["score"] for s in bundle["score_over_time"]] == [0.4, 0.9]
    assert bundle["best_attempt"]["commit_hash"] == "c2"
    assert bundle["model_revisions_served"] == ["rel-0", "rel-1"]
    assert bundle["token_accounting"] == {
        "inference_calls": 3,
        "prompt_tokens": 130,
        "completion_tokens": 64,
    }
    first, second = bundle["attempts"]
    assert first["inference_calls"] == 2 and first["prompt_tokens"] == 110
    assert second["release_ids"] == ["rel-1"]
    # the revision check the issue asks for: later attempt on a newer release
    assert [(a["commit_hash"], a["release_ids"]) for a in bundle["attempts"]] == [
        ("c1", ["rel-0"]),
        ("c2", ["rel-1"]),
    ]


def test_bundle_without_references_falls_back_to_the_parent_coordinate(tmp_path):
    # A report rebuilt without references (e.g. partial recovery) still
    # correlates: the journal coordinate is (agent, parent), with the
    # gateway's 12-char truncation prefix-matching the full parent hash.
    parent = "f" * 40
    journal = CallJournal(tmp_path / "journal.jsonl")
    journal.append(_record("a1", parent[:12], "r1", release="rel-0", prompt=3, completion=1))
    journal.append(_record("a2", parent[:12], "r-other-agent"))
    bundle = build_result_bundle(journal, [_report("a1", "c" * 40, 0.5, parent=parent)], run_id="run-1")

    (attempt,) = bundle["attempts"]
    assert attempt["inference_calls"] == 1
    assert attempt["inference_record_ids"] == ["r1"]


def test_bundle_filters_by_run_and_tolerates_unscored(tmp_path):
    journal = CallJournal(tmp_path / "journal.jsonl")
    journal.append(_record("a1", "p1", "r1", run="run-1"))
    journal.append(_record("a1", "p1", "r9", run="run-OTHER"))
    reports = [_report("a1", "c1", None, status="crashed", parent="p1", references=("r1", "r9"))]

    bundle = build_result_bundle(journal, reports, run_id="run-1")
    assert bundle["best_attempt"] is None
    assert bundle["attempts"][0]["inference_calls"] == 1  # run-OTHER excluded


def test_middleware_captures_release_and_usage(tmp_path):
    class Downstream:
        async def __call__(self, scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"x-reef-agent-record-id", b"rec-1"),
                        (b"x-reef-release-id", b"rel-42"),
                    ],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps(
                        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
                    ).encode(),
                }
            )

    journal = CallJournal(tmp_path / "journal.jsonl")
    mw = ReefGatewayMiddleware(Downstream(), scenario="s", journal=journal)

    async def run():
        async def receive():
            return {"type": "http.request", "body": b"{}", "more_body": False}

        async def send(message):
            pass

        await mw(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": [(b"x-coral-agent-id", b"a1"), (b"x-coral-session-id", b"c1")],
            },
            receive,
            send,
        )

    asyncio.run(run())
    (record,) = journal.records()
    assert record.release_id == "rel-42"
    assert record.prompt_tokens == 7
    assert record.completion_tokens == 3


def test_middleware_usage_from_final_sse_chunk(tmp_path):
    class Downstream:
        async def __call__(self, scope, receive, send):
            frames = (
                'data: {"choices": [{"delta": {"content": "x"}}]}\n\n'
                'data: {"choices": [], "reef": {"agent_record_id": "rec-s"}, '
                '"usage": {"prompt_tokens": 11, "completion_tokens": 4}}\n\n'
                "data: [DONE]\n\n"
            )
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")],
                }
            )
            await send({"type": "http.response.body", "body": frames.encode()})

    journal = CallJournal(tmp_path / "journal.jsonl")
    mw = ReefGatewayMiddleware(Downstream(), scenario="s", journal=journal)

    async def run():
        async def receive():
            return {"type": "http.request", "body": b"{}", "more_body": False}

        async def send(message):
            pass

        await mw(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": [(b"x-coral-agent-id", b"a1"), (b"x-coral-session-id", b"c1")],
            },
            receive,
            send,
        )

    asyncio.run(run())
    (record,) = journal.records()
    assert record.agent_record_id == "rec-s"
    assert record.prompt_tokens == 11
    assert record.completion_tokens == 4
