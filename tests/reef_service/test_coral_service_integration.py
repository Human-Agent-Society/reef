"""End-to-end: adapter -> real reef service -> receipt -> journal -> report.

The reef service is real (``reef.service.app.create_app`` with a stub
inference backend, exactly as reef's own service tests build it); the
middleware forwards to it over real HTTP. What LiteLLM would contribute in
production — an opaque proxy hop — is the pass-through the shim reproduces.

Requires the reef package importable (run from a reef checkout); skipped
otherwise so the adapter's own suite stays standalone.
"""

from __future__ import annotations

import asyncio
import json

import pytest

reef_service = pytest.importorskip("reef.service.app", reason="requires a reef checkout")

from aiohttp.test_utils import TestClient, TestServer

from recipes.beta.coral.journal import CallJournal
from recipes.beta.coral.middleware import ReefGatewayMiddleware
from recipes.beta.coral.reporter import AttemptReport
from reef.dispatcher import build_default_dispatcher
from reef.runtime.inference import InferenceBackend
from reef.storage.sqlite import SQLiteScenarioStorage


class _EchoBackend(InferenceBackend):
    async def inference(self, artifact, path, payload):
        del artifact, path, payload
        return {"choices": [{"message": {"content": "a proposed config"}}]}


class _HttpForwarder:
    """ASGI app that forwards to the reef test client — the LiteLLM stand-in."""

    def __init__(self, client: TestClient) -> None:
        self.client = client

    async def __call__(self, scope, receive, send):
        headers = {bytes(k).decode("latin-1"): bytes(v).decode("latin-1") for k, v in scope["headers"]}
        body = bytearray()
        while True:
            message = await receive()
            body.extend(message.get("body", b""))
            if not message.get("more_body"):
                break
        upstream = await self.client.post(scope["path"], data=bytes(body), headers=headers)
        payload = await upstream.read()
        await send(
            {
                "type": "http.response.start",
                "status": upstream.status,
                "headers": [(k.encode(), v.encode()) for k, v in upstream.headers.items()],
            }
        )
        await send({"type": "http.response.body", "body": payload})


def _coral_scope():
    return {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer sk-coral-agent-1-deadbeef"),
            (b"x-coral-agent-id", b"agent-1"),
            (b"x-coral-session-id", b"commit-a"),
        ],
    }


async def _post_through(mw, scope, body: bytes):
    sent = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        sent.append(message)

    await mw(scope, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    payload = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, payload


@pytest.mark.unit
def test_full_loop_against_real_reef_service(tmp_path):
    async def run():
        reef_client = TestClient(
            TestServer(
                reef_service.create_app(
                    build_default_dispatcher(scenario_storage=SQLiteScenarioStorage()),
                    inference_backend=_EchoBackend(),
                )
            )
        )
        await reef_client.start_server()
        try:
            journal = CallJournal(tmp_path / "calls.jsonl")
            mw = ReefGatewayMiddleware(
                _HttpForwarder(reef_client),
                scenario="coral-demo",
                journal=journal,
                extra_tags={"coral-run": "run-1"},
            )

            # inference through the adapter: reef must accept (scenario header
            # stamped by us, not by CORAL) and hand back a receipt
            status, payload = await _post_through(
                mw, _coral_scope(), json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
            )
            assert status == 200
            assert json.loads(payload)["choices"][0]["message"]["content"] == "a proposed config"

            (record,) = journal.records()
            assert record.agent_record_id, "reef receipt must be captured"
            assert record.agent_id == "agent-1"

            # grader finalizes later: report with exact references
            report = AttemptReport(
                scenario="coral-demo",
                agent_id="agent-1",
                commit_hash="commit-a",
                score=0.9,
                status="improved",
                run_id="run-1",
                references=tuple(journal.record_ids_since(0, "agent-1", "commit-a")),
            )
            assert report.references == (record.agent_record_id,)

            first = await reef_client.post(
                "/reef/report", json=report.payload(), headers={"x-reef-scenario": "coral-demo"}
            )
            assert first.status == 200
            echoed = (await first.json())["agent_record_id"]

            # duplicate grader run: identical resend dedups, no double count
            retry = await reef_client.post(
                "/reef/report", json=report.payload(), headers={"x-reef-scenario": "coral-demo"}
            )
            assert retry.status == 200
            assert (await retry.json())["agent_record_id"] == echoed

            # conflicting resend (same id, different score) must be rejected
            conflicting = dict(report.payload())
            conflicting["score"] = 0.1
            conflict = await reef_client.post(
                "/reef/report", json=conflicting, headers={"x-reef-scenario": "coral-demo"}
            )
            assert conflict.status == 409
        finally:
            await reef_client.close()

    asyncio.run(run())
