"""Serve mode: transparent forwarding, session stamping, capture, SSE."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from reef_client.serve import CaptureStore, ServeConfig, build_handler, has_tools
from reef_client.sse import SSEAccumulator, synthesize_sse_events

COMPLETION: dict[str, Any] = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1234,
    "model": "qwen3-8b",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "héllo ✓ world",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "reverse_string", "arguments": '{"s": "abc"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
}


@pytest.mark.unit
def test_sse_accumulator_round_trip_with_synthesized_events() -> None:
    events = synthesize_sse_events(COMPLETION, include_usage=True, agent_record_id="receipt-1")
    payload = "".join(events).encode()
    accumulator = SSEAccumulator()
    # Feed in awkward 7-byte slices to split multi-byte UTF-8 and SSE lines.
    for offset in range(0, len(payload), 7):
        accumulator.feed(payload[offset : offset + 7])
    accumulator.finish()

    assert accumulator.done
    completion = accumulator.completion
    assert completion is not None
    assert completion["choices"][0]["message"] == COMPLETION["choices"][0]["message"]
    assert completion["choices"][0]["finish_reason"] == "tool_calls"
    assert completion["usage"] == COMPLETION["usage"]
    assert accumulator.agent_record_id == "receipt-1"


@pytest.mark.unit
def test_sse_accumulator_ignores_keepalives_and_partial_events() -> None:
    accumulator = SSEAccumulator()
    accumulator.feed(b": ping\n\n")
    accumulator.feed(b'data: {"id": "x", "choices": [{"index": 0, "delta": {"role": "assistant"}}]}\n\n')
    accumulator.feed(b'data: {"choices": [{"index": 0, "delta": {"content": "hi"}}]}\n\n')
    accumulator.feed(b"data: [DONE]\n\n")
    assert accumulator.done
    completion = accumulator.completion
    assert completion is not None
    assert completion["choices"][0]["message"]["content"] == "hi"


@pytest.mark.unit
def test_sse_accumulator_accumulates_tool_call_argument_fragments() -> None:
    events = [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "c1", "type": "function", "function": {"name": "rev", "arguments": ""}}
                        ]
                    },
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"s"'}}]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": ': "ab"}'}}]}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    accumulator = SSEAccumulator()
    for event in events:
        accumulator.feed(f"data: {json.dumps(event)}\n\n".encode())
    accumulator.feed(b"data: [DONE]\n\n")
    completion = accumulator.completion
    assert completion is not None
    tool_call = completion["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["function"] == {"name": "rev", "arguments": '{"s": "ab"}'}


class _Upstream:
    """Stub model endpoint: canned completion, buffered or SSE, with a receipt."""

    def __init__(self, completion: dict[str, Any] | None = None, *, sse: bool = False, status: int = 200):
        self.completion = completion if completion is not None else COMPLETION
        self.sse = sse
        self.status = status
        self.requests: list[dict[str, Any]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                server.requests.append(
                    {
                        "path": self.path,
                        "headers": {name.lower(): value for name, value in self.headers.items()},
                        "body": json.loads(self.rfile.read(length) or b"{}"),
                    }
                )
                if server.sse and server.status == 200:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for event in synthesize_sse_events(
                        server.completion,
                        include_usage=True,
                        agent_record_id="receipt-1",
                    ):
                        data = event.encode()
                        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                        self.wfile.flush()
                        time.sleep(0.01)
                    self.wfile.write(b"0\r\n\r\n")
                    return
                payload = json.dumps(server.completion).encode()
                self.send_response(server.status)
                self.send_header("x-reef-agent-record-id", "receipt-1")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: Any) -> None:
                return

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


class _Proxy:
    def __init__(self, upstream_url: str, **overrides: Any):
        self.store = CaptureStore()
        config = ServeConfig(upstream=upstream_url, **overrides)
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(config, self.store))
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def _post(
    url: str, payload: dict[str, Any], headers: dict[str, str] | None = None
) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **(headers or {})}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers.items())


@pytest.mark.unit
def test_serve_stamps_session_and_captures_buffered_turn() -> None:
    upstream = _Upstream()
    proxy = _Proxy(
        upstream.url,
        session_header="x-openclawrl-session-id",
        stamp_when=lambda body, method: method == "POST" and has_tools(body),
    )
    try:
        status, _, _ = _post(f"{proxy.url}/_session", {"id": "personal-3"})
        assert status == 200

        with_tools = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function"}]}
        status, body, headers = _post(f"{proxy.url}/v1/chat/completions", with_tools)
        assert status == 200
        assert json.loads(body)["choices"][0]["message"]["content"] == "héllo ✓ world"
        assert headers["x-reef-agent-record-id"] == "receipt-1"
        assert upstream.requests[0]["headers"]["x-openclawrl-session-id"] == "personal-3"

        # Side task (no tools): forwarded but not stamped.
        _post(f"{proxy.url}/v1/chat/completions", {"model": "m", "messages": []})
        assert "x-openclawrl-session-id" not in upstream.requests[1]["headers"]

        turns = proxy.store.snapshot()
        assert len(turns) == 2
        assert turns[0]["session_id"] == "personal-3"
        assert turns[0]["receipt"] == "receipt-1"
        assert turns[0]["request"]["tools"]
        assert turns[0]["response"]["choices"][0]["finish_reason"] == "tool_calls"
        assert turns[1]["session_id"] == ""
    finally:
        proxy.close()
        upstream.close()


@pytest.mark.unit
def test_serve_relays_sse_passthrough_and_reconstructs_capture() -> None:
    upstream = _Upstream(sse=True)
    proxy = _Proxy(upstream.url)
    try:
        status, body, headers = _post(
            f"{proxy.url}/v1/chat/completions",
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert status == 200
        assert headers["Content-Type"] == "text/event-stream"
        assert "x-reef-agent-record-id" not in headers
        # Byte-identical passthrough of the upstream event stream.
        assert body.decode() == "".join(
            synthesize_sse_events(COMPLETION, include_usage=True, agent_record_id="receipt-1")
        )

        # The client-facing stream still parses as a valid completion.
        accumulator = SSEAccumulator()
        accumulator.feed(body)
        accumulator.finish()
        assert accumulator.completion["choices"][0]["message"] == COMPLETION["choices"][0]["message"]
        assert accumulator.agent_record_id == "receipt-1"

        turns = proxy.store.snapshot()
        assert len(turns) == 1
        assert turns[0]["client_stream"] is True
        assert turns[0]["receipt"] == "receipt-1"
        assert turns[0]["response"]["choices"][0]["message"] == COMPLETION["choices"][0]["message"]
        assert turns[0]["response"]["usage"] == COMPLETION["usage"]
    finally:
        proxy.close()
        upstream.close()


@pytest.mark.unit
def test_serve_force_non_stream_synthesizes_sse_for_streaming_clients() -> None:
    upstream = _Upstream()  # buffered only
    proxy = _Proxy(upstream.url, force_non_stream=True)
    try:
        status, body, headers = _post(
            f"{proxy.url}/v1/chat/completions",
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert status == 200
        assert headers["Content-Type"] == "text/event-stream"
        assert "x-reef-agent-record-id" not in headers
        # The upstream saw a buffered request: stream keys stripped.
        assert "stream" not in upstream.requests[0]["body"]
        assert "stream_options" not in upstream.requests[0]["body"]

        accumulator = SSEAccumulator()
        accumulator.feed(body)
        accumulator.finish()
        assert accumulator.done
        assert accumulator.completion["choices"][0]["message"] == COMPLETION["choices"][0]["message"]
        assert accumulator.completion["usage"] == COMPLETION["usage"]
        assert accumulator.agent_record_id == "receipt-1"

        turns = proxy.store.snapshot()
        assert len(turns) == 1
        assert turns[0]["client_stream"] is True
        assert turns[0]["request"]["messages"][0]["content"] == "hi"
    finally:
        proxy.close()
        upstream.close()


@pytest.mark.unit
def test_serve_reports_upstream_error_in_committed_stream() -> None:
    upstream = _Upstream(status=400)
    proxy = _Proxy(upstream.url, force_non_stream=True)
    try:
        status, body, headers = _post(
            f"{proxy.url}/v1/chat/completions", {"model": "m", "messages": [], "stream": True}
        )
        assert status == 200
        assert headers["Content-Type"] == "text/event-stream"

        data_lines = [line.removeprefix("data: ") for line in body.decode().splitlines() if line.startswith("data: ")]
        assert len(data_lines) == 1
        error = json.loads(data_lines[0])["error"]
        assert error["code"] == 400
        assert error["type"] == "reef_client_serve"
        assert json.loads(error["message"]) == COMPLETION
    finally:
        proxy.close()
        upstream.close()


@pytest.mark.unit
def test_serve_reports_502_when_upstream_is_down() -> None:
    proxy = _Proxy("http://127.0.0.1:1")  # nothing listening
    try:
        status, body, _ = _post(f"{proxy.url}/v1/chat/completions", {"model": "m", "messages": []})
        assert status == 502
        assert "reef-client serve" in json.loads(body)["error"]["message"]
    finally:
        proxy.close()


@pytest.mark.unit
def test_serve_captures_endpoint_snapshots_and_clears() -> None:
    upstream = _Upstream()
    proxy = _Proxy(upstream.url)
    try:
        _post(f"{proxy.url}/v1/chat/completions", {"model": "m", "messages": []})
        with urllib.request.urlopen(f"{proxy.url}/_captures", timeout=10) as response:
            turns = json.loads(response.read())["turns"]
        assert len(turns) == 1
        assert turns[0]["path"] == "/v1/chat/completions"

        request = urllib.request.Request(f"{proxy.url}/_captures", method="DELETE")
        with urllib.request.urlopen(request, timeout=10) as response:
            assert json.loads(response.read())["cleared"] == 1
        assert proxy.store.snapshot() == []
    finally:
        proxy.close()
        upstream.close()


def _get_json(url: str) -> tuple[int, dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _delete_json(url: str) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.mark.unit
def test_serve_addressable_sessions_self_identify_by_path() -> None:
    upstream = _Upstream()
    proxy = _Proxy(
        upstream.url,
        session_header="x-openclawrl-session-id",
        stamp_when=lambda body, method: method == "POST" and has_tools(body),
    )
    try:
        status, body, _ = _post(f"{proxy.url}/_sessions", {"id": "hw-a"})
        assert status == 200
        created = json.loads(body)
        assert created["path"] == "/s/hw-a"
        assert created["url"].endswith("/s/hw-a")
        _post(f"{proxy.url}/_sessions", {"id": "hw-b"})

        _, listed = _get_json(f"{proxy.url}/_sessions")
        assert listed["sessions"] == ["hw-a", "hw-b"]

        payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function"}]}
        status, _, _ = _post(f"{proxy.url}/s/hw-a/v1/chat/completions", payload)
        assert status == 200
        _post(f"{proxy.url}/s/hw-b/v1/chat/completions", payload)

        # The prefix is stripped upstream; each call carries its own session.
        assert upstream.requests[0]["path"] == "/v1/chat/completions"
        assert upstream.requests[0]["headers"]["x-openclawrl-session-id"] == "hw-a"
        assert upstream.requests[1]["headers"]["x-openclawrl-session-id"] == "hw-b"

        turns = proxy.store.snapshot()
        assert [turn["session_id"] for turn in turns] == ["hw-a", "hw-b"]
        assert turns[0]["path"] == "/v1/chat/completions"

        _, own = _get_json(f"{proxy.url}/_sessions/hw-a/captures")
        assert [turn["session_id"] for turn in own["turns"]] == ["hw-a"]
    finally:
        proxy.close()
        upstream.close()


@pytest.mark.unit
def test_serve_addressable_sessions_coexist_with_legacy_global_mode() -> None:
    upstream = _Upstream()
    proxy = _Proxy(upstream.url, session_header="x-openclawrl-session-id")
    try:
        _post(f"{proxy.url}/_session", {"id": "legacy"})
        _post(f"{proxy.url}/_sessions", {"id": "hw-a"})

        # Bare path rides the global declaration; the prefixed path its own id.
        _post(f"{proxy.url}/v1/chat/completions", {"model": "m", "messages": []})
        _post(f"{proxy.url}/s/hw-a/v1/chat/completions", {"model": "m", "messages": []})
        assert upstream.requests[0]["headers"]["x-openclawrl-session-id"] == "legacy"
        assert upstream.requests[1]["headers"]["x-openclawrl-session-id"] == "hw-a"
    finally:
        proxy.close()
        upstream.close()


@pytest.mark.unit
def test_serve_session_lifecycle_and_validation() -> None:
    upstream = _Upstream()
    proxy = _Proxy(upstream.url)
    try:
        status, body, _ = _post(f"{proxy.url}/_sessions", {"id": "bad/id"})
        assert status == 400
        assert "match" in json.loads(body)["error"]["message"]

        status, body, _ = _post(f"{proxy.url}/s/ghost/v1/chat/completions", {"model": "m", "messages": []})
        assert status == 404
        assert "ghost" in json.loads(body)["error"]["message"]

        _post(f"{proxy.url}/_sessions", {"id": "hw-a"})
        status, _ = _delete_json(f"{proxy.url}/_sessions/hw-a")
        assert status == 200
        status, _ = _delete_json(f"{proxy.url}/_sessions/hw-a")
        assert status == 404
        status, _, _ = _post(f"{proxy.url}/s/hw-a/v1/chat/completions", {"model": "m", "messages": []})
        assert status == 404  # closed sessions stop routing
    finally:
        proxy.close()
        upstream.close()
