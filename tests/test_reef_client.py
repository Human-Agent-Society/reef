"""Stdlib ``reef_client`` wire behavior: headers, receipts, error mapping."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from reef_client import ReefClient, ReefClientError


class _Stub:
    """One-shot HTTP stub recording the last request and serving a canned reply."""

    def __init__(
        self, status: int = 200, reply: dict[str, Any] | None = None, reply_headers: dict[str, str] | None = None
    ):
        self.status = status
        self.reply = reply if reply is not None else {"ok": True}
        self.reply_headers = reply_headers or {}
        self.requests: list[dict[str, Any]] = []

    def url(self) -> str:
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
                payload = json.dumps(server.reply).encode()
                self.send_response(server.status)
                for name, value in server.reply_headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:
                server.requests.append(
                    {
                        "path": self.path,
                        "headers": {name.lower(): value for name, value in self.headers.items()},
                        "body": None,
                    }
                )
                payload = json.dumps(server.reply).encode()
                self.send_response(server.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: Any) -> None:
                return

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self._httpd = httpd
        return f"http://127.0.0.1:{httpd.server_address[1]}"

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.mark.unit
def test_post_sends_protocol_headers_and_returns_receipt() -> None:
    stub = _Stub(reply_headers={"x-reef-agent-record-id": "ad-1"})
    client = ReefClient(stub.url(), token="tok", timeout_s=5)
    body, headers = client.post(
        "/v1/chat/completions",
        "scen",
        {"model": "m", "messages": []},
        recipe="recipe",
        extra_headers={"x-example-extra": "1"},
    )
    stub.close()

    assert body == {"ok": True}
    assert headers["x-reef-agent-record-id"] == "ad-1"
    sent = stub.requests[0]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["headers"]["x-reef-scenario"] == "scen"
    assert sent["headers"]["x-reef-recipe"] == "recipe"
    assert sent["headers"]["authorization"] == "Bearer tok"
    assert sent["headers"]["x-example-extra"] == "1"
    assert sent["body"]["model"] == "m"


@pytest.mark.unit
def test_explicit_arguments_win_over_extra_headers() -> None:
    stub = _Stub()
    client = ReefClient(stub.url(), timeout_s=5)
    client.post(
        "/reef/report",
        "scen-arg",
        {"score": 1.0},
        recipe="recipe-arg",
        extra_headers={
            "X-Reef-Scenario": "scen-extra",
            "x-reef-recipe": "recipe-extra",
            "content-type": "text/plain",
        },
    )
    stub.close()

    sent = stub.requests[0]["headers"]
    assert sent["x-reef-scenario"] == "scen-arg"
    assert sent["x-reef-recipe"] == "recipe-arg"
    assert sent["content-type"] == "application/json"  # not duplicated or replaced


@pytest.mark.unit
def test_http_error_carries_status_body_and_content_type() -> None:
    stub = _Stub(status=409, reply={"detail": "scenario bound to another recipe"})
    client = ReefClient(stub.url(), timeout_s=5)
    with pytest.raises(ReefClientError) as caught:
        client.report("scen", {"score": 1.0})
    stub.close()

    assert caught.value.status == 409
    assert "scenario bound" in caught.value.body
    assert caught.value.content_type == "application/json"


@pytest.mark.unit
def test_report_hits_its_path() -> None:
    stub = _Stub()
    client = ReefClient(stub.url(), timeout_s=5)
    client.report("scen", {"score": 0.5})
    stub.close()

    assert [request["path"] for request in stub.requests] == ["/reef/report"]


@pytest.mark.unit
def test_report_serializes_typed_reports_through_to_dict() -> None:
    class _Signal:
        """Duck-typed stand-in for a recipe's reef.core.reports.ReportBase dataclass."""

        def to_dict(self, *, references=()):
            return {"score": 1.0, "references": list(references)}

    stub = _Stub()
    client = ReefClient(stub.url(), timeout_s=5)
    client.report("scen", _Signal(), references=["receipt-1"])
    stub.close()

    body = stub.requests[0]["body"]
    assert body == {"score": 1.0, "references": ["receipt-1"]}


@pytest.mark.unit
def test_report_merges_references_into_plain_dict_payloads() -> None:
    stub = _Stub()
    client = ReefClient(stub.url(), timeout_s=5)
    client.report("scen", {"score": 0.5}, references=["receipt-1"])
    stub.close()

    assert stub.requests[0]["body"] == {"references": ["receipt-1"], "score": 0.5}


@pytest.mark.unit
def test_inference_with_record_requires_receipt() -> None:
    stub = _Stub()  # no x-reef-agent-record-id header
    client = ReefClient(stub.url(), timeout_s=5)
    with pytest.raises(ReefClientError) as caught:
        client.inference_with_record("scen", "/v1/chat/completions", {"messages": []})
    stub.close()

    assert caught.value.status == 502
    assert "agent-record-id" in str(caught.value)


@pytest.mark.unit
def test_get_sends_token_and_returns_body() -> None:
    stub = _Stub(reply={"releases": [{"operation": "training"}, {"operation": "inference"}]})
    client = ReefClient(stub.url(), token="tok", timeout_s=5)
    body = client.get("/reef/scenarios/scen/releases")
    stub.close()

    assert body == {"releases": [{"operation": "training"}, {"operation": "inference"}]}
    sent = stub.requests[0]
    assert sent["path"] == "/reef/scenarios/scen/releases"
    assert sent["headers"]["authorization"] == "Bearer tok"


@pytest.mark.unit
def test_get_http_error_carries_status() -> None:
    stub = _Stub(status=404, reply={"detail": "unknown scenario"})
    client = ReefClient(stub.url(), timeout_s=5)
    with pytest.raises(ReefClientError) as caught:
        client.get("/reef/scenarios/scen/releases")
    stub.close()

    assert caught.value.status == 404
    assert "unknown scenario" in caught.value.body
