"""End-to-end smoke test for the reef-pi wrapper: run agent → capture receipts → report."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import textwrap
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

from reef.harness.harness_wrapper import report, run_agent


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return b"{}"


def _write_spool_entry(
    directory: Path, scenario: str, state: str, *, sequence: int = 1, receipt: str | None = "receipt-1"
) -> Path:
    scenario_key = hashlib.sha256(scenario.encode()).hexdigest()
    path = directory / f"{scenario_key}-{sequence:020d}-run.{state}.json"
    path.write_text(json.dumps({"reef_url": "http://reef", "scenario": scenario, "turns": [{"receipt": receipt}]}))
    return path


def _make_fake_pi(tmp_path: Path, reef_port: int) -> Path:
    """A fake pi binary that makes one HTTP call to its models.json baseUrl."""
    binary = tmp_path / "fake-pi"
    binary.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json, os, urllib.request
            from pathlib import Path
            agent_dir = Path(os.environ["PI_CODING_AGENT_DIR"])
            models = json.loads((agent_dir / "models.json").read_text())
            base_url = list(models["providers"].values())[0]["baseUrl"]
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5).read()
            """
        )
    )
    binary.chmod(0o755)
    return binary


def _make_fake_opencode(tmp_path: Path) -> Path:
    """A fake opencode binary that reads opencode.json and calls the provider baseURL."""
    binary = tmp_path / "fake-opencode"
    binary.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json, os, urllib.request
            from pathlib import Path
            config_dir = Path(os.environ["OPENCODE_CONFIG_DIR"])
            config = json.loads((config_dir / "opencode.json").read_text())
            base_url = list(config["provider"].values())[0]["options"]["baseURL"]
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5).read()
            """
        )
    )
    binary.chmod(0o755)
    return binary


def _make_compose(tmp_path: Path, reef_port: int) -> str:
    """A minimal pi composition directory with models.json pointing at reef."""
    compose = tmp_path / "compose"
    compose.mkdir()
    (compose / "AGENTS.md").write_text("be concise\n")
    (compose / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "qwen": {
                        "api": "openai-completions",
                        "apiKey": "dummy",
                        "baseUrl": f"http://127.0.0.1:{reef_port}/v1",
                        "models": [{"id": "qwen3-8b"}],
                    }
                }
            }
        )
        + "\n"
    )
    return str(compose)


@pytest.mark.unit
def test_run_agent_captures_receipts_and_report_posts_them(tmp_path) -> None:
    """run_agent starts a proxy that captures receipts; report POSTs them to reef."""
    import http.server
    import threading

    receipt_id = "test-receipt-123"
    reports: list[dict] = []

    class FakeReefHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            if self.path.startswith("/v1/chat/completions"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("x-reef-agent-record-id", receipt_id)
                self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())
            elif self.path == "/reef/report":
                reports.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')

        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), FakeReefHandler)
    reef_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    compose = _make_compose(tmp_path, reef_port)
    binary = _make_fake_pi(tmp_path, reef_port)

    env = {**os.environ, "REEF_HARNESS_CAPTURES_DIR": str(tmp_path)}
    with patch.dict(os.environ, env):
        with contextlib.suppress(SystemExit):
            run_agent(str(binary), compose, "test-scenario", "pi", "PI_CODING_AGENT_DIR", ["-p", "fix the bug"])

        (captures_file,) = tmp_path.glob("*.pending.json")
        assert captures_file.exists()
        data = json.loads(captures_file.read_text())
        receipts = [t["receipt"] for t in data["turns"] if t.get("receipt")]
        assert receipt_id in receipts

        report("test-scenario", "pi", 0.0, "missed the empty-token case")

        assert len(reports) == 1
        assert reports[0]["score"] == 0.0
        assert reports[0]["feedback"] == "missed the empty-token case"
        assert receipt_id in reports[0]["references"]
        assert not captures_file.exists()

    server.shutdown()


@pytest.mark.unit
def test_same_scenario_runs_remain_independently_reportable(tmp_path) -> None:
    """Completing another run must not overwrite the first run's receipts."""
    import http.server
    import threading

    receipt_ids = iter(("first-receipt", "second-receipt"))
    reports: list[dict] = []

    class FakeReefHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            if self.path.startswith("/v1/chat/completions"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("x-reef-agent-record-id", next(receipt_ids))
                self.end_headers()
                self.wfile.write(b'{"choices":[{"message":{"content":"ok"}}]}')
            elif self.path == "/reef/report":
                reports.append(body)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeReefHandler)
    reef_port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    compose = _make_compose(tmp_path, reef_port)
    binary = _make_fake_pi(tmp_path, reef_port)

    with patch.dict(os.environ, {"REEF_HARNESS_CAPTURES_DIR": str(tmp_path)}):
        for _ in range(2):
            with contextlib.suppress(SystemExit):
                run_agent(str(binary), compose, "same-scenario", "pi", "PI_CODING_AGENT_DIR", ["-p", "hi"])

        report("same-scenario", "pi", 1.0, "first")
        report("same-scenario", "pi", 0.0, "second")

    assert [item["references"] for item in reports] == [["first-receipt"], ["second-receipt"]]
    server.shutdown()


@pytest.mark.unit
def test_run_completed_during_report_remains_pending(tmp_path) -> None:
    """A reporter consumes only the spool entry that it claimed before I/O."""
    import http.server
    import threading

    receipt_ids = iter(("old-receipt", "new-receipt"))
    reports: list[dict] = []
    report_started = threading.Event()
    finish_report = threading.Event()

    class FakeReefHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            if self.path.startswith("/v1/chat/completions"):
                self.send_response(200)
                self.send_header("x-reef-agent-record-id", next(receipt_ids))
                self.end_headers()
                self.wfile.write(b"{}")
            elif self.path == "/reef/report":
                reports.append(body)
                if body["references"] == ["old-receipt"]:
                    report_started.set()
                    assert finish_report.wait(5)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeReefHandler)
    reef_port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    compose = _make_compose(tmp_path, reef_port)
    binary = _make_fake_pi(tmp_path, reef_port)

    with patch.dict(os.environ, {"REEF_HARNESS_CAPTURES_DIR": str(tmp_path)}):
        with contextlib.suppress(SystemExit):
            run_agent(str(binary), compose, "same-scenario", "pi", "PI_CODING_AGENT_DIR", ["-p", "old"])

        report_errors: list[BaseException] = []

        def report_old_run() -> None:
            try:
                report("same-scenario", "pi", 1.0, "old")
            except BaseException as error:
                report_errors.append(error)

        reporter = threading.Thread(target=report_old_run)
        reporter.start()
        assert report_started.wait(5)

        with contextlib.suppress(SystemExit):
            run_agent(str(binary), compose, "same-scenario", "pi", "PI_CODING_AGENT_DIR", ["-p", "new"])

        finish_report.set()
        reporter.join(5)
        assert not reporter.is_alive()
        assert report_errors == []

        report("same-scenario", "pi", 0.0, "new")

    assert [item["references"] for item in reports] == [["old-receipt"], ["new-receipt"]]
    server.shutdown()


@pytest.mark.unit
def test_failed_report_restores_claim_for_retry(tmp_path) -> None:
    pending_file = _write_spool_entry(tmp_path, "scenario", "pending")
    failure = urllib.error.HTTPError("http://reef/reef/report", 503, "unavailable", {}, io.BytesIO(b"try later"))

    with (
        patch.dict(os.environ, {"REEF_HARNESS_CAPTURES_DIR": str(tmp_path)}),
        patch("reef.harness.harness_wrapper.urllib.request.urlopen", side_effect=[failure, _Response()]),
    ):
        with pytest.raises(SystemExit, match=r"report failed \(503\)"):
            report("scenario", "pi", 1.0, "retry me")
        assert pending_file.exists()

        report("scenario", "pi", 1.0, "retry me")

    assert not pending_file.exists()


@pytest.mark.unit
def test_per_receipt_report_posts_one_report_for_each_capture(tmp_path) -> None:
    """--per-receipt fans the run's score across its receipts as separate
    reports, one reference each, so every exchange batches on its own."""
    scenario_key = hashlib.sha256(b"scenario").hexdigest()
    captures = tmp_path / f"{scenario_key}-{1:020d}-run.pending.json"
    captures.write_text(
        json.dumps(
            {
                "reef_url": "http://reef",
                "scenario": "scenario",
                "turns": [{"receipt": "receipt-1"}, {"receipt": "receipt-2"}],
            }
        )
    )
    posted: list[dict] = []

    def record_report(req, timeout=None):
        posted.append(json.loads(req.data))
        return _Response()

    with (
        patch.dict(os.environ, {"REEF_HARNESS_CAPTURES_DIR": str(tmp_path)}),
        patch("reef.harness.harness_wrapper.urllib.request.urlopen", record_report),
    ):
        report("scenario", "pi", 0.0, "per turn", per_receipt=True)

    assert [item["references"] for item in posted] == [["receipt-1"], ["receipt-2"]]
    assert {item["score"] for item in posted} == {0.0}
    assert not any(tmp_path.glob("*.json"))


@pytest.mark.unit
def test_report_recovers_reused_process_id_before_newer_run(tmp_path) -> None:
    claimed_file = _write_spool_entry(
        tmp_path,
        "scenario",
        f"reporting-{os.getpid()}-previous-process-abandoned",
        receipt="abandoned-receipt",
    )
    pending_file = _write_spool_entry(tmp_path, "scenario", "pending", sequence=2, receipt="newer-receipt")
    reported_references: list[list[str]] = []

    def record_report(request, timeout):
        assert timeout == 30
        reported_references.append(json.loads(request.data)["references"])
        return _Response()

    with (
        patch.dict(os.environ, {"REEF_HARNESS_CAPTURES_DIR": str(tmp_path)}),
        patch("reef.harness.harness_wrapper._process_start_id", return_value="current-process"),
        patch("reef.harness.harness_wrapper.urllib.request.urlopen", record_report),
    ):
        report("scenario", "pi", 1.0, "recovered")
        report("scenario", "pi", 0.0, "newer")

    assert not claimed_file.exists()
    assert not pending_file.exists()
    assert reported_references == [["abandoned-receipt"], ["newer-receipt"]]


@pytest.mark.unit
def test_receiptless_run_does_not_block_newer_receipts(tmp_path) -> None:
    empty_file = _write_spool_entry(tmp_path, "scenario", "pending", receipt=None)
    pending_file = _write_spool_entry(tmp_path, "scenario", "pending", sequence=2, receipt="valid-receipt")
    reported_references: list[str] = []

    def record_report(request, timeout):
        assert timeout == 30
        reported_references.extend(json.loads(request.data)["references"])
        return _Response()

    with (
        patch.dict(os.environ, {"REEF_HARNESS_CAPTURES_DIR": str(tmp_path)}),
        patch("reef.harness.harness_wrapper.urllib.request.urlopen", record_report),
    ):
        report("scenario", "pi", 1.0, "valid")

    assert not empty_file.exists()
    assert not pending_file.exists()
    assert reported_references == ["valid-receipt"]


def _make_opencode_compose(tmp_path: Path, reef_port: int) -> str:
    """A minimal opencode composition directory with opencode.json pointing at reef."""
    compose = tmp_path / "opencode-compose"
    compose.mkdir()
    (compose / "AGENTS.md").write_text("be concise\n")
    (compose / "opencode.json").write_text(
        json.dumps(
            {
                "autoupdate": False,
                "share": "disabled",
                "permission": {"*": "allow"},
                "defaultModel": "qwen/qwen3-8b",
                "defaultProvider": "qwen",
                "provider": {
                    "qwen": {
                        "npm": "@ai-sdk/openai-compatible",
                        "options": {
                            "baseURL": f"http://127.0.0.1:{reef_port}/v1",
                            "apiKey": "dummy",
                        },
                        "models": {"qwen3-8b": {"name": "Qwen3 8B"}},
                    }
                },
            }
        )
        + "\n"
    )
    return str(compose)


@pytest.mark.unit
def test_opencode_run_agent_captures_receipts_and_report_posts_them(tmp_path) -> None:
    """The opencode adapter path: reads opencode.json, rewrites provider.*.options.baseURL."""
    import http.server
    import threading

    receipt_id = "opencode-receipt-456"
    reports: list[dict] = []

    class FakeReefHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            if self.path.startswith("/v1/chat/completions"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("x-reef-agent-record-id", receipt_id)
                self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())
            elif self.path == "/reef/report":
                reports.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')

        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), FakeReefHandler)
    reef_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    compose = _make_opencode_compose(tmp_path, reef_port)
    binary = _make_fake_opencode(tmp_path)

    env = {**os.environ, "REEF_HARNESS_CAPTURES_DIR": str(tmp_path)}
    with patch.dict(os.environ, env):
        with contextlib.suppress(SystemExit):
            run_agent(str(binary), compose, "oc-scenario", "opencode", "OPENCODE_CONFIG_DIR", ["run", "hi"])

        (captures_file,) = tmp_path.glob("*.pending.json")
        assert captures_file.exists()
        data = json.loads(captures_file.read_text())
        receipts = [t["receipt"] for t in data["turns"] if t.get("receipt")]
        assert receipt_id in receipts

        report("oc-scenario", "opencode", 1.0, "good job")

        assert len(reports) == 1
        assert reports[0]["score"] == 1.0
        assert receipt_id in reports[0]["references"]
        assert not captures_file.exists()

    server.shutdown()
