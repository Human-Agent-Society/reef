"""End-to-end smoke test for the reef-pi wrapper: run agent → capture receipts → report."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
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
def test_report_carries_the_installed_release_as_metadata(tmp_path) -> None:
    """report reads the sidecar beside the compose dir and stamps the report
    with the release the client is running."""
    compose = tmp_path / "reef-harness" / "pi-agent"
    compose.mkdir(parents=True)
    (tmp_path / "reef-harness" / ".reef-harness-release").write_text(
        json.dumps({"release_id": "rel-42"}), encoding="utf-8"
    )
    scenario_key = hashlib.sha256(b"scenario").hexdigest()
    captures = tmp_path / f"{scenario_key}-{1:020d}-run.pending.json"
    captures.write_text(json.dumps({"reef_url": "http://reef", "scenario": "scenario", "turns": [{"receipt": "r1"}]}))
    posted: list[dict] = []

    def record(req, timeout=None):
        posted.append(json.loads(req.data))
        return _Response()

    with (
        patch.dict(
            os.environ,
            {"REEF_HARNESS_CAPTURES_DIR": str(tmp_path), "REEF_HARNESS_COMPOSE": str(compose)},
        ),
        patch("reef.harness.harness_wrapper.urllib.request.urlopen", record),
    ):
        report("scenario", "pi", 0.0, "note")

    assert posted[0]["metadata"] == {"client_release": "rel-42"}


@pytest.mark.unit
def test_run_agent_tags_records_with_the_installed_release(tmp_path) -> None:
    """run_agent reads the sidecar and sends x-reef-tag-release, so the record
    keeps which release answered."""
    from reef.harness import harness_wrapper

    compose = tmp_path / "reef-harness" / "pi-agent"
    compose.mkdir(parents=True)
    (tmp_path / "reef-harness" / ".reef-harness-release").write_text(
        json.dumps({"release_id": "rel-7"}), encoding="utf-8"
    )
    assert harness_wrapper._installed_release(str(compose)) == "rel-7"
    assert harness_wrapper._installed_release(str(tmp_path / "missing")) is None


@pytest.mark.unit
def test_partial_per_receipt_failure_retries_only_the_unsent(tmp_path) -> None:
    """When a later per-receipt post fails, the restored claim holds only the
    receipts that never went out, so a retry cannot duplicate reports."""
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
    failure = urllib.error.HTTPError("http://reef/reef/report", 503, "unavailable", {}, io.BytesIO(b"later"))
    posted: list[dict] = []

    def flaky(req, timeout=None):
        posted.append(json.loads(req.data))
        if len(posted) == 2:
            raise failure
        return _Response()

    with (
        patch.dict(os.environ, {"REEF_HARNESS_CAPTURES_DIR": str(tmp_path)}),
        patch("reef.harness.harness_wrapper.urllib.request.urlopen", flaky),
    ):
        with pytest.raises(SystemExit, match=r"report failed \(503\)"):
            report("scenario", "pi", 0.0, "per turn", per_receipt=True)

        restored = json.loads(captures.read_text())
        assert [turn["receipt"] for turn in restored["turns"]] == ["receipt-2"]

        report("scenario", "pi", 0.0, "per turn", per_receipt=True)

    assert [item["references"] for item in posted] == [["receipt-1"], ["receipt-2"], ["receipt-2"]]


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


def _make_native_compose(tmp_path: Path, reef_port: int) -> str:
    """A minimal native tree: the rendered model binding pointing at reef, rules, and empty tool and hook dirs."""
    compose = tmp_path / "native-compose" / "native"
    (compose / "tools").mkdir(parents=True)
    (compose / "hooks").mkdir()
    (compose / "RULES.md").write_text("Answer in one word.\n")
    (compose / "models.json").write_text(
        json.dumps(
            {"api": "openai", "base_url": f"http://127.0.0.1:{reef_port}", "api_key": "dummy", "model": "qwen3-8b"}
        )
        + "\n"
    )
    return str(compose)


def _make_native_launcher(tmp_path: Path) -> Path:
    """reef-native as the installed console script would be: this interpreter running the loop."""
    import sys

    root = Path(__file__).resolve().parents[2]
    binary = tmp_path / "reef-native"
    binary.write_text(
        f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(root)!r})\n"
        "from reef.harness.native import main\nsys.exit(main())\n"
    )
    binary.chmod(0o755)
    return binary


@pytest.mark.unit
def test_native_run_agent_drives_the_real_loop_through_the_proxy_and_reports(tmp_path) -> None:
    """The native adapter path: the wrapper rewrites models.json base_url to the proxy, the loop reads it
    through REEF_NATIVE_DIR, its session log lands beside the installed tree, and the receipt reports."""
    import http.server
    import threading

    receipt_id = "native-receipt-789"
    reports: list[dict] = []
    seen: list[dict] = []

    class FakeReefHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            if self.path.startswith("/v1/chat/completions"):
                seen.append({"headers": dict(self.headers), "body": body})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("x-reef-agent-record-id", receipt_id)
                self.end_headers()
                self.wfile.write(
                    json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}).encode()
                )
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

    compose = _make_native_compose(tmp_path, reef_port)
    binary = _make_native_launcher(tmp_path)

    env = {**os.environ, "REEF_HARNESS_CAPTURES_DIR": str(tmp_path)}
    env.pop("REEF_NATIVE_SESSION_DIR", None)
    with patch.dict(os.environ, env, clear=True):
        with contextlib.suppress(SystemExit):
            run_agent(str(binary), compose, "native-scenario", "native", "REEF_NATIVE_DIR", ["-p", "say ok"])

        # The loop talked to reef through the proxy: the scenario header rode along and the rules were the system prompt.
        (request,) = seen
        assert request["headers"].get("x-reef-scenario") == "native-scenario"
        assert request["body"]["messages"][0] == {"role": "system", "content": "Answer in one word."}
        # The session log outlived the temp copy, beside the installed tree.
        session = Path(compose) / "sessions" / "session.jsonl"
        assert session.exists() and '"turn/end"' in session.read_text()

        (captures_file,) = tmp_path.glob("*.pending.json")
        data = json.loads(captures_file.read_text())
        assert receipt_id in [t["receipt"] for t in data["turns"] if t.get("receipt")]

        report("native-scenario", "native", 1.0, "answered")

        assert len(reports) == 1 and receipt_id in reports[0]["references"]
        assert not captures_file.exists()

    server.shutdown()


@pytest.mark.unit
@pytest.mark.parametrize("adapter", ["pi", "opencode", "claude", "codex", "dsh", "hermes", "native"])
def test_wrapper_reads_and_rewrites_every_adapters_binding_from_its_descriptor(tmp_path, adapter) -> None:
    """The descriptor names where the binding renders Reef's address; the wrapper reads it back from the
    installed tree and the temp copy equals a fresh render at the proxy, byte for byte, with the tree untouched."""
    from pathlib import PurePosixPath

    from reef.harness.adapters import get_adapter
    from reef.harness.harness_wrapper import _create_temp_composition, _extract_reef_url
    from reef.harness.model_binding import ModelBinding
    from reef.harness.render import render_composition

    descriptor = get_adapter(adapter)
    api = next(iter(descriptor.model_binding))
    reef = ModelBinding(base_url="http://127.0.0.1:8900", model="qwen3-8b", api_key="dummy", api=api)
    files = render_composition([("rules", {"text": "Be brief."}), *reef.compose_nodes(descriptor)], descriptor)
    root = tmp_path / "tree"
    for relative, text in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text, encoding="utf-8")
    _, subdir = descriptor.compose_relocation()
    compose = root / subdir

    assert _extract_reef_url(adapter, compose) == "http://127.0.0.1:8900"
    temp = Path(_create_temp_composition(adapter, str(compose), 41234))
    proxied = ModelBinding(base_url="http://127.0.0.1:41234", model="qwen3-8b", api_key="dummy", api=api)
    expected = render_composition([("rules", {"text": "Be brief."}), *proxied.compose_nodes(descriptor)], descriptor)
    for relative, text in expected.items():
        assert (temp / PurePosixPath(relative).relative_to(subdir)).read_text(encoding="utf-8") == text
    # The installed tree keeps Reef's address: only the temp copy was rewritten.
    for relative, text in files.items():
        assert (root / relative).read_text(encoding="utf-8") == text
    shutil.rmtree(temp)


# -- the binding lookup follows the descriptor's key path, not the first URL in the file ------


def _pi_tree(tmp_path: Path, models: dict) -> str:
    compose = tmp_path / "pi-tree" / "pi-agent"
    compose.mkdir(parents=True)
    (compose / "models.json").write_text(json.dumps(models, indent=2, sort_keys=True) + "\n")
    return str(compose)


def _opencode_tree(tmp_path: Path, config: dict) -> str:
    compose = tmp_path / "oc-tree" / "opencode"
    compose.mkdir(parents=True)
    (compose / "opencode.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return str(compose)


@pytest.mark.unit
def test_wrapper_ignores_urls_that_are_not_the_binding(tmp_path) -> None:
    """opencode's own $schema key and an mcp server url sort before the provider; neither is Reef and neither is rewritten."""
    from reef.harness.harness_wrapper import _create_temp_composition, _extract_reef_url

    compose = _opencode_tree(
        tmp_path,
        {
            "$schema": "https://opencode.ai/config.json",
            "mcp": {"docs": {"type": "remote", "url": "https://mcp.example.com/sse"}},
            "provider": {"reef": {"options": {"baseURL": "http://127.0.0.1:8900/v1", "apiKey": "dummy"}}},
        },
    )
    assert _extract_reef_url("opencode", Path(compose)) == "http://127.0.0.1:8900"
    temp = Path(_create_temp_composition("opencode", compose, 41234))
    written = json.loads((temp / "opencode.json").read_text())
    assert written["$schema"] == "https://opencode.ai/config.json"
    assert written["mcp"]["docs"]["url"] == "https://mcp.example.com/sse"
    assert written["provider"]["reef"]["options"]["baseURL"] == "http://127.0.0.1:41234/v1"
    shutil.rmtree(temp)


@pytest.mark.unit
def test_wrapper_picks_the_reef_provider_among_several_by_the_parent_key(tmp_path) -> None:
    """A second pi provider that sorts first keeps its own endpoint; the entry under the template's parent key is Reef."""
    from reef.harness.harness_wrapper import WrapperError, _create_temp_composition, _extract_reef_url

    providers = {
        "anthropic": {"api": "anthropic-messages", "baseUrl": "https://api.anthropic.com", "apiKey": "x"},
        "reef": {"api": "openai-completions", "baseUrl": "http://127.0.0.1:8900/v1", "apiKey": "dummy"},
    }
    compose = _pi_tree(tmp_path, {"providers": providers})
    assert _extract_reef_url("pi", Path(compose)) == "http://127.0.0.1:8900"
    temp = Path(_create_temp_composition("pi", compose, 41234))
    written = json.loads((temp / "models.json").read_text())["providers"]
    assert written["anthropic"]["baseUrl"] == "https://api.anthropic.com"
    assert written["reef"]["baseUrl"] == "http://127.0.0.1:41234/v1"
    shutil.rmtree(temp)
    # Two providers and neither named as the template names it: the wrapper says so instead of guessing.
    compose = _pi_tree(
        tmp_path / "two", {"providers": {"a": {"baseUrl": "http://a/v1"}, "b": {"baseUrl": "http://b/v1"}}}
    )
    with pytest.raises(WrapperError, match="2 entries hold a URL at providers/reef/baseUrl"):
        _extract_reef_url("pi", Path(compose))


@pytest.mark.unit
def test_wrapper_normalizes_the_rewritten_url_to_the_templates_suffix(tmp_path) -> None:
    """A bare origin in the tree still sends the agent to /v1 at the proxy, and a Reef behind a path prefix keeps it."""
    from reef.harness.harness_wrapper import _create_temp_composition, _extract_reef_url

    compose = _pi_tree(tmp_path, {"providers": {"reef": {"baseUrl": "http://127.0.0.1:8900", "apiKey": "d"}}})
    assert _extract_reef_url("pi", Path(compose)) == "http://127.0.0.1:8900"
    temp = Path(_create_temp_composition("pi", compose, 41234))
    assert (
        json.loads((temp / "models.json").read_text())["providers"]["reef"]["baseUrl"] == "http://127.0.0.1:41234/v1"
    )
    shutil.rmtree(temp)
    compose = _pi_tree(tmp_path / "prefix", {"providers": {"reef": {"baseUrl": "http://gw.example/reef/v1"}}})
    assert _extract_reef_url("pi", Path(compose)) == "http://gw.example/reef"


@pytest.mark.unit
def test_wrapper_refuses_a_binding_file_that_leaves_the_composition(tmp_path, monkeypatch) -> None:
    from dataclasses import replace

    from reef.harness.adapters import get_adapter
    from reef.harness.harness_wrapper import WrapperError, _extract_reef_url

    base = get_adapter("native")
    for path, message in (
        ("native/../outside.json", "escapes the tree"),
        ("elsewhere/models.json", "outside the composition"),
    ):
        targets = dict(base.config_targets)
        targets["models"] = replace(targets["models"], path=path)
        monkeypatch.setattr(
            "reef.harness.harness_wrapper.get_adapter", lambda name, d=replace(base, config_targets=targets): d
        )
        with pytest.raises(WrapperError, match=message):
            _extract_reef_url("native", tmp_path)


@pytest.mark.unit
def test_wrapper_captures_the_beta_messages_path_claude_code_posts(tmp_path) -> None:
    """The proxy matches the request path with its query; the Anthropic SDK posts /v1/messages?beta=true."""
    import http.server
    import threading

    receipt_id = "claude-receipt-1"

    class FakeReefHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("x-reef-agent-record-id", receipt_id)
            self.end_headers()
            self.wfile.write(json.dumps({"content": [{"type": "text", "text": "ok"}]}).encode())

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), FakeReefHandler)
    reef_port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    compose = tmp_path / "claude-tree" / "claude"
    compose.mkdir(parents=True)
    (compose / "settings.json").write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{reef_port}", "ANTHROPIC_AUTH_TOKEN": "d"}})
        + "\n"
    )
    binary = tmp_path / "fake-claude"
    binary.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json, os, urllib.request
            from pathlib import Path
            settings = json.loads((Path(os.environ["CLAUDE_CONFIG_DIR"]) / "settings.json").read_text())
            base = settings["env"]["ANTHROPIC_BASE_URL"]
            req = urllib.request.Request(
                f"{base}/v1/messages?beta=true",
                data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5).read()
            """
        )
    )
    binary.chmod(0o755)
    env = {**os.environ, "REEF_HARNESS_CAPTURES_DIR": str(tmp_path)}
    with patch.dict(os.environ, env):
        with contextlib.suppress(SystemExit):
            run_agent(str(binary), str(compose), "claude-scenario", "claude", "CLAUDE_CONFIG_DIR", ["-p", "hi"])
        (captures_file,) = tmp_path.glob("*.pending.json")
        data = json.loads(captures_file.read_text())
        assert [t["receipt"] for t in data["turns"]] == [receipt_id]
    server.shutdown()
