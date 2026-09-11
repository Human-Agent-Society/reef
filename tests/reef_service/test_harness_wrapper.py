"""End-to-end smoke test for the reef-pi wrapper: run agent -> capture receipts -> report."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import textwrap
import urllib.error
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from reef.harness.client.wrapper import harness, main, report, run_agent, setup


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
    """A minimal pi composition directory with models.json pointing at reef, under the provider name the install renders."""
    compose = tmp_path / "compose"
    compose.mkdir()
    (compose / "AGENTS.md").write_text("be concise\n")
    (compose / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "reef": {
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
        patch("reef.harness.client.wrapper.urllib.request.urlopen", side_effect=[failure, _Response()]),
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
        patch("reef.harness.client.wrapper.urllib.request.urlopen", record_report),
    ):
        report("scenario", "pi", 0.0, "per turn", per_receipt=True)

    assert [item["references"] for item in posted] == [["receipt-1"], ["receipt-2"]]
    assert {item["score"] for item in posted} == {0.0}
    assert not any(tmp_path.glob("*.json"))


@pytest.mark.unit
def test_report_carries_the_installed_release_as_metadata(tmp_path) -> None:
    """report reads the release file beside the compose dir and stamps the report
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
        patch("reef.harness.client.wrapper.urllib.request.urlopen", record),
    ):
        report("scenario", "pi", 0.0, "note")

    assert posted[0]["metadata"] == {"client_release": "rel-42"}


@pytest.mark.unit
def test_run_agent_tags_records_with_the_installed_release(tmp_path) -> None:
    """run_agent reads the release file and sends x-reef-tag-release, so the record
    keeps which release answered."""
    from reef.harness.client import wrapper as harness_wrapper

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
        patch("reef.harness.client.wrapper.urllib.request.urlopen", flaky),
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
        patch("reef.harness.client.wrapper._process_start_id", return_value="current-process"),
        patch("reef.harness.client.wrapper.urllib.request.urlopen", record_report),
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
        patch("reef.harness.client.wrapper.urllib.request.urlopen", record_report),
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
        "from reef.harness.runners.native import main\nsys.exit(main())\n"
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
    from reef.harness.client.wrapper import _create_temp_composition, _extract_reef_url
    from reef.harness.episodes.model_binding import ModelBinding
    from reef.harness.tree.render import render_composition

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
    from reef.harness.client.wrapper import _create_temp_composition, _extract_reef_url

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
    from reef.harness.client.wrapper import WrapperError, _create_temp_composition, _extract_reef_url

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
    from reef.harness.client.wrapper import _create_temp_composition, _extract_reef_url

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
def test_wrapper_takes_the_token_from_the_binding_when_the_shell_has_none(tmp_path, monkeypatch) -> None:
    """The install wrote the token at the binding's key path; a later shell needs no REEF_TOKEN, one that sets it
    still wins, and a second provider's key in the same file is never Reef's, whatever the two are called."""
    from reef.harness.client.wrapper import _reef_token

    reef = {"api": "openai-completions", "baseUrl": "http://127.0.0.1:8900/v1", "apiKey": "from-binding"}
    other = {"api": "anthropic-messages", "baseUrl": "https://api.anthropic.com", "apiKey": "other"}
    compose = _pi_tree(tmp_path, {"providers": {"anthropic": other, "reef": reef}})
    monkeypatch.delenv("REEF_TOKEN", raising=False)
    assert _reef_token("pi", compose) == "from-binding"
    monkeypatch.setenv("REEF_TOKEN", "from-shell")
    assert _reef_token("pi", compose) == "from-shell"
    monkeypatch.delenv("REEF_TOKEN")
    # An install without REEF_TOKEN left Reef's key empty: nothing to send, and never the other provider's key.
    empty = _pi_tree(tmp_path / "empty", {"providers": {"anthropic": other, "reef": {**reef, "apiKey": ""}}})
    assert _reef_token("pi", empty) is None
    assert _reef_token("pi", "") is None
    # Reef reached by a host named reef, beside a provider that sorts after it: the key path settles it.
    docker = _pi_tree(
        tmp_path / "docker",
        {"providers": {"reef": {**reef, "baseUrl": "http://reef:8900/v1"}, "zai": {**other, "apiKey": "zk"}}},
    )
    assert _reef_token("pi", docker) == "from-binding"


@pytest.mark.unit
@pytest.mark.parametrize("adapter", ["pi", "opencode", "claude", "codex", "dsh", "hermes", "native"])
def test_wrapper_reads_the_token_back_from_every_adapters_binding(tmp_path, adapter, monkeypatch) -> None:
    """Every adapter's binding file, in its own format (JSON, TOML, YAML, dotenv), yields the token the install
    rendered into it, byte for byte, characters a text search would cut at included."""
    from reef.harness.adapters import get_adapter
    from reef.harness.client.wrapper import _reef_token
    from reef.harness.episodes.model_binding import ModelBinding
    from reef.harness.tree.render import render_composition

    descriptor = get_adapter(adapter)
    token = "sk-abc,def;g h\"i'j<k>l\\m"
    monkeypatch.delenv("REEF_TOKEN", raising=False)
    for api in descriptor.model_binding:
        reef = ModelBinding(base_url="http://127.0.0.1:8900", model="qwen3-8b", api_key=token, api=api)
        files = render_composition([("rules", {"text": "Be brief."}), *reef.compose_nodes(descriptor)], descriptor)
        root = tmp_path / api
        for relative, text in files.items():
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_text(text, encoding="utf-8")
        _, subdir = descriptor.compose_relocation()
        assert _reef_token(adapter, str(root / subdir)) == token


@pytest.mark.unit
def test_wrapper_refuses_a_binding_file_that_leaves_the_composition(tmp_path, monkeypatch) -> None:
    from dataclasses import replace

    from reef.harness.adapters import get_adapter
    from reef.harness.client.wrapper import WrapperError, _extract_reef_url

    base = get_adapter("native")
    for path, message in (
        ("native/../outside.json", "escapes the tree"),
        ("elsewhere/models.json", "outside the composition"),
    ):
        targets = dict(base.config_targets)
        targets["models"] = replace(targets["models"], path=path)
        descriptor = replace(base, config_targets=targets)
        monkeypatch.setattr("reef.harness.client.wrapper.get_adapter", lambda name, d=descriptor: d)
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


# -- reef-<adapter> harness: submit native manual training ---------------------


class _FakeReef:
    """A reef that records every POST: inference answers with a receipt, the request route with ``answer``."""

    def __init__(self, answer: dict, *, status: int = 200, receipt: str = "ask-receipt") -> None:
        import http.server
        import threading

        self.seen: list[dict] = []
        seen = self.seen

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                seen.append(
                    {"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body}
                )
                if self.path.startswith("/v1/chat/completions"):
                    code, payload = 200, {"choices": [{"message": {"content": "ok"}}]}
                    extra = {"x-reef-agent-record-id": receipt}
                elif self.path == "/reef/train":
                    code, payload, extra = status, answer, {}
                else:
                    code, payload, extra = 200, {}, {}
                raw = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                for name, value in extra.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.port = self._server.server_address[1]

    def posts(self, path: str) -> list[dict]:
        return [call for call in self.seen if call["path"] == path]

    def close(self) -> None:
        self._server.shutdown()


def _ask_tree(tmp_path: Path, port: int, *, with_release_file: bool = True) -> tuple[str, Path]:
    """A pi composition bound to the reef at ``port``, the release file beside it, and an empty spool directory."""
    compose = _make_compose(tmp_path, port)
    if with_release_file:
        (tmp_path / ".reef-harness-release").write_text(json.dumps({"release_id": "rel-3"}), encoding="utf-8")
    captures = tmp_path / "captures"
    captures.mkdir()
    return compose, captures


def _ask_env(captures: Path, compose: str, **extra: str) -> dict[str, str]:
    env = {**os.environ, "REEF_HARNESS_CAPTURES_DIR": str(captures), "REEF_HARNESS_COMPOSE": compose, **extra}
    if "REEF_TOKEN" not in extra:
        env.pop("REEF_TOKEN", None)
    return env


@pytest.mark.unit
def test_harness_submits_training_and_preserves_the_last_sessions_receipts(tmp_path, capsys) -> None:
    """Manual training carries the session id without fabricating feedback."""
    reef = _FakeReef({"agent_record_id": "q-1", "scenario": "ask-scenario", "request_type": "train"})
    compose, captures = _ask_tree(tmp_path, reef.port)
    binary = _make_fake_pi(tmp_path, reef.port)
    with patch.dict(os.environ, _ask_env(captures, compose, REEF_TOKEN="tok"), clear=True):
        with contextlib.suppress(SystemExit):
            run_agent(str(binary), compose, "ask-scenario", "pi", "PI_CODING_AGENT_DIR", ["-p", "hi"])
        harness("ask-scenario", "pi", compose, "text me when you are blocked")
    reef.close()

    (request,) = reef.posts("/reef/train")
    assert set(request["body"]) == {"text", "session", "release_id"}
    assert request["body"]["text"] == "text me when you are blocked"
    assert request["body"]["release_id"] == "rel-3"
    # The proxy stamps a session tag on every call, so the request names the session that ran; the spool carries it.
    session = request["body"]["session"]
    uuid.UUID(session)
    (call,) = [seen for seen in reef.seen if seen["path"].startswith("/v1/chat/completions")]
    assert call["headers"]["x-reef-tag-session"] == session  # the tag rode the model call the session made
    assert request["headers"]["x-reef-scenario"] == "ask-scenario"
    assert request["headers"]["authorization"] == "Bearer tok"
    assert request["headers"]["content-type"] == "application/json"
    assert not reef.posts("/reef/report")
    (pending,) = captures.glob("*.pending.json")
    assert json.loads(pending.read_text())["turns"][0]["receipt"] == "ask-receipt"
    out = capsys.readouterr().out
    assert "reef-pi: training request q-1 accepted" in out


@pytest.mark.unit
def test_harness_without_spooled_receipts_submits_training(tmp_path, capsys) -> None:
    reef = _FakeReef({"agent_record_id": "q-2", "scenario": "ask-scenario", "request_type": "train"})
    compose, captures = _ask_tree(tmp_path, reef.port)
    with patch.dict(os.environ, _ask_env(captures, compose), clear=True):
        harness("ask-scenario", "pi", compose, "read papers first")
    reef.close()

    (request,) = reef.seen
    assert request["path"] == "/reef/train"
    assert request["body"]["text"] == "read papers first"
    assert request["body"]["release_id"] == "rel-3"
    assert request["headers"]["authorization"] == "Bearer dummy"  # no REEF_TOKEN in the shell: models.json's apiKey
    out = capsys.readouterr().out
    assert "reef-pi: training request q-2 accepted" in out
    uuid.UUID(request["body"]["session"])


@pytest.mark.unit
def test_run_agent_reaches_reef_with_the_bindings_token_when_the_shell_has_none(tmp_path) -> None:
    """The proxy and the agent's own extensions carry the token the install wrote, so a plain shell runs the tree."""
    reef = _FakeReef({"agent_record_id": "q-3", "scenario": "ask-scenario", "request_type": "train"})
    compose, captures = _ask_tree(tmp_path, reef.port)
    binary = _make_fake_pi(tmp_path, reef.port)
    with patch.dict(os.environ, _ask_env(captures, compose), clear=True), contextlib.suppress(SystemExit):
        run_agent(str(binary), compose, "ask-scenario", "pi", "PI_CODING_AGENT_DIR", ["-p", "hi"])
    reef.close()

    (call,) = [seen for seen in reef.seen if seen["path"].startswith("/v1/chat/completions")]
    assert call["headers"]["authorization"] == "Bearer dummy"  # models.json's apiKey, written by the install


@pytest.mark.unit
def test_harness_names_the_session_the_spool_recorded(tmp_path) -> None:
    """A saved capture supplies the session id and remains available for a later report."""
    reef = _FakeReef({"agent_record_id": "q-3", "scenario": "ask-scenario", "request_type": "train"})
    compose, captures = _ask_tree(tmp_path, reef.port)
    key = hashlib.sha256(b"ask-scenario").hexdigest()
    spooled = {
        "reef_url": f"http://127.0.0.1:{reef.port}",
        "scenario": "ask-scenario",
        "turns": [{"receipt": None, "session_id": ""}, {"receipt": "r-9", "session_id": "sess-9"}],
    }
    (captures / f"{key}-{1:020d}-run.pending.json").write_text(json.dumps(spooled), encoding="utf-8")
    with patch.dict(os.environ, _ask_env(captures, compose), clear=True):
        harness("ask-scenario", "pi", compose, "text me")
    reef.close()

    (request,) = reef.posts("/reef/train")
    assert request["body"]["session"] == "sess-9"
    assert not reef.posts("/reef/report")
    (pending,) = captures.glob("*.pending.json")
    assert json.loads(pending.read_text()) == spooled


@pytest.mark.unit
def test_harness_auto_mode_refusal_keeps_the_spool(tmp_path) -> None:
    reef = _FakeReef({"error": "training requests require training_mode='manual'"}, status=400)
    compose, captures = _ask_tree(tmp_path, reef.port)
    pending = _write_spool_entry(captures, "ask-scenario", "pending")
    with (
        patch.dict(os.environ, _ask_env(captures, compose), clear=True),
        pytest.raises(SystemExit, match="training requests require training_mode='manual'"),
    ):
        harness("ask-scenario", "pi", compose, "ignore the rules")
    reef.close()

    assert [call["path"] for call in reef.seen] == ["/reef/train"]
    assert pending.exists()


@pytest.mark.unit
def test_harness_rejected_body_exits_with_the_status_and_the_detail(tmp_path) -> None:
    reef = _FakeReef({"error": "release_id must be a string"}, status=400)
    compose, captures = _ask_tree(tmp_path, reef.port)
    pending = _write_spool_entry(captures, "ask-scenario", "pending")
    with (
        patch.dict(os.environ, _ask_env(captures, compose), clear=True),
        pytest.raises(SystemExit, match=r"request failed \(400\): .*release_id must be a string"),
    ):
        harness("ask-scenario", "pi", compose, "text me")
    reef.close()

    assert [call["path"] for call in reef.seen] == ["/reef/train"]
    assert pending.exists()


@pytest.mark.unit
def test_harness_200_without_a_record_id_exits_with_the_body_not_a_traceback(tmp_path) -> None:
    reef = _FakeReef({"scenario": "ask-scenario", "request_type": "train"})
    compose, captures = _ask_tree(tmp_path, reef.port)
    with (
        patch.dict(os.environ, _ask_env(captures, compose), clear=True),
        pytest.raises(SystemExit, match=r"answered 200 without an agent_record_id: .*ask-scenario"),
    ):
        harness("ask-scenario", "pi", compose, "text me")
    reef.close()

    assert [call["path"] for call in reef.seen] == ["/reef/train"]


@pytest.mark.unit
def test_harness_unreachable_exits_and_keeps_the_spool(tmp_path) -> None:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    compose, captures = _ask_tree(tmp_path, port)
    pending = _write_spool_entry(captures, "ask-scenario", "pending")
    with (
        patch.dict(os.environ, _ask_env(captures, compose), clear=True),
        pytest.raises(SystemExit, match=f"reef-pi: reef unreachable at http://127.0.0.1:{port}"),
    ):
        harness("ask-scenario", "pi", compose, "text me")
    assert pending.exists()


@pytest.mark.unit
def test_harness_without_the_release_file_sends_nothing(tmp_path) -> None:
    reef = _FakeReef({"agent_record_id": "q-0", "scenario": "ask-scenario", "request_type": "train"})
    compose, captures = _ask_tree(tmp_path, reef.port, with_release_file=False)
    pending = _write_spool_entry(captures, "ask-scenario", "pending")
    with patch.dict(os.environ, _ask_env(captures, compose), clear=True):
        with pytest.raises(SystemExit, match=r"no \.reef-harness-release release file at .*nothing was sent"):
            harness("ask-scenario", "pi", compose, "text me")
        with pytest.raises(SystemExit, match="the request is empty"):
            harness("ask-scenario", "pi", compose, "   ")
    reef.close()

    assert reef.seen == []
    assert pending.exists()


@pytest.mark.unit
def test_main_dispatches_harness_with_the_words_joined(tmp_path) -> None:
    asked: list[tuple] = []
    env = {
        "REEF_HARNESS_BINARY": "fake-pi",
        "REEF_HARNESS_COMPOSE": str(tmp_path),
        "REEF_HARNESS_SCENARIO": "ask-scenario",
        "REEF_HARNESS_ADAPTER": "pi",
        "REEF_HARNESS_ENV_VAR": "PI_CODING_AGENT_DIR",
    }
    with (
        patch.dict(os.environ, env),
        patch("reef.harness.client.wrapper.harness", lambda *args: asked.append(args)),
        patch("sys.argv", ["reef-pi", "harness", "text", "me", "when", "you", "are", "blocked"]),
    ):
        main()
    assert asked == [("ask-scenario", "pi", str(tmp_path), "text me when you are blocked")]


def test_a_clear_from_the_agent_empties_what_the_wrapper_would_spool() -> None:
    """DELETE /_captures after an in session report: publish_turn spools nothing, so no later report resends them."""
    from reef_client.serve import CapturedTurn

    from reef.harness.client.wrapper import CaptureProxy

    proxy = CaptureProxy("http://127.0.0.1:9", "clear-scenario", None, tags={"release": "r"})
    proxy.start()
    try:
        proxy._store.add(CapturedTurn("", "/v1/chat/completions", 200, None, None, "r-1", False, 0.0))
        assert len(proxy._store.snapshot()) == 1
        request = urllib.request.Request(f"http://127.0.0.1:{proxy.port}/_captures", method="DELETE")
        with urllib.request.urlopen(request, timeout=5) as response:
            assert json.loads(response.read())["cleared"] == 1
        assert proxy.publish_turn() == 0
    finally:
        proxy.stop()


# -- reef-<adapter> setup: check off what the newest release requires ---------------------------


class _ReleasesReef:
    """A reef whose GET /reef/harness/releases answers ``rows``; every request is recorded."""

    def __init__(self, rows: list[dict]) -> None:
        import http.server
        import threading

        self.seen: list[dict] = []
        seen = self.seen

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}})
                found = self.path == "/reef/harness/releases"
                raw = json.dumps({"scenario": "setup-scenario", "releases": rows} if found else {}).encode()
                self.send_response(200 if found else 404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.port = self._server.server_address[1]

    def close(self) -> None:
        self._server.shutdown()


def _row(release_id: str, requires: list[dict] | None = None, *, pending: bool = False) -> dict:
    row: dict = {"release_id": release_id, "pending": pending, "current": False, "operation": "training"}
    if requires is not None:
        row["metrics"] = {"training_request": {"id": "q-1", "text": "text me", "requires": requires}}
    return row


def _setup_tree(tmp_path: Path, port: int, release_info: dict) -> tuple[str, Path]:
    """A pi composition bound to the reef at ``port`` with the given release file beside it."""
    compose = _make_compose(tmp_path, port)
    path = tmp_path / ".reef-harness-release"
    path.write_text(json.dumps(release_info, indent=2) + "\n", encoding="utf-8")
    return compose, path


@pytest.mark.unit
def test_setup_lists_the_head_rows_items_runs_checks_after_yes_and_records_the_check_offs(tmp_path, capsys) -> None:
    """The newest row that is not pending is the head; ``--yes`` runs each unmet check; a passing one is checked
    off in the release file, a failing one is not; a later run does not run a checked off item again."""
    ran = tmp_path / "ran"
    rows = [
        _row("v1"),
        _row(
            "v2",
            [
                {"name": "TWILIO_SID", "kind": "env", "check": "TWILIO_SID"},
                {"name": "notify", "kind": "permission", "check": f"touch {ran}"},
                {"name": "twilio", "kind": "service", "check": "exit 3"},
            ],
        ),
        _row("v3", [{"name": "later", "kind": "env"}], pending=True),
    ]
    reef = _ReleasesReef(rows)
    compose, release_file = _setup_tree(tmp_path, reef.port, {"release_id": "v1", "requires": [], "setup": []})
    env = _ask_env(tmp_path / "captures", compose, REEF_TOKEN="tok", TWILIO_SID="AC123")
    with patch.dict(os.environ, env, clear=True):
        assert setup("setup-scenario", "pi", compose, yes=True) == 1
    assert ran.exists()
    assert capsys.readouterr().out.splitlines() == [
        "reef-pi setup: release v2 requires 3 item(s)",
        "  TWILIO_SID (env): TWILIO_SID",
        "    met",
        f"  notify (permission): touch {ran}",
        "    met",
        "  twilio (service): exit 3",
        "    not met (exit 3)",
        "reef-pi setup: 1 item(s) not met: twilio",
    ]
    (call,) = reef.seen
    assert call["path"] == "/reef/harness/releases"
    assert call["headers"]["x-reef-scenario"] == "setup-scenario" and call["headers"]["authorization"] == "Bearer tok"
    record = json.loads(release_file.read_text(encoding="utf-8"))
    assert [item["name"] for item in record["setup"]] == ["TWILIO_SID", "notify"]
    assert all(isinstance(item["checked_at"], float) for item in record["setup"])
    assert record["release_id"] == "v1" and record["requires"] == [] and not list(tmp_path.glob(".*.part"))
    # Checked off items are not run again; the failing one fails again and the release file is left alone.
    ran.unlink()
    before = release_file.read_bytes()
    with patch.dict(os.environ, env, clear=True):
        assert setup("setup-scenario", "pi", compose, yes=True) == 1
    assert not ran.exists() and release_file.read_bytes() == before
    out = capsys.readouterr().out
    assert out.count("    met (checked off)") == 2 and "    not met (exit 3)" in out
    # A check off by hand runs nothing, and with every item met the status is 0.
    with patch.dict(os.environ, env, clear=True):
        assert setup("setup-scenario", "pi", compose, marks=("twilio",)) == 0
    out = capsys.readouterr().out
    assert "    met (marked by hand)" in out and out.splitlines()[-1].startswith("reef-pi setup: every item is met")
    assert [item["name"] for item in json.loads(release_file.read_text())["setup"]] == [
        "TWILIO_SID",
        "notify",
        "twilio",
    ]
    # An unknown name is refused with the list, and nothing changes.
    before = release_file.read_bytes()
    with patch.dict(os.environ, env, clear=True):
        assert setup("setup-scenario", "pi", compose, marks=("nope",)) == 2
    err = capsys.readouterr().err
    assert "no item named nope; release v2 requires TWILIO_SID, notify, twilio" in err
    assert release_file.read_bytes() == before
    reef.close()


@pytest.mark.unit
def test_setup_asks_before_running_a_check_and_reads_an_unset_variable_without_asking(tmp_path, capsys) -> None:
    ran = tmp_path / "ran"
    rows = [
        _row("v1"),
        _row(
            "v2", [{"name": "notify", "kind": "permission", "check": f"touch {ran}"}, {"name": "SMTP", "kind": "env"}]
        ),
    ]
    reef = _ReleasesReef(rows)
    compose, release_file = _setup_tree(tmp_path, reef.port, {"release_id": "v1"})
    env = _ask_env(tmp_path / "captures", compose)
    env.pop("SMTP", None)
    with patch.dict(os.environ, env, clear=True), patch("sys.stdin", io.StringIO("n\n")):
        assert setup("setup-scenario", "pi", compose) == 1
    assert not ran.exists() and "setup" not in json.loads(release_file.read_text())
    out = capsys.readouterr().out
    assert "    run it? [y/N] " in out and "    skipped" in out and "    not set" in out
    assert out.splitlines()[-1] == "reef-pi setup: 2 item(s) not met: notify, SMTP"
    # A yes runs it; the variable is read from the environment, its check being its name.
    with patch.dict(os.environ, {**env, "SMTP": "smtp.example"}, clear=True), patch("sys.stdin", io.StringIO("y\n")):
        assert setup("setup-scenario", "pi", compose) == 0
    assert ran.exists()
    assert [item["name"] for item in json.loads(release_file.read_text())["setup"]] == ["notify", "SMTP"]
    reef.close()


@pytest.mark.unit
def test_setup_without_a_release_file_a_reef_or_any_item_says_so(tmp_path, capsys) -> None:
    import socket

    reef = _ReleasesReef([_row("v1"), _row("v2", []), _row("v3", [{"name": "later", "kind": "env"}], pending=True)])
    compose, _ = _setup_tree(tmp_path, reef.port, {"release_id": "v1"})
    with patch.dict(os.environ, _ask_env(tmp_path / "captures", compose), clear=True):
        assert setup("setup-scenario", "pi", compose, yes=True) == 0
    assert capsys.readouterr().out == "reef-pi setup: release v2 requires nothing\n"
    reef.close()
    (tmp_path / ".reef-harness-release").unlink()
    with (
        patch.dict(os.environ, _ask_env(tmp_path / "captures", compose), clear=True),
        pytest.raises(SystemExit, match=r"no \.reef-harness-release release file at .*nowhere to record a check off"),
    ):
        setup("setup-scenario", "pi", compose)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    (tmp_path / "down").mkdir()
    compose, _ = _setup_tree(tmp_path / "down", port, {"release_id": "v1"})
    with (
        patch.dict(os.environ, _ask_env(tmp_path / "captures", compose), clear=True),
        pytest.raises(SystemExit, match=f"reef-pi: reef unreachable at http://127.0.0.1:{port}"),
    ):
        setup("setup-scenario", "pi", compose)


@pytest.mark.unit
def test_run_agent_prints_the_unmet_list_once_and_runs_the_session_without_a_check(tmp_path, capsys) -> None:
    reef = _FakeReef({"agent_record_id": "q-1", "scenario": "ask-scenario", "request_type": "train"})
    ran = tmp_path / "ran"
    release_info = {
        "release_id": "v2",
        "requires": [
            {"name": "notify", "kind": "permission", "check": f"touch {ran}"},
            {"name": "TWILIO_SID", "kind": "env", "check": "TWILIO_SID"},
        ],
        "setup": [{"name": "TWILIO_SID", "checked_at": 1.0}],
    }
    compose, _ = _setup_tree(tmp_path, reef.port, release_info)
    binary = _make_fake_pi(tmp_path, reef.port)
    captures = tmp_path / "captures"
    captures.mkdir()
    with patch.dict(os.environ, _ask_env(captures, compose), clear=True), contextlib.suppress(SystemExit):
        run_agent(str(binary), compose, "ask-scenario", "pi", "PI_CODING_AGENT_DIR", ["-p", "hi"])
    reef.close()
    err = capsys.readouterr().err
    assert err.splitlines() == [
        "reef-pi: this release requires setup you have not checked off; run reef-pi setup:",
        f"  notify (permission): touch {ran}",
    ]
    assert not ran.exists()
    # The session ran through the proxy as always: its receipt is spooled.
    (spooled,) = captures.glob("*.pending.json")
    assert json.loads(spooled.read_text())["turns"][0]["receipt"] == "ask-receipt"


@pytest.mark.unit
def test_main_dispatches_setup_with_yes_and_marks_and_exits_with_its_status(tmp_path) -> None:
    called: list[tuple] = []
    env = {
        "REEF_HARNESS_BINARY": "fake-pi",
        "REEF_HARNESS_COMPOSE": str(tmp_path),
        "REEF_HARNESS_SCENARIO": "setup-scenario",
        "REEF_HARNESS_ADAPTER": "pi",
        "REEF_HARNESS_ENV_VAR": "PI_CODING_AGENT_DIR",
    }
    with (
        patch.dict(os.environ, env),
        patch("reef.harness.client.wrapper.setup", lambda *args, **kwargs: called.append((args, kwargs)) or 1),
        patch("sys.argv", ["reef-pi", "setup", "--yes", "--mark", "a", "--mark", "b"]),
        pytest.raises(SystemExit) as exited,
    ):
        main()
    assert exited.value.code == 1
    assert called == [(("setup-scenario", "pi", str(tmp_path)), {"yes": True, "marks": ("a", "b")})]


def _chain_row(release_id: str, parent: str | None, requires: list[dict] | None = None, *, pending: bool = False):
    return {**_row(release_id, requires, pending=pending), "parent_release_id": parent}


@pytest.mark.unit
def test_setup_reads_the_chains_union_and_release_names_a_pending_row(tmp_path, capsys) -> None:
    """What a release requires is every item over its chain, as the manifest lists it; ``--release`` names any
    catalog row, a pending one included; an unknown id, a catalog with nothing served and a row without an id
    each say what they are, and an unknown ``--mark`` is exit 2 even when nothing is required."""
    rows = [
        _chain_row("v1", None),
        _chain_row("v2", "v1", [{"name": "TWILIO_SID", "kind": "env"}]),
        _chain_row("v3", "v2", []),
        _chain_row("v4", "v3", [{"name": "later", "kind": "env", "check": "LATER"}], pending=True),
    ]
    reef = _ReleasesReef(rows)
    compose, release_file = _setup_tree(tmp_path, reef.port, {"release_id": "v1"})
    env = _ask_env(tmp_path / "captures", compose, TWILIO_SID="AC1", LATER="x")
    with patch.dict(os.environ, env, clear=True):
        assert setup("setup-scenario", "pi", compose, yes=True) == 0
    assert capsys.readouterr().out.splitlines() == [
        "reef-pi setup: release v3 requires 1 item(s)",
        "  TWILIO_SID (env)",
        "    met",
        "reef-pi setup: every item is met; install the release when the notice offers it",
    ]
    # The pending v4 by name: its chain's item, checked off already, and its own.
    with patch.dict(os.environ, env, clear=True):
        assert setup("setup-scenario", "pi", compose, yes=True, release="v4") == 0
    out = capsys.readouterr().out.splitlines()
    assert out[:5] == [
        "reef-pi setup: release v4 requires 2 item(s)",
        "  TWILIO_SID (env)",
        "    met (checked off)",
        "  later (env): LATER",
        "    met",
    ]
    assert [item["name"] for item in json.loads(release_file.read_text())["setup"]] == ["TWILIO_SID", "later"]
    with patch.dict(os.environ, env, clear=True):
        assert setup("setup-scenario", "pi", compose, release="nope") == 2
    assert capsys.readouterr().err == "reef-pi setup: no release nope in the catalog\n"
    reef.close()
    # Nothing served yet: said so, and there is nothing to check off.
    reef = _ReleasesReef([_chain_row("v9", None, [{"name": "x", "kind": "env"}], pending=True)])
    (tmp_path / "waiting").mkdir()
    compose, _ = _setup_tree(tmp_path / "waiting", reef.port, {"release_id": "v1"})
    with patch.dict(os.environ, _ask_env(tmp_path / "captures", compose), clear=True):
        assert setup("setup-scenario", "pi", compose, yes=True) == 0
    assert capsys.readouterr().out == "reef-pi setup: no served release yet\n"
    reef.close()
    # A row without an id prints without one; an unknown mark is refused before the nothing required return.
    reef = _ReleasesReef([{"pending": False, "operation": "creation", "current": True}])
    (tmp_path / "bare").mkdir()
    compose, _ = _setup_tree(tmp_path / "bare", reef.port, {"release_id": "v1"})
    with patch.dict(os.environ, _ask_env(tmp_path / "captures", compose), clear=True):
        assert setup("setup-scenario", "pi", compose) == 0
        assert setup("setup-scenario", "pi", compose, marks=("nope",)) == 2
    captured = capsys.readouterr()
    assert captured.out == "reef-pi setup: requires nothing\n"
    assert captured.err == "reef-pi setup: no item named nope; requires nothing\n"
    reef.close()


@pytest.mark.unit
def test_setup_runs_an_item_again_when_its_check_changed_since_the_check_off(tmp_path, capsys) -> None:
    """A check off records the check it stood for: an item whose check differs counts as unmet everywhere the
    check offs are read, setup runs it again, and the new record carries the new check."""
    from reef.harness.client.wrapper import _unmet

    ran = tmp_path / "ran"
    item = {"name": "notify", "kind": "permission", "check": f"touch {ran}"}
    stale = {"name": "notify", "checked_at": 1.0, "check": "touch elsewhere"}
    assert _unmet([item], [stale]) == [item]
    assert _unmet([item], [{**stale, "check": item["check"]}]) == [] and _unmet([item], [{"name": "notify"}]) == []
    reef = _ReleasesReef([_row("v1"), _row("v2", [item])])
    compose, release_file = _setup_tree(tmp_path, reef.port, {"release_id": "v1", "setup": [stale]})
    with patch.dict(os.environ, _ask_env(tmp_path / "captures", compose), clear=True):
        assert setup("setup-scenario", "pi", compose, yes=True) == 0
    assert ran.exists()
    out = capsys.readouterr().out.splitlines()
    assert out[1:4] == [
        f"  notify (permission): touch {ran}",
        "    the check changed since it was checked off",
        "    met",
    ]
    (record,) = json.loads(release_file.read_text())["setup"]
    assert record["name"] == "notify" and record["check"] == item["check"] and record["checked_at"] != 1.0
    # Marked by hand, the record carries the item's check too.
    reef.close()
    reef = _ReleasesReef([_row("v1"), _row("v2", [{**item, "check": "exit 1"}])])
    (tmp_path / "marked").mkdir()
    compose, release_file = _setup_tree(tmp_path / "marked", reef.port, {"release_id": "v1", "setup": [record]})
    with patch.dict(os.environ, _ask_env(tmp_path / "captures", compose), clear=True):
        assert setup("setup-scenario", "pi", compose, marks=("notify",)) == 0
    (record,) = json.loads(release_file.read_text())["setup"]
    assert record["check"] == "exit 1"
    reef.close()


@pytest.mark.unit
def test_main_passes_release_to_setup_only_when_named(tmp_path) -> None:
    called: list[tuple] = []
    env = {
        "REEF_HARNESS_BINARY": "fake-pi",
        "REEF_HARNESS_COMPOSE": str(tmp_path),
        "REEF_HARNESS_SCENARIO": "setup-scenario",
        "REEF_HARNESS_ADAPTER": "pi",
        "REEF_HARNESS_ENV_VAR": "PI_CODING_AGENT_DIR",
    }
    with (
        patch.dict(os.environ, env),
        patch("reef.harness.client.wrapper.setup", lambda *args, **kwargs: called.append((args, kwargs)) or 0),
        patch("sys.argv", ["reef-pi", "setup", "--release", "v4"]),
        pytest.raises(SystemExit) as exited,
    ):
        main()
    assert exited.value.code == 0
    assert called == [(("setup-scenario", "pi", str(tmp_path)), {"yes": False, "marks": (), "release": "v4"})]


# -- reef-<adapter> doctor: one report of what the install needs ---------------------------------


class _DoctorReef:
    """A reef whose status route checks the bearer and whose catalog names one served head."""

    def __init__(self, token: str, head: str) -> None:
        import http.server
        import threading

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/reef/status":
                    ok = self.headers.get("Authorization") == f"Bearer {token}"
                    code, payload = (200, {"scenarios": {}}) if ok else (401, {"error": "invalid service token"})
                elif self.path == "/reef/harness/releases":
                    code, payload = 200, {"releases": [{"release_id": head, "pending": False}]}
                else:
                    code, payload = 404, {}
                raw = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.port = self._server.server_address[1]

    def close(self) -> None:
        self._server.shutdown()


@pytest.mark.unit
def test_doctor_reports_every_line_and_exits_by_the_worst_of_them(tmp_path, capsys, monkeypatch) -> None:
    from reef.harness.client.wrapper import doctor

    reef = _DoctorReef(token="dummy", head="rel-3")
    compose, _ = _ask_tree(tmp_path, reef.port)  # models.json binds the token dummy; release file names rel-3
    binary = tmp_path / "fake-pi"
    binary.write_text("#!/bin/sh\necho 0.84.2\n")
    binary.chmod(0o755)
    monkeypatch.delenv("REEF_TOKEN", raising=False)
    monkeypatch.setattr("shutil.which", lambda command: f"/usr/bin/{command}" if command == "rg" else None)
    assert doctor("doc-scenario", "pi", compose, str(binary)) == 1  # fd is missing
    out = capsys.readouterr().out.splitlines()
    assert any(line.startswith("ok  interpreter") and "reef " in line for line in out)
    assert any(line.startswith("ok  service") and "token accepted" in line for line in out)
    assert any(line.startswith("ok  binary") and "0.84.2" in line for line in out)
    assert any(line.startswith("ok  tool") and "rg at /usr/bin/rg" in line for line in out)
    assert any(line.startswith("!!  tool") and "fd missing: install fd" in line for line in out)
    assert any(line.startswith("ok  release") and "rel-3 installed, the served head" in line for line in out)
    monkeypatch.setattr("shutil.which", lambda command: f"/usr/bin/{command}")
    assert doctor("doc-scenario", "pi", compose, str(binary)) == 0
    # A wrong token in the shell wins over the binding's, and the service says so.
    monkeypatch.setenv("REEF_TOKEN", "wrong")
    assert doctor("doc-scenario", "pi", compose, str(binary)) == 1
    out = capsys.readouterr().out
    assert "!!  service" in out and "401" in out
    reef.close()
