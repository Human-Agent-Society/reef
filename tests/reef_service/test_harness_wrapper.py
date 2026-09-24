"""End-to-end smoke test for the reef-pi wrapper: run agent -> capture receipts -> report."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import textwrap
import urllib.error
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from reef.core.training_request import CLIENT_COMMANDS
from reef.harness.client.wrapper import (
    harness,
    main,
    report,
    run_agent,
    setup,
    setup_json,
    setup_run,
    setup_set,
    update,
)


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
def test_run_agent_puts_the_harness_binary_first_on_path(tmp_path) -> None:
    """An evolved tool that runs ``pi`` gets this harness's own binary, wherever
    the install put it, even when no pi is on the person's PATH."""
    compose = _make_compose(tmp_path, 1)
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    binary = bin_dir / "pi"
    seen = tmp_path / "path.txt"
    binary.write_text(f'#!/usr/bin/env python3\nimport os\nopen({str(seen)!r}, "w").write(os.environ["PATH"])\n')
    binary.chmod(0o755)

    env = {**os.environ, "REEF_HARNESS_CAPTURES_DIR": str(tmp_path)}
    with patch.dict(os.environ, env), contextlib.suppress(SystemExit):
        run_agent(str(binary), compose, "test-scenario", "pi", "PI_CODING_AGENT_DIR", ["-p", "review"])

    entries = seen.read_text().split(os.pathsep)
    assert entries[0] == str(bin_dir.resolve())
    assert os.environ["PATH"].split(os.pathsep)[0] in entries[1:]


@pytest.mark.unit
def test_pi_sessions_outlive_the_run_so_a_later_run_can_resume_them(tmp_path) -> None:
    """pi saves sessions under its agent dir, which a run points at a temp copy;
    a second run lists what the first saved, as ``pi --resume`` does."""
    compose = _make_compose(tmp_path, 1)
    binary = tmp_path / "fake-pi"
    seen = tmp_path / "seen.txt"
    binary.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import os
            from pathlib import Path
            sessions = Path(os.environ["PI_CODING_AGENT_DIR"]) / "sessions" / "--project--"
            open({str(seen)!r}, "a").write(",".join(sorted(p.name for p in sessions.glob("*.jsonl"))) + "\\n")
            sessions.mkdir(parents=True, exist_ok=True)
            (sessions / f"{{len(list(sessions.glob('*.jsonl')))}}.jsonl").write_text("{{}}\\n")
            """
        )
    )
    binary.chmod(0o755)

    env = {**os.environ, "REEF_HARNESS_CAPTURES_DIR": str(tmp_path)}
    env.pop("PI_CODING_AGENT_SESSION_DIR", None)
    with patch.dict(os.environ, env, clear=True):
        for prompt in ("first", "second"):
            with contextlib.suppress(SystemExit):
                run_agent(str(binary), compose, "test-scenario", "pi", "PI_CODING_AGENT_DIR", ["-p", prompt])

    assert seen.read_text().splitlines() == ["", "0.jsonl"]
    assert sorted(p.name for p in (Path(compose) / "sessions" / "--project--").iterdir()) == ["0.jsonl", "1.jsonl"]


@pytest.mark.unit
def test_hermes_state_database_outlives_the_run_so_a_later_run_can_resume_it(tmp_path) -> None:
    """hermes reads its sessions from state.db; the run finds a database already kept in the installed tree."""
    compose = tmp_path / "compose"
    compose.mkdir()
    (compose / "config.yaml").write_text(
        yaml.safe_dump({"model": {"provider": "custom", "base_url": "http://127.0.0.1:1/v1", "api_key": "dummy"}})
    )
    binary = tmp_path / "fake-hermes"
    seen = tmp_path / "seen.txt"
    binary.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import os, sqlite3, sys
            from pathlib import Path
            path = Path(os.environ["HERMES_HOME"]) / "state.db"
            header = path.read_bytes()[:16]
            database = sqlite3.connect(path)
            database.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT)")
            open({str(seen)!r}, "a").write(f"{{header!r}} {{[row for (row,) in database.execute('SELECT id FROM sessions')]}}\\n")
            database.execute("INSERT INTO sessions VALUES (?)", (sys.argv[-1],))
            database.commit()
            """
        )
    )
    binary.chmod(0o755)

    with patch.dict(os.environ, {**os.environ, "REEF_HARNESS_CAPTURES_DIR": str(tmp_path)}):
        for prompt in ("first", "second"):
            with contextlib.suppress(SystemExit):
                run_agent(str(binary), str(compose), "test-scenario", "hermes", "HERMES_HOME", ["-q", prompt])

    assert seen.read_text().splitlines() == ["b'SQLite format 3\\x00' []", "b'SQLite format 3\\x00' ['first']"]


@pytest.mark.unit
def test_claude_settings_file_outlives_the_run_with_its_mode(tmp_path) -> None:
    """A kept file the binary creates, or renames a new file over, is copied back with its mode after the run;
    a later run reads it through the link."""
    compose = tmp_path / "claude-tree" / "claude"
    compose.mkdir(parents=True)
    (compose / "settings.json").write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:1"}}) + "\n")
    binary = tmp_path / "fake-claude"
    seen = tmp_path / "seen.txt"
    binary.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import json, os, sys
            from pathlib import Path
            path = Path(os.environ["CLAUDE_CONFIG_DIR"]) / ".claude.json"
            state = json.loads(path.read_text()) if path.exists() else {{}}
            open({str(seen)!r}, "a").write(f"{{path.is_symlink()}} {{state}}\\n")
            staging = path.with_name(".claude.json.tmp")
            staging.write_text(json.dumps({{"numStartups": state.get("numStartups", 0) + 1}}))
            staging.chmod(0o600)
            os.replace(staging, path)
            """
        )
    )
    binary.chmod(0o755)

    with patch.dict(os.environ, {**os.environ, "REEF_HARNESS_CAPTURES_DIR": str(tmp_path)}):
        for prompt in ("first", "second"):
            with contextlib.suppress(SystemExit):
                run_agent(str(binary), str(compose), "test-scenario", "claude", "CLAUDE_CONFIG_DIR", ["-p", prompt])

    assert seen.read_text().splitlines() == ["False {}", "True {'numStartups': 1}"]
    kept = compose / ".claude.json"
    assert json.loads(kept.read_text()) == {"numStartups": 2}
    assert kept.stat().st_mode & 0o777 == 0o600
    assert sorted(path.name for path in compose.iterdir()) == [".claude.json", "projects", "settings.json"]


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
    """A reef that records every call: inference answers with a receipt, the request route with ``answer``,
    ``GET /reef/harness/releases`` with ``rows``, the request's record route with ``record`` (404 without one) and
    the scenario's promote route with ``promote``."""

    def __init__(
        self,
        answer: dict,
        *,
        status: int = 200,
        receipt: str = "ask-receipt",
        rows: list[dict] | None = None,
        record: dict | None = None,
        promote: dict | None = None,
    ) -> None:
        import http.server
        import threading

        self.seen: list[dict] = []
        seen = self.seen

        class Handler(http.server.BaseHTTPRequestHandler):
            def _answer(self, code: int, payload: dict, extra: dict | None = None) -> None:
                raw = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                for name, value in (extra or {}).items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                seen.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}})
                if self.path == "/reef/harness/releases":
                    self._answer(200, {"scenario": "ask-scenario", "releases": rows or []})
                elif self.path == "/reef/harness/requests/q-1/progress" and record is not None:
                    self._answer(200, record)
                else:
                    self._answer(404, {})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                seen.append(
                    {"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body}
                )
                if self.path.startswith("/v1/chat/completions"):
                    self._answer(
                        200, {"choices": [{"message": {"content": "ok"}}]}, {"x-reef-agent-record-id": receipt}
                    )
                elif self.path == "/reef/train":
                    self._answer(status, answer)
                elif self.path == "/reef/scenarios/ask-scenario/promote" and promote is not None:
                    self._answer(200, promote)
                else:
                    self._answer(200, {})

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
    assert set(request["body"]) == {"text", "session", "release_id", "client"}
    client = request["body"]["client"]
    assert client["platform"] == sys.platform and set(client["commands"]) == set(CLIENT_COMMANDS)
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
    out = capsys.readouterr().out.splitlines()
    assert out[-3] == "reef-pi: training request q-1 accepted"
    # The link to the request's page follows, with the scenario and the shell's token as query parameters.
    link = f"http://127.0.0.1:{reef.port}/reef/harness/requests/q-1/page?scenario=ask-scenario&token=tok"
    assert out[-2] == f"reef-pi: watch it here: {link}"
    assert out[-1] == "reef-pi: reef is running the step; add --wait to stay here, or check /versions later"


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


def _main_env(tmp_path: Path, scenario: str = "ask-scenario") -> dict[str, str]:
    """The five settings the install bakes into the wrapper."""
    return {
        "REEF_HARNESS_BINARY": "fake-pi",
        "REEF_HARNESS_COMPOSE": str(tmp_path),
        "REEF_HARNESS_SCENARIO": scenario,
        "REEF_HARNESS_ADAPTER": "pi",
        "REEF_HARNESS_ENV_VAR": "PI_CODING_AGENT_DIR",
    }


@pytest.mark.unit
@pytest.mark.parametrize("command", ["evolve", "harness"])
def test_main_dispatches_evolve_with_the_words_joined_and_exits_with_its_status(tmp_path: Path, command: str) -> None:
    asked: list[tuple] = []
    with (
        patch.dict(os.environ, _main_env(tmp_path)),
        patch("reef.harness.client.wrapper.harness", lambda *args, **kwargs: asked.append((args, kwargs)) or 2),
    ):
        with (
            patch("sys.argv", ["reef-pi", command, "text", "me", "when", "you", "are", "blocked"]),
            pytest.raises(SystemExit) as exited,
        ):
            main()
        assert exited.value.code == 2
        # The flags read the same after the request as before it.
        with (
            patch("sys.argv", ["reef-pi", command, "text me", "--wait", "--timeout", "30"]),
            pytest.raises(SystemExit),
        ):
            main()
    assert asked == [
        (("ask-scenario", "pi", str(tmp_path), "text me when you are blocked"), {"wait": False, "timeout_s": 1800.0}),
        (("ask-scenario", "pi", str(tmp_path), "text me"), {"wait": True, "timeout_s": 30.0}),
    ]


CREATION_ROW = {"release_id": "rel-0", "parent_release_id": None, "operation": "creation", "pending": False}


def _step_row(release_id: str, metrics: dict, *, pending: bool = False, request_id: str = "q-1") -> dict:
    """The catalog row of the step that consumed request ``request_id``, with the result ``metrics`` carry."""
    return {
        "release_id": release_id,
        "parent_release_id": "rel-0",
        "operation": "training",
        "pending": pending,
        "metrics": {"training_request": {"id": request_id, "text": "text me when you are blocked"}, **metrics},
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("row", "status", "lines"),
    [
        (
            _step_row(
                "rel-1111-selected",
                {"selected": True, "proposal_notes": {"review": {"uncovered": ["idle detection", "two way replies"]}}},
            ),
            0,
            [
                "reef-pi: 'text me when you are blocked' is published as release rel-1111. Restart reef-pi to install "
                "it (the update notice offers it).",
                "reef-pi: not covered: idle detection; two way replies",
                "reef-pi: next: reef-pi setup, then reef-pi update",
            ],
        ),
        (
            _step_row("rel-3333-pending", {"selected": True}, pending=True),
            0,
            [
                "reef-pi: 'text me when you are blocked' is ready as release rel-3333. This release changes an "
                "extension, so read it before it runs: /versions v1 opens the page, /versions v1 install "
                "serves it. Page: {page}",
                "reef-pi: next: /versions v1 install in a reef-pi session, or reef-pi setup and reef-pi update",
            ],
        ),
        (
            _step_row(
                "rel-0", {"selected": False, "selection": {"reason": "candidate missed the floor on 1 of 1 tasks"}}
            ),
            1,
            [
                "reef-pi: 'text me when you are blocked' did not pass the checks (candidate missed the floor on 1 of 1 "
                "tasks). Nothing changed; rephrase or split the request."
            ],
        ),
        (
            _step_row("rel-0", {"skipped": "no proposal"}),
            1,
            ["reef-pi: 'text me when you are blocked' produced no change (no proposal). Nothing changed."],
        ),
        (
            # The proposer's own reason rides the skip line when the step recorded one.
            _step_row(
                "rel-0",
                {
                    "skipped": "no proposal",
                    "proposal_notes": {"failure": "model call failed after 58.2 s (max_tokens=16384): timed out"},
                },
            ),
            1,
            [
                "reef-pi: 'text me when you are blocked' produced no change (no proposal: model call failed after "
                "58.2 s (max_tokens=16384): timed out). Nothing changed."
            ],
        ),
    ],
    ids=["selected", "pending", "rejected", "skipped", "skipped-failure"],
)
def test_harness_wait_prints_the_result_line_and_exits_by_it(tmp_path, capsys, row, status, lines) -> None:
    reef = _FakeReef(
        {"agent_record_id": "q-1", "scenario": "ask-scenario", "request_type": "train"}, rows=[CREATION_ROW, row]
    )
    compose, captures = _ask_tree(tmp_path, reef.port)
    with patch.dict(os.environ, _ask_env(captures, compose), clear=True):
        exit_status = harness(
            "ask-scenario", "pi", compose, "text me when you are blocked", wait=True, timeout_s=5, poll_s=0.01
        )
    reef.close()

    assert exit_status == status
    out = capsys.readouterr().out.splitlines()
    upstream = f"http://127.0.0.1:{reef.port}"
    assert out[:3] == [
        "reef-pi: training request q-1 accepted",
        f"reef-pi: watch it here: {upstream}/reef/harness/requests/q-1/page?scenario=ask-scenario&token=dummy",
        "reef-pi: reef is running the step; waiting up to 5 s for its result",
    ]
    # The pending line names the step's page link, the scenario and the token as query parameters.
    page = f"{upstream}/reef/harness/releases/1/page?scenario=ask-scenario&token=dummy"
    # Without a terminal (pytest's stdin is none) the next step is printed as commands, never asked.
    assert out[3:] == [line.replace("{page}", page) for line in lines]
    assert [call["path"] for call in reef.seen] == ["/reef/train", "/reef/harness/releases"]
    assert reef.seen[1]["headers"]["authorization"] == "Bearer dummy"  # the catalog read carries the token too


class _Tty(io.StringIO):
    """A stdin that reads as a terminal, so the wait hands over the next step and reads the answers from here."""

    def isatty(self) -> bool:
        return True


@pytest.mark.unit
def test_harness_wait_hands_over_the_next_step_on_a_terminal(tmp_path, capsys) -> None:
    """A selected release asks to install and runs the setup prompts then the update; a pending one names its page,
    asks to promote, promotes and installs the new head; a declined step prints the commands; a failed install
    keeps its status."""
    answer = {"agent_record_id": "q-1", "scenario": "ask-scenario", "request_type": "train"}
    calls: list[tuple[str, tuple, dict]] = []
    statuses = {"update": 0}

    def fake(name: str):
        return lambda *args, **kwargs: calls.append((name, args, kwargs)) or statuses.get(name, 0)

    selected = _step_row("rel-1111-selected", {"selected": True})
    pending = _step_row("rel-3333-pending", {"selected": True}, pending=True)
    promoted = {"scenario": "ask-scenario", "release_id": "rel-4444-promoted", "content_id": "c"}
    cases = [
        (
            "selected-yes",
            selected,
            "\n",
            0,
            "reef-pi: Install now? [Y/n] reef-pi: Installed release rel-1111. Restart reef-pi to use it.",
        ),
        (
            "selected-no",
            selected,
            "n\n",
            0,
            "reef-pi: Install now? [Y/n] reef-pi: next: reef-pi setup, then reef-pi update",
        ),
        (
            "pending-yes",
            pending,
            "y\n",
            0,
            "reef-pi: Promote now? [y/N] reef-pi: promoted; the served head is release rel-4444",
        ),
        (
            "pending-no",
            pending,
            "\n",
            0,
            "reef-pi: Promote now? [y/N] reef-pi: next: /versions v1 install in a reef-pi session, or reef-pi setup and reef-pi update",
        ),
        ("selected-failed", selected, "yes\n", 3, "reef-pi: Install now? [Y/n] "),
    ]
    for name, row, typed, status, line in cases:
        calls.clear()
        statuses["update"] = 3 if name == "selected-failed" else 0
        reef = _FakeReef(answer, rows=[CREATION_ROW, row], promote=promoted)
        (tmp_path / name).mkdir()
        compose, captures = _ask_tree(tmp_path / name, reef.port)
        with (
            patch.dict(os.environ, _ask_env(captures, compose), clear=True),
            patch("sys.stdin", _Tty(typed)),
            patch("reef.harness.client.wrapper.setup", fake("setup")),
            patch("reef.harness.client.wrapper.update", fake("update")),
        ):
            exit_status = harness("ask-scenario", "pi", compose, "text me", wait=True, timeout_s=5, poll_s=0.01)
        reef.close()
        out = capsys.readouterr().out.splitlines()
        assert exit_status == status, name
        # The answer is read from the terminal, so the question and what follows share a line here.
        assert line in out, (name, out)
        installed = [(call, kwargs) for call, _, kwargs in calls]
        if name == "selected-yes":
            assert installed == [
                ("setup", {"release": "rel-1111-selected"}),
                ("update", {"release": "rel-1111-selected"}),
            ]
            assert all(args == ("ask-scenario", "pi", compose) for _, args, _ in calls)
        elif name == "pending-yes":
            assert "reef-pi: read the change first: reef-pi page 1" in out
            (promote_call,) = reef.posts("/reef/scenarios/ask-scenario/promote")
            assert promote_call["body"] == {"release_id": "rel-3333-pending"}
            assert promote_call["headers"]["authorization"] == "Bearer dummy"
            assert installed == [
                ("setup", {"release": "rel-4444-promoted"}),
                ("update", {"release": "rel-4444-promoted"}),
            ]
            assert out[-1] == "reef-pi: Installed release rel-4444. Restart reef-pi to use it."
        elif name == "selected-failed":
            assert installed == [
                ("setup", {"release": "rel-1111-selected"}),
                ("update", {"release": "rel-1111-selected"}),
            ]
            assert not any("Installed release" in item for item in out)
        else:
            assert installed == [] and not reef.posts("/reef/scenarios/ask-scenario/promote")


@pytest.mark.unit
def test_harness_wait_gives_up_at_the_timeout_and_without_it_says_how_to_follow(tmp_path, capsys) -> None:
    # The catalog holds another request's step only: ours is still running.
    rows = [CREATION_ROW, _step_row("rel-1111-selected", {"selected": True}, request_id="q-other")]
    reef = _FakeReef(
        {"agent_record_id": "q-1", "scenario": "ask-scenario", "request_type": "train"},
        rows=rows,
        record={"agent_record_id": "q-1", "state": "queued"},  # queued the whole wait
    )
    compose, captures = _ask_tree(tmp_path, reef.port)
    with patch.dict(os.environ, _ask_env(captures, compose), clear=True):
        assert harness("ask-scenario", "pi", compose, "text me", wait=True, timeout_s=0.05, poll_s=0.01) == 2
        assert harness("ask-scenario", "pi", compose, "text me") == 0
    reef.close()

    out = capsys.readouterr().out.splitlines()
    assert out[3] == "reef-pi: no result yet for 'text me' after 0.05 s; /versions shows it when it settles"
    assert len([call for call in reef.seen if call["path"] == "/reef/harness/releases"]) >= 2
    link = f"http://127.0.0.1:{reef.port}/reef/harness/requests/q-1/page?scenario=ask-scenario&token=dummy"
    assert out[-3:] == [
        "reef-pi: training request q-1 accepted",
        f"reef-pi: watch it here: {link}",
        "reef-pi: reef is running the step; add --wait to stay here, or check /versions later",
    ]


@pytest.mark.unit
def test_harness_wait_says_once_when_the_record_shows_the_step_started(tmp_path, capsys) -> None:
    """Until progress reports a running step the request waits; once it starts the wait says so, once,
    and stops reading the record. A record the service does not answer is no reason to stop waiting."""
    rows = [CREATION_ROW, _step_row("rel-1111-selected", {"selected": True}, request_id="q-other")]
    answer = {"agent_record_id": "q-1", "scenario": "ask-scenario", "request_type": "train"}
    record_path = "/reef/harness/requests/q-1/progress"
    reef = _FakeReef(answer, rows=rows, record={"agent_record_id": "q-1", "state": "running"})
    compose, captures = _ask_tree(tmp_path, reef.port)
    with patch.dict(os.environ, _ask_env(captures, compose), clear=True):
        assert harness("ask-scenario", "pi", compose, "text me", wait=True, timeout_s=0.1, poll_s=0.01) == 2
    reef.close()
    out = capsys.readouterr().out.splitlines()
    assert out[3] == "reef-pi: the step started; usually one to three minutes"
    assert out[4].startswith("reef-pi: no result yet for 'text me' after 0.1 s") and len(out) == 5
    record_reads = [call for call in reef.seen if call["path"] == record_path]
    assert len(record_reads) == 1 and record_reads[0]["headers"]["authorization"] == "Bearer dummy"
    assert len([call for call in reef.seen if call["path"] == "/reef/harness/releases"]) >= 3
    # Queued (explicit progress): no line, and the record is read again at every poll.
    reef = _FakeReef(answer, rows=rows, record={"agent_record_id": "q-1", "state": "queued"})
    (tmp_path / "queued").mkdir()
    compose, captures = _ask_tree(tmp_path / "queued", reef.port)
    with patch.dict(os.environ, _ask_env(captures, compose), clear=True):
        assert harness("ask-scenario", "pi", compose, "text me", wait=True, timeout_s=0.1, poll_s=0.01) == 2
    reef.close()
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 4 and out[3].startswith("reef-pi: no result yet")
    assert len([call for call in reef.seen if call["path"] == record_path]) >= 3
    # Unanswered (404) twice in a row: the service no longer knows the request (its scenario was reset), so the
    # wait ends with one line and exit 1 instead of polling until the timeout.
    reef = _FakeReef(answer, rows=rows, record=None)
    (tmp_path / "missing").mkdir()
    compose, captures = _ask_tree(tmp_path / "missing", reef.port)
    with patch.dict(os.environ, _ask_env(captures, compose), clear=True):
        assert harness("ask-scenario", "pi", compose, "text me", wait=True, timeout_s=5.0, poll_s=0.01) == 1
    reef.close()
    out = capsys.readouterr().out.splitlines()
    assert out[3] == "reef-pi: request q-1 is no longer on the service (its scenario was reset); ask again"
    assert len(out) == 4 and len([call for call in reef.seen if call["path"] == record_path]) == 2


@pytest.mark.unit
def test_main_dispatches_page_with_the_step_and_the_print_flag(tmp_path) -> None:
    fetched: list[tuple] = []
    with (
        patch.dict(os.environ, _main_env(tmp_path)),
        patch("reef.harness.client.wrapper.page", lambda *args, **kwargs: fetched.append((args, kwargs)) or 0),
    ):
        for argv in (["reef-pi", "page", "3", "--print"], ["reef-pi", "page", "v3"]):
            with patch("sys.argv", argv), pytest.raises(SystemExit) as exited:
                main()
            assert exited.value.code == 0
        with patch("sys.argv", ["reef-pi", "page", "three"]), pytest.raises(SystemExit) as refused:
            main()
        assert refused.value.code == 2
    assert fetched == [
        (("ask-scenario", "pi", str(tmp_path), 3), {"open_page": False}),
        (("ask-scenario", "pi", str(tmp_path), 3), {"open_page": True}),
    ]


@pytest.mark.unit
def test_main_prints_its_own_usage_for_help_then_runs_the_agent(tmp_path, capsys) -> None:
    ran: list[list[str]] = []
    with (
        patch.dict(os.environ, _main_env(tmp_path)),
        patch("reef.harness.client.wrapper.run_agent", lambda *args: ran.append(args[-1])),
    ):
        for argv in (["reef-pi", "--help"], ["reef-pi", "-h"], ["reef-pi", "help"]):
            with patch("sys.argv", argv):
                main()
    out = capsys.readouterr().out
    assert out.count("reef-pi: run pi through reef's capture proxy, or one of") == 3
    for line in (
        "reef-pi report --score S",
        'reef-pi evolve "<what it should do>" [--wait] [--timeout SECONDS]',
        "reef-pi page <step> [--print]",
        "reef-pi doctor",
        "reef-pi setup [--yes] [--mark NAME] [--release ID]",
        "reef-pi setup --json | --set NAME=VALUE | --run NAME [--release ID]",
        "reef-pi update [--release ID]",
        "Anything else runs pi with the same arguments; --help and -h print its help after this.",
    ):
        assert line in out
    # pi's own help follows the wrapper's for --help and -h; help alone hands nothing to pi.
    assert ran == [["--help"], ["-h"]]


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
    """A reef whose GET /reef/harness/releases answers ``rows``, whose step pages are ``pages`` and whose install
    route serves ``install``; every request is recorded, and a step without a page answers the route's 404 text."""

    def __init__(self, rows: list[dict], *, pages: dict[int, str] | None = None, install: str | None = None) -> None:
        import http.server
        import threading

        self.seen: list[dict] = []
        seen = self.seen

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}})
                step = re.fullmatch(r"/reef/harness/releases/(\d+)/page", self.path)
                if self.path == "/reef/harness/releases":
                    code, kind = 200, "application/json"
                    raw = json.dumps({"scenario": "setup-scenario", "releases": rows}).encode()
                elif step is not None and int(step.group(1)) in (pages or {}):
                    code, kind, raw = 200, "text/html", (pages or {})[int(step.group(1))].encode()
                elif self.path.startswith("/reef/harness/install?") and install is not None:
                    code, kind, raw = 200, "text/x-shellscript", install.encode()
                elif step is not None:
                    scenario = self.headers.get("x-reef-scenario")
                    code, kind = 404, "text/plain"
                    raw = f"scenario {scenario!r} has no step {step.group(1)}: the catalog holds steps 0 to {len(rows) - 1}".encode()
                else:
                    code, kind, raw = 404, "application/json", b"{}"
                self.send_response(code)
                self.send_header("Content-Type", kind)
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
def test_setup_asks_before_running_a_check_and_asks_for_an_unset_variables_value(tmp_path, capsys) -> None:
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
    # The value is asked for; stdin ended, so nothing is stored.
    assert "    value for SMTP (blank to skip): " in out and not (tmp_path / ".reef-harness-env").exists()
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
        pytest.raises(SystemExit, match=r"no \.reef-harness-release release file at .*no installed release to set up"),
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
def test_run_agent_refuses_unmet_requirements_and_shows_setup_without_running_checks(tmp_path, capsys) -> None:
    reef = _FakeReef({"agent_record_id": "q-1", "scenario": "ask-scenario", "request_type": "train"})
    ran = tmp_path / "ran"
    release_info = {
        "release_id": "v2",
        "requires": [
            {"name": "notify", "kind": "permission", "check": f"touch {ran}", "prompt": "Allow notifications"},
            {"name": "TWILIO_SID", "kind": "env", "check": "TWILIO_SID"},
        ],
        "setup": [{"name": "TWILIO_SID", "checked_at": 1.0}],
    }
    compose, _ = _setup_tree(tmp_path, reef.port, release_info)
    binary = _make_fake_pi(tmp_path, reef.port)
    captures = tmp_path / "captures"
    captures.mkdir()
    with (
        patch.dict(os.environ, _ask_env(captures, compose), clear=True),
        patch("reef.harness.client.wrapper.CaptureProxy") as proxy,
        pytest.raises(SystemExit) as exited,
    ):
        run_agent(str(binary), compose, "ask-scenario", "pi", "PI_CODING_AGENT_DIR", ["-p", "hi"])
    reef.close()
    assert exited.value.code == 3
    proxy.assert_not_called()
    err = capsys.readouterr().err
    assert err.splitlines() == [
        "reef-pi: cannot start agent; this release has unmet requirements:",
        f"  notify (permission): touch {ran}",
        "    Allow notifications",
        "reef-pi: run reef-pi setup --release v2, then start the agent again",
    ]
    assert not ran.exists()
    assert not list(captures.glob("*.pending.json"))


@pytest.mark.unit
@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("checked_off", [False, True])
@pytest.mark.parametrize("has_check", [False, True])
def test_run_agent_requires_program_on_path_even_when_checked_off(
    tmp_path, capsys, installed: bool, checked_off: bool, has_check: bool
) -> None:
    reef = _FakeReef({"agent_record_id": "q-1", "scenario": "ask-scenario", "request_type": "train"})
    ran = tmp_path / "check-ran"
    program = tmp_path / "reef-required-pdf-tool"
    if installed:
        program.write_text("#!/bin/sh\nexit 0\n")
        program.chmod(0o755)
    item = {"name": program.name, "kind": "binary", "prompt": "Install the PDF reader and add it to PATH"}
    if has_check:
        item["check"] = f"touch {ran}"
    release_info = {
        "release_id": "v2",
        "requires": [item],
        "setup": [{"name": program.name, "check": item.get("check")}] if checked_off else [],
    }
    compose, _ = _setup_tree(tmp_path, reef.port, release_info)
    binary = _make_fake_pi(tmp_path, reef.port)
    captures = tmp_path / "captures"
    captures.mkdir()
    env = _ask_env(captures, compose, PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    try:
        with patch.dict(os.environ, env, clear=True), pytest.raises(SystemExit) as exited:
            run_agent(str(binary), compose, "ask-scenario", "pi", "PI_CODING_AGENT_DIR", ["-p", "hi"])
    finally:
        reef.close()
    err = capsys.readouterr().err
    if installed and (not has_check or checked_off):
        assert exited.value.code == 0
        assert "cannot start agent" not in err
        (spooled,) = captures.glob("*.pending.json")
        assert json.loads(spooled.read_text())["turns"][0]["receipt"] == "ask-receipt"
    else:
        assert exited.value.code == 3
        assert "cannot start agent" in err and item["prompt"] in err
        assert "reef-pi setup --release v2" in err
        assert not list(captures.glob("*.pending.json"))
        assert not reef.posts("/v1/chat/completions")
        if not installed:
            assert f"{program.name} is not on PATH; install it before starting the agent" in err
    assert not ran.exists()


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
        "reef-pi setup: every item is met; reef-pi update installs the release",
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
    assert capsys.readouterr().err == (
        "reef-pi setup: no release nope in the catalog\n"
        f"catalog: http://127.0.0.1:{reef.port} (scenario 'setup-scenario'). "
        "Check the service and scenario, then refresh the release list before retrying.\n"
    )
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


# -- setup --json, --set, --run, the env file, update -------------------------------------------


def _env_file(tmp_path: Path) -> Path:
    return tmp_path / ".reef-harness-env"


@pytest.mark.unit
def test_setup_json_lists_the_items_with_met_from_the_environment_the_env_file_and_the_check_offs(
    tmp_path, capsys
) -> None:
    """One JSON object, the release and its items with check, prompt and whether each is met; nothing runs and
    nothing is written; an unknown release is exit 2 and nothing served is a null release with no items."""
    rows = [
        _row("v1"),
        _row(
            "v2",
            [
                {"name": "SHELL_VAR", "kind": "env"},
                {"name": "file-var", "kind": "env", "check": "FILE_VAR"},
                {"name": "notify", "kind": "permission", "check": "exit 1"},
                {"name": "twilio", "kind": "service", "check": "exit 3", "prompt": "Connect the Twilio account"},
                {"name": "REEF_AWAY_PHONE", "kind": "env", "prompt": "The phone number to text"},
            ],
        ),
    ]
    reef = _ReleasesReef(rows)
    release_info = {"release_id": "v1", "setup": [{"name": "notify", "checked_at": 1.0, "check": "exit 1"}]}
    compose, release_file = _setup_tree(tmp_path, reef.port, release_info)
    _env_file(tmp_path).write_text("# stored by setup\n\nFILE_VAR = from-file\nno equals sign\n", encoding="utf-8")
    before = release_file.read_text()
    env = _ask_env(tmp_path / "captures", compose, SHELL_VAR="1")
    with patch.dict(os.environ, env, clear=True):
        assert setup_json("setup-scenario", "pi", compose) == 0
    assert json.loads(capsys.readouterr().out) == {
        "release_id": "v2",
        "items": [
            {"name": "SHELL_VAR", "kind": "env", "check": None, "prompt": None, "met": True},
            {"name": "file-var", "kind": "env", "check": "FILE_VAR", "prompt": None, "met": True},
            {"name": "notify", "kind": "permission", "check": "exit 1", "prompt": None, "met": True},
            {
                "name": "twilio",
                "kind": "service",
                "check": "exit 3",
                "prompt": "Connect the Twilio account",
                "met": False,
            },
            {
                "name": "REEF_AWAY_PHONE",
                "kind": "env",
                "check": None,
                "prompt": "The phone number to text",
                "met": False,
            },
        ],
    }
    assert release_file.read_text() == before and [call["path"] for call in reef.seen] == ["/reef/harness/releases"]
    with patch.dict(os.environ, env, clear=True):
        assert setup_json("setup-scenario", "pi", compose, release="nope") == 2
    assert capsys.readouterr().err == (
        "reef-pi setup: no release nope in the catalog\n"
        f"catalog: http://127.0.0.1:{reef.port} (scenario 'setup-scenario'). "
        "Check the service and scenario, then refresh the release list before retrying.\n"
    )
    reef.close()
    reef = _ReleasesReef([_row("v9", [{"name": "x", "kind": "env"}], pending=True)])
    (tmp_path / "waiting").mkdir()
    compose, _ = _setup_tree(tmp_path / "waiting", reef.port, {"release_id": "v1"})
    with patch.dict(os.environ, _ask_env(tmp_path / "captures", compose), clear=True):
        assert setup_json("setup-scenario", "pi", compose) == 0
    assert json.loads(capsys.readouterr().out) == {"release_id": None, "items": []}
    reef.close()


@pytest.mark.unit
def test_setup_set_stores_an_env_value_readable_by_the_person_alone_and_checks_the_item_off(tmp_path, capsys) -> None:
    """The value lands in the env file under the item's variable (its check, else its name), mode 0600, and the
    check off is recorded; a name the release does not require, a non-env item, a malformed assignment and a
    value that is empty or spans lines are refused with exit 2 and write nothing."""
    rows = [
        _row("v1"),
        _row(
            "v2",
            [
                {"name": "REEF_AWAY_PHONE", "kind": "env", "prompt": "The phone number to text"},
                {"name": "twilio-sid", "kind": "env", "check": "TWILIO_SID"},
                {"name": "notify", "kind": "permission", "check": "exit 1"},
            ],
        ),
    ]
    reef = _ReleasesReef(rows)
    compose, release_file = _setup_tree(tmp_path, reef.port, {"release_id": "v1"})
    env = _ask_env(tmp_path / "captures", compose)
    with patch.dict(os.environ, env, clear=True):
        assert setup_set("setup-scenario", "pi", compose, "REEF_AWAY_PHONE=+1 555 0100 ") == 0
        assert setup_set("setup-scenario", "pi", compose, "twilio-sid=AC1") == 0
    path = _env_file(tmp_path)
    assert capsys.readouterr().out.splitlines() == [
        f"reef-pi setup: REEF_AWAY_PHONE stored in {path}",
        f"reef-pi setup: twilio-sid stored in {path}",
    ]
    assert path.read_text(encoding="utf-8") == "REEF_AWAY_PHONE=+1 555 0100\nTWILIO_SID=AC1\n"
    assert path.stat().st_mode & 0o777 == 0o600 and not list(tmp_path.glob(".*.part"))
    recorded = json.loads(release_file.read_text())["setup"]
    assert [(item["name"], item["check"]) for item in recorded] == [
        ("REEF_AWAY_PHONE", None),
        ("twilio-sid", "TWILIO_SID"),
    ]
    refusals = [
        ("nope=1", "reef-pi setup: no item named nope; release v2 requires REEF_AWAY_PHONE, twilio-sid, notify"),
        (
            "notify=1",
            "reef-pi setup: notify is a permission item; --set stores the value of an env item, --run runs a check",
        ),
        ("REEF_AWAY_PHONE", "reef-pi setup: --set takes NAME=VALUE"),
        ("=1", "reef-pi setup: --set takes NAME=VALUE"),
        ("REEF_AWAY_PHONE= ", "reef-pi setup: the value for REEF_AWAY_PHONE must be one non-empty line"),
        ("REEF_AWAY_PHONE=a\nb", "reef-pi setup: the value for REEF_AWAY_PHONE must be one non-empty line"),
    ]
    for assignment, message in refusals:
        with patch.dict(os.environ, env, clear=True):
            assert setup_set("setup-scenario", "pi", compose, assignment) == 2, assignment
        assert capsys.readouterr().err == message + "\n"
    assert path.read_text(encoding="utf-8") == "REEF_AWAY_PHONE=+1 555 0100\nTWILIO_SID=AC1\n"
    reef.close()


@pytest.mark.unit
def test_setup_run_records_a_passing_check_and_not_a_failing_one(tmp_path, capsys) -> None:
    ran = tmp_path / "ran"
    rows = [
        _row("v1"),
        _row(
            "v2",
            [
                {"name": "notify", "kind": "permission", "check": f"touch {ran}"},
                {"name": "twilio", "kind": "service", "check": "exit 3"},
                {"name": "SMTP", "kind": "env"},
                {"name": "manual", "kind": "service"},
            ],
        ),
    ]
    reef = _ReleasesReef(rows)
    compose, release_file = _setup_tree(tmp_path, reef.port, {"release_id": "v1"})
    env = _ask_env(tmp_path / "captures", compose)
    env.pop("SMTP", None)
    with patch.dict(os.environ, env, clear=True):
        assert setup_run("setup-scenario", "pi", compose, "notify") == 0
        assert setup_run("setup-scenario", "pi", compose, "twilio") == 1
        assert setup_run("setup-scenario", "pi", compose, "SMTP") == 1
        assert setup_run("setup-scenario", "pi", compose, "manual") == 1
        assert setup_run("setup-scenario", "pi", compose, "nope") == 2
    assert ran.exists()
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        "reef-pi setup: notify met",
        "reef-pi setup: twilio not met (exit 3)",
        "reef-pi setup: SMTP not met (SMTP is not set; --set SMTP=VALUE stores it)",
        "reef-pi setup: manual not met (no check; --mark manual checks it off by hand)",
    ]
    assert captured.err == "reef-pi setup: no item named nope; release v2 requires notify, twilio, SMTP, manual\n"
    (recorded,) = json.loads(release_file.read_text())["setup"]
    assert recorded["name"] == "notify" and recorded["check"] == f"touch {ran}"
    # An env item's check is reading its variable: set, it is met and checked off.
    with patch.dict(os.environ, {**env, "SMTP": "smtp.example"}, clear=True):
        assert setup_run("setup-scenario", "pi", compose, "SMTP") == 0
    assert capsys.readouterr().out == "reef-pi setup: SMTP met\n"
    assert [item["name"] for item in json.loads(release_file.read_text())["setup"]] == ["notify", "SMTP"]
    reef.close()


@pytest.mark.unit
def test_setup_asks_for_an_env_value_shows_the_prompt_and_hides_a_credential(tmp_path, capsys) -> None:
    """An unmet env item shows its prompt sentence and asks for the value, without echo when the name looks like
    a credential; the value is stored in the env file and the item checked off; a blank answer leaves it unmet;
    ``--yes`` asks nothing."""
    rows = [
        _row("v1"),
        _row(
            "v2",
            [
                {
                    "name": "REEF_AWAY_PHONE",
                    "kind": "env",
                    "prompt": "The phone number to text, with the country code",
                },
                {"name": "sms", "kind": "env", "check": "REEF_SMS_TOKEN", "prompt": "The SMS API token"},
                {"name": "SKIPPED", "kind": "env"},
            ],
        ),
    ]
    reef = _ReleasesReef(rows)
    compose, release_file = _setup_tree(tmp_path, reef.port, {"release_id": "v1"})
    env = _ask_env(tmp_path / "captures", compose)
    asked: list[tuple[str, str]] = []
    answers = iter(("+15550100", ""))

    def typed(prompt: str) -> str:
        asked.append(("input", prompt))
        return next(answers)

    def hidden(prompt: str) -> str:
        asked.append(("getpass", prompt))
        return "tok-123"

    with (
        patch.dict(os.environ, env, clear=True),
        patch("builtins.input", typed),
        patch("reef.harness.client.wrapper.getpass.getpass", hidden),
    ):
        assert setup("setup-scenario", "pi", compose) == 1
    path = _env_file(tmp_path)
    assert capsys.readouterr().out.splitlines() == [
        "reef-pi setup: release v2 requires 3 item(s)",
        "  REEF_AWAY_PHONE (env)",
        "    The phone number to text, with the country code",
        f"    met (stored in {path})",
        "  sms (env): REEF_SMS_TOKEN",
        "    The SMS API token",
        f"    met (stored in {path})",
        "  SKIPPED (env)",
        "    not set",
        "reef-pi setup: 1 item(s) not met: SKIPPED",
    ]
    assert asked == [
        ("input", "    value for REEF_AWAY_PHONE (blank to skip): "),
        ("getpass", "    value for REEF_SMS_TOKEN (not echoed; blank to skip): "),
        ("input", "    value for SKIPPED (blank to skip): "),
    ]
    assert path.read_text(encoding="utf-8") == "REEF_AWAY_PHONE=+15550100\nREEF_SMS_TOKEN=tok-123\n"
    assert path.stat().st_mode & 0o777 == 0o600
    assert [item["name"] for item in json.loads(release_file.read_text())["setup"]] == ["REEF_AWAY_PHONE", "sms"]
    # A second run finds the stored values met, and --yes never asks for the one still missing.
    asked.clear()
    with patch.dict(os.environ, env, clear=True), patch("builtins.input", typed):
        assert setup("setup-scenario", "pi", compose, yes=True) == 1
    assert asked == []
    assert capsys.readouterr().out.splitlines()[1:] == [
        "  REEF_AWAY_PHONE (env)",
        "    met (checked off)",
        "  sms (env): REEF_SMS_TOKEN",
        "    met (checked off)",
        "  SKIPPED (env)",
        "    not set",
        "reef-pi setup: 1 item(s) not met: SKIPPED",
    ]
    reef.close()


INSTALL_SCRIPT = textwrap.dedent(
    """\
    #!/bin/sh
    set -eu
    printf '%s\\n' "$1" > "$1/dest-seen"
    printf '%s\\n' "${REEF_TOKEN:-}" > "$1/token-seen"
    cp "$1/.reef-harness-release" "$1/release-before"
    printf '{"release_id": "v2", "files": []}\\n' > "$1/.reef-harness-release"
    echo "reef: done"
    """
)


@pytest.mark.unit
def test_update_runs_the_fetched_install_script_for_the_install_root_and_refuses_unmet_items_first(
    tmp_path, capsys
) -> None:
    """The script comes from the install route with the token and the scenario header and runs with bash for the
    install root, the token in its environment; an unmet item is printed and nothing fetched, exit 3; an env item
    the environment meets is checked off before the script runs, so its evaluation sees it; a failing script, an unknown
    release and a route that has no script are exit 1."""
    rows = [_row("v1"), _row("v2", [{"name": "SMTP", "kind": "env", "prompt": "The SMTP host"}])]
    reef = _ReleasesReef(rows, install=INSTALL_SCRIPT)
    compose, release_file = _setup_tree(tmp_path, reef.port, {"release_id": "v1"})
    env = _ask_env(
        tmp_path / "captures",
        compose,
        REEF_TOKEN="tok",
        REEF_HARNESS_DEST=str(Path(compose).resolve().parent),
        REEF_SERVICE_URL=f"http://127.0.0.1:{reef.port}/",
        REEF_SCENARIO="setup-scenario",
    )
    env.pop("SMTP", None)
    with patch.dict(os.environ, env, clear=True):
        assert update("setup-scenario", "pi", compose) == 3
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.splitlines() == [
        "reef-pi update: release v2 requires setup first:",
        "  SMTP (env)",
        "reef-pi update: run reef-pi setup, then update again",
    ]
    assert [call["path"] for call in reef.seen] == ["/reef/harness/releases"] and not (tmp_path / "dest-seen").exists()
    with patch.dict(os.environ, {**env, "SMTP": "smtp.example"}, clear=True):
        assert update("setup-scenario", "pi", compose) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "reef-pi update: installed release v2"
    install_call = reef.seen[-1]
    assert install_call["path"] == "/reef/harness/install?adapter=pi"
    assert install_call["headers"]["x-reef-scenario"] == "setup-scenario"
    assert install_call["headers"]["authorization"] == "Bearer tok"
    root = Path(compose).resolve().parent
    assert (tmp_path / "dest-seen").read_text().strip() == str(root)
    assert (tmp_path / "token-seen").read_text().strip() == "tok"
    assert [item["name"] for item in json.loads((tmp_path / "release-before").read_text())["setup"]] == ["SMTP"]
    assert json.loads(release_file.read_text())["release_id"] == "v2"
    assert not list(Path(tempfile.gettempdir()).glob("reef-harness-install-*.sh"))
    with patch.dict(os.environ, {**env, "SMTP": "smtp.example"}, clear=True):
        assert update("setup-scenario", "pi", compose, release="v2") == 0
        assert update("setup-scenario", "pi", compose, release="nope") == 1
    assert reef.seen[-2]["path"] == "/reef/harness/install?adapter=pi&release_id=v2"
    assert capsys.readouterr().err == (
        "reef-pi update: no release nope in the catalog\n"
        f"catalog: http://127.0.0.1:{reef.port} (scenario 'setup-scenario'). "
        "Check the service and scenario, then refresh the release list before retrying.\n"
    )
    reef.close()
    for name, install, message in (
        ("failing", "#!/bin/sh\nexit 7\n", "reef-pi update: the install script exited 7"),
        ("missing", None, "reef-pi update: install script read failed (404): {}"),
    ):
        reef = _ReleasesReef([_row("v1"), _row("v2", [])], install=install)
        (tmp_path / name).mkdir()
        compose, _ = _setup_tree(tmp_path / name, reef.port, {"release_id": "v1"})
        with patch.dict(os.environ, _ask_env(tmp_path / "captures", compose), clear=True):
            assert update("setup-scenario", "pi", compose) == 1
        assert capsys.readouterr().err.splitlines()[-1] == message
        reef.close()


@pytest.mark.parametrize("operation", ["update", "setup-set"])
@pytest.mark.parametrize("changed", ["scenario", "service", "missing-binding"])
def test_session_setup_and_update_recover_using_the_sessions_service_and_scenario(
    tmp_path, capsys, operation, changed
) -> None:
    reef = _ReleasesReef([_row("v2", [{"name": "SMTP", "kind": "env"}])], install=INSTALL_SCRIPT)
    other = _ReleasesReef([_row("other-release")])
    compose, release_file = _setup_tree(
        tmp_path, other.port if changed == "service" else reef.port, {"release_id": "v1"}
    )
    if changed == "missing-binding":
        (Path(compose) / "models.json").unlink()
    disk_scenario = "other-scenario" if changed == "scenario" else "setup-scenario"
    env = _ask_env(
        tmp_path / "captures",
        compose,
        REEF_TOKEN="session-token",
        SMTP="smtp.example",
        REEF_HARNESS_DEST=str(Path(compose).parent),
        REEF_SERVICE_URL=f"http://127.0.0.1:{reef.port}",
        REEF_SCENARIO="setup-scenario",
    )
    try:
        with patch.dict(os.environ, env, clear=True):
            if operation == "update":
                assert update(disk_scenario, "pi", compose, release="v2") == 0
                assert json.loads(release_file.read_text())["release_id"] == "v2"
                assert (tmp_path / "dest-seen").read_text().strip() == str(tmp_path.resolve())
            else:
                assert setup_set(disk_scenario, "pi", compose, "SMTP=new-value", release="v2") == 0
                assert "SMTP=new-value" in (tmp_path / ".reef-harness-env").read_text()
        assert capsys.readouterr().err == ""
        assert other.seen == []
        assert all(call["headers"]["x-reef-scenario"] == "setup-scenario" for call in reef.seen)
        assert all(call["headers"]["authorization"] == "Bearer session-token" for call in reef.seen)
        if operation == "update":
            assert reef.seen[-1]["path"] == "/reef/harness/install?adapter=pi&release_id=v2"
    finally:
        reef.close()
        other.close()


def test_session_recovery_does_not_substitute_another_catalog_or_release(tmp_path, capsys) -> None:
    reef = _ReleasesReef([_row("new-head")], install=INSTALL_SCRIPT)
    other = _ReleasesReef([_row("selected-release")], install=INSTALL_SCRIPT)
    compose, release_file = _setup_tree(tmp_path, other.port, {"release_id": "installed"})
    before = release_file.read_bytes()
    env = _ask_env(
        tmp_path / "captures",
        compose,
        REEF_HARNESS_DEST=str(tmp_path),
        REEF_SCENARIO="setup-scenario",
        REEF_SERVICE_URL=f"http://127.0.0.1:{reef.port}",
    )
    # The active session is unauthenticated; the overwritten binding's token belongs to another service.
    models_file = Path(compose) / "models.json"
    models = json.loads(models_file.read_text())
    models["providers"]["reef"]["apiKey"] = "other-service-token"
    models_file.write_text(json.dumps(models))
    try:
        with patch.dict(os.environ, env, clear=True):
            assert update("other-scenario", "pi", compose, release="selected-release") == 1
        error = capsys.readouterr().err
        assert "no release selected-release" in error and f"127.0.0.1:{reef.port}" in error
        assert "other-service-token" not in error
        assert len(reef.seen) == 1 and reef.seen[0]["path"] == "/reef/harness/releases"
        assert "authorization" not in reef.seen[0]["headers"]
        assert other.seen == [] and release_file.read_bytes() == before
        assert not (tmp_path / "dest-seen").exists()
    finally:
        reef.close()
        other.close()


def test_session_does_not_redirect_an_explicit_update_of_another_installation(tmp_path, capsys) -> None:
    reef = _ReleasesReef([_row("v2")], install=INSTALL_SCRIPT)
    compose, release_file = _setup_tree(tmp_path, reef.port, {"release_id": "v1"})
    env = _ask_env(
        tmp_path / "captures",
        compose,
        REEF_HARNESS_DEST=str(tmp_path / "another-installation"),
        REEF_SERVICE_URL="http://session-service.invalid",
        REEF_SCENARIO="session-scenario",
    )
    try:
        with patch.dict(os.environ, env, clear=True):
            assert update("setup-scenario", "pi", compose, release="v2") == 0
        assert capsys.readouterr().err == ""
        assert json.loads(release_file.read_text())["release_id"] == "v2"
        assert all(call["headers"]["x-reef-scenario"] == "setup-scenario" for call in reef.seen)
    finally:
        reef.close()


def _make_env_dump_binary(tmp_path: Path) -> Path:
    """A fake agent that writes its environment to ``env.json`` beside itself and makes no call."""
    binary = tmp_path / "fake-env-dump"
    binary.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json, os
            from pathlib import Path
            Path(__file__).with_name("env.json").write_text(json.dumps(dict(os.environ)))
            """
        )
    )
    binary.chmod(0o755)
    return binary


@pytest.mark.unit
def test_run_agent_sets_the_env_files_variables_under_the_shells_and_exports_the_wrapper_path(
    tmp_path, capsys
) -> None:
    """Each env file variable reaches the agent unless the shell sets it; an env item the file meets is not warned
    about; ``REEF_HARNESS_WRAPPER`` names the wrapper at the install root when the install wrote one."""
    reef = _FakeReef({"agent_record_id": "q-1", "scenario": "ask-scenario", "request_type": "train"})
    release_info = {
        "release_id": "v2",
        "requires": [{"name": "FILE_ONLY", "kind": "env"}, {"name": "BOTH", "kind": "env"}],
    }
    compose, _ = _setup_tree(tmp_path, reef.port, release_info)
    _env_file(tmp_path).write_text("FILE_ONLY=from-file\nBOTH=from-file\n", encoding="utf-8")
    wrapper = tmp_path / "reef-pi"
    wrapper.write_text("#!/bin/sh\n")
    binary = _make_env_dump_binary(tmp_path)
    captures = tmp_path / "captures"
    captures.mkdir()
    env = _ask_env(captures, compose, BOTH="from-shell")
    with patch.dict(os.environ, env, clear=True), contextlib.suppress(SystemExit):
        run_agent(str(binary), compose, "ask-scenario", "pi", "PI_CODING_AGENT_DIR", ["-p", "hi"])
    seen = json.loads((tmp_path / "env.json").read_text())
    assert (seen["FILE_ONLY"], seen["BOTH"]) == ("from-file", "from-shell")
    assert seen["REEF_HARNESS_WRAPPER"] == str(Path(compose).resolve().parent / "reef-pi")
    assert seen["REEF_HARNESS_DEST"] == str(Path(compose).resolve().parent)
    assert capsys.readouterr().err == ""
    # Without a wrapper at the install root nothing names one, and the shell's own setting is kept.
    wrapper.unlink()
    with (
        patch.dict(os.environ, {**env, "REEF_HARNESS_WRAPPER": "/elsewhere/reef-pi"}, clear=True),
        contextlib.suppress(SystemExit),
    ):
        run_agent(str(binary), compose, "ask-scenario", "pi", "PI_CODING_AGENT_DIR", ["-p", "hi"])
    assert json.loads((tmp_path / "env.json").read_text())["REEF_HARNESS_WRAPPER"] == "/elsewhere/reef-pi"
    reef.close()


@pytest.mark.unit
def test_main_dispatches_the_setup_forms_and_update(tmp_path) -> None:
    called: list[tuple[str, tuple, dict]] = []

    def record(name: str):
        return lambda *args, **kwargs: called.append((name, args, kwargs)) or 0

    argvs = [
        ["reef-pi", "setup", "--json", "--release", "v4"],
        ["reef-pi", "setup", "--set", "REEF_AWAY_PHONE=+1 555"],
        ["reef-pi", "setup", "--run", "notify", "--release", "v4"],
        ["reef-pi", "update"],
        ["reef-pi", "update", "--release", "v4"],
    ]
    with (
        patch.dict(os.environ, _main_env(tmp_path, "setup-scenario")),
        patch("reef.harness.client.wrapper.setup_json", record("setup_json")),
        patch("reef.harness.client.wrapper.setup_set", record("setup_set")),
        patch("reef.harness.client.wrapper.setup_run", record("setup_run")),
        patch("reef.harness.client.wrapper.update", record("update")),
    ):
        for argv in argvs:
            with patch("sys.argv", argv), pytest.raises(SystemExit) as exited:
                main()
            assert exited.value.code == 0
        # The three forms exclude one another.
        with patch("sys.argv", ["reef-pi", "setup", "--json", "--run", "x"]), pytest.raises(SystemExit) as exited:
            main()
        assert exited.value.code == 2
    root = ("setup-scenario", "pi", str(tmp_path))
    assert called == [
        ("setup_json", root, {"release": "v4"}),
        ("setup_set", (*root, "REEF_AWAY_PHONE=+1 555"), {}),
        ("setup_run", (*root, "notify"), {"release": "v4"}),
        ("update", root, {}),
        ("update", root, {"release": "v4"}),
    ]


# -- reef-<adapter> doctor: one report of what the install needs ---------------------------------


class _DoctorReef:
    """A reef that serves only the harness routes, checks the bearer on them, and names one served head."""

    def __init__(self, token: str, head: str, rows: list[dict] | None = None) -> None:
        import http.server
        import threading

        catalog = rows if rows is not None else [{"release_id": head, "pending": False}]

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                # Only the harness routes, the ones the API platform exposes too; no service wide status.
                if self.headers.get("Authorization") != f"Bearer {token}":
                    code, payload = 401, {"error": "invalid service token"}
                elif self.path == "/reef/harness/releases":
                    code, payload = 200, {"releases": catalog}
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
def test_doctor_names_a_release_that_waits_for_review(tmp_path, capsys, monkeypatch) -> None:
    """A pending release is served to nobody until a person promotes it; doctor says so
    with the step page, and stops saying so once a promote row names it."""
    from reef.harness.client.wrapper import doctor

    held = [{"release_id": "rel-3", "pending": False}, {"release_id": "rel-4", "pending": True}]
    reef = _DoctorReef(token="dummy", head="rel-3", rows=held)
    compose, _ = _ask_tree(tmp_path, reef.port)
    binary = tmp_path / "fake-pi"
    binary.write_text("#!/bin/sh\necho 0.84.2\n")
    binary.chmod(0o755)
    monkeypatch.delenv("REEF_TOKEN", raising=False)
    monkeypatch.setattr("shutil.which", lambda command: f"/usr/bin/{command}")
    assert doctor("doc-scenario", "pi", compose, str(binary)) == 0
    out = capsys.readouterr().out.splitlines()
    assert any(line.startswith("ok  release") and "rel-3 installed, the served head" in line for line in out)
    (review,) = [line for line in out if line.startswith("ok  review")]
    assert f"rel-4 waits for your review: http://127.0.0.1:{reef.port}/reef/harness/releases/1/page" in review
    reef.close()

    promoted = [
        *held,
        {"release_id": "rel-5", "pending": False, "operation": "promote", "rollback_target_release_id": "rel-4"},
    ]
    reef = _DoctorReef(token="dummy", head="rel-3", rows=promoted)
    (tmp_path / "promoted").mkdir()
    compose, _ = _ask_tree(tmp_path / "promoted", reef.port)
    assert doctor("doc-scenario", "pi", compose, str(binary)) == 0
    assert not [line for line in capsys.readouterr().out.splitlines() if line.startswith("ok  review")]
    reef.close()


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


@pytest.mark.unit
def test_doctor_links_a_release_awaiting_review_with_the_page_query(tmp_path, capsys, monkeypatch) -> None:
    """The review row's page link carries the scenario and the token, so it opens from a browser as the wait's."""
    from reef.harness.client.wrapper import doctor

    pending = _step_row("rel-2222-pending", {"selected": True}, pending=True)
    reef = _ReleasesReef([_row("rel-1"), pending])
    compose, _ = _setup_tree(tmp_path, reef.port, {"release_id": "rel-1"})
    binary = tmp_path / "fake-pi"
    binary.write_text("#!/bin/sh\necho 0.84.2\n")
    binary.chmod(0o755)
    monkeypatch.delenv("REEF_TOKEN", raising=False)
    monkeypatch.setattr("shutil.which", lambda command: f"/usr/bin/{command}")
    assert doctor("doc-scenario", "pi", compose, str(binary)) == 0
    out = capsys.readouterr().out.splitlines()
    assert any(line.startswith("ok  release") and "rel-1 installed, the served head" in line for line in out)
    page = f"http://127.0.0.1:{reef.port}/reef/harness/releases/1/page?scenario=doc-scenario&token=dummy"
    assert out[-1].startswith("ok  review") and out[-1].endswith(f"rel-2222 waits for your review: {page}")
    reef.close()


# -- reef-<adapter> page: a step's page as a file the desktop opens ------------------------------


@pytest.mark.unit
def test_page_writes_the_step_page_to_the_cache_prints_its_path_and_opens_it(tmp_path, capsys, monkeypatch) -> None:
    from reef.harness.client.wrapper import page

    reef = _ReleasesReef([_row("rel-1"), _row("rel-2")], pages={1: "<html>step 1</html>"})
    compose, _ = _setup_tree(tmp_path, reef.port, {"release_id": "rel-1"})
    opened: list[Path] = []
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("REEF_TOKEN", "tok")
    monkeypatch.setattr("reef.harness.client.wrapper._open_in_browser", lambda path: opened.append(path) or True)
    assert page("team/scenario", "pi", compose, 1) == 0
    expected = tmp_path / "cache" / "reef-harness" / "team-scenario-step-1.html"
    assert capsys.readouterr().out.strip() == str(expected)
    assert expected.read_text(encoding="utf-8") == "<html>step 1</html>"
    assert opened == [expected]
    (request,) = [call for call in reef.seen if call["path"] == "/reef/harness/releases/1/page"]
    assert request["headers"]["x-reef-scenario"] == "team/scenario"
    assert request["headers"]["authorization"] == "Bearer tok"
    # --print writes and prints the path and opens nothing; a missing step is the route's 404 text.
    assert page("team/scenario", "pi", compose, 1, open_page=False) == 0
    assert opened == [expected]
    with pytest.raises(SystemExit, match=r"page read failed \(404\): scenario 'team/scenario' has no step 9"):
        page("team/scenario", "pi", compose, 9)
    reef.close()


@pytest.mark.unit
def test_setup_meets_a_binary_item_by_looking_on_path_before_any_check_runs(tmp_path, capsys) -> None:
    """A binary item with no check is met once its program is on PATH; one that is not there is not met and says
    to install it; a binary that names a check is looked for first and the check runs on the confirmation a
    permission's does, so no check runs for a program the machine does not have."""
    ran = tmp_path / "ran"
    never = tmp_path / "never"
    rows = [
        _row("v1"),
        _row(
            "v2",
            [
                {"name": "pdftotext", "kind": "binary", "prompt": "Install poppler for the PDF reader"},
                {"name": "ffmpeg", "kind": "binary"},
                {"name": "gh", "kind": "binary", "check": f"touch {ran}"},
                {"name": "wkhtmltopdf", "kind": "binary", "check": f"touch {never}"},
            ],
        ),
    ]
    reef = _ReleasesReef(rows)
    compose, release_file = _setup_tree(tmp_path, reef.port, {"release_id": "v1"})
    env = _ask_env(tmp_path / "captures", compose)
    on_path = {"pdftotext": "/usr/local/bin/pdftotext", "gh": "/usr/bin/gh"}
    with (
        patch.dict(os.environ, env, clear=True),
        patch("shutil.which", lambda command: on_path.get(command)),
    ):
        assert setup("setup-scenario", "pi", compose, yes=True) == 1
    assert capsys.readouterr().out.splitlines() == [
        "reef-pi setup: release v2 requires 4 item(s)",
        "  pdftotext (binary)",
        "    Install poppler for the PDF reader",
        "    found at /usr/local/bin/pdftotext",
        "  ffmpeg (binary)",
        "    ffmpeg is not on PATH; install it, then run setup again",
        f"  gh (binary): touch {ran}",
        "    found at /usr/bin/gh",
        "    met",
        f"  wkhtmltopdf (binary): touch {never}",
        "    wkhtmltopdf is not on PATH; install it, then run setup again",
        "reef-pi setup: 2 item(s) not met: ffmpeg, wkhtmltopdf",
    ]
    # The check of a program that is not there never ran: the look comes first, and it decided.
    assert ran.exists() and not never.exists()
    assert [item["name"] for item in json.loads(release_file.read_text())["setup"]] == ["pdftotext", "gh"]
    reef.close()


@pytest.mark.unit
def test_doctor_reports_a_required_program_and_runs_no_check_of_its_own(tmp_path, capsys, monkeypatch) -> None:
    """Each ``binary`` item of the installed release is a program row, ok when the program is on PATH and not
    when it is missing, with the item's prompt as the hint. The check an item names is a command the release
    wrote: it runs in setup, where the person confirms it, and never here."""
    from reef.harness.client.wrapper import doctor

    ran = tmp_path / "ran"
    reef = _DoctorReef(token="dummy", head="rel-3")
    compose, _ = _ask_tree(tmp_path, reef.port)
    (tmp_path / ".reef-harness-release").write_text(
        json.dumps(
            {
                "release_id": "rel-3",
                "requires": [
                    {"name": "pdftotext", "kind": "binary", "prompt": "brew install poppler"},
                    {"name": "ffmpeg", "kind": "binary", "check": f"touch {ran}"},
                    {"name": "REEF_AWAY_PHONE", "kind": "env"},
                ],
            }
        ),
        encoding="utf-8",
    )
    binary = tmp_path / "fake-pi"
    binary.write_text("#!/bin/sh\necho 0.84.2\n")
    binary.chmod(0o755)
    monkeypatch.delenv("REEF_TOKEN", raising=False)
    monkeypatch.setattr("shutil.which", lambda command: None if command == "pdftotext" else f"/usr/bin/{command}")
    assert doctor("doc-scenario", "pi", compose, str(binary)) == 1  # pdftotext is missing
    out = capsys.readouterr().out.splitlines()
    assert any(line.startswith("!!  program") and "pdftotext missing: brew install poppler" in line for line in out)
    assert any(line.startswith("ok  program") and "ffmpeg at /usr/bin/ffmpeg" in line for line in out)
    # Only the binary items become program rows, and nothing the release wrote ran.
    assert not [line for line in out if "REEF_AWAY_PHONE" in line]
    assert not ran.exists()
    monkeypatch.setattr("shutil.which", lambda command: f"/usr/bin/{command}")
    assert doctor("doc-scenario", "pi", compose, str(binary)) == 0
    reef.close()
