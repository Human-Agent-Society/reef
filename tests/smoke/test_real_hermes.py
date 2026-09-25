"""Smoke of the real hermes binary: render, episode, and trajectory against the
real thing. The hermetic suites drive scripted fakes, so every format and
environment assumption about the real Hermes Agent lives here: a pinned
hermes runs one episode against a stdlib OpenAI-compatible stub and must read
our rendered config.yaml, reach the stub with the SOUL.md text and the bound
key, write a session snapshot our reader parses, make no extra model call,
and leave no unexplained residue. Skipped unless REEF_REAL_HERMES_BINARY
points at an installed hermes (.github/workflows/harness-smoke.yml provides
one)."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import threading
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.client.wrapper import _create_temp_composition, run_agent
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.episodes.run import run_episode
from reef.harness.tree.render import render_composition

REAL_HERMES = os.environ.get("REEF_REAL_HERMES_BINARY", "")

pytestmark = pytest.mark.skipif(not REAL_HERMES, reason="REEF_REAL_HERMES_BINARY does not name a real hermes binary")

MODEL = "smoke-model"
RULES_SENTENCE = "The reef smoke marker phrase is TIDEPOOL-STONE-41."


class StubOpenAI(ThreadingHTTPServer):
    """A canned /v1 endpoint that records every request it is sent. It asks for the ``tool_calls``, (tool name,
    arguments) pairs, one per answer, before it replies READY."""

    def __init__(self, tool_calls: Sequence[tuple[str, dict[str, str]]] = ()) -> None:
        super().__init__(("127.0.0.1", 0), StubHandler)
        self.requests: list[tuple[str, str | None, str]] = []
        self.tool_calls = list(tool_calls)


class StubHandler(BaseHTTPRequestHandler):
    server: StubOpenAI

    def log_message(self, format: str, *args: Any) -> None:
        pass  # keep the harness's stderr out of pytest output

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8")
        self.server.requests.append((self.path, self.headers.get("Authorization"), body))
        if not self.path.endswith("/chat/completions"):
            self.send_response(404)  # hermes probes the base URL as a local endpoint first
            self.end_headers()
            return
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        request = json.loads(body)
        answered = sum(1 for message in request.get("messages", []) if message.get("role") == "tool")
        delta: dict[str, Any] = {"role": "assistant", "content": "READY"}
        finish = "stop"
        if answered < len(self.server.tool_calls):
            name, arguments = self.server.tool_calls[answered]
            call = {"name": name, "arguments": json.dumps(arguments)}
            tool_call = {"index": 0, "id": f"call-{answered}", "type": "function", "function": call}
            delta = {"role": "assistant", "tool_calls": [tool_call]}
            finish = "tool_calls"
        if request.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {"choices": [{"index": 0, "delta": delta}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage},
            ]
            for chunk in chunks:
                event = {"id": "chatcmpl-smoke", "object": "chat.completion.chunk", "model": MODEL, **chunk}
                self.wfile.write(b"data: " + json.dumps(event).encode("utf-8") + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            payload = {
                "id": "chatcmpl-smoke",
                "object": "chat.completion",
                "model": MODEL,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "READY"}, "finish_reason": "stop"}
                ],
                "usage": usage,
            }
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)


def test_real_hermes_episode_renders_runs_and_cleans_up() -> None:
    server = StubOpenAI()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    descriptor = get_adapter("hermes")
    try:
        binding = ModelBinding(
            base_url=f"http://127.0.0.1:{server.server_address[1]}", model=MODEL, api_key="smoke-key-1234"
        )
        nodes = [("rules", {"text": RULES_SENTENCE}), *binding.compose_nodes(descriptor)]
        files = render_composition(nodes, descriptor)
        # The task starts with a dash on purpose: it must pass through -q as the query.
        result = run_episode(
            descriptor, files, "- Reply with exactly the word READY", binary=REAL_HERMES, timeout=300.0
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result.exit_code == 0, (result.stdout, result.stderr)
    completions = [(auth, body) for path, auth, body in server.requests if path.endswith("/chat/completions")]
    # (a) Exactly one model call: the title call is off. It carried the rules text and the bound key.
    assert len(completions) == 1, [path for path, _, _ in server.requests]
    assert RULES_SENTENCE in completions[0][1] and "- Reply with exactly the word READY" in completions[0][1]
    assert completions[0][0] == "Bearer smoke-key-1234"
    # (b) The snapshot the real binary wrote parses through our reader and carries the exchange.
    kinds = [event["type"] for event in result.trajectory]
    assert kinds == ["session", "message", "message"], kinds
    assert [event["role"] for event in result.trajectory[1:]] == ["user", "assistant"]
    assert result.trajectory[2]["content"] == "READY"
    # (c) The cleanup audit found nothing outside the declared whitelist.
    assert result.residue == ()


def test_real_hermes_reads_the_scanner_off_and_finds_the_commands_in_an_episode_and_a_session(tmp_path) -> None:
    """hermes's own config readers, run in its venv on the rendered home: the tirith scanner is off (it would download
    at boot and block "$REEF_HARNESS_WRAPPER" commands), and the commands root is found beside an episode home and,
    through REEF_HARNESS_DEST, beside the temp copy a reef-hermes session runs in. It is still found when the tree
    sets skills.external_dirs itself, empty or to a directory of its own."""
    descriptor = get_adapter("hermes")
    binding = ModelBinding(base_url="http://127.0.0.1:9", model=MODEL, api_key="smoke-key-1234")
    command = ("agent_command", {"name": "probe", "text": "Reply with PROBE."})
    probe = (
        "import json\n"
        "from agent.skill_utils import get_external_skills_dirs\n"
        "from tools.tirith_security import _load_security_config\n"
        "print(json.dumps([_load_security_config()['tirith_enabled'], [str(p) for p in get_external_skills_dirs()]]))\n"
    )
    env = {name: value for name, value in os.environ.items() if name not in ("REEF_HARNESS_DEST", "TIRITH_ENABLED")}
    extra = tmp_path / "extra-skills"
    extra.mkdir()
    for tree, listed in (("default", None), ("empty", []), ("own", [str(extra.resolve())])):
        config = [] if listed is None else [("config", {"data": {"skills": {"external_dirs": listed}}})]
        root = tmp_path / tree / "reef-harness"
        nodes = [*config, command, *binding.compose_nodes(descriptor)]
        for relative, text in render_composition(nodes, descriptor).items():
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_text(text, encoding="utf-8")
        found = [*(listed or []), str((root / "hermes-commands").resolve())]
        session_home = _create_temp_composition("hermes", str(root / "hermes"), 9)
        try:
            layouts = {
                "episode": {**env, "HERMES_HOME": str(root / "hermes")},
                "session": {**env, "HERMES_HOME": session_home, "REEF_HARNESS_DEST": str(root)},
            }
            for layout, layout_env in layouts.items():
                result = subprocess.run(
                    [str(Path(REAL_HERMES).parent / "python"), "-c", probe],
                    env=layout_env,
                    cwd=tmp_path,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                assert result.returncode == 0, (tree, layout, result.stderr)
                assert json.loads(result.stdout.splitlines()[-1]) == [False, found], (tree, layout)
        finally:
            shutil.rmtree(session_home)


def test_real_hermes_episode_makes_no_background_review_call_after_ten_tool_calls() -> None:
    """hermes reviews the skill library in a background agent after ten tool calls in a turn; the defaults turn that
    off, so an episode makes only its own model calls and writes no skill into the tree. The tree has no rules, like
    a scenario's seed tree, so hermes writes its default SOUL.md, which is not residue."""
    server = StubOpenAI(tool_calls=[("terminal", {"command": f"echo step-{step}"}) for step in range(10)])
    threading.Thread(target=server.serve_forever, daemon=True).start()
    descriptor = get_adapter("hermes")
    try:
        binding = ModelBinding(
            base_url=f"http://127.0.0.1:{server.server_address[1]}", model=MODEL, api_key="smoke-key-1234"
        )
        files = render_composition(binding.compose_nodes(descriptor), descriptor)
        result = run_episode(
            descriptor, files, "Run ten echo commands, then reply READY", binary=REAL_HERMES, timeout=300.0
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result.exit_code == 0, (result.stdout, result.stderr)
    completions = [body for path, _, body in server.requests if path.endswith("/chat/completions")]
    assert not [body for body in completions if "update the skill library" in body]
    assert len(completions) == 11, len(completions)  # ten tool calls and the answer
    assert result.residue == ()


def test_real_hermes_episode_that_loads_a_skill_leaves_no_residue() -> None:
    """A skill_view of a tree skill makes hermes count the load in skills/.usage.json under a lock file, with no
    config key to turn that off. Both are hermes's own state, so the episode reports no residue."""
    server = StubOpenAI(tool_calls=[("skill_view", {"name": "notes"})])
    threading.Thread(target=server.serve_forever, daemon=True).start()
    descriptor = get_adapter("hermes")
    try:
        binding = ModelBinding(
            base_url=f"http://127.0.0.1:{server.server_address[1]}", model=MODEL, api_key="smoke-key-1234"
        )
        skill = ("skill", {"name": "notes", "text": "# Notes\n\nKeep notes short: NOTES-MARKER-7."})
        files = render_composition([skill, *binding.compose_nodes(descriptor)], descriptor)
        result = run_episode(
            descriptor, files, "Load the notes skill, then reply READY", binary=REAL_HERMES, timeout=300.0
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result.exit_code == 0, (result.stdout, result.stderr)
    completions = [body for path, _, body in server.requests if path.endswith("/chat/completions")]
    tool_results = [message for message in json.loads(completions[-1])["messages"] if message.get("role") == "tool"]
    # The skill loaded, which is when hermes counts it.
    assert len(tool_results) == 1 and "NOTES-MARKER-7" in tool_results[0]["content"], tool_results
    assert result.residue == ()


def test_real_hermes_session_keeps_its_snapshot_and_log_in_the_installed_tree(tmp_path, monkeypatch) -> None:
    """A reef-hermes session runs on a temp copy of the home that the wrapper removes; the session snapshot and the
    log hermes writes stay in the installed tree, so a second session keeps the first one's files."""
    server = StubOpenAI()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    descriptor = get_adapter("hermes")
    root = tmp_path / "reef-harness"
    binding = ModelBinding(
        base_url=f"http://127.0.0.1:{server.server_address[1]}", model=MODEL, api_key="smoke-key-1234"
    )
    for relative, text in render_composition(binding.compose_nodes(descriptor), descriptor).items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text, encoding="utf-8")
    monkeypatch.setenv("REEF_HARNESS_CAPTURES_DIR", str(tmp_path / "captures"))
    home = root / "hermes"
    sessions: list[list[str]] = []
    try:
        for prompt in ("Reply with exactly the word READY", "Reply READY again"):
            with contextlib.suppress(SystemExit):
                run_agent(
                    REAL_HERMES, str(home), "smoke", "hermes", "HERMES_HOME", ["chat", "-Q", "--oneshot", "-q", prompt]
                )
            sessions.append(sorted(path.name for path in (home / "sessions").glob("session_*.json")))
    finally:
        server.shutdown()
        server.server_close()

    assert len(sessions[0]) == 1 and len(sessions[1]) == 2 and sessions[0][0] in sessions[1], sessions
    for name in sessions[1]:
        assert json.loads((home / "sessions" / name).read_text())["messages"][-1]["content"] == "READY"
    assert (home / "logs" / "agent.log").is_file()


def test_real_hermes_reads_the_curator_off(tmp_path) -> None:
    """hermes's own curator check, run in its venv on the rendered home: the curator is off, so a session start seeds
    no curator state in the skills directory, which in a reef-hermes session is the installed release's."""
    descriptor = get_adapter("hermes")
    home = tmp_path / "hermes"
    for relative, text in render_composition([], descriptor).items():
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(text, encoding="utf-8")
    probe = (
        "import json\nfrom agent import curator\nprint(json.dumps([curator.is_enabled(), curator.should_run_now()]))\n"
    )
    result = subprocess.run(
        [str(Path(REAL_HERMES).parent / "python"), "-c", probe],
        env={**os.environ, "HERMES_HOME": str(home)},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == [False, False]
    assert not (home / "skills" / ".curator_state").exists()
