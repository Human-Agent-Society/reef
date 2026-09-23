"""Smoke the pinned Codex CLI through Reef's complete harness path.

The hermetic suite uses a scripted binary. These tests exercise the real CLI
against a local Responses API stub: the runs prove rules, skills, binding,
trajectory collection, cleanup, and temporary proxy path. The workflow
supplies the pinned binary through ``REEF_REAL_CODEX_BINARY``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.episodes.run import run_episode
from reef.harness.tree.render import render_composition

REAL_CODEX = os.environ.get("REEF_REAL_CODEX_BINARY", "")

pytestmark = pytest.mark.skipif(not REAL_CODEX, reason="REEF_REAL_CODEX_BINARY does not name a real Codex binary")

MODEL = "smoke-model"
RULES_MARKER = "The Reef Codex rules marker is TIDEPOOL-CODEX-41."
SKILL_MARKER = "The Reef Codex skill marker is TIDEPOOL-SKILL-27."


def _response(status: str, output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "resp_reef_smoke",
        "object": "response",
        "created_at": 1,
        "status": status,
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": MODEL,
        "output": output,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": False,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 2,
        },
        "user": None,
        "metadata": {},
    }


class StubResponses(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), StubHandler)
        self.requests: list[tuple[str, str, str]] = []


class StubHandler(BaseHTTPRequestHandler):
    server: StubResponses

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8")
        self.server.requests.append((self.path, self.headers.get("Authorization", ""), body))

        message = {
            "id": "msg_reef_smoke",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "READY", "annotations": [], "logprobs": []}],
        }
        events = [
            {"type": "response.created", "response": _response("in_progress", []), "sequence_number": 0},
            {
                "type": "response.output_item.done",
                "item": message,
                "output_index": 0,
                "sequence_number": 1,
            },
            {"type": "response.completed", "response": _response("completed", [message]), "sequence_number": 2},
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event in events:
            self.wfile.write(b"data: " + json.dumps(event).encode("utf-8") + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


def _server() -> tuple[StubResponses, str]:
    server = StubResponses()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _binding(base_url: str) -> ModelBinding:
    return ModelBinding(base_url=base_url, model=MODEL, api_key="reef-smoke-key", api="responses")


def _bound_files(*nodes: tuple[str, dict[str, Any]], binding: ModelBinding) -> dict[str, str]:
    descriptor = get_adapter("codex")
    return render_composition([*nodes, *binding.compose_nodes(descriptor)], descriptor)


def _capturing_wrapper(tmp_path: Path, capture: Path) -> str:
    wrapper = tmp_path / "codex-capturing"
    wrapper.write_text(
        "#!/bin/sh\n"
        f'"{REAL_CODEX}" "$@"\n'
        "status=$?\n"
        f'find "$CODEX_HOME/sessions" -name "*.jsonl" -exec cp {{}} "{capture}" \\;\n'
        "exit $status\n"
    )
    wrapper.chmod(0o755)
    return str(wrapper)


def test_real_codex_accepts_every_admitted_tuning_key() -> None:
    server, base_url = _server()
    try:
        files = _bound_files(
            (
                "config",
                {
                    "data": {
                        "model_auto_compact_token_limit": 100_000,
                        "model_auto_compact_token_limit_scope": "total",
                        "model_context_window": 200_000,
                        "model_reasoning_effort": "low",
                        "model_reasoning_summary": "none",
                        "model_verbosity": "low",
                        "tool_output_token_limit": 1_000,
                    }
                },
            ),
            binding=_binding(base_url),
        )
        result = run_episode(get_adapter("codex"), files, "Reply READY", binary=REAL_CODEX, timeout=120.0)
    finally:
        server.shutdown()
        server.server_close()
    assert result.exit_code == 0, result.stderr


def test_real_codex_renders_runs_collects_and_cleans_up(tmp_path: Path) -> None:
    server, base_url = _server()
    try:
        files = _bound_files(
            ("rules", {"text": RULES_MARKER}),
            ("skill", {"name": "reef-smoke", "text": f"# Reef smoke skill\n\n{SKILL_MARKER}"}),
            ("agent_command", {"name": "reef-command", "text": "Reef smoke command"}),
            binding=_binding(base_url),
        )
        capture = Path(os.environ.get("REEF_REAL_CODEX_SESSION_OUT", tmp_path / "real-codex-session.jsonl"))
        result = run_episode(
            get_adapter("codex"),
            files,
            "Reply with exactly the word READY",
            binary=_capturing_wrapper(tmp_path, capture),
            timeout=120.0,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert server.requests
    path, authorization, body = server.requests[0]
    assert path == "/v1/responses"
    assert authorization == "Bearer reef-smoke-key"
    assert RULES_MARKER in body
    # Codex advertises skill metadata first and reads the body only when the
    # model chooses the skill; discovery is the adapter contract here. A
    # command is a skill in the same root under $CODEX_HOME, which a
    # reef-codex session lists too.
    assert re.search(r"`r0` = `[^`]*/codex/skills`", body)
    assert "reef-smoke: Reef smoke skill (file: r0/reef-smoke/SKILL.md)" in body
    assert "reef-command: Reef smoke command (file: r0/reef-command/SKILL.md)" in body
    # config.toml sets no approval_policy; the episode still runs with approval never, so it never waits.
    assert "Approval policy is currently never" in body and "# Escalation Requests" not in body
    assert any(event.get("type") == "session_meta" for event in result.trajectory)
    assert any(event.get("payload", {}).get("type") == "task_complete" for event in result.trajectory)
    assert result.residue == ()
    assert capture.is_file() and capture.stat().st_size > 0


def test_real_codex_episode_offers_no_web_search_when_the_tree_turns_it_on() -> None:
    """The tree's web_search is for a person's session; the episode argv turns it off again, and without that
    override the same tree offers the model the hosted web_search tool."""
    descriptor = get_adapter("codex")
    pin = descriptor.argv.index('web_search="disabled"')
    unpinned = dataclasses.replace(descriptor, argv=descriptor.argv[: pin - 1] + descriptor.argv[pin + 1 :])
    tools: dict[str, list[str]] = {}
    for label, adapter in (("pinned", descriptor), ("unpinned", unpinned)):
        server, base_url = _server()
        try:
            files = _bound_files(("config", {"data": {"web_search": "live"}}), binding=_binding(base_url))
            result = run_episode(adapter, files, "Reply READY", binary=REAL_CODEX, timeout=120.0)
        finally:
            server.shutdown()
            server.server_close()
        assert result.exit_code == 0, result.stderr
        tools[label] = [tool.get("type", "") for _, _, body in server.requests for tool in json.loads(body)["tools"]]
    assert tools["pinned"] and "web_search" not in tools["pinned"]
    assert "web_search" in tools["unpinned"]


def test_real_codex_session_config_lets_the_model_ask_the_person_to_leave_the_sandbox(tmp_path: Path) -> None:
    """A reef-codex session runs the interactive CLI on the rendered config.toml, which sets no approval_policy:
    Codex keeps its on-request default and tells the model how to ask the person before a command runs outside
    the sandbox, which the wrapper's calls to reef need. The rendered tree holds no rule that skips that ask."""
    files = _bound_files(binding=_binding("http://127.0.0.1:9"))
    assert not any(path.startswith("codex/rules/") for path in files)
    for relative, text in files.items():
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(text, encoding="utf-8")
    result = subprocess.run(
        [REAL_CODEX, "debug", "prompt-input", "hi"],
        env={**os.environ, "HOME": str(tmp_path), "CODEX_HOME": str(tmp_path / "codex")},
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    permissions = next(
        part["text"]
        for item in json.loads(result.stdout)
        for part in item.get("content", [])
        if "<permissions instructions>" in part.get("text", "")
    )
    assert "# Escalation Requests" in permissions and "Approval policy is currently never" not in permissions
