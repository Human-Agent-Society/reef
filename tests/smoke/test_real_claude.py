"""Smoke of the real Claude Code binary: every part an evolved tree may hold reaches the model.

The hermetic suites check the rendered files. Here the pinned Claude Code runs against a stdlib Anthropic
Messages stub that records each request, once as an episode and once as a person's session: the layer over an
empty own config directory, as ``reef-client harness run`` starts it. Skipped unless REEF_REAL_CLAUDE_BINARY
points at an installed Claude Code (.github/workflows/harness-smoke.yml provides one).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from tests.support.harness_render import write_layer

from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.episodes.run import run_episode
from reef.harness.sync import layer_install
from reef.harness.tree import HarnessTree, render_tree

REAL_CLAUDE = os.environ.get("REEF_REAL_CLAUDE_BINARY", "")

pytestmark = pytest.mark.skipif(not REAL_CLAUDE, reason="REEF_REAL_CLAUDE_BINARY does not name a real Claude Code")

MODEL = "claude-smoke"
#: The hook runs the extension at session start; Claude Code adds what it prints to the model's context.
HOOKS = {"SessionStart": [{"hooks": [{"type": "command", "command": 'node "{root}/claude/extensions/probe.mjs"'}]}]}
PARTS = {
    "rules.md": "The reef smoke rules marker is TIDEPOOL-RULES-41.\n",
    "skills/probe/SKILL.md": "---\nname: probe\ndescription: The reef smoke skill marker is TIDEPOOL-SKILL-23.\n---\n",
    "commands/ship.md": "The reef smoke command marker is TIDEPOOL-COMMAND-31.\n",
    "memory/MEMORY.md": "- The reef smoke memory marker is TIDEPOOL-MEMORY-17.\n",
    "extensions/probe.mjs": 'console.log("The reef smoke hook marker is TIDEPOOL-HOOK-37.");\n',
    "config/primary.json": json.dumps({"hooks": HOOKS}),
}
MARKERS = ("TIDEPOOL-RULES-41", "TIDEPOOL-SKILL-23", "TIDEPOOL-COMMAND-31", "TIDEPOOL-MEMORY-17", "TIDEPOOL-HOOK-37")
#: The variables an episode sets so Claude Code makes no network call of its own.
QUIET = {
    "DISABLE_AUTOUPDATER": "1",
    "DISABLE_TELEMETRY": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}


class StubMessages(ThreadingHTTPServer):
    """A canned Anthropic Messages endpoint that records every request body it is sent."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), StubHandler)
        self.bodies: list[str] = []


class StubHandler(BaseHTTPRequestHandler):
    server: StubMessages

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8")
        self.server.bodies.append(body)
        usage = {"input_tokens": 1, "output_tokens": 1}
        message = {"id": "msg_smoke", "type": "message", "role": "assistant", "model": MODEL, "usage": usage}
        if not json.loads(body).get("stream"):
            payload = {**message, "content": [{"type": "text", "text": "READY"}], "stop_reason": "end_turn"}
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        events = [
            {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "READY"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
            {"type": "message_stop"},
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event in events:
            self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())


@pytest.mark.parametrize("form", ["episode", "layer"])
def test_real_claude_takes_every_part_of_a_tree_in_an_episode_and_over_a_persons_own_config(
    tmp_path: Path, form: str
) -> None:
    """Every part an evolved tree may hold reaches the model, in an episode and on a person's machine, where Claude
    Code loads the person's own config directory, empty here, and the release as the adapter's layer over it. The
    prompt runs the command, whose name a plugin's command takes after reef:."""
    server = StubMessages()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    descriptor = get_adapter("claude")
    binding = ModelBinding(base_url=url, model=MODEL, api_key="dummy", api="anthropic")
    try:
        if form == "episode":
            files = render_tree(HarnessTree(PARTS), descriptor, binding.config_overlays(descriptor))
            result = run_episode(descriptor, files, "/ship", binary=REAL_CLAUDE, timeout=120.0)
            assert result.exit_code == 0, (result.stdout, result.stderr)
        else:
            layer = layer_install(HarnessTree(PARTS), descriptor, binding, url, str(tmp_path / "install"))
            own = tmp_path / "own"
            own.mkdir()
            env = {**os.environ, **QUIET, **write_layer(tmp_path / "install", layer), "CLAUDE_CONFIG_DIR": str(own)}
            argv = [REAL_CLAUDE, *layer.args, "-p", "/reef:ship", "--output-format", "json"]
            done = subprocess.run(
                argv, env=env, cwd=tmp_path, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120
            )
            assert done.returncode == 0, (done.stdout, done.stderr)
    finally:
        server.shutdown()
        server.server_close()
    requests = "".join(server.bodies)
    assert [marker for marker in MARKERS if marker not in requests] == []
