"""Real Pi smoke for the Meta-Harness host-to-Harbor bash bridge.

No provider call is made: a local OpenAI-compatible stub asks Pi to invoke
``bash`` once, and a fake Harbor environment records the forwarded command.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.model_binding import ModelBinding
from reef.harness.render import render_composition

REAL_PI = os.environ.get("REEF_REAL_PI_BINARY", "")
EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "recipes" / "meta_harness" / "examples" / "terminal_bench"
MODEL = "meta-harness-bridge-smoke"

pytestmark = pytest.mark.skipif(not REAL_PI, reason="REEF_REAL_PI_BINARY does not name a real Pi binary")


class StubOpenAI(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), StubHandler)
        self.requests: list[dict[str, Any]] = []


class StubHandler(BaseHTTPRequestHandler):
    server: StubOpenAI

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        self._json({"object": "list", "data": [{"id": MODEL, "object": "model"}]})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", "0"))))
        self.server.requests.append(body)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        chunks = self._tool_call_chunks() if len(self.server.requests) == 1 else self._final_chunks()
        for chunk in chunks:
            event = {
                "id": "chatcmpl-meta-harness-bridge",
                "object": "chat.completion.chunk",
                "model": MODEL,
                **chunk,
            }
            self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    @staticmethod
    def _tool_call_chunks() -> list[dict[str, Any]]:
        return [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_bridge",
                                    "type": "function",
                                    "function": {
                                        "name": "bash",
                                        "arguments": '{"command":"printf BRIDGE_OK"}',
                                    },
                                }
                            ],
                        },
                    }
                ]
            },
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        ]

    @staticmethod
    def _final_chunks() -> list[dict[str, Any]]:
        return [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "done"}}]},
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
            },
        ]

    def _json(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class FakeHarborEnvironment:
    def __init__(self) -> None:
        self.commands: list[tuple[str, int]] = []

    async def exec(self, *, command: str, timeout_sec: int) -> SimpleNamespace:
        self.commands.append((command, timeout_sec))
        return SimpleNamespace(stdout="BRIDGE_OK", stderr="", return_code=0)


def test_real_pi_executes_bash_through_harbor_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        del sys.modules[name]
    composition = importlib.import_module("harness.composition")
    pi_bridge = importlib.import_module("harness.pi_bridge")
    server = StubOpenAI()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    environment = FakeHarborEnvironment()

    async def run_episode():
        descriptor = get_adapter("pi")
        binding = ModelBinding(
            base_url=f"http://127.0.0.1:{server.server_address[1]}",
            model=MODEL,
            api_key="dummy",
        )
        candidate = composition.genesis_composition()
        files = render_composition((*candidate.nodes, *binding.compose_nodes(descriptor)), descriptor)
        runner = pi_bridge.PiEpisodeRunner(descriptor, binary=Path(REAL_PI), timeout_s=60)
        loop = asyncio.get_running_loop()
        with pi_bridge.HarborExecBridge(environment.exec, loop) as bridge:
            return await asyncio.to_thread(
                runner.run,
                files,
                "Use bash once, then finish.",
                bridge_url=bridge.url,
                bridge_token=bridge.token,
            )

    try:
        result = asyncio.run(run_episode())
    finally:
        server.shutdown()
        server.server_close()

    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert len(server.requests) == 2
    assert environment.commands == [("printf BRIDGE_OK", 600)]
    assert result.trajectory
    assert result.usage == {"input_tokens": 30, "cached_input_tokens": 0, "output_tokens": 7}
    assert result.estimated_cost_usd > 0
    assert result.provider_reported_cost_usd == 0
