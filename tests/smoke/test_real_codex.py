"""Smoke the pinned Codex CLI through Reef's complete harness path.

The hermetic suite uses a scripted binary. These tests exercise the real CLI
against a local Responses API stub: the runs prove rules, skills, binding,
trajectory collection, cleanup, and temporary proxy path. The workflow
supplies the pinned binary through ``REEF_REAL_CODEX_BINARY``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import tomllib
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from reef.core.model_metadata import ModelMetadata
from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.episodes.run import run_episode
from reef.harness.episodes.trajectory import final_assistant_text
from reef.harness.tree.render import render_composition
from reef.service.install_script import render_install_script

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

    def do_GET(self) -> None:
        if self.path != "/v1/models":
            self.send_error(404)
            return
        payload = {"data": [{"id": MODEL, "context_length": 640_000, "supported_parameters": ["reasoning"]}]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

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


def model_binding(base_url: str) -> ModelBinding:
    return ModelBinding(base_url=base_url, model=MODEL, api_key="reef-smoke-key", api="responses").with_metadata()


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
            binding=model_binding(base_url),
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
            binding=model_binding(base_url),
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
    assert "fallback metadata" not in result.stdout + result.stderr
    started = next(
        event["payload"] for event in result.trajectory if event.get("payload", {}).get("type") == "task_started"
    )
    assert started["model_context_window"] == 608_000  # Codex reserves 5% of the declared 640k window.
    assert server.requests
    path, authorization, body = server.requests[0]
    assert path == "/v1/responses"
    assert authorization == "Bearer reef-smoke-key"
    assert RULES_MARKER in body
    # Codex advertises skill metadata first and reads the body only when the
    # model chooses the skill; discovery is the adapter contract here.
    assert "reef-smoke: Reef smoke skill" in body and ".agents/skills" in body
    assert any(event.get("type") == "session_meta" for event in result.trajectory)
    assert any(event.get("payload", {}).get("type") == "task_complete" for event in result.trajectory)
    assert result.residue == ()
    assert capture.is_file() and capture.stat().st_size > 0
    assert final_assistant_text(result.trajectory) == "READY"


@pytest.mark.parametrize("selected_model", [MODEL, "gpt-5.4", "gpt-6-astra"])
def test_installed_codex_reads_catalog_after_client_relocation(tmp_path: Path, selected_model: str) -> None:
    server, base_url = _server()
    try:
        descriptor = get_adapter("codex")
        composition = render_composition([("rules", {"text": RULES_MARKER})], descriptor)
        bound = _bound_files(("rules", {"text": RULES_MARKER}), binding=model_binding(base_url))
        script = tmp_path / "install.sh"
        script.write_text(
            render_install_script(
                descriptor=descriptor,
                files=composition,
                release_id="v-test",
                content_id="c-test",
                scenario="smoke",
                binding_files={path: bound[path] for path in ("codex/config.toml", "codex/models.json")},
            )
        )
        prefix = tmp_path / "prefix"
        binary = prefix / "node_modules/.bin/codex"
        binary.parent.mkdir(parents=True)
        binary.symlink_to(Path(REAL_CODEX).resolve())
        dest = tmp_path / "installed"
        env = {**os.environ, "HOME": str(tmp_path / "home"), "REEF_PYTHON": sys.executable}
        env.pop("REEF_TOKEN", None)
        installed = subprocess.run(
            ["sh", str(script), str(dest), str(prefix)], env=env, capture_output=True, text=True, timeout=30
        )
        assert installed.returncode == 0, installed.stderr
        config = tomllib.loads((dest / "codex/config.toml").read_text())
        assert config["model_catalog_json"] == "models.json"
        # The install writes the wrapper outside the tree, beside its record in the fake home.
        root_digest = hashlib.sha256(os.fsencode(os.path.realpath(dest))).hexdigest()
        run = subprocess.run(
            [
                str(tmp_path / "home" / ".reef" / "installs" / root_digest / "reef-codex"),
                "exec",
                "--json",
                "--strict-config",
                "--skip-git-repo-check",
                "--model",
                selected_model,
                "Reply READY",
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert run.returncode == 0, (run.stdout, run.stderr)
        assert "fallback metadata" not in run.stdout + run.stderr
        assert '"text":"READY"' in run.stdout
        sessions = [
            json.loads(line)
            for path in (dest / "codex/sessions").rglob("*.jsonl")
            for line in path.read_text().splitlines()
        ]
        started = next(
            event["payload"] for event in sessions if event.get("payload", {}).get("type") == "task_started"
        )
        expected_window = 608_000 if selected_model == MODEL else 258_400
        assert started["model_context_window"] == expected_window
        request_body = json.loads(server.requests[-1][2])
        assert request_body["model"] == selected_model
        if selected_model == "gpt-5.4":
            assert any(tool.get("name") == "apply_patch" for tool in request_body["tools"])
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("reasoning", [True, False])
def test_codex_sends_reasoning_parameters_only_when_supported(reasoning: bool) -> None:
    server, base_url = _server()
    try:
        binding = replace(model_binding(base_url), metadata=ModelMetadata(640_000, reasoning))
        files = _bound_files(binding=binding)
        result = run_episode(get_adapter("codex"), files, "Reply READY", binary=REAL_CODEX, timeout=30.0)
        assert result.exit_code == 0, result.stderr
        assert "fallback metadata" not in result.stdout + result.stderr
        body = json.loads(server.requests[0][2])
        if reasoning:
            assert body["reasoning"]["effort"] == "medium"
        else:
            assert body.get("reasoning", {}) == {}
    finally:
        server.shutdown()
        server.server_close()


def test_metadata_preserves_codex_unknown_model_instructions() -> None:
    server, base_url = _server()
    try:
        descriptor = get_adapter("codex")
        binding = ModelBinding(base_url=base_url, model=MODEL, api="responses")
        before = run_episode(descriptor, _bound_files(binding=binding), "Reply READY", binary=REAL_CODEX, timeout=30.0)
        assert before.exit_code == 0, before.stderr
        assert "fallback metadata" in before.stdout
        after = run_episode(
            descriptor, _bound_files(binding=binding.with_metadata()), "Reply READY", binary=REAL_CODEX, timeout=30.0
        )
        assert after.exit_code == 0, after.stderr
        assert "fallback metadata" not in after.stdout + after.stderr
        before_body = json.loads(server.requests[0][2])
        after_body = json.loads(server.requests[1][2])
        assert before_body["instructions"] == after_body["instructions"]
    finally:
        server.shutdown()
        server.server_close()


def model_request_config(request_json: str) -> dict[str, object]:
    """Extract native model settings from standard and Responses Lite requests."""
    request = json.loads(request_json)
    if "instructions" in request:
        instructions = request["instructions"]
        tools = request["tools"]
    else:
        instructions = next(
            item["content"] for item in request["input"] if item["type"] == "message" and item["role"] == "developer"
        )
        tools = next(item["tools"] for item in request["input"] if item["type"] == "additional_tools")
        for namespace in tools:
            if namespace["name"] != "collaboration":
                continue
            for tool in namespace["tools"]:
                if tool["name"] == "spawn_agent":
                    # Codex prepends the current catalog's model names and reasoning levels.
                    description = tool["description"]
                    tool["description"] = description[description.index("Spawns an agent") :]
    return {
        "model": request["model"],
        "instructions": instructions,
        "tools": tools,
        "reasoning": request.get("reasoning", {}),
    }


def test_codex_bundled_model_catalog_matches_the_pin() -> None:
    from reef.harness.adapters.codex.quirks import bundled_model_catalog

    descriptor = get_adapter("codex")
    result = run_episode(
        replace(descriptor, argv=("debug", "models", "--bundled")),
        render_composition([], descriptor),
        "",
        binary=REAL_CODEX,
        timeout=30.0,
    )
    assert result.exit_code == 0, result.stderr
    assert bundled_model_catalog() == {model["slug"]: model for model in json.loads(result.stdout)["models"]}


@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-6-astra"])
def test_codex_model_switch_keeps_bundled_instructions_and_tools(model: str) -> None:
    server, base_url = _server()
    try:
        descriptor = get_adapter("codex")
        switched = replace(descriptor, argv=(*descriptor.argv, "--model", model))
        binding = ModelBinding(base_url=base_url, model=MODEL, api="responses")
        before = run_episode(switched, _bound_files(binding=binding), "Reply READY", binary=REAL_CODEX, timeout=30.0)
        after = run_episode(
            switched, _bound_files(binding=binding.with_metadata()), "Reply READY", binary=REAL_CODEX, timeout=30.0
        )
        assert before.exit_code == after.exit_code == 0, (before.stderr, after.stderr)
        assert "fallback metadata" not in after.stdout + after.stderr
        assert model_request_config(server.requests[0][2]) == model_request_config(server.requests[1][2])
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("model", ["gpt-5.4", "openai/gpt-5.4", "gpt-6-astra", "openai/gpt-6-astra"])
@pytest.mark.parametrize("reasoning", [True, False])
def test_codex_native_model_applies_capabilities_and_keeps_instructions(model: str, reasoning: bool) -> None:
    server, base_url = _server()
    try:
        descriptor = get_adapter("codex")
        binding = ModelBinding(base_url=base_url, model=model, api="responses")
        before = run_episode(descriptor, _bound_files(binding=binding), "Reply READY", binary=REAL_CODEX, timeout=30.0)
        configured = replace(binding, metadata=ModelMetadata(32_000, reasoning))
        after = run_episode(
            descriptor, _bound_files(binding=configured), "Reply READY", binary=REAL_CODEX, timeout=30.0
        )
        assert before.exit_code == after.exit_code == 0, (before.stderr, after.stderr)
        assert "fallback metadata" not in after.stdout + after.stderr
        started = next(
            event["payload"] for event in after.trajectory if event.get("payload", {}).get("type") == "task_started"
        )
        assert started["model_context_window"] == 30_400
        before_body = model_request_config(server.requests[0][2])
        after_body = model_request_config(server.requests[1][2])
        assert before_body["instructions"] == after_body["instructions"]
        assert before_body["tools"] == after_body["tools"]
        if reasoning:
            assert after_body["reasoning"] == before_body["reasoning"]
        else:
            # Responses Lite keeps its context policy while omitting effort and summary.
            expected_reasoning = {"context": "all_turns"} if model.endswith("gpt-6-astra") else {}
            assert after_body["reasoning"] == expected_reasoning
    finally:
        server.shutdown()
        server.server_close()
