"""Smoke the pinned Codex CLI through Reef's complete harness path.

The hermetic suite uses a scripted binary. These tests exercise the real CLI
against a local Responses API stub: the runs prove rules, skills, binding,
trajectory collection, cleanup, and temporary proxy path. The workflow
supplies the pinned binary through ``REEF_REAL_CODEX_BINARY``.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from collections.abc import Mapping
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from tests.support.harness_render import write_layer

from reef.core.model_metadata import ModelMetadata
from reef.harness.adapters import get_adapter
from reef.harness.sync import layer_install
from reef.harness.tasks.model_binding import ModelBinding
from reef.harness.tasks.run import run_task
from reef.harness.tasks.trajectory import final_assistant_text
from reef.harness.tree import HarnessTree, render_tree

REAL_CODEX = os.environ.get("REEF_REAL_CODEX_BINARY", "")

pytestmark = pytest.mark.skipif(not REAL_CODEX, reason="REEF_REAL_CODEX_BINARY does not name a real Codex binary")

MODEL = "smoke-model"
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


def _bound_files(files: Mapping[str, str] | None = None, *, binding: ModelBinding) -> dict[str, str]:
    """The source tree ``files`` rendered for codex under ``binding``."""
    descriptor = get_adapter("codex")
    return render_tree(HarnessTree(dict(files or {})), descriptor, binding.config_overlays(descriptor))


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


def test_real_codex_renders_runs_collects_and_cleans_up(tmp_path: Path) -> None:
    server, base_url = _server()
    try:
        files = _bound_files(
            {
                "skills/reef-smoke/SKILL.md": f"# Reef smoke skill\n\n{SKILL_MARKER}",
                "commands/reef-command.md": "Reef smoke command",
            },
            binding=model_binding(base_url),
        )
        capture = Path(os.environ.get("REEF_REAL_CODEX_SESSION_OUT", tmp_path / "real-codex-session.jsonl"))
        result = run_task(
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
    # Codex advertises skill metadata first and reads the body only when the
    # model chooses the skill; discovery is the adapter contract here. The
    # skills come from the plugin reef, and a command is a skill there too.
    assert "reef:reef-smoke: Reef smoke skill" in body
    assert "reef:reef-command: Reef smoke command" in body
    # config.toml sets no approval_policy; the run still runs with approval never, so it never waits.
    assert "Approval policy is currently never" in body and "# Escalation Requests" not in body
    assert any(event.get("type") == "session_meta" for event in result.trajectory)
    assert any(event.get("payload", {}).get("type") == "task_complete" for event in result.trajectory)
    # The grader reads the stub's reply from the run the real binary wrote.
    assert final_assistant_text(result.trajectory) == "READY"
    assert result.residue == ()
    assert capture.is_file() and capture.stat().st_size > 0


def test_real_codex_session_config_lets_the_model_ask_the_person_to_leave_the_sandbox(tmp_path: Path) -> None:
    """An interactive session on the rendered config.toml, which sets no approval_policy, keeps Codex's on-request
    default and tells the model how to ask the person before a command runs outside the sandbox. The rendered tree
    holds no rule that skips that ask."""
    descriptor = get_adapter("codex")
    files = descriptor.place_root(_bound_files(binding=model_binding("http://127.0.0.1:9")), str(tmp_path))
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


@pytest.mark.parametrize("reasoning", [True, False])
def test_codex_sends_reasoning_parameters_only_when_supported(reasoning: bool) -> None:
    server, base_url = _server()
    try:
        binding = replace(model_binding(base_url), metadata=ModelMetadata(640_000, reasoning))
        files = _bound_files(binding=binding)
        result = run_task(get_adapter("codex"), files, "Reply READY", binary=REAL_CODEX, timeout=30.0)
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
        before = run_task(descriptor, _bound_files(binding=binding), "Reply READY", binary=REAL_CODEX, timeout=30.0)
        assert before.exit_code == 0, before.stderr
        assert "fallback metadata" in before.stdout
        after = run_task(
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
    from reef.harness.adapters.codex.render import bundled_model_catalog

    descriptor = get_adapter("codex")
    result = run_task(
        replace(descriptor, argv=("debug", "models", "--bundled")),
        render_tree(HarnessTree({}), descriptor),
        "",
        binary=REAL_CODEX,
        timeout=30.0,
    )
    assert result.exit_code == 0, result.stderr
    assert bundled_model_catalog() == {model["slug"]: model for model in json.loads(result.stdout)["models"]}


#: Codex 0.158.0 leaves the sentence on checklists out of gpt-5.5's own prompt, but keeps it from a model catalog,
#: so a run with model metadata on gpt-5.5 sends one sentence more than Codex does.
GPT_5_5_KEEPS_A_SENTENCE = pytest.mark.xfail(
    strict=True, reason="Codex drops gpt-5.5's checklist sentence only for its own catalog"
)


@pytest.mark.parametrize(
    "model", [pytest.param("gpt-5.5", marks=GPT_5_5_KEEPS_A_SENTENCE), "gpt-5.6-sol", "gpt-6-astra"]
)
def test_codex_model_switch_keeps_bundled_instructions_and_tools(model: str) -> None:
    server, base_url = _server()
    try:
        descriptor = get_adapter("codex")
        switched = replace(descriptor, argv=(*descriptor.argv, "--model", model))
        binding = ModelBinding(base_url=base_url, model=MODEL, api="responses")
        before = run_task(switched, _bound_files(binding=binding), "Reply READY", binary=REAL_CODEX, timeout=30.0)
        after = run_task(
            switched, _bound_files(binding=binding.with_metadata()), "Reply READY", binary=REAL_CODEX, timeout=30.0
        )
        assert before.exit_code == after.exit_code == 0, (before.stderr, after.stderr)
        assert "fallback metadata" not in after.stdout + after.stderr
        assert model_request_config(server.requests[0][2]) == model_request_config(server.requests[1][2])
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("gpt-5.5", marks=GPT_5_5_KEEPS_A_SENTENCE),
        pytest.param("openai/gpt-5.5", marks=GPT_5_5_KEEPS_A_SENTENCE),
        "gpt-5.6-sol",
        "openai/gpt-5.6-sol",
        "gpt-6-astra",
        "openai/gpt-6-astra",
    ],
)
@pytest.mark.parametrize("reasoning", [True, False])
def test_codex_native_model_applies_capabilities_and_keeps_instructions(model: str, reasoning: bool) -> None:
    server, base_url = _server()
    try:
        descriptor = get_adapter("codex")
        binding = ModelBinding(base_url=base_url, model=model, api="responses")
        before = run_task(descriptor, _bound_files(binding=binding), "Reply READY", binary=REAL_CODEX, timeout=30.0)
        configured = replace(binding, metadata=ModelMetadata(32_000, reasoning))
        after = run_task(descriptor, _bound_files(binding=configured), "Reply READY", binary=REAL_CODEX, timeout=30.0)
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
            expected_reasoning = {key: value for key, value in before_body["reasoning"].items() if key == "context"}
            assert after_body["reasoning"] == expected_reasoning
    finally:
        server.shutdown()
        server.server_close()


#: A tree with a marker in each part codex's adapter evolves.
CODEX_PARTS = {
    "skills/probe/SKILL.md": f"---\nname: probe\ndescription: {SKILL_MARKER}\n---\nProbe.\n",
    "commands/ship.md": "---\ndescription: The Reef Codex command marker is TIDEPOOL-COMMAND-33.\n---\nShip.\n",
}


@pytest.mark.parametrize("form", ["run", "layer"])
def test_real_codex_takes_every_part_of_a_tree_in_an_run_and_over_a_persons_own_config(
    tmp_path: Path, form: str
) -> None:
    """Every part an evolved tree may hold reaches the model, in a run and on a person's machine, where Codex
    keeps the person's own home, empty here, and takes the skills from the plugin the layer registers.
    Unregistering leaves the person's config without the plugin."""
    server, base_url = _server()
    descriptor = get_adapter("codex")
    binding = ModelBinding(base_url=base_url, model="gpt-6-astra", api_key="reef-smoke-key", api="responses")
    try:
        if form == "run":
            files = _bound_files(CODEX_PARTS, binding=binding)
            result = run_task(descriptor, files, "Reply READY", binary=REAL_CODEX, timeout=120.0)
            assert result.exit_code == 0, result.stderr
        else:
            own = tmp_path / "own"
            own.mkdir()
            layer = layer_install(HarnessTree(CODEX_PARTS), descriptor, binding, base_url, str(tmp_path / "install"))
            env = {**os.environ, **write_layer(tmp_path / "install", layer), "CODEX_HOME": str(own)}
            for command in layer.register:
                subprocess.run([REAL_CODEX, *command], env=env, check=True, capture_output=True)
            argv = [REAL_CODEX, *layer.args, "exec", "--json", "--skip-git-repo-check", "Reply READY"]
            done = subprocess.run(argv, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120)
            assert done.returncode == 0, done.stderr
            for command in layer.unregister:
                subprocess.run([REAL_CODEX, *command], env=env, check=True, capture_output=True)
            assert "reef" not in (own / "config.toml").read_text()
    finally:
        server.shutdown()
        server.server_close()
    request = server.requests[-1][2]
    markers = ("TIDEPOOL-SKILL-27", "TIDEPOOL-COMMAND-33")
    assert [marker for marker in markers if marker not in request] == []
