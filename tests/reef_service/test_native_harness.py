"""The native adapter: a loop inside the tree whose tools are nodes, driven through the real episode engine.

A stdlib HTTP server plays the served model: it answers by conversation state
(write a file, then read it back, then answer), so the whole loop, the tool
modules, and the trajectory run hermetically."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from reef.harness.adapters import available_adapters, get_adapter
from reef.harness.episode import run_episode
from reef.harness.model_binding import ModelBinding
from reef.harness.native import MAX_RESULT_CHARS, ToolModule, _invoke, _LoopGuard
from reef.harness.native.seed import SEED_TOOLS
from reef.harness.nodes import NODE_KINDS
from reef.harness.render import RenderError, render_composition
from reef.harness.trajectory import reader_for
from reef.train.cordis_backend import CordisBackend, Mutation, ScoreComparisonSelector
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer
from reef.train.evaluation import DefaultCandidateEvaluationPlugin
from reef.train.types import TraceBatch, TraceSample

TOOL = (
    "native_tool",
    {
        "name": "shout",
        "description": "Upper-case a string.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        "code": "def run(args, workdir):\n    return str(args.get('text', '')).upper()\n",
    },
)


def _reply(content=None, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"id": "fake", "object": "chat.completion", "choices": [{"index": 0, "message": message}]}


def _call(name, arguments, call_id):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


class _FakeModel(ThreadingHTTPServer):
    """Answers /v1/chat/completions by conversation state, so every episode gets the same script."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests: list[dict] = []
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def script(self, body: dict) -> dict:
        tool_results = [m for m in body["messages"] if m.get("role") == "tool"]
        if not tool_results:
            return _reply(tool_calls=[_call("write_file", {"path": "notes.txt", "content": "hello"}, "c1")])
        if len(tool_results) == 1:
            return _reply(tool_calls=[_call("read_file", {"path": "notes.txt"}, "c2")])
        return _reply(content=f"The file says: {tool_results[-1]['content']}")


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        self.server.requests.append(body)  # type: ignore[attr-defined]
        payload = json.dumps(self.server.script(body)).encode()  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        return None


@pytest.fixture
def fake_model():
    server = _FakeModel()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _launcher(tmp_path: Path) -> str:
    """The episode binary: this interpreter running the native loop, like an installed reef-native.

    The child gets no PYTHONPATH from the episode engine, so the source
    checkout is put on sys.path the way pytest does for this process."""
    root = Path(__file__).resolve().parents[2]
    path = tmp_path / "reef-native"
    path.write_text(
        f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(root)!r})\n"
        "from reef.harness.native import main\nsys.exit(main())\n"
    )
    path.chmod(0o755)
    return str(path)


def _seed_nodes():
    return [(entry["name"], entry["config"]) for entry in SEED_TOOLS]


def _events(trajectory, type_):
    return [event for event in trajectory if event["type"] == type_]


def test_native_adapter_is_bundled_and_renders_tools_as_modules() -> None:
    assert "native" in available_adapters()
    descriptor = get_adapter("native")
    assert descriptor.binary == "reef-native" and descriptor.trajectory_format == "native-jsonl"
    files = render_composition([("rules", {"text": "Be brief."}), TOOL], descriptor)
    module = files["native/tools/shout.py"]
    assert "NAME = 'shout'" in module and "PARAMETERS = {" in module and "def run(args, workdir):" in module
    assert files["native/RULES.md"] == "Be brief.\n"
    assert type(reader_for("native-jsonl")).__name__ == "NativeSessionReader"


def test_native_tool_is_optional_per_adapter_and_validated() -> None:
    with pytest.raises(RenderError, match="does not render native_tool"):
        render_composition([TOOL], get_adapter("pi"))
    NODE_KINDS["native_tool"](None, TOOL[1])
    with pytest.raises(ValueError, match="'description'"):
        NODE_KINDS["native_tool"](None, {"name": "x", "code": "def run(a, w): pass"})
    with pytest.raises(ValueError, match="'parameters' must be an object"):
        NODE_KINDS["native_tool"](None, {**TOOL[1], "parameters": ["nope"]})
    with pytest.raises(ValueError, match="inline credential"):
        NODE_KINDS["native_tool"](None, {**TOOL[1], "code": "KEY = 'sk-476-TOOL-KEY-0123456789abcdef'"})


def test_tool_results_carry_closed_error_codes_and_validate_arguments(tmp_path: Path) -> None:
    def run(args, workdir):
        if args.get("text") == "boom":
            raise RuntimeError("kaboom")
        return "x" * (MAX_RESULT_CHARS + 5) if args.get("text") == "long" else args["text"].upper()

    tools = {"shout": ToolModule("shout", "Upper-case a string.", TOOL[1]["parameters"], run)}
    assert _invoke(tools, "nope", "{}", tmp_path)["error"]["code"] == "UNKNOWN_TOOL"
    assert _invoke(tools, "shout", "{}", tmp_path)["error"] == {
        "code": "INVALID_ARGS",
        "message": "missing required argument 'text'",
    }
    assert _invoke(tools, "shout", '{"text": 3}', tmp_path)["error"]["code"] == "INVALID_ARGS"
    assert _invoke(tools, "shout", "not json", tmp_path)["error"]["code"] == "INVALID_ARGS"
    failed = _invoke(tools, "shout", '{"text": "boom"}', tmp_path)
    assert failed["is_error"] is True and failed["error"]["code"] == "TOOL_FAILED"
    assert failed["content"] == "Error: RuntimeError: kaboom"
    ok = _invoke(tools, "shout", '{"text": "hi"}', tmp_path)
    assert ok == {"content": "HI", "is_error": False, "arguments": {"text": "hi"}, "meta": ok["meta"]}
    assert ok["meta"]["truncated"] is False
    long = _invoke(tools, "shout", '{"text": "long"}', tmp_path)
    assert len(long["content"]) == MAX_RESULT_CHARS and long["meta"]["truncated"] is True


def test_loop_guard_reminds_at_the_thresholds_only() -> None:
    guard = _LoopGuard()
    notes = [guard.note("read_file", {"path": "a"}) for _ in range(5)]
    assert [note is not None for note in notes] == [False, False, True, False, True]
    assert guard.note("read_file", {"path": "b"}) is None


def test_native_loop_runs_seed_tools_and_logs_everything_the_model_saw(tmp_path: Path, fake_model) -> None:
    descriptor = get_adapter("native")
    binding = ModelBinding(base_url=fake_model.base_url, model="fake", api_key="dummy")
    files = render_composition(
        [*_seed_nodes(), ("rules", {"text": "Be brief."}), *binding.compose_nodes(descriptor)], descriptor
    )
    result = run_episode(descriptor, files, "put hello in notes.txt and read it back", binary=_launcher(tmp_path))
    kinds = [event["type"] for event in result.trajectory]
    assert kinds == [
        "session",
        "turn/start",
        "step/start",
        "request/header",
        "assistant/message",
        "tool/call",
        "tool/result",
        "step/end",
        "step/start",
        "assistant/message",
        "tool/call",
        "tool/result",
        "step/end",
        "step/start",
        "assistant/message",
        "step/end",
        "turn/end",
    ]
    assert [event["seq"] for event in result.trajectory] == list(range(len(kinds)))
    header = result.trajectory[0]["data"]
    assert header["version"] == 1 and header["tools"] == ["read_file", "run_bash", "write_file"]
    request = _events(result.trajectory, "request/header")[0]["data"]
    assert request["system"] == "Be brief."
    assert sorted(tool["function"]["name"] for tool in request["tools"]) == ["read_file", "run_bash", "write_file"]
    written, read = (event["data"] for event in _events(result.trajectory, "tool/result"))
    assert written["name"] == "write_file" and written["content"] == "wrote 5 characters to notes.txt"
    assert written["is_error"] is False and written["arguments"] == {"path": "notes.txt", "content": "hello"}
    assert read["name"] == "read_file" and read["content"] == "hello"
    assert _events(result.trajectory, "assistant/message")[-1]["data"]["content"] == "The file says: hello"
    assert result.trajectory[-1]["data"]["reason"] == {"kind": "completed"}
    # The first request the fake model saw is the one the log says it saw.
    first = fake_model.requests[0]
    assert first["messages"][0] == {"role": "system", "content": "Be brief."}
    assert sorted(tool["function"]["name"] for tool in first["tools"]) == ["read_file", "run_bash", "write_file"]


def test_native_harness_runs_through_the_evolution_gate(tmp_path: Path, fake_model) -> None:
    """A proposal that adds a tool node is rendered, run on both sides, and scored like any other mutation."""

    def final_answer(task, result):
        answers = [event for event in result.trajectory if event["type"] == "assistant/message"]
        return 1.0 if answers and "hello" in (answers[-1]["data"].get("content") or "") else 0.0

    binding = ModelBinding(base_url=fake_model.base_url, model="fake", api_key="dummy")
    backend = CordisBackend(
        descriptor=get_adapter("native"),
        propose=resolve_proposer(
            lambda nodes, samples, models: Mutation("create", "shout", {"name": "native_tool", "config": TOOL[1]})
        ),
        score_episode=resolve_episode_scorer(final_answer),
        tasks=("put hello in notes.txt and read it back",),
        models=binding,
        binary=_launcher(tmp_path),
        seed=SEED_TOOLS,
    )
    batch = TraceBatch("demo:trace:native", (TraceSample("a1", {"messages": []}, 0.0),))
    prepared = backend.prepare_step(batch, backend.initial_state(), 0)
    assert prepared.candidate is not None
    assert "native/tools/shout.py" in prepared.candidate.candidate_files
    evaluator = DefaultCandidateEvaluationPlugin(backend, ScoreComparisonSelector())
    decision = evaluator.decide(prepared.candidate, evaluator.evaluate(prepared.candidate))
    result = backend.settle_step(prepared, decision)
    # Both trees complete the scripted task, so the verdict is a tie and nothing publishes.
    assert result.metrics["wins"] == 0 and result.metrics["losses"] == 0 and result.metrics["ties"] == 1
    assert result.metrics["selected"] is False
