"""The native adapter: a loop inside the tree whose tools and events are nodes, driven through the real episode engine.

A stdlib HTTP server plays the served model: it answers by conversation state
(write a file, then read it back, then answer), so the whole loop, the tool
and hook modules, and the trajectory run hermetically."""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from reef_service.test_native_enforce import PROBE

from reef.harness.adapters import available_adapters, get_adapter
from reef.harness.episode import run_episode
from reef.harness.executor import SandboxExecutor
from reef.harness.model_binding import ModelBinding
from reef.harness.native import (
    _DEFAULTS,
    MAX_COMPLETION_TOKENS,
    MAX_RESULT_CHARS,
    SPILL_TAIL_CHARS,
    HookModule,
    ToolModule,
    _invoke,
    _waterfall,
    load_hooks,
    load_tools,
    run_loop,
)
from reef.harness.native.seed import SEED_GRAPH, SEED_NODES, SEED_TOOLS
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
HOOK = (
    "native_hook",
    {"name": "echo", "event": "pre_step", "code": "def listen(payload, next):\n    return next()\n"},
)
# One hook per event: retry the first failed request, block writes with a note, stop before the third step.
EVENT_HOOKS = (
    (
        "native_hook",
        {
            "name": "retry",
            "event": "request_error",
            "code": "def listen(payload, next):\n    return {'kind': 'retry', 'delay_ms': 0}\n",
        },
    ),
    (
        "native_hook",
        {
            "name": "veto",
            "event": "post_execute",
            "code": (
                "def listen(payload, next):\n"
                "    decision = next()\n"
                "    if payload['name'] == 'write_file':\n"
                "        return {'kind': 'block', 'feedback': 'writes are off', 'contexts': ['Writes are disabled here.']}\n"
                "    return decision\n"
            ),
        },
    ),
    (
        "native_hook",
        {
            "name": "stop",
            "event": "pre_step",
            "code": (
                "def listen(payload, next):\n"
                "    if payload['step'] == 3:\n"
                "        return {'kind': 'reject', 'reason': 'two steps are enough'}\n"
                "    return next()\n"
            ),
        },
    ),
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

    def status(self, body: dict) -> int:
        return 200

    def script(self, body: dict) -> dict:
        tool_results = [m for m in body["messages"] if m.get("role") == "tool"]
        if not tool_results:
            return _reply(tool_calls=[_call("write_file", {"path": "notes.txt", "content": "hello"}, "c1")])
        if len(tool_results) == 1:
            return _reply(tool_calls=[_call("read_file", {"path": "notes.txt"}, "c2")])
        return _reply(content=f"The file says: {tool_results[-1]['content']}")


class _FlakyModel(_FakeModel):
    """Fails the first request with a 500, then plays the script."""

    def status(self, body: dict) -> int:
        return 500 if len(self.requests) == 1 else 200


class _RepeatingModel(_FakeModel):
    """Asks for the same read three times, then answers."""

    def script(self, body: dict) -> dict:
        if len(self.requests) <= 3:
            return _reply(tool_calls=[_call("read_file", {"path": "missing.txt"}, f"c{len(self.requests)}")])
        return _reply(content="giving up")


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        self.server.requests.append(body)  # type: ignore[attr-defined]
        payload = json.dumps(self.server.script(body)).encode()  # type: ignore[attr-defined]
        self.send_response(self.server.status(body))  # type: ignore[attr-defined]
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


def _seed_nodes(entries=SEED_NODES):
    return [(entry["name"], entry["config"]) for entry in entries]


def _episode(tmp_path: Path, model, nodes, prompt="put hello in notes.txt and read it back"):
    descriptor = get_adapter("native")
    binding = ModelBinding(base_url=model.base_url, model="fake", api_key="dummy")
    files = render_composition([*nodes, *binding.compose_nodes(descriptor)], descriptor)
    return run_episode(descriptor, files, prompt, binary=_launcher(tmp_path))


def _events(trajectory, type_):
    return [event for event in trajectory if event["type"] == type_]


def test_native_adapter_is_bundled_and_renders_tools_as_modules() -> None:
    assert "native" in available_adapters()
    descriptor = get_adapter("native")
    assert descriptor.binary == "reef-native" and descriptor.trajectory_format == "native-jsonl"
    # The session directory is the one path the sandbox opens, so the loop can write its log inside the jail.
    assert descriptor.writable_paths == ("native/sessions",) and descriptor.trajectory_path == "native/sessions"
    files = render_composition([("rules", {"text": "Be brief."}), TOOL, HOOK], descriptor)
    module = files["native/tools/shout.py"]
    assert "NAME = 'shout'" in module and "PARAMETERS = {" in module and "def run(args, workdir):" in module
    # The config constants come after the code, so the tree's values are what the module ends with.
    assert (
        files["native/hooks/echo.py"]
        == "def listen(payload, next):\n    return next()\n\nNAME = 'echo'\nEVENT = 'pre_step'\n"
    )
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
    with pytest.raises(ValueError, match="'code' does not compile"):
        NODE_KINDS["native_tool"](None, {**TOOL[1], "code": "def run(args, workdir)\n    return 1\n"})


def test_native_hook_is_optional_per_adapter_and_bound_to_an_event() -> None:
    with pytest.raises(RenderError, match="does not render native_hook"):
        render_composition([HOOK], get_adapter("pi"))
    NODE_KINDS["native_hook"](None, HOOK[1])
    with pytest.raises(ValueError, match="'event' must be one of pre_step, pre_execute, request_error, post_execute"):
        NODE_KINDS["native_hook"](None, {**HOOK[1], "event": "tool_router"})
    with pytest.raises(ValueError, match="'code'"):
        NODE_KINDS["native_hook"](None, {"name": "x", "event": "pre_step"})
    with pytest.raises(ValueError, match="'code' does not compile"):
        NODE_KINDS["native_hook"](None, {**HOOK[1], "code": "def listen(:\n"})
    # A lone surrogate survives JSON but cannot be written as UTF-8 at render.
    with pytest.raises(ValueError, match="must be UTF-8 encodable"):
        NODE_KINDS["native_hook"](
            None, {**HOOK[1], "code": "X = '\udc80'\ndef listen(payload, next):\n    return next()\n"}
        )


def test_rendered_constants_bind_over_what_the_code_assigns(tmp_path: Path) -> None:
    sneaky = {
        "name": "sneaky",
        "event": "pre_step",
        "code": "EVENT = 'post_execute'\nNAME = 'loop_guard'\ndef listen(payload, next):\n    return next()\n",
    }
    files = render_composition([("native_hook", sneaky)], get_adapter("native"))
    for relative, text in files.items():
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(text)
    hooks = load_hooks(tmp_path / "native" / "hooks")
    assert [(hook.name, hook.event) for hook in hooks["pre_step"]] == [("sneaky", "pre_step")]
    assert hooks["post_execute"] == []


def test_hooks_form_a_waterfall_where_next_runs_once_and_a_raising_hook_is_skipped() -> None:
    calls: list[str] = []

    def a(payload, next_):
        calls.append("a")
        below = next_()
        assert next_() is below
        return {**below, "messages": [*below["messages"], "from a"]}

    def b(payload, next_):
        calls.append("b")
        raise RuntimeError("boom")

    def c(payload, next_):
        calls.append("c")
        return {"kind": "enter", "messages": ["from c"]}

    def d(payload, next_):
        calls.append("d")
        return next_()

    hooks = [HookModule(name, "pre_step", listen) for name, listen in (("a", a), ("b", b), ("c", c), ("d", d))]
    trace: list[dict] = []
    decision = _waterfall(hooks, 0, {"step": 1}, _DEFAULTS["pre_step"], trace)
    assert decision == {"kind": "enter", "messages": ["from c", "from a"]}
    # c owned the event, so d never ran; b's error fell through to c.
    assert calls == ["a", "b", "c"]
    assert trace == [
        {"hook": "b", "error": "RuntimeError: boom"},
        {"hook": "c", "owned": True, "decision": {"kind": "enter", "messages": ["from c"]}},
        {"hook": "a", "owned": False, "decision": {"kind": "enter", "messages": ["from c", "from a"]}},
    ]
    # No hooks: the loop's default, as a fresh copy.
    assert _waterfall([], 0, {}, _DEFAULTS["post_execute"], trace) == {"kind": "accept", "contexts": []}
    assert _waterfall([], 0, {}, _DEFAULTS["post_execute"], trace) is not _DEFAULTS["post_execute"]


def test_hooks_get_a_copy_so_in_place_edits_are_traced_and_bad_decisions_are_skipped() -> None:
    def in_place(payload, next_):
        decision = next_()
        decision["contexts"].append("from in_place")
        return decision

    def forgetful(payload, next_):
        next_()["contexts"].append("lost")

    def cyclic(payload, next_):
        decision = dict(next_())
        decision["self"] = decision
        return decision

    def odd_keys(payload, next_):
        return {**next_(), (1, 2): "x"}

    def fresh(payload, next_):
        return {"kind": "accept", "contexts": "not a list"}

    names = (
        ("in_place", in_place),
        ("forgetful", forgetful),
        ("cyclic", cyclic),
        ("odd_keys", odd_keys),
        ("fresh", fresh),
    )
    hooks = [HookModule(name, "post_execute", listen) for name, listen in names]
    trace: list[dict] = []
    decision = _waterfall(hooks, 0, {"step": 1}, _DEFAULTS["post_execute"], trace)
    # fresh's non-list contexts read as empty; every hook above edited a copy, so only in_place changed the decision.
    assert decision == {"kind": "accept", "contexts": ["from in_place"]}
    assert trace == [
        {"hook": "fresh", "owned": True, "decision": {"kind": "accept", "contexts": []}},
        {"hook": "odd_keys", "error": "TypeError: keys must be str, int, float, bool or None, not tuple"},
        {"hook": "cyclic", "error": "ValueError: Circular reference detected"},
        {"hook": "in_place", "owned": False, "decision": {"kind": "accept", "contexts": ["from in_place"]}},
    ]


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
    # With a spill path the whole result lands on disk and the model reads head, marker, and tail within the cap.
    spilled = _invoke(tools, "shout", '{"text": "long"}', tmp_path, spill=tmp_path / ".reef" / "spill" / "1-c1.txt")
    assert (tmp_path / ".reef" / "spill" / "1-c1.txt").read_text() == "x" * (MAX_RESULT_CHARS + 5)
    assert spilled["meta"]["truncated"] is True and spilled["meta"]["spill"] == ".reef/spill/1-c1.txt"
    assert len(spilled["content"]) <= MAX_RESULT_CHARS
    assert "[5 characters omitted; the full result is in .reef/spill/1-c1.txt]" in spilled["content"]
    assert spilled["content"].startswith("x" * 100) and spilled["content"].endswith("x" * SPILL_TAIL_CHARS)


def test_native_loop_runs_seed_tools_and_logs_everything_the_model_saw(tmp_path: Path, fake_model) -> None:
    result = _episode(tmp_path, fake_model, [*_seed_nodes(), ("rules", {"text": "Be brief."})])
    kinds = [event["type"] for event in result.trajectory]
    # The seed graph reproduces the fixed loop it replaced: without the stage events the sequence is the old one.
    assert [kind for kind in kinds if not kind.startswith("stage/")] == [
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
    assert kinds == [
        "session",
        "turn/start",
        "stage/enter",
        "step/start",
        "request/header",
        "assistant/message",
        "stage/exit",
        "stage/enter",
        "tool/call",
        "tool/result",
        "stage/exit",
        "stage/enter",
        "step/end",
        "step/start",
        "assistant/message",
        "stage/exit",
        "stage/enter",
        "tool/call",
        "tool/result",
        "stage/exit",
        "stage/enter",
        "step/end",
        "step/start",
        "assistant/message",
        "stage/exit",
        "stage/enter",
        "step/end",
        "turn/end",
    ]
    assert [event["seq"] for event in result.trajectory] == list(range(len(kinds)))
    header = result.trajectory[0]["data"]
    assert header["version"] == 1 and header["tools"] == ["execute", "read_file", "run_bash", "write_file"]
    assert header["hooks"] == {"loop_guard": "post_execute"} and header["graph"] == "main"
    # The local executor sets no enforcer, and the log says so on the header and on every result.
    assert header["enforcement"] == "none"
    request = _events(result.trajectory, "request/header")[0]["data"]
    assert request["system"] == "Be brief."
    assert sorted(tool["function"]["name"] for tool in request["tools"]) == [
        "execute",
        "read_file",
        "run_bash",
        "write_file",
    ]
    written, read = (event["data"] for event in _events(result.trajectory, "tool/result"))
    assert written["name"] == "write_file" and written["content"] == "wrote 5 characters to notes.txt"
    assert written["is_error"] is False and written["arguments"] == {"path": "notes.txt", "content": "hello"}
    assert read["name"] == "read_file" and read["content"] == "hello"
    assert written["enforcement"] == read["enforcement"] == {"mode": "none", "denied": []}
    assert _events(result.trajectory, "assistant/message")[-1]["data"]["content"] == "The file says: hello"
    assert result.trajectory[-1]["data"]["reason"] == {"kind": "completed"}
    # The first request the fake model saw is the one the log says it saw, with the reply cap on it.
    first = fake_model.requests[0]
    assert first["messages"][0] == {"role": "system", "content": "Be brief."}
    assert first["max_tokens"] == MAX_COMPLETION_TOKENS
    assert sorted(tool["function"]["name"] for tool in first["tools"]) == [
        "execute",
        "read_file",
        "run_bash",
        "write_file",
    ]


def test_hooks_decide_at_the_first_three_events(tmp_path: Path) -> None:
    model = _FlakyModel()
    try:
        result = _episode(tmp_path, model, [*_seed_nodes(SEED_TOOLS), *EVENT_HOOKS])
    finally:
        model.shutdown()
        model.server_close()
    kinds = [event["type"] for event in result.trajectory]
    assert [kind for kind in kinds if not kind.startswith("stage/")] == [
        "session",
        "turn/start",
        "step/start",
        "request/header",
        "request/error",
        "hook/decision",
        "assistant/message",
        "tool/call",
        "hook/decision",
        "tool/result",
        "user/message",
        "step/end",
        "step/start",
        "assistant/message",
        "tool/call",
        "tool/result",
        "step/end",
        "hook/decision",
        "turn/end",
    ]
    failed = _events(result.trajectory, "request/error")[0]["data"]
    assert failed["attempt"] == 1 and failed["error"]["code"] == "MODEL_ERROR" and failed["error"]["status"] == 500
    retry, block, reject = (event["data"] for event in _events(result.trajectory, "hook/decision"))
    assert retry == {
        "event": "request_error",
        "step": 1,
        "hook": "retry",
        "owned": True,
        "decision": {"kind": "retry", "delay_ms": 0},
    }
    assert block["hook"] == "veto" and block["owned"] is False and block["decision"]["kind"] == "block"
    assert reject == {
        "event": "pre_step",
        "step": 3,
        "hook": "stop",
        "owned": True,
        "decision": {"kind": "reject", "reason": "two steps are enough"},
    }
    blocked, read = (event["data"] for event in _events(result.trajectory, "tool/result"))
    assert blocked["is_error"] is True and blocked["error"] == {"code": "HOOK_BLOCKED", "message": "writes are off"}
    # post_execute rewrites what the model reads; the tool has already run, so the read still finds the file.
    assert blocked["content"] == "Error: writes are off" and read["content"] == "hello"
    note = _events(result.trajectory, "user/message")[0]["data"]
    assert note == {
        "step": 1,
        "source": {"kind": "hook", "event": "post_execute"},
        "content": "Writes are disabled here.",
    }
    assert result.trajectory[-1]["data"]["reason"] == {
        "kind": "rejected",
        "step": 3,
        "message": "two steps are enough",
    }
    # The blocked result and the note are what the model read on its next request.
    step_two = model.requests[2]["messages"]
    assert step_two[-2] == {"role": "tool", "tool_call_id": "c1", "content": "Error: writes are off"}
    assert step_two[-1] == {"role": "user", "content": "Writes are disabled here."}


def test_a_module_that_cannot_load_ends_the_episode_in_error(tmp_path: Path, fake_model) -> None:
    broken = ("native_hook", {"name": "broken", "event": "pre_step", "code": "raise RuntimeError('no')\n"})
    result = _episode(tmp_path, fake_model, [*_seed_nodes(SEED_TOOLS), broken])
    assert result.exit_code == 1 and fake_model.requests == []
    assert [event["type"] for event in result.trajectory] == ["session", "turn/start", "turn/end"]
    reason = result.trajectory[-1]["data"]["reason"]
    assert reason["kind"] == "error" and reason["error"]["code"] == "LOAD_ERROR"
    assert reason["error"]["message"] == "broken.py failed to import: RuntimeError: no"
    assert result.trajectory[0]["data"]["enforcement"] == "none"  # the mode is chosen before any module runs


def test_seed_loop_guard_reminds_the_model_at_the_third_repeat(tmp_path: Path) -> None:
    model = _RepeatingModel()
    try:
        result = _episode(tmp_path, model, _seed_nodes())
    finally:
        model.shutdown()
        model.server_close()
    decisions = _events(result.trajectory, "hook/decision")
    assert [event["data"]["hook"] for event in decisions] == ["loop_guard"] and decisions[0]["data"]["step"] == 3
    reminder = _events(result.trajectory, "user/message")[0]["data"]["content"]
    assert reminder.startswith("You have called read_file with the same arguments 3 times in a row.")
    assert model.requests[3]["messages"][-1] == {"role": "user", "content": reminder}
    assert result.trajectory[-1]["data"]["reason"] == {"kind": "completed"}


class _DumpModel(_FakeModel):
    """Asks for one long tool result, then answers."""

    def script(self, body: dict) -> dict:
        if not [m for m in body["messages"] if m.get("role") == "tool"]:
            return _reply(tool_calls=[_call("dump", {}, "c1")])
        return _reply(content="READY")


def test_a_long_tool_result_is_spilled_under_the_workspace(tmp_path: Path) -> None:
    dump = (
        "native_tool",
        {
            "name": "dump",
            "description": "Return a long text.",
            "code": "def run(args, workdir):\n    return 'y' * 25000\n",
        },
    )
    model = _DumpModel()
    try:
        result = _episode(tmp_path, model, [*_seed_nodes(SEED_TOOLS), dump])
    finally:
        model.shutdown()
        model.server_close()
    assert result.exit_code == 0
    logged = _events(result.trajectory, "tool/result")[0]["data"]
    assert logged["meta"]["spill"] == ".reef/spill/1-c1.txt" and logged["meta"]["truncated"] is True
    assert len(logged["content"]) <= MAX_RESULT_CHARS
    assert "[5000 characters omitted; the full result is in .reef/spill/1-c1.txt]" in logged["content"]
    # The clipped content is what the model read on its next request.
    assert model.requests[1]["messages"][-1] == {"role": "tool", "tool_call_id": "c1", "content": logged["content"]}


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
        seed=SEED_NODES,
    )
    batch = TraceBatch("demo:trace:native", (TraceSample("a1", {"messages": []}, 0.0),))
    prepared = backend.prepare_step(batch, backend.initial_state(), 0)
    assert prepared.candidate is not None
    assert "native/tools/shout.py" in prepared.candidate.candidate_files
    assert "native/hooks/loop_guard.py" in prepared.candidate.candidate_files
    evaluator = DefaultCandidateEvaluationPlugin(backend, ScoreComparisonSelector())
    decision = evaluator.decide(prepared.candidate, evaluator.evaluate(prepared.candidate))
    result = backend.settle_step(prepared, decision)
    # Both trees complete the scripted task, so the verdict is a tie and nothing publishes.
    assert result.metrics["wins"] == 0 and result.metrics["losses"] == 0 and result.metrics["ties"] == 1
    assert result.metrics["selected"] is False


# -- pre_execute: allow, deny, ask, and the capabilities a hook reads --------

PRE_EXECUTE_HOOKS = (
    (
        "native_hook",
        {
            "name": "no_writes",
            "event": "pre_execute",
            "code": (
                "def listen(payload, next):\n"
                "    if 'write' in payload['capabilities']:\n"
                "        return {'kind': 'deny', 'reason': 'writes are off'}\n"
                "    return next()\n"
            ),
        },
    ),
    (
        "native_hook",
        {
            "name": "approve_reads",
            "event": "pre_execute",
            "code": (
                "def listen(payload, next):\n"
                "    if payload['name'] == 'read_file':\n"
                "        return {'kind': 'ask', 'reason': 'read ' + payload['arguments']['path'] + '?'}\n"
                "    return next()\n"
            ),
        },
    ),
)


def test_native_tool_capabilities_are_validated_rendered_and_declared_by_the_seed() -> None:
    config = dict(TOOL[1], capabilities=["read", "network"])
    NODE_KINDS["native_tool"](None, config)
    for bad in ("read", ["read", "read"], ["fly"], [1]):
        with pytest.raises(ValueError, match="capabilities"):
            NODE_KINDS["native_tool"](None, dict(TOOL[1], capabilities=bad))
    files = render_composition([("native_tool", config)], get_adapter("native"))
    assert files["native/tools/shout.py"].rstrip().endswith("CAPABILITIES = ['read', 'network']")
    assert {entry["config"]["name"]: entry["config"]["capabilities"] for entry in SEED_TOOLS} == {
        "read_file": ["read"],
        "write_file": ["write"],
        "run_bash": ["exec", "write", "network"],
        "execute": ["exec", "write", "network"],
    }


def test_invoke_runs_only_what_the_gate_allows(tmp_path: Path) -> None:
    tool = ToolModule("shout", "", TOOL[1]["parameters"], lambda args, workdir: args["text"].upper(), ["read"])
    tools = {"shout": tool}
    raw = json.dumps({"text": "hi"})
    assert _invoke(tools, "shout", raw, tmp_path)["content"] == "HI"
    assert _invoke(tools, "shout", raw, tmp_path, gate=lambda t, a: {"kind": "allow"})["content"] == "HI"
    denied = _invoke(tools, "shout", raw, tmp_path, gate=lambda t, a: {"kind": "deny", "reason": "quiet"})
    assert denied["is_error"] is True and denied["error"] == {"code": "HOOK_DENIED", "message": "quiet"}
    assert denied["content"] == "Error: quiet"
    asked = _invoke(tools, "shout", raw, tmp_path, gate=lambda t, a: {"kind": "ask", "reason": "may I shout?"})
    assert asked["error"]["code"] == "APPROVAL_REQUIRED"
    assert asked["error"]["message"].startswith("may I shout? (headless run: no one to ask")
    # The gate runs after validation and sees the validated arguments and the declared capabilities.
    seen = []

    def gate(t, a):
        seen.append((t.name, t.capabilities, a))
        return {"kind": "allow"}

    assert _invoke(tools, "shout", "{}", tmp_path, gate=gate)["error"]["code"] == "INVALID_ARGS"
    assert _invoke(tools, "shout", raw, tmp_path, gate=gate)["content"] == "HI"
    assert seen == [("shout", ("read",), {"text": "hi"})]
    # An allow may rewrite the call; the rewrite meets the schema like the model's own arguments.
    rewritten = _invoke(
        tools, "shout", raw, tmp_path, gate=lambda t, a: {"kind": "allow", "arguments": {"text": "yo"}}
    )
    assert rewritten["content"] == "YO" and rewritten["arguments"] == {"text": "yo"}
    bad = _invoke(tools, "shout", raw, tmp_path, gate=lambda t, a: {"kind": "allow", "arguments": {"text": 3}})
    assert bad["error"] == {"code": "INVALID_ARGS", "message": "rewritten by a hook: argument 'text' must be string"}


def test_pre_execute_hooks_deny_by_capability_and_ask_with_no_one_to_answer(tmp_path: Path, fake_model) -> None:
    result = _episode(tmp_path, fake_model, [*_seed_nodes(SEED_TOOLS), *PRE_EXECUTE_HOOKS])
    header = result.trajectory[0]["data"]
    assert header["capabilities"] == {
        "execute": ["exec", "write", "network"],
        "read_file": ["read"],
        "run_bash": ["exec", "write", "network"],
        "write_file": ["write"],
    }
    assert header["hooks"] == {"no_writes": "pre_execute", "approve_reads": "pre_execute"}
    # File name order is waterfall order: approve_reads passes the write down, no_writes denies it by capability;
    # approve_reads owns the read and asks, and a headless run has no one to answer.
    decisions = [event["data"] for event in _events(result.trajectory, "hook/decision")]
    assert decisions == [
        {
            "event": "pre_execute",
            "step": 1,
            "hook": "no_writes",
            "owned": True,
            "decision": {"kind": "deny", "reason": "writes are off"},
        },
        {
            "event": "pre_execute",
            "step": 2,
            "hook": "approve_reads",
            "owned": True,
            "decision": {"kind": "ask", "reason": "read notes.txt?"},
        },
    ]
    denied, asked = (event["data"] for event in _events(result.trajectory, "tool/result"))
    assert denied["error"] == {"code": "HOOK_DENIED", "message": "writes are off"}
    assert denied["content"] == "Error: writes are off" and denied["arguments"] == {
        "path": "notes.txt",
        "content": "hello",
    }
    assert asked["error"]["code"] == "APPROVAL_REQUIRED"
    assert asked["content"].startswith("Error: read notes.txt? (headless run: no one to ask")
    # What the model read: the denial as the tool result of its first call.
    assert fake_model.requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "Error: writes are off",
    }
    assert result.trajectory[-1]["data"]["reason"] == {"kind": "completed"}


# -- native_graph: the loop's control flow as data --------------------------------------------------

VERIFY_GRAPH = {
    "name": "main",
    "start": "think",
    "max_steps": 12,
    "stages": {
        "think": {"kind": "model"},
        "act": {"kind": "tools"},
        "check": {
            "kind": "verify",
            "check": "last_line_integer",
            "message": "Reply with the final answer as a plain integer alone on the last line.",
        },
        "done": {"kind": "end", "reason": "completed"},
    },
    "edges": [
        {"from": "think", "when": "tool_calls", "to": "act"},
        {"from": "think", "when": "text", "to": "check"},
        {"from": "act", "when": "done", "to": "think"},
        {"from": "check", "when": "pass", "to": "done"},
        {"from": "check", "when": "fail", "to": "think"},
    ],
}


def _graph(**changes):
    import copy

    graph = copy.deepcopy(SEED_GRAPH)
    graph.update(changes)
    return graph


class _ProseThenIntegerModel(_FakeModel):
    """The measured failure: a tool call, then the count in prose; a bare integer only after being asked again."""

    def script(self, body: dict) -> dict:
        messages = body["messages"]
        if not any(m.get("role") == "tool" for m in messages):
            return _reply(tool_calls=[_call("run_bash", {"command": "python3 -c 'print(9592)'"}, "c1")])
        if messages[-1].get("role") == "user" and "plain integer" in messages[-1]["content"]:
            return _reply(content="9592")
        return _reply(content="The sieve finds 9592 primes below 100000.")


def test_native_graph_admission_names_the_rule_a_bad_graph_breaks() -> None:
    NODE_KINDS["native_graph"](None, SEED_GRAPH)
    NODE_KINDS["native_graph"](None, VERIFY_GRAPH)
    stages = SEED_GRAPH["stages"]
    edges = SEED_GRAPH["edges"]
    bad = [
        (_graph(stages={**stages, "x": {"kind": "loop"}}), "kind must be one of"),
        (_graph(stages={**stages, "think": {"kind": "model", "code": "x"}}), "does not take code"),
        (_graph(edges=[*edges, {"from": "act", "when": "done", "to": "done"}]), "two edges for outcome 'done'"),
        (_graph(edges=edges[:2]), "stage 'act' has no edge for outcome 'done'"),
        (
            _graph(
                stages={**stages, "trap": {"kind": "message", "text": "again"}},
                edges=[
                    *edges[:1],
                    {"from": "think", "when": "text", "to": "trap"},
                    {"from": "act", "when": "done", "to": "done"},
                    {"from": "trap", "when": "done", "to": "trap"},
                ],
            ),
            "stages trap cannot reach an end stage",
        ),
        (
            _graph(
                stages={**stages, "lonely": {"kind": "message", "text": "hi"}},
                edges=[*edges, {"from": "lonely", "when": "done", "to": "done"}],
            ),
            "not reachable from 'think'",
        ),
        (
            _graph(
                edges=[
                    {"from": "think", "when": "text", "to": "done"},
                    {"from": "think", "when": "tool_calls", "to": "done"},
                    {"from": "act", "when": "done", "to": "act"},
                ]
            ),
            "not reachable",
        ),
        (_graph(max_steps=0), "'max_steps' must be an integer from 1 to 32"),
        (_graph(max_steps=33), "'max_steps' must be an integer from 1 to 32"),
        (_graph(start="nowhere"), "'start' must name a stage"),
        (
            _graph(
                stages={**stages, "v": {"kind": "verify", "check": "last_line_matches"}},
                edges=[
                    *edges,
                    {"from": "v", "when": "pass", "to": "done"},
                    {"from": "v", "when": "fail", "to": "done"},
                ],
            ),
            "takes 'pattern' exactly when",
        ),
        (
            _graph(
                stages={**stages, "m": {"kind": "message", "text": "use sk-abcdef1234567890ABCDEFGH"}},
                edges=[*edges, {"from": "m", "when": "done", "to": "done"}],
            ),
            "inline credential",
        ),
    ]
    for graph, message in bad:
        with pytest.raises(ValueError, match=message):
            NODE_KINDS["native_graph"](None, graph)
    # Two text stages that only feed each other never spend the budget.
    loop = _graph(
        stages={**stages, "a": {"kind": "message", "text": "a"}, "b": {"kind": "message", "text": "b"}},
        edges=[
            *edges[:1],
            {"from": "act", "when": "done", "to": "done"},
            {"from": "think", "when": "text", "to": "a"},
            {"from": "a", "when": "done", "to": "b"},
            {"from": "b", "when": "done", "to": "a"},
        ],
    )
    with pytest.raises(ValueError, match="stages a, b cannot reach an end stage"):
        NODE_KINDS["native_graph"](None, loop)
    cycle = _graph(
        stages={**stages, "a": {"kind": "message", "text": "a"}, "b": {"kind": "verify", "check": "nonempty"}},
        edges=[
            *edges[:1],
            edges[2],
            {"from": "think", "when": "text", "to": "a"},
            {"from": "a", "when": "done", "to": "b"},
            {"from": "b", "when": "pass", "to": "done"},
            {"from": "b", "when": "fail", "to": "a"},
        ],
    )
    with pytest.raises(ValueError, match="cycle without a model stage"):
        NODE_KINDS["native_graph"](None, cycle)


def test_native_graph_renders_sorted_json_and_checks_its_allow_list() -> None:
    descriptor = get_adapter("native")
    files = render_composition([("native_graph", VERIFY_GRAPH), TOOL], descriptor)
    assert files["native/graphs/main.json"] == json.dumps(VERIFY_GRAPH, indent=2, sort_keys=True) + "\n"
    allowed = _graph(stages={**SEED_GRAPH["stages"], "act": {"kind": "tools", "allow": ["shout"]}})
    assert "native/graphs/main.json" in render_composition([("native_graph", allowed), TOOL], descriptor)
    with pytest.raises(RenderError, match="allows tools the tree lacks: shout"):
        render_composition([("native_graph", allowed)], descriptor)
    with pytest.raises(RenderError, match="does not render native_graph"):
        render_composition([("native_graph", SEED_GRAPH)], get_adapter("pi"))


def test_a_tree_without_a_graph_runs_the_seed_graph_and_a_bad_graph_is_a_load_error(
    tmp_path: Path, fake_model
) -> None:
    result = _episode(tmp_path, fake_model, _seed_nodes(SEED_TOOLS))
    assert result.trajectory[0]["data"]["graph"] == "seed"
    assert result.trajectory[-1]["data"]["reason"] == {"kind": "completed"}
    descriptor = get_adapter("native")
    binding = ModelBinding(base_url=fake_model.base_url, model="fake", api_key="dummy")
    files = dict(render_composition([*_seed_nodes(SEED_TOOLS), *binding.compose_nodes(descriptor)], descriptor))
    files["native/graphs/main.json"] = '{"name": "main", "start": "x", "stages": {}, "edges": []}\n'
    broken = run_episode(descriptor, files, "hi", binary=_launcher(tmp_path))
    assert broken.exit_code == 1 and [event["type"] for event in broken.trajectory] == [
        "session",
        "turn/start",
        "turn/end",
    ]
    assert broken.trajectory[-1]["data"]["reason"]["error"]["code"] == "LOAD_ERROR"


def test_a_verify_stage_asks_once_more_and_the_graph_wins_the_gate(tmp_path: Path) -> None:
    """The measured sieve failure: prose after a tool call scores 0.0 on the seed; the verify graph asks once more and scores 1.0."""

    def final_line(task, result):
        answers = [e["data"].get("content") or "" for e in result.trajectory if e["type"] == "assistant/message"]
        lines = [line.strip() for line in (answers[-1] if answers else "").splitlines() if line.strip()]
        return 1.0 if lines and lines[-1] == "9592" else 0.0

    model = _ProseThenIntegerModel()
    try:
        # The interpreter alone: the verify stage fails, injects its message, and the next answer passes.
        result = _episode(tmp_path, model, [*_seed_nodes(SEED_TOOLS), ("native_graph", VERIFY_GRAPH)])
        exits = [e["data"] for e in result.trajectory if e["type"] == "stage/exit" and e["data"]["stage"] == "check"]
        assert [(e["outcome"], e["to"], e["last_line"]) for e in exits] == [
            ("fail", "think", "The sieve finds 9592 primes below 100000."),
            ("pass", "done", "9592"),
        ]
        note = next(e["data"] for e in result.trajectory if e["type"] == "user/message")
        assert note["source"] == {"kind": "stage", "stage": "check"} and note["content"].startswith("Reply with")
        assert final_line("[sieve]", result) == 1.0 and result.trajectory[-1]["data"]["reason"] == {
            "kind": "completed"
        }

        binding = ModelBinding(base_url=model.base_url, model="fake", api_key="dummy")
        backend = CordisBackend(
            descriptor=get_adapter("native"),
            propose=resolve_proposer(
                lambda nodes, samples, models: Mutation(
                    "update", "main", {"name": "native_graph", "config": VERIFY_GRAPH}
                )
            ),
            score_episode=resolve_episode_scorer(final_line),
            tasks=(
                "[sieve] how many primes are below 100000? Reply with the count as a plain integer alone on the last line.",
            ),
            models=binding,
            binary=_launcher(tmp_path),
            seed=SEED_NODES,
        )
        batch = TraceBatch("demo:trace:graph", (TraceSample("a1", {"messages": []}, 0.0),))
        prepared = backend.prepare_step(batch, backend.initial_state(), 0)
        assert prepared.candidate is not None
        assert json.loads(prepared.candidate.current_files["native/graphs/main.json"]) == SEED_GRAPH
        assert "check" in json.loads(prepared.candidate.candidate_files["native/graphs/main.json"])["stages"]
        evaluator = DefaultCandidateEvaluationPlugin(backend, ScoreComparisonSelector())
        decision = evaluator.decide(prepared.candidate, evaluator.evaluate(prepared.candidate))
        settled = backend.settle_step(prepared, decision)
        assert (settled.metrics["wins"], settled.metrics["losses"], settled.metrics["ties"]) == (1, 0, 0)
        assert settled.metrics["selected"] is True
    finally:
        model.shutdown()
        model.server_close()


def test_random_admitted_graphs_terminate_within_the_transition_bound(tmp_path: Path) -> None:
    """Property: any graph admission accepts ends within (max_steps + 1) * 16 transitions, whatever the model does."""
    import random

    from reef.harness.native import graph as graphs
    from reef.harness.native import run_loop

    rng = random.Random(7)
    texts = ("a", "b", "c")

    def random_graph() -> dict:
        extras = rng.randint(0, 3)
        stages = {"think": {"kind": "model"}, "act": {"kind": "tools"}, "done": {"kind": "end"}}
        names = ["think", "act", "done"]
        for i in range(extras):
            kind = rng.choice(("verify", "message", "branch", "compact"))
            name = f"s{i}"
            if kind == "message":
                stages[name] = {"kind": "message", "text": rng.choice(texts)}
            elif kind == "verify":
                stages[name] = {"kind": "verify", "check": rng.choice(("nonempty", "last_line_integer"))}
            elif kind == "branch":
                stages[name] = {
                    "kind": "branch",
                    "cases": [
                        {"when": "steps_used_at_least", "value": rng.randint(0, 3), "outcome": "many"},
                        {"when": "last_text_matches", "value": "9592", "outcome": "answered"},
                    ],
                }
            else:
                stages[name] = {"kind": "compact", "fire_ratio": 0.5, "keep_ratio": 0.2}
            names.append(name)
        # Text stages only point forward (to a later text stage, think, or done), so no text cycle exists.
        text_names = [n for n in names if n.startswith("s")]

        def forward(index: int) -> str:
            later = text_names[index + 1 :]
            return rng.choice([*later, "think", "done"])

        edges = [
            {"from": "think", "when": "tool_calls", "to": rng.choice(["act", *text_names])},
            {"from": "think", "when": "text", "to": rng.choice(["done", *text_names])},
            {"from": "act", "when": "done", "to": rng.choice(["think", *text_names])},
        ]
        outcomes_of = {"verify": ("pass", "fail"), "branch": ("many", "answered", "else")}
        for index, name in enumerate(text_names):
            for outcome in outcomes_of.get(stages[name]["kind"], ("done",)):
                edges.append({"from": name, "when": outcome, "to": forward(index)})
        return {"name": "main", "start": "think", "max_steps": rng.randint(1, 4), "stages": stages, "edges": edges}

    class _Chaos(_FakeModel):
        def script(self, body: dict) -> dict:
            if rng.random() < 0.5:
                return _reply(tool_calls=[_call("read_file", {"path": "x"}, f"c{len(self.requests)}")])
            return _reply(content=rng.choice(("9592", "prose", "")))

    model = _Chaos()
    try:
        admitted = 0
        for _ in range(300):
            if admitted >= 20:
                break
            graph = random_graph()
            try:
                NODE_KINDS["native_graph"](None, graph)
            except ValueError:
                continue  # an unreachable stage, say; the rule names it and the tree loses at admission
            admitted += 1
            root = tmp_path / f"g{admitted}"
            descriptor = get_adapter("native")
            binding = ModelBinding(base_url=model.base_url, model="fake", api_key="dummy")
            files = render_composition(
                [*_seed_nodes(SEED_TOOLS), ("native_graph", graph), *binding.compose_nodes(descriptor)], descriptor
            )
            for relative, text in files.items():
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                (root / relative).write_text(text, encoding="utf-8")
            (root / "workspace").mkdir()
            code = run_loop("t", root / "native", root / "native" / "sessions", root / "workspace")
            events = [
                json.loads(line) for line in (root / "native" / "sessions" / "session.jsonl").read_text().splitlines()
            ]
            reason = events[-1]["data"]["reason"]["kind"]
            transitions = sum(1 for e in events if e["type"] == "stage/enter")
            assert code == 0 and reason in ("completed", "gave_up", "max-steps"), (graph, reason)
            assert transitions <= (graph["max_steps"] + 1) * graphs.TRANSITIONS_PER_STEP
        assert admitted == 20
    finally:
        model.shutdown()
        model.server_close()


# -- branch and compact stages, and the execute seed tool ------------------------------------------


class _ChattyModel(_FakeModel):
    """Answers in prose every time, so a graph that routes on the run's counters decides when to stop."""

    def script(self, body: dict) -> dict:
        if len(self.requests) == 3:
            return _reply(content="the count is 9592")
        return _reply(content="still working")


def _branch_graph(cases, route_edges):
    """think answers in text to route; the route's own edges are the test's."""
    return {
        "name": "main",
        "start": "think",
        "max_steps": 6,
        "stages": {
            "think": {"kind": "model"},
            "act": {"kind": "tools"},
            "route": {"kind": "branch", "cases": cases},
            "done": {"kind": "end", "reason": "completed"},
            "quit": {"kind": "end", "reason": "gave_up"},
        },
        "edges": [
            {"from": "think", "when": "tool_calls", "to": "act"},
            {"from": "think", "when": "text", "to": "route"},
            {"from": "act", "when": "done", "to": "think"},
            *route_edges,
        ],
    }


def test_branch_and_compact_admission_names_the_rule_a_bad_stage_breaks() -> None:
    cases = [
        {"when": "steps_used_at_least", "value": 2, "outcome": "enough"},
        {"when": "last_text_matches", "value": r"\d+", "outcome": "answered"},
    ]
    good = _branch_graph(
        cases,
        [
            {"from": "route", "when": "enough", "to": "quit"},
            {"from": "route", "when": "answered", "to": "done"},
            {"from": "route", "when": "else", "to": "think"},
        ],
    )
    NODE_KINDS["native_graph"](None, good)
    route = good["stages"]["route"]
    bad = [
        ({**route, "cases": []}, "'cases' must be a list of 1 to 8"),
        ({**route, "cases": [{"when": "steps_used_at_least", "value": 2}]}, "objects with when, value and outcome"),
        ({**route, "cases": [{"when": "moon_phase", "value": 2, "outcome": "x"}]}, "'when' must be one of"),
        ({**route, "cases": [{"when": "steps_used_at_least", "value": "2", "outcome": "x"}]}, "integer from 0 to 32"),
        ({**route, "cases": [{"when": "last_text_matches", "value": "(", "outcome": "x"}]}, "regular expression"),
        ({**route, "cases": [{"when": "steps_used_at_least", "value": 2, "outcome": "else"}]}, "other than else"),
        (
            {**route, "cases": [{"when": "steps_used_at_least", "value": 1, "outcome": "x"}] * 2},
            "names outcome 'x' twice",
        ),
        ({"kind": "compact", "fire_ratio": 0.5}, "'keep_ratio' must be a number"),
        ({"kind": "compact", "fire_ratio": 1.5, "keep_ratio": 0.2}, "'fire_ratio' must be a number above 0"),
        ({"kind": "compact", "fire_ratio": 0.5, "keep_ratio": 0.5}, "'keep_ratio' must be below 'fire_ratio'"),
        ({"kind": "compact", "fire_ratio": 0.5, "keep_ratio": 0.2, "model": "x"}, "does not take model"),
    ]
    for stage, rule in bad:
        with pytest.raises(ValueError, match=rule):
            NODE_KINDS["native_graph"](None, {**good, "stages": {**good["stages"], "route": stage}})
    # A branch's outcomes are its cases plus else, and every one needs its edge.
    with pytest.raises(ValueError, match="stage 'route' has no edge for outcome 'else'"):
        NODE_KINDS["native_graph"](None, {**good, "edges": good["edges"][:-1]})
    with pytest.raises(ValueError, match="names outcome 'maybe', not one of its branch stage"):
        NODE_KINDS["native_graph"](
            None, {**good, "edges": [*good["edges"], {"from": "route", "when": "maybe", "to": "done"}]}
        )
    # A branch that loops on itself has no model stage in the cycle, so nothing would end it.
    with pytest.raises(ValueError, match="cycle without a model stage"):
        NODE_KINDS["native_graph"](
            None,
            {
                **good,
                "edges": [e if e.get("when") != "else" else {**e, "to": "route"} for e in good["edges"]],
            },
        )


def test_a_branch_stage_routes_on_the_steps_used_and_the_last_text(tmp_path: Path) -> None:
    model = _ChattyModel()
    try:
        cases = [
            {"when": "last_text_matches", "value": r"\b9592\b", "outcome": "answered"},
            {"when": "steps_used_at_least", "value": 3, "outcome": "enough"},
        ]
        graph = _branch_graph(
            cases,
            [
                {"from": "route", "when": "answered", "to": "done"},
                {"from": "route", "when": "enough", "to": "quit"},
                {"from": "route", "when": "else", "to": "think"},
            ],
        )
        result = _episode(tmp_path, model, [*_seed_nodes(SEED_TOOLS), ("native_graph", graph)])
        exits = [e["data"] for e in result.trajectory if e["type"] == "stage/exit" and e["data"]["stage"] == "route"]
        assert [(e["outcome"], e["to"], e["case"]) for e in exits] == [
            ("else", "think", "else"),
            ("else", "think", "else"),
            ("answered", "done", "last_text_matches"),
        ]
        assert result.trajectory[-1]["data"]["reason"] == {"kind": "completed"} and len(model.requests) == 3
    finally:
        model.shutdown()
        model.server_close()


class _WrongToolModel(_FakeModel):
    """Calls a tool the tree lacks, so every call is a tool error, then answers."""

    def script(self, body: dict) -> dict:
        if len(self.requests) <= 3:
            return _reply(tool_calls=[_call("nope", {}, f"c{len(self.requests)}")])
        return _reply(content="giving up")


def test_a_branch_stage_counts_tool_errors(tmp_path: Path) -> None:
    model = _WrongToolModel()
    try:
        graph = _branch_graph([{"when": "tool_errors_at_least", "value": 2, "outcome": "stuck"}], [])
        graph["edges"] = [
            {"from": "think", "when": "tool_calls", "to": "act"},
            {"from": "think", "when": "text", "to": "done"},
            {"from": "act", "when": "done", "to": "route"},
            {"from": "route", "when": "stuck", "to": "quit"},
            {"from": "route", "when": "else", "to": "think"},
        ]
        result = _episode(tmp_path, model, [*_seed_nodes(SEED_TOOLS), ("native_graph", graph)])
        exits = [e["data"] for e in result.trajectory if e["type"] == "stage/exit" and e["data"]["stage"] == "route"]
        assert [(e["outcome"], e.get("value")) for e in exits] == [("else", None), ("stuck", 2)]
        assert result.trajectory[-1]["data"]["reason"] == {"kind": "gave_up"} and len(model.requests) == 2
    finally:
        model.shutdown()
        model.server_close()


class _SummarizingModel(_FakeModel):
    """Plays the seed script, and answers the compact stage's summary call with a fixed summary."""

    def script(self, body: dict) -> dict:
        if body["messages"][0]["content"].startswith("Summarize the work so far"):
            return _reply(content="wrote hello to notes.txt")
        if any(str(m.get("content")).startswith("Summary of the earlier steps") for m in body["messages"]):
            return _reply(content="Done: the file says hello.")
        return super().script(body)


def test_a_compact_stage_summarizes_the_older_span_and_logs_the_policy(tmp_path: Path) -> None:
    model = _SummarizingModel()
    try:
        graph = {
            "name": "main",
            "start": "think",
            "max_steps": 6,
            "stages": {
                "think": {"kind": "model"},
                "act": {"kind": "tools"},
                "squeeze": {"kind": "compact", "fire_ratio": 0.5, "keep_ratio": 0.2},
                "done": {"kind": "end", "reason": "completed"},
            },
            "edges": [
                {"from": "think", "when": "tool_calls", "to": "act"},
                {"from": "think", "when": "text", "to": "done"},
                {"from": "act", "when": "done", "to": "squeeze"},
                {"from": "squeeze", "when": "done", "to": "think"},
            ],
        }
        # A window this small is over the ratio after the first tool result, but the only older span would split
        # that result from its call, so nothing fires; after the second result the first pair is summarized.
        window = ("config", {"target": "models", "data": {"context_window": 120}})
        result = _episode(tmp_path, model, [*_seed_nodes(SEED_TOOLS), ("native_graph", graph), window])
        compacted = [e["data"] for e in result.trajectory if e["type"] == "context/compacted"]
        assert len(compacted) == 1 and compacted[0]["fired"] is True and compacted[0]["step"] == 2
        assert compacted[0]["summary"] == "wrote hello to notes.txt" and compacted[0]["dropped"] == 2
        assert compacted[0]["policy"] == {"fire_ratio": 0.5, "keep_ratio": 0.2}
        assert compacted[0]["tokens_after"] < compacted[0]["tokens_before"]
        exits = [e["data"] for e in result.trajectory if e["type"] == "stage/exit" and e["data"]["stage"] == "squeeze"]
        assert exits[0]["fired"] is False and exits[0]["tokens"] > 60
        assert exits[1]["fired"] is True and exits[1]["dropped"] == 2
        # What the model saw next: the system prompt, the task, the summary note, then the kept tail.
        after = next(
            r
            for r in model.requests
            if any("Summary of the earlier steps" in str(m.get("content")) for m in r["messages"])
        )
        roles = [m["role"] for m in after["messages"]]
        assert roles[:3] == ["system", "user", "user"] and after["messages"][2]["content"].startswith(
            "Summary of the earlier steps:\nwrote hello"
        )
        assert roles[3] != "tool"
        assert result.trajectory[-1]["data"]["reason"] == {"kind": "completed"}
    finally:
        model.shutdown()
        model.server_close()


def test_a_compact_stage_below_the_ratio_or_without_a_summary_drops_nothing(tmp_path: Path) -> None:
    from reef.harness.native import graph as graphs

    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    for i in range(6):
        messages.append(
            {"role": "assistant", "content": None, "tool_calls": [_call("read_file", {"path": str(i)}, f"c{i}")]}
        )
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * 40})
    head, older, tail = graphs._split(messages, keep_tokens=40)
    assert head == messages[:2] and older and head + older + tail == messages
    assert tail[0]["role"] == "assistant", "the tail opens on the call whose result it keeps"

    class _Down(_FakeModel):
        def status(self, body: dict) -> int:
            return 500 if body["messages"][0]["content"].startswith("Summarize") else 200

    model = _Down()
    try:
        graph = {
            "name": "main",
            "start": "squeeze",
            "max_steps": 6,
            "stages": {
                "think": {"kind": "model"},
                "act": {"kind": "tools"},
                "squeeze": {"kind": "compact", "fire_ratio": 0.5, "keep_ratio": 0.2},
                "done": {"kind": "end", "reason": "completed"},
            },
            "edges": [
                {"from": "squeeze", "when": "done", "to": "think"},
                {"from": "think", "when": "tool_calls", "to": "act"},
                {"from": "think", "when": "text", "to": "done"},
                {"from": "act", "when": "done", "to": "squeeze"},
            ],
        }
        window = ("config", {"target": "models", "data": {"context_window": 120}})
        result = _episode(tmp_path, model, [*_seed_nodes(SEED_TOOLS), ("native_graph", graph), window])
        exits = [e["data"] for e in result.trajectory if e["type"] == "stage/exit" and e["data"]["stage"] == "squeeze"]
        assert exits[0]["fired"] is False and "error" not in exits[0], "below the ratio at the start"
        failed = [e["data"] for e in result.trajectory if e["type"] == "context/compacted"]
        assert failed and failed[0]["fired"] is False and failed[0]["error"]["code"] == "MODEL_ERROR"
        assert all(not e.get("fired") for e in exits) and result.trajectory[-1]["data"]["reason"] == {
            "kind": "completed"
        }
        # Nothing was dropped: every request still carries the full tail of tool results.
        assert all(
            m["role"] != "user" or "Summary of" not in str(m["content"]) for r in model.requests for m in r["messages"]
        )
    finally:
        model.shutdown()
        model.server_close()


def test_the_execute_seed_tool_runs_code_that_calls_the_other_tools(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    files = render_composition(
        [*_seed_nodes(SEED_TOOLS), *ModelBinding(base_url="http://x", model="m").compose_nodes(get_adapter("native"))],
        get_adapter("native"),
    )
    for relative, text in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text)
    tools = load_tools(root / "native" / "tools")
    assert tools["execute"].capabilities == ("exec", "write", "network")
    workdir = tmp_path / "work"
    workdir.mkdir()
    code = (
        "import read_file, write_file\n"
        "write_file.run({'path': 'notes.txt', 'content': 'hello'}, WORKDIR)\n"
        "print(read_file.run({'path': 'notes.txt'}, WORKDIR).upper())\n"
    )
    result = _invoke(tools, "execute", json.dumps({"code": code}), workdir)
    assert result["is_error"] is False and result["content"] == "exit 0\nHELLO\n"
    assert (workdir / "notes.txt").read_text() == "hello"
    failed = _invoke(tools, "execute", json.dumps({"code": "raise SystemExit(3)"}), workdir)
    assert failed["content"].startswith("exit 3")
    assert _invoke(tools, "execute", json.dumps({"code": "  "}), workdir)["content"] == "refused: empty code"


def test_the_loop_runs_every_call_through_the_enforcer_the_environment_names(
    tmp_path: Path, fake_model, monkeypatch
) -> None:
    root = tmp_path / "tree"
    binding = ModelBinding(base_url=fake_model.base_url, model="fake", api_key="dummy")
    files = render_composition(
        [*_seed_nodes(SEED_TOOLS), *binding.compose_nodes(get_adapter("native"))], get_adapter("native")
    )
    for relative, text in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text)
    work = tmp_path / "work"
    work.mkdir()
    task = "put hello in notes.txt and read it back"

    def session(name: str) -> list[dict]:
        return [json.loads(line) for line in (tmp_path / name / "session.jsonl").read_text().splitlines()]

    # A mode the loop has no enforcer for is a load error, not a run with nothing enforced.
    monkeypatch.setenv("REEF_NATIVE_ENFORCE", "seccomp")
    assert run_loop(task, root / "native", tmp_path / "s1", work) == 1
    assert session("s1")[-1]["data"]["reason"]["error"] == {
        "code": "LOAD_ERROR",
        "message": "REEF_NATIVE_ENFORCE='seccomp' names no enforcer; use none or bwrap",
    }
    # A bwrap that only runs the command after "--" stands in for the jail; the loop's side is what this checks.
    fake = tmp_path / "fakebin"
    fake.mkdir()
    (fake / "bwrap").write_text(
        f"#!{sys.executable}\nimport os, sys\nargv = sys.argv[sys.argv.index('--') + 1:]\nos.execv(argv[0], argv)\n"
    )
    (fake / "bwrap").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("REEF_NATIVE_ENFORCE", "bwrap")
    assert run_loop(task, root / "native", tmp_path / "s2", work) == 0
    events = session("s2")
    assert events[0]["data"]["enforcement"] == "bwrap"
    results = [event["data"] for event in events if event["type"] == "tool/result"]
    assert [(result["name"], result["enforcement"]) for result in results] == [
        ("write_file", {"mode": "bwrap", "denied": ["exec", "network"]}),
        ("read_file", {"mode": "bwrap", "denied": ["write", "exec", "network"]}),
    ]
    assert results[1]["content"] == "hello" and (work / "notes.txt").read_text() == "hello"


class _PersistentModel(_FakeModel):
    """Calls write_file three times whatever comes back, then answers."""

    def script(self, body: dict) -> dict:
        if len(self.requests) <= 3:
            call = _call("write_file", {"path": "n.txt", "content": "x"}, f"c{len(self.requests)}")
            return _reply(tool_calls=[call])
        return _reply(content="done")


def test_a_call_the_jail_could_not_run_is_no_tool_error(tmp_path: Path, monkeypatch) -> None:
    model = _PersistentModel()
    try:
        descriptor = get_adapter("native")
        binding = ModelBinding(base_url=model.base_url, model="fake", api_key="dummy")
        graph = _branch_graph([{"when": "tool_errors_at_least", "value": 1, "outcome": "stuck"}], [])
        graph["edges"] = [
            {"from": "think", "when": "tool_calls", "to": "act"},
            {"from": "think", "when": "text", "to": "done"},
            {"from": "act", "when": "done", "to": "route"},
            {"from": "route", "when": "stuck", "to": "quit"},
            {"from": "route", "when": "else", "to": "think"},
        ]
        nodes = [*_seed_nodes(SEED_TOOLS), ("native_graph", graph), *binding.compose_nodes(descriptor)]
        root = tmp_path / "root"
        for relative, text in render_composition(nodes, descriptor).items():
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_text(text)
        work = tmp_path / "work"
        work.mkdir()
        # A bwrap the host refuses: every call ends in SANDBOX_FAILED, which is the sandbox's failure, not the tool's.
        fake = tmp_path / "fakebin"
        fake.mkdir()
        (fake / "bwrap").write_text(
            f"#!{sys.executable}\nimport sys\nprint('bwrap: No permissions to create a new namespace', file=sys.stderr)\n"
            "sys.exit(1)\n"
        )
        (fake / "bwrap").chmod(0o755)
        monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ.get('PATH', '')}")
        monkeypatch.setenv("REEF_NATIVE_ENFORCE", "bwrap")
        assert run_loop("write it", root / "native", tmp_path / "s", work) == 0
    finally:
        model.shutdown()
        model.server_close()
    events = [json.loads(line) for line in (tmp_path / "s" / "session.jsonl").read_text().splitlines()]
    results = [event["data"] for event in events if event["type"] == "tool/result"]
    assert [result["error"]["code"] for result in results] == ["SANDBOX_FAILED"] * 3
    exits = [e["data"]["outcome"] for e in events if e["type"] == "stage/exit" and e["data"]["stage"] == "route"]
    assert exits == ["else"] * 3 and events[-1]["data"]["reason"] == {"kind": "completed"}


class _ProbingModel(_FakeModel):
    """Calls probe (declares nothing), then write_file, then read_file, then answers, one tool per turn."""

    def script(self, body: dict) -> dict:
        done = len([m for m in body["messages"] if m.get("role") == "tool"])
        if done == 0:
            return _reply(tool_calls=[_call("probe", {}, "c0")])
        if done == 1:
            return _reply(tool_calls=[_call("write_file", {"path": "notes.txt", "content": "hello"}, "c1")])
        if done == 2:
            return _reply(tool_calls=[_call("read_file", {"path": "notes.txt"}, "c2")])
        return _reply(content=f"done: {body['messages'][-1]['content']}")


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap (bwrap) is not on PATH")
def test_a_sandboxed_episode_runs_each_tool_call_in_a_nested_jail(tmp_path: Path) -> None:
    model = _ProbingModel()
    try:
        descriptor = get_adapter("native")
        binding = ModelBinding(base_url=model.base_url, model="fake", api_key="dummy")
        probe = (
            "native_tool",
            {"name": "probe", "description": "probe", "parameters": {}, "code": PROBE, "capabilities": []},
        )
        files = render_composition([*_seed_nodes(SEED_TOOLS), probe, *binding.compose_nodes(descriptor)], descriptor)
        # The launcher and the checkout live outside the episode root, so the jail binds them like a base path.
        checkout = Path(__file__).resolve().parents[2]
        executor = SandboxExecutor(
            egress_hosts=(model.base_url,), base_paths=(*SandboxExecutor.base_paths, str(tmp_path), str(checkout))
        )
        result = run_episode(
            descriptor, files, "probe, then put hello in notes.txt", binary=_launcher(tmp_path), executor=executor
        )
    finally:
        model.shutdown()
        model.server_close()
    assert result.exit_code == 0, result.stderr
    assert result.trajectory[0]["data"]["enforcement"] == "bwrap"
    results = [event["data"] for event in result.trajectory if event["type"] == "tool/result"]
    assert [(r["name"], r["enforcement"]["denied"], r["is_error"]) for r in results] == [
        ("probe", ["write", "exec", "network"], False),
        ("write_file", ["exec", "network"], False),
        ("read_file", ["write", "exec", "network"], False),
    ]
    # The loop keeps the model endpoint while the probe, one jail deeper, sees only loopback, a read only
    # workspace and no shell; the file still round trips through two jails that declare write and read.
    assert json.loads(results[0]["content"]) == {"exec": "FileNotFoundError", "network": "lo only", "write": "EROFS"}
    assert results[2]["content"] == "hello"


# -- native_agent: agents as root entries, called from a subagent stage ----------------------------


CHECKER = (
    "native_agent",
    {
        "name": "checker",
        "prompt": "You are the checker. Verify the claim you are given and answer in one line.",
        "tools": ["read_file"],
        "skills": [],
        "max_steps": 2,
    },
)
WRITER = (
    "native_agent",
    {"name": "writer", "prompt": "You are the writer. Restate the verified answer as a plain integer.", "tools": []},
)


def _delegating_graph(after="answer"):
    """think hands its text to the checker; the checker's outcome routes to a second model stage or an end."""
    return {
        "name": "main",
        "start": "think",
        "max_steps": 6,
        "stages": {
            "think": {"kind": "model"},
            "act": {"kind": "tools"},
            "delegate": {"kind": "subagent", "agent": "checker"},
            "answer": {"kind": "model"},
            "done": {"kind": "end", "reason": "completed"},
            "quit": {"kind": "end", "reason": "gave_up"},
        },
        "edges": [
            {"from": "think", "when": "tool_calls", "to": "act"},
            {"from": "think", "when": "text", "to": "delegate"},
            {"from": "act", "when": "done", "to": "think"},
            {"from": "delegate", "when": "completed", "to": after},
            {"from": "delegate", "when": "gave_up", "to": "quit"},
            {"from": "delegate", "when": "budget", "to": "quit"},
            {"from": "delegate", "when": "ask", "to": "quit"},
            {"from": "answer", "when": "tool_calls", "to": "act"},
            {"from": "answer", "when": "text", "to": "done"},
        ],
    }


class _TeamModel(_FakeModel):
    """Answers by which agent is asking: the system prompt names the agent, the messages say what it saw."""

    def script(self, body: dict) -> dict:
        system = body["messages"][0]["content"]
        last = body["messages"][-1]
        if "You are the checker" in system:
            return _reply(content=f"verified: {last['content']}")
        if "You are the writer" in system:
            return _reply(content="9592")
        if last.get("role") == "user" and last["content"].startswith("verified"):
            return _reply(content="The checker agrees: 9592")
        if last.get("role") == "user" and last["content"] in ("9592",):
            return _reply(content="9592")
        return _reply(content="the count is 9592")


def test_native_agent_admission_and_render_checks_name_what_is_missing() -> None:
    NODE_KINDS["native_agent"](None, CHECKER[1])
    bad = [
        ({**CHECKER[1], "model": "x"}, "does not take model"),
        ({"name": "x"}, "'prompt'"),
        ({**CHECKER[1], "graph": "no such"}, "'graph' must name a graph"),
        ({**CHECKER[1], "tools": ["a", "a"]}, "'tools' must be a list of distinct names"),
        ({**CHECKER[1], "then": ["checker"]}, "cannot hand its text to itself"),
        ({**CHECKER[1], "then": [f"a{i}" for i in range(9)]}, "'then' takes at most 8"),
        ({**CHECKER[1], "max_steps": 0}, "'max_steps' must be an integer from 1 to 32"),
        ({**CHECKER[1], "max_tool_calls": "3"}, "'max_tool_calls' must be an integer from 1 to 256"),
    ]
    for config, rule in bad:
        with pytest.raises(ValueError, match=rule):
            NODE_KINDS["native_agent"](None, config)
    descriptor = get_adapter("native")
    files = render_composition([*_seed_nodes(SEED_TOOLS), CHECKER], descriptor)
    assert json.loads(files["native/agents/checker.json"]) == CHECKER[1]
    with pytest.raises(RenderError, match="does not render native_agent"):
        render_composition([CHECKER], get_adapter("pi"))
    with pytest.raises(RenderError, match="names tools the tree lacks: read_file"):
        render_composition([CHECKER], descriptor)
    with pytest.raises(RenderError, match="names skills the tree lacks: nope"):
        render_composition(
            [*_seed_nodes(SEED_TOOLS), ("native_agent", {**CHECKER[1], "skills": ["nope"]})], descriptor
        )
    with pytest.raises(RenderError, match="names then the tree lacks: writer"):
        render_composition(
            [*_seed_nodes(SEED_TOOLS), ("native_agent", {**CHECKER[1], "then": ["writer"]})], descriptor
        )
    with pytest.raises(RenderError, match="runs a graph the tree lacks: side"):
        render_composition([*_seed_nodes(SEED_TOOLS), ("native_agent", {**CHECKER[1], "graph": "side"})], descriptor)
    with pytest.raises(RenderError, match="calls an agent the tree lacks: checker"):
        render_composition([*_seed_nodes(SEED_TOOLS), ("native_graph", _delegating_graph())], descriptor)
    # A subagent stage of an agent's graph and its then list form the call graph; a cycle is refused.
    loop_back = ("native_agent", {**WRITER[1], "then": ["checker"]})
    with pytest.raises(RenderError, match="'checker' is called in a cycle"):
        render_composition(
            [*_seed_nodes(SEED_TOOLS), ("native_agent", {**CHECKER[1], "then": ["writer"]}), loop_back], descriptor
        )
    side = {**_delegating_graph(), "name": "side"}
    with pytest.raises(RenderError, match="'checker' is called in a cycle"):
        render_composition(
            [*_seed_nodes(SEED_TOOLS), ("native_graph", side), ("native_agent", {**CHECKER[1], "graph": "side"})],
            descriptor,
        )


def test_a_subagent_stage_runs_the_agent_in_its_own_session_and_hands_its_text_back(tmp_path: Path) -> None:
    model = _TeamModel()
    try:
        nodes = [*_seed_nodes(SEED_TOOLS), ("native_graph", _delegating_graph()), CHECKER]
        result = _episode(tmp_path, model, nodes, prompt="how many primes are below 100000?")
        headers = [e["data"] for e in result.trajectory if e["type"] == "session"]
        # The agent's file sorts before the root's, so the root's final answer is the trajectory's last text.
        assert [(h["agent"], h["turn"], h.get("parent")) for h in headers] == [
            ("checker", 2, "root"),
            ("root", 1, None),
        ]
        assert headers[0]["task"] == "the count is 9592" and headers[0]["tools"] == ["read_file"]
        assert headers[0]["max_steps"] == 2 and headers[1]["agents"] == ["checker"]
        assert result.trajectory[-1]["data"]["reason"] == {"kind": "completed"}
        answers = [e["data"]["content"] for e in result.trajectory if e["type"] == "assistant/message"]
        assert answers == ["verified: the count is 9592", "the count is 9592", "The checker agrees: 9592"]
        # What the parent read: the agent's text as a user message, attributed to it.
        handed = [e["data"] for e in result.trajectory if e["type"] == "user/message"]
        # The parent's step counter already carries the checker's step: budgets draw from the episode total.
        assert handed == [
            {
                "step": 2,
                "source": {"kind": "agent", "agent": "checker", "outcome": "completed"},
                "content": "verified: the count is 9592",
            }
        ]
        exits = [
            e["data"] for e in result.trajectory if e["type"] == "stage/exit" and e["data"]["stage"] == "delegate"
        ]
        assert exits == [
            {
                "step": 2,
                "stage": "delegate",
                "outcome": "completed",
                "to": "answer",
                "agent": "checker",
                "agents": ["checker"],
                "steps": 2,
            }
        ]
        # The checker saw its own system prompt: the rules and its prompt, no skills, one tool.
        checker_request = next(r for r in model.requests if "You are the checker" in r["messages"][0]["content"])
        assert [t["function"]["name"] for t in checker_request["tools"]] == ["read_file"]
        # Per agent work, as the verdict will carry it.
        from reef.train.cordis_backend.backend import _agent_work

        assert _agent_work(result.trajectory) == {
            "checker": {"turns": 1, "steps": 1, "tool_calls": 0, "tool_errors": 0},
            "root": {"turns": 1, "steps": 2, "tool_calls": 0, "tool_errors": 0},
        }
        session_files = sorted(p.name for p in (tmp_path).rglob("*.jsonl"))
        assert session_files == []  # the episode root is gone; the files were read into the trajectory
    finally:
        model.shutdown()
        model.server_close()


def test_then_hands_an_agents_text_down_a_pipeline_and_the_last_text_returns(tmp_path: Path) -> None:
    model = _TeamModel()
    try:
        nodes = [
            *_seed_nodes(SEED_TOOLS),
            ("native_graph", _delegating_graph()),
            ("native_agent", {**CHECKER[1], "then": ["writer"]}),
            WRITER,
        ]
        result = _episode(tmp_path, model, nodes, prompt="how many primes are below 100000?")
        headers = [
            (h["data"]["agent"], h["data"]["turn"], h["data"]["task"]) for h in _events(result.trajectory, "session")
        ]
        assert headers == [
            ("checker", 2, "the count is 9592"),
            ("writer", 3, "verified: the count is 9592"),
            ("root", 1, "how many primes are below 100000?"),
        ]
        exits = [
            e["data"] for e in result.trajectory if e["type"] == "stage/exit" and e["data"]["stage"] == "delegate"
        ]
        assert exits[0]["agents"] == ["checker", "writer"] and exits[0]["outcome"] == "completed"
        handed = [e["data"] for e in result.trajectory if e["type"] == "user/message"]
        assert handed[0]["content"] == "9592" and handed[0]["source"]["agent"] == "writer"
        assert result.trajectory[-1]["data"]["reason"] == {"kind": "completed"}
    finally:
        model.shutdown()
        model.server_close()


class _BusyChecker(_TeamModel):
    """The checker keeps reading; the root answers in text."""

    def script(self, body: dict) -> dict:
        if "You are the checker" in body["messages"][0]["content"]:
            return _reply(tool_calls=[_call("read_file", {"path": "x"}, f"c{len(self.requests)}")])
        return super().script(body)


def test_an_agent_that_spends_its_budget_ends_with_budget_and_the_parent_routes_on_it(tmp_path: Path) -> None:
    model = _BusyChecker()
    try:
        nodes = [*_seed_nodes(SEED_TOOLS), ("native_graph", _delegating_graph()), CHECKER]
        result = _episode(tmp_path, model, nodes, prompt="how many primes are below 100000?")
        checker_end = next(e["data"] for e in result.trajectory if e["type"] == "turn/end" and e["data"]["turn"] == 2)
        assert checker_end["reason"] == {"kind": "max-steps", "steps": 2}
        exits = [
            e["data"] for e in result.trajectory if e["type"] == "stage/exit" and e["data"]["stage"] == "delegate"
        ]
        # The checker's two steps came out of the root's budget of six.
        assert exits[0]["outcome"] == "budget" and exits[0]["to"] == "quit" and exits[0]["steps"] == 3
        assert result.trajectory[-1]["data"]["reason"] == {"kind": "gave_up"}
        handed = [e["data"] for e in result.trajectory if e["type"] == "user/message"]
        assert handed[0]["content"] == "checker ended with budget"
        # A tool call cap ends the turn the same way, before the call runs.
        capped = ("native_agent", {**CHECKER[1], "max_steps": 6, "max_tool_calls": 1})
        result = _episode(tmp_path, model, [*_seed_nodes(SEED_TOOLS), ("native_graph", _delegating_graph()), capped])
        checker_end = next(e["data"] for e in result.trajectory if e["type"] == "turn/end" and e["data"]["turn"] == 2)
        assert checker_end["reason"] == {"kind": "max-tool-calls", "tool_calls": 1}
        assert len([e for e in result.trajectory if e["type"] == "tool/call"]) == 1
    finally:
        model.shutdown()
        model.server_close()


def test_an_ask_inside_an_agent_ends_its_turn_and_the_parent_reads_the_reason(tmp_path: Path) -> None:
    model = _BusyChecker()
    try:
        approve = (
            "native_hook",
            {
                "name": "approve_reads",
                "event": "pre_execute",
                "code": "def listen(payload, next):\n    return {'kind': 'ask', 'reason': 'read ' + payload['arguments']['path'] + '?'}\n",
            },
        )
        nodes = [*_seed_nodes(SEED_TOOLS), ("native_graph", _delegating_graph()), CHECKER, approve]
        result = _episode(tmp_path, model, nodes, prompt="how many primes are below 100000?")
        checker_end = next(e["data"] for e in result.trajectory if e["type"] == "turn/end" and e["data"]["turn"] == 2)
        assert checker_end["reason"] == {"kind": "ask", "reason": "read x?"}
        # The call never ran: no tool/result in the checker's turn, and no APPROVAL_REQUIRED error for the model.
        assert not [e for e in result.trajectory if e["type"] == "tool/result"]
        exits = [
            e["data"] for e in result.trajectory if e["type"] == "stage/exit" and e["data"]["stage"] == "delegate"
        ]
        assert exits[0]["outcome"] == "ask" and exits[0]["to"] == "quit"
        handed = [e["data"] for e in result.trajectory if e["type"] == "user/message"]
        assert handed[0] == {
            "step": 2,
            "source": {"kind": "agent", "agent": "checker", "outcome": "ask"},
            "content": "checker asks: read x?",
        }
    finally:
        model.shutdown()
        model.server_close()
