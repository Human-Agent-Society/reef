"""The native adapter: a loop inside the tree whose tools and events are nodes, driven through the real episode engine.

A stdlib HTTP server plays the served model: it answers by conversation state
(write a file, then read it back, then answer), so the whole loop, the tool
and hook modules, and the trajectory run hermetically."""

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
from reef.harness.native import (
    _DEFAULTS,
    MAX_RESULT_CHARS,
    SPILL_TAIL_CHARS,
    HookModule,
    ToolModule,
    _invoke,
    _waterfall,
    load_hooks,
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
    assert header["version"] == 1 and header["tools"] == ["read_file", "run_bash", "write_file"]
    assert header["hooks"] == {"loop_guard": "post_execute"} and header["graph"] == "main"
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
            kind = rng.choice(("verify", "message"))
            name = f"s{i}"
            stages[name] = (
                {"kind": "message", "text": rng.choice(texts)}
                if kind == "message"
                else {"kind": "verify", "check": rng.choice(("nonempty", "last_line_integer"))}
            )
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
        for index, name in enumerate(text_names):
            for outcome in ("pass", "fail") if stages[name]["kind"] == "verify" else ("done",):
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
