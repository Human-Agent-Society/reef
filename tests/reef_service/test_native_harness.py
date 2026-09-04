"""The native adapter: a loop inside the tree whose tools and seams are nodes, driven through the real episode engine.

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
from reef.harness.native import _DEFAULTS, MAX_RESULT_CHARS, HookModule, ToolModule, _invoke, _waterfall, load_hooks
from reef.harness.native.seed import SEED_NODES, SEED_TOOLS
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
HOOK = ("native_hook", {"name": "echo", "seam": "pre_step", "code": "def listen(payload, next):\n    return next()\n"})
# One hook per seam: retry the first failed request, block writes with a note, stop before the third step.
SEAM_HOOKS = (
    (
        "native_hook",
        {
            "name": "retry",
            "seam": "request_error",
            "code": "def listen(payload, next):\n    return {'kind': 'retry', 'delay_ms': 0}\n",
        },
    ),
    (
        "native_hook",
        {
            "name": "veto",
            "seam": "post_execute",
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
            "seam": "pre_step",
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
        == "def listen(payload, next):\n    return next()\n\nNAME = 'echo'\nSEAM = 'pre_step'\n"
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


def test_native_hook_is_optional_per_adapter_and_bound_to_a_seam() -> None:
    with pytest.raises(RenderError, match="does not render native_hook"):
        render_composition([HOOK], get_adapter("pi"))
    NODE_KINDS["native_hook"](None, HOOK[1])
    with pytest.raises(ValueError, match="'seam' must be one of pre_step, request_error, post_execute"):
        NODE_KINDS["native_hook"](None, {**HOOK[1], "seam": "tool_router"})
    with pytest.raises(ValueError, match="'code'"):
        NODE_KINDS["native_hook"](None, {"name": "x", "seam": "pre_step"})
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
        "seam": "pre_step",
        "code": "SEAM = 'post_execute'\nNAME = 'loop_guard'\ndef listen(payload, next):\n    return next()\n",
    }
    files = render_composition([("native_hook", sneaky)], get_adapter("native"))
    for relative, text in files.items():
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(text)
    hooks = load_hooks(tmp_path / "native" / "hooks")
    assert [(hook.name, hook.seam) for hook in hooks["pre_step"]] == [("sneaky", "pre_step")]
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
    # c owned the seam, so d never ran; b's error fell through to c.
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


def test_native_loop_runs_seed_tools_and_logs_everything_the_model_saw(tmp_path: Path, fake_model) -> None:
    result = _episode(tmp_path, fake_model, [*_seed_nodes(), ("rules", {"text": "Be brief."})])
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
    assert header["hooks"] == {"loop_guard": "post_execute"}
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


def test_hooks_decide_at_the_three_seams(tmp_path: Path) -> None:
    model = _FlakyModel()
    try:
        result = _episode(tmp_path, model, [*_seed_nodes(SEED_TOOLS), *SEAM_HOOKS])
    finally:
        model.shutdown()
        model.server_close()
    kinds = [event["type"] for event in result.trajectory]
    assert kinds == [
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
        "seam": "request_error",
        "step": 1,
        "hook": "retry",
        "owned": True,
        "decision": {"kind": "retry", "delay_ms": 0},
    }
    assert block["hook"] == "veto" and block["owned"] is False and block["decision"]["kind"] == "block"
    assert reject == {
        "seam": "pre_step",
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
        "source": {"kind": "hook", "seam": "post_execute"},
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
    broken = ("native_hook", {"name": "broken", "seam": "pre_step", "code": "raise RuntimeError('no')\n"})
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
