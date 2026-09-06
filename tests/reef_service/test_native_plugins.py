"""The native tree as live effects: node plugins install into a host and uninstall on dispose, and the interpreter reads the host at each step."""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.runners.native import DEFAULT_SYSTEM_PROMPT, LoadError, Session, _Loop
from reef.harness.runners.native.graph import DEFAULT_CONTEXT_WINDOW, Run, narrow_allow
from reef.harness.runners.native.host import NativeHost
from reef.harness.runners.native.plugins import NATIVE_PLUGINS, LoaderOrder
from reef.harness.runners.native.seed import SEED_GRAPH, SEED_NODES
from reef.harness.tree.nodes import NODE_KINDS
from reef.harness.tree.render import render_composition
from reef.train.cordis_backend import compose
from reef.train.cordis_backend.compose import FiberState
from reef.train.cordis_backend.compose.loader import Loader


def _entry(id_: str, kind: str, **config):
    return {"id": id_, "name": kind, "config": config}


def _tool(name: str, body: str = "    return 'ok'", id_: str | None = None):
    return _entry(
        id_ or name,
        "native_tool",
        name=name,
        description=f"the {name} tool",
        parameters={"type": "object", "properties": {}},
        code=f"def run(args, workdir):\n{body}\n",
    )


def _hook(name: str, event: str = "pre_step"):
    return _entry(name, "native_hook", name=name, event=event, code="def listen(payload, next):\n    return next()\n")


def _live(tmp_path: Path) -> tuple[NativeHost, Loader]:
    ctx = compose.Context()
    host = NativeHost(mount_dir=tmp_path / "mount")
    ctx.provide("native", host)
    return host, Loader(ctx, NATIVE_PLUGINS.get)


def _states(loader: Loader) -> dict[str, FiberState | None]:
    return {str(e.options["id"]): (None if e.fiber is None else e.fiber.state) for e in loader.entries()}


def _error(loader: Loader, id_: str) -> str:
    fiber = loader.resolve(id_).fiber
    assert fiber is not None
    return str(fiber.error)


def test_the_plugin_table_names_what_the_loop_consumes_and_injects_the_host() -> None:
    assert set(NATIVE_PLUGINS) == {
        "config",
        "rules",
        "skill",
        "native_tool",
        "native_hook",
        "native_graph",
        "native_agent",
    }
    assert set(NATIVE_PLUGINS) < set(NODE_KINDS)
    assert all(plugin.inject == ("native",) and plugin.name == name for name, plugin in NATIVE_PLUGINS.items())


def test_every_kind_installs_into_the_host_and_leaves_nothing_when_the_entry_leaves(tmp_path: Path) -> None:
    host, loader = _live(tmp_path)
    loader.root.update(
        [
            *SEED_NODES,
            _entry("r1", "rules", text="Be brief."),
            _entry("s1", "skill", name="tidy", text="Keep files tidy.\n"),
            _entry("helper", "native_agent", name="helper", prompt="You check answers."),
            _entry("window", "config", target="models", data={"context_window": 4096}),
        ]
    )
    assert set(_states(loader).values()) == {FiberState.ACTIVE}
    assert list(host.tools) == ["execute", "read_file", "run_bash", "write_file"]
    assert {hook.name: event for event, hooks in host.hooks.items() for hook in hooks} == {
        "loop_guard": "post_execute"
    }
    main = host.graph("main")
    assert main.source == "main" and set(main.stages) == set(SEED_GRAPH["stages"])
    assert list(host.agents) == ["helper"] and host.agents["helper"]["prompt"] == "You check answers."
    assert host.system_prompt() == "Be brief.\n\nKeep files tidy."
    assert host.system_prompt(skills=[], prompt="Own prompt") == "Be brief.\n\nOwn prompt"
    assert host.context_window == 4096
    mount = tmp_path / "mount"
    assert sorted(p.name for p in (mount / "tools").glob("*.py")) == [
        "execute.py",
        "read_file.py",
        "run_bash.py",
        "write_file.py",
    ]
    assert [p.name for p in (mount / "hooks").glob("*.py")] == ["loop_guard.py"]
    # The modules run from the mount directory, and execute finds its siblings beside it there.
    assert host.tools["read_file"].run({"path": "nope"}, str(tmp_path)) == "no such file: nope"
    ran = host.tools["execute"].run(
        {"code": "import read_file\nprint(read_file.run({'path': 'nope'}, WORKDIR))"}, str(tmp_path)
    )
    assert "no such file: nope" in ran

    assert list((mount / "tools" / "__pycache__").glob("*.pyc"))  # execute's subprocess cached a sibling's bytecode

    loader.root.update([])
    assert list(loader.entries()) == []
    assert host.tools == {} and host.agents == {} and all(not hooks for hooks in host.hooks.values())
    assert host.graph("main").source == "seed"
    assert host.system_prompt() == DEFAULT_SYSTEM_PROMPT and host.context_window == DEFAULT_CONTEXT_WINDOW
    assert not list(mount.rglob("*"))  # no module, no bytecode, no empty directory


def test_a_host_without_a_mount_directory_refuses_a_module(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    ctx = compose.Context()
    host = NativeHost()
    ctx.provide("native", host)
    loader = Loader(ctx, NATIVE_PLUGINS.get)
    with caplog.at_level(logging.ERROR):
        loader.root.update([_tool("shout")])
    assert _states(loader) == {"shout": FiberState.FAILED}
    assert "no mount directory" in _error(loader, "shout")
    with pytest.raises(LoadError, match="no mount directory"):
        host.mount_module("native_tool", _tool("shout")["config"])


def test_rules_and_windows_follow_the_tree_order_the_render_uses(tmp_path: Path) -> None:
    host, loader = _live(tmp_path)
    r1, r2, r3 = (_entry(f"r{i}", "rules", text=text) for i, text in enumerate(("first", "second", "third"), 1))
    c1 = _entry("c1", "config", target="models", data={"context_window": 1000})
    c2 = _entry("c2", "config", target="models", data={"context_window": 2000})
    assert host.order is None
    loader.root.update([r1, r2, r3, c1, c2])
    assert isinstance(host.order, LoaderOrder)  # the plugins hand the host the loader's order
    assert host.system_prompt() == "first\n\nsecond\n\nthird" and host.context_window == 2000
    # A reorder with no config change restarts nothing; the order is still the tree's.
    loader.root.update([r2, r1, r3, c2, c1])
    assert host.system_prompt() == "second\n\nfirst\n\nthird" and host.context_window == 1000
    # A changed entry moved among unchanged siblings, and one sibling removed.
    r3_revised = _entry("r3", "rules", text="third, revised")
    loader.root.update([r3_revised, r1, c1])
    assert host.system_prompt() == "third, revised\n\nfirst" and host.context_window == 1000


def test_narrow_allow_keeps_an_agent_inside_what_its_parent_sees() -> None:
    assert narrow_allow(None, None) is None
    assert narrow_allow(("a", "b"), None) == ("a", "b")
    assert narrow_allow(None, ["b", "c"]) == ("b", "c")
    assert narrow_allow(("a", "b"), ["b", "c"]) == ("b",)
    assert narrow_allow(("a", "b"), []) == ()
    assert narrow_allow((), ["a"]) == ()


def test_an_update_reinstalls_the_entry_in_place(tmp_path: Path) -> None:
    host, loader = _live(tmp_path)
    loader.root.update(
        [
            _entry("r1", "rules", text="first"),
            _entry("r2", "rules", text="second"),
            _tool("shout", "    return 'one'"),
            _hook("gate", "pre_step"),
        ]
    )
    assert host.system_prompt() == "first\n\nsecond"
    assert host.tools["shout"].run({}, ".") == "one"
    loader.root.update(
        [
            _entry("r1", "rules", text="first, revised"),
            _entry("r2", "rules", text="second"),
            _tool("shout", "    return 'two'"),
            _hook("gate", "post_execute"),
        ]
    )
    assert set(_states(loader).values()) == {FiberState.ACTIVE}
    assert host.system_prompt() == "first, revised\n\nsecond"  # a revised rule keeps its place in the tree
    assert host.tools["shout"].run({}, ".") == "two"
    assert (tmp_path / "mount" / "tools" / "shout.py").read_text(encoding="utf-8").count("return 'two'") == 1
    assert [hook.name for hook in host.hooks["post_execute"]] == ["gate"] and host.hooks["pre_step"] == []


def test_a_failing_entry_is_absent_from_the_host_and_its_siblings_stand(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    host, loader = _live(tmp_path)
    entries = [
        _tool("good"),
        _entry(
            "broken",
            "native_tool",
            name="broken",
            description="compiles, then fails to import",
            parameters={},
            code="raise RuntimeError('boom')\n\n\ndef run(args, workdir):\n    return 1\n",
        ),
        _tool("good", id_="twin"),
        _entry("pinned", "config", target="models", data={"model": "other", "context_window": 8}),
        _entry("primary", "config", target="primary", data={"a": 1}),
        _entry("window", "config", target="models", data={"context_window": 0}),
    ]
    with caplog.at_level(logging.ERROR):
        loader.root.update(entries)
    states = _states(loader)
    assert states["good"] is FiberState.ACTIVE
    assert {states[id_] for id_ in ("broken", "twin", "pinned", "primary", "window")} == {FiberState.FAILED}
    assert "broken.py failed to import: RuntimeError: boom" in _error(loader, "broken")
    assert "one name, one module" in _error(loader, "twin")
    assert "cannot set model" in _error(loader, "pinned")
    assert "never reads" in _error(loader, "primary")
    assert "positive integer" in _error(loader, "window")
    assert list(host.tools) == ["good"] and host.tools["good"].run({}, ".") == "ok"
    assert sorted(p.name for p in (tmp_path / "mount" / "tools").glob("*.py")) == ["good.py"]
    assert host.context_window == DEFAULT_CONTEXT_WINDOW

    # A fix through update recovers the entry; nothing else moves.
    fixed = {**entries[1], "config": {**entries[1]["config"], "code": "def run(args, workdir):\n    return 2\n"}}
    loader.root.update([entries[0], fixed, *entries[2:]])
    assert _states(loader)["broken"] is FiberState.ACTIVE
    assert list(host.tools) == ["broken", "good"] and host.tools["broken"].run({}, ".") == 2


def test_kinds_the_loop_never_reads_resolve_to_no_plugin(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    host, loader = _live(tmp_path)
    with caplog.at_level(logging.ERROR):
        loader.root.update(
            [
                _entry("cmd", "agent_command", name="cmd", text="a command"),
                _entry("ext", "code_extension", name="ext", code="x = 1\n"),
                _tool("good"),
            ]
        )
    assert _states(loader) == {"cmd": None, "ext": None, "good": FiberState.ACTIVE}
    assert list(host.tools) == ["good"]


def test_the_episode_form_reads_the_same_host_from_the_rendered_files(tmp_path: Path) -> None:
    descriptor = get_adapter("native")
    nodes = [
        *((entry["name"], entry["config"]) for entry in SEED_NODES),
        ("rules", {"text": "Be brief."}),
        ("skill", {"name": "tidy", "text": "Keep files tidy."}),
        ("native_agent", {"name": "helper", "prompt": "You check answers."}),
        ("config", {"target": "models", "data": {"context_window": 4096}}),
    ]
    binding = ModelBinding(base_url="http://127.0.0.1:9", model="fake", api_key="dummy")
    for path, text in render_composition([*nodes, *binding.compose_nodes(descriptor)], descriptor).items():
        (tmp_path / "root" / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "root" / path).write_text(text, encoding="utf-8")
    files = NativeHost.from_root(tmp_path / "root" / "native")

    live, loader = _live(tmp_path)
    loader.root.update(
        [{"id": str(index), "name": kind, "config": config} for index, (kind, config) in enumerate(nodes)]
    )

    assert list(files.tools) == list(live.tools)
    assert {n: t.declaration() for n, t in files.tools.items()} == {n: t.declaration() for n, t in live.tools.items()}
    assert {e: [h.name for h in hs] for e, hs in files.hooks.items()} == {
        e: [h.name for h in hs] for e, hs in live.hooks.items()
    }
    assert files.agents == live.agents
    assert files.graph("main").source == live.graph("main").source == "main"
    assert files.graph("main").edges == live.graph("main").edges
    assert files.system_prompt() == live.system_prompt() == "Be brief.\n\nKeep files tidy."
    assert files.system_prompt(skills=["tidy"], prompt="P") == live.system_prompt(skills=["tidy"], prompt="P")
    assert files.context_window == live.context_window == 4096


class _ScriptedModel(ThreadingHTTPServer):
    """Answers each request with the next scripted assistant message and keeps every request body."""

    def __init__(self, replies: list[dict]) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.replies = list(replies)
        self.requests: list[dict] = []
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        server: _ScriptedModel = self.server  # type: ignore[assignment]
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        server.requests.append(body)
        message = server.replies.pop(0)
        payload = json.dumps({"choices": [{"message": message}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        pass


def _shout_call() -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "shout", "arguments": "{}"}}],
    }


def test_the_interpreter_reads_the_host_between_steps_and_logs_a_new_header(tmp_path: Path) -> None:
    host, loader = _live(tmp_path)
    rule = _entry("r1", "rules", text="Be brief.")
    whisper = _tool("whisper", "    return 'quiet'")
    seen_by_hook = _hook("witness", "pre_step")
    seen_by_hook["config"][
        "code"
    ] = "SEEN = []\n\ndef listen(payload, next):\n    SEEN.append(payload['messages'][0]['content'])\n    return next()\n"
    loader.root.update([rule, whisper, seen_by_hook])
    model = _ScriptedModel([_shout_call(), _shout_call(), _shout_call(), _shout_call()])
    try:
        session_dir = tmp_path / "sessions"
        session = Session(session_dir / "session.jsonl")
        loop = _Loop(session, tmp_path, session_dir)
        binding = ModelBinding(base_url=model.base_url, model="fake", api_key="dummy")
        run = Run(loop, "say it", binding, host, tmp_path)
        graph = host.graph("main")
        run.max_steps = graph.max_steps

        assert run.model(graph, {"kind": "model"}) == "tool_calls"
        assert run.tools_stage(graph, {"kind": "tools"}) == "done"
        # A mount between two steps: the tool the model asked for, and nothing else.
        loader.root.update([rule, whisper, seen_by_hook, _tool("shout", "    return 'LOUD'")])
        assert run.model(graph, {"kind": "model"}) == "tool_calls"
        assert run.tools_stage(graph, {"kind": "tools"}) == "done"
        # Then only the prompt changes.
        loader.root.update([_entry("r1", "rules", text="Be brief and loud."), whisper, seen_by_hook, _tool("shout")])
        assert run.model(graph, {"kind": "model"}) == "tool_calls"
        # Then nothing changes: no new header.
        assert run.model(graph, {"kind": "model"}) == "tool_calls"
        session.close()
    finally:
        model.shutdown()
        model.server_close()

    events = [json.loads(line) for line in (session_dir / "session.jsonl").read_text(encoding="utf-8").splitlines()]
    headers = [e["data"] for e in events if e["type"] == "request/header"]
    assert [h["system"] for h in headers] == ["Be brief.", "Be brief.", "Be brief and loud."]
    assert [[t["function"]["name"] for t in h["tools"]] for h in headers] == [
        ["whisper"],
        ["shout", "whisper"],
        ["shout", "whisper"],
    ]
    results = [e["data"] for e in events if e["type"] == "tool/result"]
    assert results[0]["error"]["code"] == "UNKNOWN_TOOL" and results[1]["content"] == "LOUD"
    first, second, third, fourth = model.requests
    assert first["messages"][0]["content"] == "Be brief."
    assert [t["function"]["name"] for t in first["tools"]] == ["whisper"]
    assert [t["function"]["name"] for t in second["tools"]] == ["shout", "whisper"]
    assert third["messages"][0]["content"] == fourth["messages"][0]["content"] == "Be brief and loud."
    # The pre_step hook saw the prompt the model was about to see, at every step.
    witness = host.hooks["pre_step"][0]
    assert witness.listen.__globals__["SEEN"] == ["Be brief.", "Be brief.", "Be brief and loud.", "Be brief and loud."]
