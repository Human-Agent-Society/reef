"""The native_loop kind end to end: admission, render, host, plugin, backend review, the context API and the serve form."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from reef_service.test_harness_recipe import MODEL, batch, run_backend_step
from reef_service.test_native_harness import CHECKER, _FakeModel
from reef_service.test_native_harness import _reply as _completion
from reef_service.test_native_harness import _seed_nodes
from reef_service.test_native_serve import (
    SEED_EDGES,
    SEED_STAGES,
    _call,
    _entry,
    _events,
    _FakeReef,
    _graph,
    _reply,
    _running,
    _session_file,
    _tool,
    _tree,
    _turn,
    _typed,
    _wait,
)

from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.episodes.run import EpisodeResult
from reef.harness.runners.native import LoadError, LoopModule, load_loop, run_loop, serve
from reef.harness.runners.native.enforce import ToolFailed
from reef.harness.runners.native.graph import LOOP_LOG_CHARS, TRANSITIONS_PER_STEP
from reef.harness.runners.native.host import NativeHost
from reef.harness.runners.native.plugins import NATIVE_PLUGINS
from reef.harness.runners.native.seed import SEED_NODES, SEED_TOOLS
from reef.harness.runners.native.selftools import SelfTools
from reef.harness.tree.nodes import (
    ALWAYS_REVIEWED_KINDS,
    FLAT_TREE_REFUSAL,
    NATIVE_GRAPH_MAX_STEPS,
    NODE_KINDS,
    validate_native_loop,
)
from reef.harness.tree.render import RenderError, render_composition, render_native_module
from reef.train.cordis_backend import CordisBackend, Mutation
from reef.train.cordis_backend.backend import _stage_path, admit_mutations, tree_files
from reef.train.cordis_backend.compose import Context, FiberState
from reef.train.cordis_backend.compose.loader import Loader
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer

# The seed graph as code: ask the model, run its tool calls while it asks for them, return on text.
LOOP_CODE = "def run_turn(ctx):\n    while ctx.model() == 'tool_calls':\n        ctx.run_tools()\n"
LOOP = ("native_loop", {"name": "main", "code": LOOP_CODE})

# A harness shaped like the native loop that records which loop file and skills the root got, so the scorer can
# prefer a candidate that carries either.
NATIVE_FAKE = """\
#!/usr/bin/env python3
import json, os
from pathlib import Path

root = Path(os.environ["REEF_NATIVE_DIR"])
sessions = Path(os.environ["REEF_NATIVE_SESSION_DIR"])
sessions.mkdir(parents=True, exist_ok=True)
loops = sorted(path.stem for path in (root / "loops").glob("*.py"))
skills = sorted(path.parent.name for path in (root / "skills").glob("*/SKILL.md"))
events = [
    {"type": "session", "seq": 0, "time": 0, "data": {"agent": "root", "loop": loops[0] if loops else None, "skills": skills}},
    {"type": "turn/end", "seq": 1, "time": 0, "data": {"turn": 1, "reason": {"kind": "completed"}}},
]
(sessions / "session.jsonl").write_text("".join(json.dumps(event) + "\\n" for event in events))
"""


def _bare_module(name: str = "loop") -> ModuleType:
    module = ModuleType(name)
    module.run_turn = lambda ctx: None  # type: ignore[attr-defined]
    return module


def _states(loader: Loader) -> dict[str, FiberState | None]:
    return {str(e.options["id"]): (None if e.fiber is None else e.fiber.state) for e in loader.entries()}


def _error(loader: Loader, id_: str) -> str:
    fiber = loader.resolve(id_).fiber
    assert fiber is not None
    return str(fiber.error)


def _write_root(tmp_path: Path, entries: list[dict]) -> Path:
    """A rendered native root with its entries list, as the episode engine lays one out."""
    descriptor = get_adapter("native")
    binding = ModelBinding(base_url="http://127.0.0.1:9", model="fake", api_key="dummy")
    nodes = [(entry["name"], entry["config"]) for entry in entries]
    files = render_composition([*nodes, *binding.compose_nodes(descriptor)], descriptor)
    for relative, text in {**files, **tree_files(descriptor, entries)}.items():
        (tmp_path / "root" / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "root" / relative).write_text(text, encoding="utf-8")
    return tmp_path / "root" / "native"


# -- admission ---------------------------------------------------------------------------------------------------


def test_native_loop_admission_reads_the_code_and_never_runs_it() -> None:
    NODE_KINDS["native_loop"](None, LOOP[1])
    assert validate_native_loop({**LOOP[1], "max_steps": 3})["max_steps"] == 3
    # Top level code that would end the process if it ran: admission is a static read of the syntax tree.
    NODE_KINDS["native_loop"](None, {**LOOP[1], "code": f"raise SystemExit(3)\n{LOOP_CODE}"})
    assert frozenset({"native_loop"}) == ALWAYS_REVIEWED_KINDS and set(NODE_KINDS) > ALWAYS_REVIEWED_KINDS


def test_native_loop_admission_refuses_what_could_not_take_a_turn() -> None:
    with pytest.raises(ValueError, match="'code' must define a top level run_turn"):
        NODE_KINDS["native_loop"](None, {**LOOP[1], "code": "def run(ctx):\n    return\n"})
    # No parameter cannot take the context; an async def hands back a coroutine; an assignment is not a def.
    for code in ("def run_turn():\n    return\n", "async def run_turn(ctx):\n    return\n", "run_turn = print\n"):
        with pytest.raises(ValueError, match="must define a top level run_turn"):
            NODE_KINDS["native_loop"](None, {**LOOP[1], "code": code})
    with pytest.raises(ValueError, match="'code' does not compile"):
        NODE_KINDS["native_loop"](None, {**LOOP[1], "code": "def run_turn(ctx)\n    return\n"})
    with pytest.raises(ValueError, match="inline credential"):
        NODE_KINDS["native_loop"](None, {**LOOP[1], "code": f"KEY = 'sk-476-LOOP-KEY-0123456789abcdef'\n{LOOP_CODE}"})
    with pytest.raises(ValueError, match="'code'"):
        NODE_KINDS["native_loop"](None, {"name": "main"})
    with pytest.raises(ValueError, match="'name'"):
        NODE_KINDS["native_loop"](None, {"code": LOOP_CODE})
    for bad in (0, NATIVE_GRAPH_MAX_STEPS + 1, True, "3", 2.0):
        with pytest.raises(ValueError, match=f"'max_steps' must be an integer from 1 to {NATIVE_GRAPH_MAX_STEPS}"):
            NODE_KINDS["native_loop"](None, {**LOOP[1], "max_steps": bad})
    for edge in (1, NATIVE_GRAPH_MAX_STEPS):
        assert validate_native_loop({**LOOP[1], "max_steps": edge})["max_steps"] == edge
    with pytest.raises(ValueError, match="native_loop node does not take start"):
        NODE_KINDS["native_loop"](None, {**LOOP[1], "start": "think"})


def test_run_turn_is_the_binding_the_module_body_leaves() -> None:
    taking = "def run_turn(ctx):\n    return\n"
    bare = "def run_turn():\n    return\n"
    # The last statement that binds the name at module scope decides: a def that takes the context after one that
    # does not, or after a rebinding inside an if.
    assert validate_native_loop({**LOOP[1], "code": bare + taking})["code"] == bare + taking
    nested = "if True:\n    run_turn = None\n"
    assert validate_native_loop({**LOOP[1], "code": taking + nested + taking})["code"] == taking + nested + taking
    for rebinding in (
        bare,
        "run_turn = None\n",
        "run_turn: object = print\n",
        "x, run_turn = 1, print\n",
        "del run_turn\n",
        "run_turn += 1\n",
        "async def run_turn(ctx):\n    return\n",
        "class run_turn:\n    pass\n",
        "from os import getcwd as run_turn\n",
        # Inside a compound statement at module scope: the read follows it, and no binding there is a top level def.
        nested,
        "if True:\n    def run_turn(ctx):\n        return\n",
        "for run_turn in (print,):\n    pass\n",
        "with open(__file__) as run_turn:\n    pass\n",
        "while False:\n    pass\nelse:\n    run_turn = None\n",
        "try:\n    pass\nexcept Exception as run_turn:\n    pass\n",
        "try:\n    pass\nfinally:\n    del run_turn\n",
        "match 1:\n    case run_turn:\n        pass\n",
        # A walrus binds in the scope that holds it, a comprehension's and a def's own header included.
        "(run_turn := None)\n",
        "[(run_turn := f) for f in ()]\n",
        "def other(a=(run_turn := None)):\n    return\n",
        *(["type run_turn = int\n"] if sys.version_info >= (3, 12) else []),
    ):
        with pytest.raises(ValueError, match="must define a top level run_turn"):
            validate_native_loop({**LOOP[1], "code": taking + rebinding})
    # A bare annotation, an attribute write and a copy under another name leave the def standing, and so does a
    # binding of another scope: a lambda's parameter, a comprehension's target, a local of another def or a class.
    standing = taking + "run_turn: object\nrun_turn.__doc__ = 'd'\nother = run_turn\n"
    standing += "f = lambda run_turn: run_turn\n[run_turn for run_turn in ()]\n"
    standing += "def g():\n    run_turn = None\n\nclass C:\n    run_turn = None\n"
    assert validate_native_loop({**LOOP[1], "code": standing})["code"] == standing


def test_a_coding_cookie_reads_the_same_at_admission_and_at_boot(tmp_path: Path) -> None:
    # The render writes UTF-8 and the loop compiles the bytes, so a latin-1 cookie reads the two bytes of an e acute
    # as two characters; admission compiles the same bytes, so it admits the module the loop will run.
    code = "# -*- coding: latin-1 -*-\nS = 'caf\u00e9'\n" + LOOP_CODE
    assert validate_native_loop({**LOOP[1], "code": code})["code"] == code
    host = NativeHost(mount_dir=tmp_path / "mount")
    host.mount_module("native_loop", {**LOOP[1], "code": code})
    mounted = host.loop
    assert mounted is not None and mounted.module.S == "caf\u00c3\u00a9"
    root = _write_root(
        tmp_path, [*SEED_NODES, {"id": "loop", "name": "native_loop", "config": {**LOOP[1], "code": code}}]
    )
    (root / "tree.json").unlink()
    booted = NativeHost.from_root(root).loop
    assert booted is not None and booted.module.S == "caf\u00c3\u00a9"
    with pytest.raises(ValueError, match="'code' does not compile: unknown encoding: no-such-codec"):
        validate_native_loop({**LOOP[1], "code": "# -*- coding: no-such-codec -*-\n" + LOOP_CODE})
    # A UTF-8 BOM is read off the same bytes: the compile and the parse at admission accept it as the import does,
    # for a loop as for a tool, so the module mounts and boots from the file the render writes.
    bom = "\ufeff" + LOOP_CODE
    assert validate_native_loop({**LOOP[1], "code": bom})["code"] == bom
    NODE_KINDS["native_tool"](
        None, {"name": "t", "description": "d", "code": "\ufeffdef run(args, workdir):\n    return 1\n"}
    )
    host.dispose()
    host.mount_module("native_loop", {**LOOP[1], "code": bom})
    mounted = host.loop
    assert mounted is not None and callable(mounted.module.run_turn)
    root = _write_root(
        tmp_path / "bom", [*SEED_NODES, {"id": "loop", "name": "native_loop", "config": {**LOOP[1], "code": bom}}]
    )
    (root / "tree.json").unlink()
    assert (root / "loops" / "main.py").read_bytes().startswith(b"\xef\xbb\xbf")
    booted = NativeHost.from_root(root).loop
    assert booted is not None and booted.name == "main" and callable(booted.module.run_turn)


# -- render ------------------------------------------------------------------------------------------------------


def test_native_loop_renders_one_module_per_tree_at_the_descriptor_path() -> None:
    descriptor = get_adapter("native")
    assert descriptor.node_paths["native_loop"] == "native/loops/{name}.py"
    module = render_composition([LOOP], descriptor)["native/loops/main.py"]
    # The code byte for byte, then the node's name and budget bound as constants the file form reads back.
    assert module == LOOP_CODE.rstrip() + "\n\nNAME = 'main'\nMAX_STEPS = 12\n"
    budgeted = render_composition([("native_loop", {**LOOP[1], "max_steps": 3})], descriptor)
    assert budgeted["native/loops/main.py"].endswith("MAX_STEPS = 3\n")
    with pytest.raises(RenderError, match="does not render native_loop nodes"):
        render_composition([LOOP], get_adapter("pi"))
    other = ("native_loop", {**LOOP[1], "name": "other"})
    with pytest.raises(RenderError, match="one loop per tree: native_loop nodes 'main' and 'other'"):
        render_composition([LOOP, other], descriptor)
    with pytest.raises(RenderError, match="one loop per tree"):
        render_composition([LOOP, LOOP], descriptor)


# -- host --------------------------------------------------------------------------------------------------------


def test_the_host_holds_one_loop_and_every_add_returns_its_inverse(tmp_path: Path) -> None:
    host = NativeHost(mount_dir=tmp_path / "mount")
    assert host.loop is None
    first = LoopModule("main", None, 12, _bare_module())
    remove = host.add_loop(first)
    assert host.loop is first
    with pytest.raises(LoadError, match="one loop per tree: 'main' is already installed, so 'other' cannot be"):
        host.add_loop(LoopModule("other", None, 12, _bare_module()))
    remove()
    assert host.loop is None
    remove()  # a second call finds nothing of its own to take out
    host.add_loop(first)
    host.dispose()
    assert host.loop is None


def test_mount_module_writes_imports_and_registers_the_loop_and_the_remover_leaves_nothing(tmp_path: Path) -> None:
    host = NativeHost(mount_dir=tmp_path / "mount")
    uninstall = host.mount_module("native_loop", {**LOOP[1], "max_steps": 5})
    path = tmp_path / "mount" / "loops" / "main.py"
    assert path.read_text(encoding="utf-8") == render_native_module("native_loop", {**LOOP[1], "max_steps": 5})
    loop = host.loop
    assert loop is not None and (loop.name, loop.path, loop.max_steps) == ("main", path, 5)
    assert callable(loop.module.run_turn) and (loop.module.NAME, loop.module.MAX_STEPS) == ("main", 5)
    with pytest.raises(LoadError, match="native_loop 'main' is already installed; one name, one module"):
        host.mount_module("native_loop", LOOP[1])
    with pytest.raises(LoadError, match="one loop per tree"):
        host.mount_module("native_loop", {**LOOP[1], "name": "other"})
    assert not (tmp_path / "mount" / "loops" / "other.py").exists()  # a failed mount leaves nothing behind
    uninstall()
    assert host.loop is None and not (tmp_path / "mount" / "loops").exists()
    # The loader's own guard, past admission: a module without a callable run_turn is refused at import.
    with pytest.raises(LoadError, match=r"loop broken\.py defines no run_turn\(ctx\)"):
        host.mount_module("native_loop", {"name": "broken", "code": "run_turn = None\n"})
    # Nothing behind: the file went, and the directory made for it went with it, for a tool as for a loop.
    assert host.loop is None and not (tmp_path / "mount" / "loops").exists()
    with pytest.raises(LoadError, match=r"t\.py failed to import: RuntimeError: x"):
        host.mount_module("native_tool", {"name": "t", "description": "d", "code": "raise RuntimeError('x')\n"})
    assert not (tmp_path / "mount" / "tools").exists()
    with pytest.raises(LoadError, match="no mount directory"):
        NativeHost().mount_module("native_loop", LOOP[1])


def test_the_loop_plugin_admits_then_mounts_and_a_second_loop_fails(tmp_path: Path, caplog) -> None:
    ctx = Context()
    host = NativeHost(mount_dir=tmp_path / "mount")
    ctx.provide("native", host)
    loader = Loader(ctx, NATIVE_PLUGINS.get)
    with caplog.at_level(logging.ERROR):
        loader.root.update(
            [
                {"id": "loop", "name": "native_loop", "config": {**LOOP[1], "max_steps": 4}},
                {"id": "twin", "name": "native_loop", "config": {**LOOP[1], "name": "other"}},
                {"id": "bad", "name": "native_loop", "config": {"name": "bad", "code": "x = 1\n"}},
            ]
        )
    assert _states(loader) == {"loop": FiberState.ACTIVE, "twin": FiberState.FAILED, "bad": FiberState.FAILED}
    assert "one loop per tree" in _error(loader, "twin")
    assert "must define a top level run_turn" in _error(loader, "bad")
    loop = host.loop
    assert loop is not None and (loop.name, loop.max_steps) == ("main", 4)
    assert [path.name for path in (tmp_path / "mount" / "loops").glob("*.py")] == ["main.py"]
    loader.root.update([])
    assert host.loop is None and not list((tmp_path / "mount").rglob("*"))


def test_from_root_boots_the_loop_from_the_entries_list_and_from_the_files(tmp_path: Path) -> None:
    entries = [*SEED_NODES, {"id": "loop", "name": "native_loop", "config": {**LOOP[1], "max_steps": 4}}]
    root = _write_root(tmp_path, entries)
    host = NativeHost.from_root(root, tmp_path / "mounts")
    loop = host.loop
    assert loop is not None and (loop.name, loop.max_steps) == ("main", 4)
    assert loop.path == tmp_path / "mounts" / "loops" / "main.py" and host.graph("main").source == "main"
    host.dispose()
    assert host.loop is None and not (tmp_path / "mounts").exists()

    (root / "tree.json").unlink()
    from_files = NativeHost.from_root(root)
    loop = from_files.loop
    assert loop is not None and (loop.name, loop.max_steps, loop.path) == ("main", 4, root / "loops" / "main.py")
    assert callable(loop.module.run_turn)

    (root / "loops" / "other.py").write_text(render_native_module("native_loop", {**LOOP[1], "name": "other"}))
    with pytest.raises(LoadError, match=r"one loop per tree: loops/ holds main\.py, other\.py"):
        NativeHost.from_root(root)
    (root / "loops" / "other.py").unlink()
    # A hand edited file cannot run unchecked: the constants and the code meet admission again.
    (root / "loops" / "main.py").write_text(f"{LOOP_CODE}\nNAME = 'main'\nMAX_STEPS = 99\n", encoding="utf-8")
    with pytest.raises(LoadError, match=r"loop main\.py cannot run: .*'max_steps' must be an integer"):
        NativeHost.from_root(root)
    (root / "loops" / "main.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(LoadError, match=r"loop main\.py cannot run: .*must define a top level run_turn"):
        NativeHost.from_root(root)
    (root / "loops" / "main.py").unlink()
    assert NativeHost.from_root(root).loop is None and load_loop(tmp_path / "nowhere") is None


def test_the_file_form_admits_the_text_before_the_import_runs_the_module(tmp_path: Path) -> None:
    loops = tmp_path / "loops"
    loops.mkdir()
    marker = tmp_path / "ran"
    ran = f"open({str(marker)!r}, 'w').write('ran')\n"
    for code, reason in (
        (f"{ran}KEY = 'sk-476-LOOP-KEY-0123456789abcdef'\n{LOOP_CODE}", "inline credential"),
        (f"{ran}x = 1\n", "must define a top level run_turn"),
        (f"{ran}{LOOP_CODE}run_turn = None\n", "must define a top level run_turn"),
    ):
        (loops / "main.py").write_text(f"{code}\nNAME = 'main'\nMAX_STEPS = 12\n", encoding="utf-8")
        with pytest.raises(LoadError, match=rf"loop main\.py cannot run: .*{reason}"):
            load_loop(loops)
        assert not marker.exists()  # the top level never ran
    # The header the render wrote is read the same way: a budget past the bound is refused before the import.
    (loops / "main.py").write_text(f"{ran}{LOOP_CODE}\nNAME = 'main'\nMAX_STEPS = 99\n", encoding="utf-8")
    with pytest.raises(LoadError, match=r"loop main\.py cannot run: .*max_steps"):
        load_loop(loops)
    assert not marker.exists()
    # A header that is not the two literals the render wrote is refused before the import too: a static read cannot
    # follow what the import would bind, and the module would have run by the time admission saw it.
    for header in (
        "MAX_STEPS = -1\n",
        "MAX_STEPS = 12\nif True:\n    MAX_STEPS = 99\n",
        "MAX_STEPS = int('12')\n",
        "MAX_STEPS: int = 12\n",
        "MAX_STEPS = 12\nMAX_STEPS += 1\n",
        "MAX_STEPS = 12\ndel NAME\n",
    ):
        (loops / "main.py").write_text(f"{ran}{LOOP_CODE}\nNAME = 'main'\n{header}", encoding="utf-8")
        with pytest.raises(
            LoadError, match=r"loop main\.py cannot run: the header is not the literals the render wrote"
        ):
            load_loop(loops)
        assert not marker.exists()
    (loops / "main.py").write_text(f"{ran}{LOOP_CODE}\nNAME = 'main'\nMAX_STEPS = 12\n", encoding="utf-8")
    loaded = load_loop(loops)
    assert loaded is not None and loaded.name == "main" and marker.read_text() == "ran"
    # The loop's own locals of those names are the loop's business: the read is of the module scope.
    local = "def run_turn(ctx):\n    NAME = MAX_STEPS = ctx.max_steps\n    return\n"
    (loops / "main.py").write_text(f"{local}\nNAME = 'main'\nMAX_STEPS = 12\n", encoding="utf-8")
    loaded = load_loop(loops)
    assert loaded is not None and (loaded.name, loaded.max_steps) == ("main", 12)


def test_a_module_that_exits_at_import_is_a_load_error_in_the_file_form(tmp_path: Path) -> None:
    root = _write_root(tmp_path, [*SEED_NODES, {"id": "loop", "name": "native_loop", "config": LOOP[1]}])
    (root / "tree.json").unlink()
    (root / "loops" / "main.py").write_text(f"raise SystemExit(3)\n{LOOP_CODE}\nNAME = 'main'\n", encoding="utf-8")
    with pytest.raises(LoadError, match=r"main\.py failed to import: SystemExit: 3"):
        NativeHost.from_root(root)
    # Through the whole file form: the session opens, and the turn ends with LOAD_ERROR instead of the process.
    sessions, work = tmp_path / "sessions", tmp_path / "work"
    work.mkdir()
    exit_code = run_loop("hello", root, sessions, work)
    events = _events(sessions / "session.jsonl")
    assert exit_code == 1 and events[0]["type"] == "session" and events[0]["data"]["tree"] == "files"
    error = events[-1]["data"]["reason"]["error"]
    assert error["code"] == "LOAD_ERROR" and error["message"] == "main.py failed to import: SystemExit: 3"


def test_a_module_that_does_not_parse_is_a_load_error_in_the_file_form(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_root(tmp_path, [*SEED_NODES, {"id": "loop", "name": "native_loop", "config": LOOP[1]}])
    (root / "tree.json").unlink()
    (root / "loops" / "main.py").write_text("def run_turn(ctx)\n    return\n\nNAME = 'main'\n", encoding="utf-8")
    refusal = r"loop main\.py cannot run: code does not compile: .*\(main\.py, line 1\)"
    with pytest.raises(LoadError, match=refusal):
        NativeHost.from_root(root)
    # Through the whole file form, as a tool file with the same typo: the turn ends with LOAD_ERROR, not the process.
    sessions, work = tmp_path / "sessions", tmp_path / "work"
    work.mkdir()
    exit_code = run_loop("hello", root, sessions, work)
    events = _events(sessions / "session.jsonl")
    assert exit_code == 1 and events[0]["type"] == "session" and events[-1]["type"] == "turn/end"
    error = events[-1]["data"]["reason"]["error"]
    assert error["code"] == "LOAD_ERROR" and error["message"].startswith(
        "loop main.py cannot run: code does not compile: "
    )
    err = capsys.readouterr().err
    assert "[reef-native] loop main.py cannot run: code does not compile: " in err and "Traceback" not in err


# -- the flat tree -----------------------------------------------------------------------------------------------


def _group(**config: Any) -> dict[str, Any]:
    """A group entry's options: children in place of a config, which the compose loader mounts through the same plugins."""
    child = {"id": "hidden", "name": "native_loop", "config": {**LOOP[1], **config}}
    return {"name": "rules", "group": True, "config": [child]}


class _Trial:
    """The serve state the self tools read: the live entries, and a try_mount that records what reached it."""

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self.entries = entries
        self.tried: list[Any] = []

    def live_entries(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self.entries]

    def try_mount(self, mutations: Any, try_id: str) -> dict[str, Any]:
        self.tried.append(mutations)
        return {"try_id": try_id, "mounted": True}


def test_a_group_wrapped_loop_is_refused_wherever_entries_are_admitted(tmp_path: Path) -> None:
    descriptor = get_adapter("native")
    entries = [dict(entry) for entry in SEED_NODES]
    admitted, refusal = admit_mutations(entries, [Mutation("create", "g", _group())], descriptor)
    assert refusal == f"mutation create 'g' rejected: {FLAT_TREE_REFUSAL}" and admitted == entries
    wrapped = Mutation("update", "main", {"group": True, "config": _group()["config"]})
    assert (
        admit_mutations(entries, [wrapped], descriptor)[1] == f"mutation update 'main' rejected: {FLAT_TREE_REFUSAL}"
    )
    listed = Mutation("create", "g", {"name": "rules", "config": [{"text": "x"}]})
    assert admit_mutations(entries, [listed], descriptor)[1] == (
        "mutation create 'g' rejected: entry config must be an object, got list"
    )
    unlisted = Mutation("update", "main", {"config": None})
    assert admit_mutations(entries, [unlisted], descriptor)[1] == (
        "mutation update 'main' rejected: entry config must be an object, got NoneType"
    )
    # An update merges, so one that names the kind and leaves the config alone keeps the entry's config.
    assert admit_mutations(entries, [Mutation("update", "main", {"name": "native_graph"})], descriptor)[1] is None
    with pytest.raises(RenderError, match="rules node config must be an object, got list"):
        render_composition([("rules", _group()["config"])], descriptor)
    root = _write_root(tmp_path, entries)
    (root / "tree.json").write_text(json.dumps([*entries, {"id": "g", **_group()}]), encoding="utf-8")
    with pytest.raises(LoadError, match=rf"tree\.json entry 'g' cannot load: {FLAT_TREE_REFUSAL}"):
        NativeHost.from_root(root, tmp_path / "mounts")
    assert not (tmp_path / "mounts").exists()
    (root / "tree.json").write_text(json.dumps([*entries, {"id": "g", "name": "rules"}]), encoding="utf-8")
    with pytest.raises(
        LoadError, match=r"tree\.json entry 'g' cannot load: entry config must be an object, got NoneType"
    ):
        NativeHost.from_root(root, tmp_path / "mounts")
    trial = _Trial(entries)
    with pytest.raises(ToolFailed, match=f"create 'g': {FLAT_TREE_REFUSAL}"):
        SelfTools(trial).try_({"mutations": [{"op": "create", "id": "g", "options": _group()}]}, str(tmp_path))
    assert trial.tried == []


def test_harness_try_refuses_an_update_that_renames_a_live_loops_kind(tmp_path: Path) -> None:
    trial = _Trial([*SEED_NODES, {"id": "main-loop", "name": "native_loop", "config": LOOP[1]}])
    reason = "native_loop is reviewed code: propose it, a person promotes it"
    skill = {"name": "skill", "config": {"name": "s", "text": "# s"}}
    # The live entry's kind counts as the options' does: an update that renames the loop's kind is refused before
    # the admission that would refuse the rename, and a skill updated into a loop is refused by its new kind.
    for mutation in (
        {"op": "update", "id": "main-loop", "options": skill},
        {"op": "update", "id": "main-loop", "options": {"config": {**LOOP[1], "max_steps": 3}}},
        {"op": "update", "id": "read_file", "options": {"name": "native_loop", "config": LOOP[1]}},
        {"op": "remove", "id": "main-loop"},
    ):
        with pytest.raises(ToolFailed, match=reason):
            SelfTools(trial).try_({"mutations": [mutation]}, str(tmp_path))
    assert trial.tried == []
    result = SelfTools(trial).try_({"mutations": [{"op": "create", "id": "s", "options": skill}]}, str(tmp_path))
    assert result["mounted"] is True and len(trial.tried) == 1


# -- backend -----------------------------------------------------------------------------------------------------


def _score(task: str, result) -> float:
    header = result.trajectory[0]["data"]
    return 1.0 if header.get("loop") or "marker" in header.get("skills", []) else 0.0


def _backend(tmp_path: Path, propose, score=_score, **options) -> CordisBackend:
    binary = tmp_path / "fake-native"
    binary.write_text(NATIVE_FAKE)
    binary.chmod(0o755)
    return CordisBackend(
        descriptor=get_adapter("native"),
        propose=resolve_proposer(propose),
        score_episode=resolve_episode_scorer(score),
        tasks=("task one",),
        models=MODEL,
        binary=str(binary),
        **options,
    )


def test_a_win_that_creates_a_loop_is_pending_whatever_review_kinds_lists(tmp_path: Path) -> None:
    loop = Mutation("create", "loop", {"name": "native_loop", "config": LOOP[1]})
    skill = Mutation("create", "s1", {"name": "skill", "config": {"name": "marker", "text": "# marker"}})
    held = _backend(tmp_path, lambda n, s, m: loop)
    result = run_backend_step(held, batch(), held.initial_state())
    assert result.metrics["selected"] is True and result.pending is True
    free = _backend(tmp_path, lambda n, s, m: skill)
    result = run_backend_step(free, batch(), free.initial_state())
    assert result.metrics["selected"] is True and result.pending is False
    listed = _backend(tmp_path, lambda n, s, m: skill, review_kinds=("skill",))
    assert run_backend_step(listed, batch(), listed.initial_state()).pending is True
    # A list that names another kind holds the loop just the same, and so does the step that takes the loop out.
    unlisted = _backend(tmp_path, lambda n, s, m: loop, review_kinds=("skill",))
    with_loop = run_backend_step(unlisted, batch(), unlisted.initial_state())
    assert with_loop.metrics["selected"] is True and with_loop.pending is True
    assert with_loop.state is not None and "loop" in [entry["id"] for entry in with_loop.state["entries"]]
    without = _backend(
        tmp_path,
        lambda n, s, m: Mutation("remove", "loop"),
        score=lambda task, result: 0.0 if result.trajectory[0]["data"].get("loop") else 1.0,
    )
    removed = run_backend_step(without, batch(), with_loop.state)
    assert removed.metrics["selected"] is True and removed.pending is True
    assert removed.state is not None and "loop" not in [entry["id"] for entry in removed.state["entries"]]


def test_a_loop_turn_has_no_stages_and_a_loop_error_is_the_path_error() -> None:
    error = {"code": "LOOP_ERROR", "message": "loop main took more than 208 transitions"}
    trajectory = [
        {"type": "session", "data": {"agent": "root", "loop": "main", "graph": None}},
        {"type": "turn/start", "data": {"turn": 1}},
        {"type": "loop/enter", "data": {"name": "main"}},
        {"type": "step/start", "data": {"step": 1}},
        {"type": "loop/exit", "data": {"reason": "error"}},
        {"type": "turn/end", "data": {"turn": 1, "reason": {"kind": "error", "error": error}}},
    ]
    assert _stage_path(trajectory) == {"stages": [], "reason": "error", "error": error}
    completed = [*trajectory[:4], {"type": "turn/end", "data": {"turn": 1, "reason": {"kind": "completed"}}}]
    assert _stage_path(completed) == {"stages": [], "reason": "completed"}


# -- the context API and the serve form (the interpreter half) --------------------------------------------

PROMPT = "put hello in notes.txt and read it back"
#: The seed graph written as code: think, act while the model calls tools, done on text.
SEED_LOOP = "def run_turn(ctx):\n    while True:\n        if ctx.model() == 'text':\n            return\n        ctx.run_tools()\n"


@dataclass(frozen=True)
class _Given:
    """A loop as a test hands it in: rendered into the tree as a native_loop entry, and its module for direct calls."""

    name: str
    max_steps: int
    code: str
    module: ModuleType


def _loop(code: str, name: str = "main", max_steps: int = 12) -> _Given:
    module = ModuleType(f"reef_native_loop_{name}")
    exec(compile(code, f"<native_loop {name}>", "exec"), module.__dict__)
    return _Given(name, max_steps, code, module)


class _TextModel(_FakeModel):
    """Answers every request in text."""

    def script(self, body: dict) -> dict:
        return _completion(content="done")


class _TeamModel(_FakeModel):
    """Answers by who asks: the checker verifies the claim, the root states it and then agrees."""

    def script(self, body: dict) -> dict:
        system = body["messages"][0]["content"]
        last = body["messages"][-1]
        if "You are the checker" in system:
            return _completion(content=f"verified: {last['content']}")
        if last.get("role") == "user" and last["content"].startswith("verified"):
            return _completion(content="The checker agrees: 9592")
        return _completion(content="the count is 9592")


@pytest.fixture
def fake_model():
    server = _FakeModel()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def reef():
    server = _FakeReef()
    try:
        yield server
    finally:
        server.close()


def _root(directory: Path, model: _FakeModel, nodes) -> Path:
    """The rendered files of ``nodes`` and the model binding under ``directory``; the native root inside."""
    descriptor = get_adapter("native")
    binding = ModelBinding(base_url=model.base_url, model="fake", api_key="dummy")
    for relative, text in render_composition([*nodes, *binding.compose_nodes(descriptor)], descriptor).items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return directory / "native"


def _run(directory: Path, model: _FakeModel, nodes, loop: _Given | None, prompt: str = PROMPT):
    """One in process turn on a rendered root; the exit status and the root session's events."""
    if loop is not None:
        nodes = [*nodes, ("native_loop", {"name": loop.name, "code": loop.code, "max_steps": loop.max_steps})]
    root = _root(directory / "root", model, nodes)
    sessions, work = directory / "sessions", directory / "work"
    work.mkdir(parents=True)
    return run_loop(prompt, root, sessions, work), _events(sessions / "session.jsonl")


def _comparable(events: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    """The events with the frame that differs between a graph and a loop taken out: the stage and loop events, the header field."""
    result = []
    for event in events:
        if event["type"].startswith(("stage/", "loop/")):
            continue
        data = json.loads(json.dumps(event["data"]))
        for key in ("loop", "cwd"):
            data.pop(key, None)
        if isinstance(data.get("meta"), dict):
            data["meta"].pop("duration_ms", None)
        result.append((event["type"], data))
    return result


# -- the interpreter -----------------------------------------------------------------------------------------


def test_the_seed_loop_as_code_logs_the_same_events_as_the_seed_graph(tmp_path: Path, fake_model) -> None:
    graph_exit, from_graph = _run(tmp_path / "graph", fake_model, _seed_nodes(), None)
    loop_exit, from_loop = _run(tmp_path / "loop", fake_model, _seed_nodes(), _loop(SEED_LOOP))
    assert graph_exit == loop_exit == 0
    assert _comparable(from_loop) == _comparable(from_graph)
    assert [event["type"] for event in from_loop] == [
        "session",
        "turn/start",
        "loop/enter",
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
        "loop/exit",
        "turn/end",
    ]
    assert [event["seq"] for event in from_loop] == list(range(len(from_loop)))
    # The header names the loop and keeps naming the graph the agents fall back to; a graph run names no loop.
    assert from_loop[0]["data"]["loop"] == "main" and from_loop[0]["data"]["graph"] == "main"
    assert from_graph[0]["data"]["loop"] is None and from_graph[0]["data"]["graph"] == "main"
    assert _typed(from_loop, "loop/enter") == [{"name": "main"}]
    assert _typed(from_loop, "loop/exit") == [{"reason": "completed"}]
    assert from_loop[-1]["data"]["reason"] == {"kind": "completed"}
    assert _stage_path(from_loop) == {"stages": [], "reason": "completed"}


def test_the_context_reads_the_turn_and_say_is_a_loop_message(tmp_path: Path) -> None:
    code = (
        "def run_turn(ctx):\n"
        "    ctx.log('before', {'step': ctx.step, 'tools': list(ctx.tools), 'prompt': ctx.prompt,\n"
        "                       'messages': len(ctx.messages), 'last': ctx.last, 'max_steps': ctx.max_steps})\n"
        "    ctx.say('Answer in one line.')\n"
        "    ctx.model()\n"
        "    ctx.messages.clear()\n"
        "    ctx.last.clear()\n"
        "    ctx.log('after', {'step': ctx.step, 'text': ctx.text(), 'messages': len(ctx.messages),\n"
        "                      'last': ctx.last['role'], 'roles': [m['role'] for m in ctx.messages]})\n"
    )
    model = _TextModel()
    try:
        exit_code, events = _run(tmp_path, model, _seed_nodes(SEED_TOOLS), _loop(code, max_steps=3))
    finally:
        model.shutdown()
        model.server_close()
    assert exit_code == 0 and events[-1]["data"]["reason"] == {"kind": "completed"}
    assert _typed(events, "loop/before") == [
        {
            "step": 0,
            "tools": ["execute", "read_file", "run_bash", "write_file"],
            "prompt": PROMPT,
            "messages": 2,
            "last": {},
            "max_steps": 3,
        }
    ]
    # Copies: clearing what the context handed out changed nothing the run holds.
    assert _typed(events, "loop/after") == [
        {
            "step": 1,
            "text": "done",
            "messages": 4,
            "last": "assistant",
            "roles": ["system", "user", "user", "assistant"],
        }
    ]
    assert _typed(events, "user/message") == [
        {"step": 0, "source": {"kind": "loop", "loop": "main"}, "content": "Answer in one line."}
    ]
    assert [m["content"] for m in model.requests[0]["messages"][1:]] == [PROMPT, "Answer in one line."]


def test_log_events_are_namespaced_clipped_and_json(tmp_path: Path) -> None:
    code = (
        "import pathlib\n"
        "def run_turn(ctx):\n"
        "    ctx.log('note', {'n': 1, 'path': pathlib.Path('/x')})\n"
        "    ctx.log('big', {'blob': 'x' * 5000})\n"
        "    for event, data in (('turn/end', {}), ('exit', {}), ('enter', {}), ('', {}), ('ok', 'not an object')):\n"
        "        try:\n"
        "            ctx.log(event, data)\n"
        "        except ValueError as exc:\n"
        "            ctx.log('refused', {'event': event, 'error': str(exc)})\n"
    )
    model = _TextModel()
    try:
        exit_code, events = _run(tmp_path, model, _seed_nodes(SEED_TOOLS), _loop(code))
    finally:
        model.shutdown()
        model.server_close()
    assert exit_code == 0
    assert _typed(events, "loop/note") == [{"n": 1, "path": "/x"}]
    big = _typed(events, "loop/big")[0]
    assert big["truncated"] is True and len(big["text"]) == LOOP_LOG_CHARS and big["text"].startswith('{"blob": "xxx')
    refused = _typed(events, "loop/refused")
    assert [entry["event"] for entry in refused] == ["turn/end", "exit", "enter", "", "ok"]
    assert all("must match" in entry["error"] for entry in refused[:4]) and "must be an object" in refused[4]["error"]
    # No core event was written by the loop: the only turn/end is the frame's, after loop/exit.
    assert [event["type"] for event in events if not event["type"].startswith("loop/")] == [
        "session",
        "turn/start",
        "turn/end",
    ]
    assert [event["type"] for event in events][-2:] == ["loop/exit", "turn/end"]


def test_a_loop_that_raises_ends_the_turn_with_loop_error(tmp_path: Path) -> None:
    code = "def run_turn(ctx):\n    ctx.model()\n    raise RuntimeError('boom')\n"
    model = _TextModel()
    try:
        exit_code, events = _run(tmp_path, model, _seed_nodes(SEED_TOOLS), _loop(code))
    finally:
        model.shutdown()
        model.server_close()
    assert exit_code == 1
    assert [event["type"] for event in events][-4:] == ["assistant/message", "step/end", "loop/exit", "turn/end"]
    assert _typed(events, "loop/exit") == [{"reason": "error"}]
    assert events[-1]["data"]["reason"] == {
        "kind": "error",
        "error": {"code": "LOOP_ERROR", "message": "RuntimeError: boom"},
    }
    path = _stage_path(events)
    assert path["stages"] == [] and path["reason"] == "error" and path["error"]["code"] == "LOOP_ERROR"


def test_a_loop_that_never_returns_hits_the_transition_cap(tmp_path: Path) -> None:
    code = "def run_turn(ctx):\n    while True:\n        ctx.run_tools()\n"
    model = _TextModel()
    try:
        exit_code, events = _run(tmp_path, model, _seed_nodes(SEED_TOOLS), _loop(code, max_steps=2))
    finally:
        model.shutdown()
        model.server_close()
    limit = 3 * TRANSITIONS_PER_STEP
    assert exit_code == 1 and not model.requests
    assert events[-1]["data"]["reason"]["error"] == {
        "code": "LOOP_ERROR",
        "message": f"loop 'main' took more than {limit} transitions",
    }
    assert _typed(events, "loop/exit") == [{"reason": "error"}]


def test_max_steps_ends_the_turn_as_a_graph_does(tmp_path: Path, fake_model) -> None:
    code = "def run_turn(ctx):\n    while True:\n        ctx.model()\n        ctx.run_tools()\n"
    exit_code, events = _run(tmp_path, fake_model, _seed_nodes(SEED_TOOLS), _loop(code, max_steps=2))
    assert exit_code == 0 and len(_typed(events, "step/start")) == 2
    assert _typed(events, "loop/exit") == [{"reason": "max-steps"}]
    assert events[-1]["data"]["reason"] == {"kind": "max-steps", "steps": 2}


def test_the_turns_end_crosses_a_loops_except_exception(tmp_path: Path) -> None:
    # Bounded: a handler that could catch the end would then fail on the events, not hold the test open.
    code = (
        "def run_turn(ctx):\n"
        "    for _ in range(3):\n"
        "        try:\n"
        "            ctx.model()\n"
        "        except Exception:\n"
        "            ctx.log('swallowed', {})\n"
    )
    model = _TextModel()
    try:
        exit_code, events = _run(tmp_path, model, _seed_nodes(SEED_TOOLS), _loop(code, max_steps=1))
    finally:
        model.shutdown()
        model.server_close()
    assert exit_code == 0 and not _typed(events, "loop/swallowed")
    assert events[-1]["data"]["reason"] == {"kind": "max-steps", "steps": 1}


def test_agent_runs_an_agent_in_its_own_session_and_appends_its_text(tmp_path: Path) -> None:
    code = (
        "def run_turn(ctx):\n"
        "    ctx.model()\n"
        "    outcome, text = ctx.agent('checker')\n"
        "    ctx.log('checked', {'outcome': outcome, 'text': text, 'step': ctx.step})\n"
        "    ctx.model()\n"
        "    try:\n"
        "        ctx.agent('nobody')\n"
        "    except ValueError as exc:\n"
        "        ctx.log('refused', {'error': str(exc)})\n"
    )
    model = _TeamModel()
    try:
        exit_code, events = _run(
            tmp_path,
            model,
            [*_seed_nodes(SEED_TOOLS), CHECKER],
            _loop(code, max_steps=6),
            prompt="how many primes are below 100000?",
        )
    finally:
        model.shutdown()
        model.server_close()
    assert exit_code == 0 and events[-1]["data"]["reason"] == {"kind": "completed"}
    assert [e["content"] for e in _typed(events, "assistant/message")] == [
        "the count is 9592",
        "The checker agrees: 9592",
    ]
    # The checker's step came out of the root's budget, as a subagent stage draws it.
    assert _typed(events, "loop/checked") == [
        {"outcome": "completed", "text": "verified: the count is 9592", "step": 2}
    ]
    assert _typed(events, "user/message") == [
        {
            "step": 2,
            "source": {"kind": "agent", "agent": "checker", "outcome": "completed"},
            "content": "verified: the count is 9592",
        }
    ]
    assert _typed(events, "loop/refused") == [{"error": "agent 'nobody' is not in the tree"}]
    checker = _events(tmp_path / "sessions" / "agents" / "002-checker.jsonl")
    header = checker[0]["data"]
    assert header["agent"] == "checker" and header["parent"] == "root" and header["graph"] == "seed"
    assert header["task"] == "the count is 9592" and header["max_steps"] == 2
    assert checker[-1]["data"]["reason"] == {"kind": "completed"}


class _CheckerFails(_FakeModel):
    """Text for the root; a 500 for the checker."""

    def status(self, body: dict[str, Any]) -> int:
        return 500 if "You are the checker" in body["messages"][0]["content"] else 200

    def script(self, body: dict[str, Any]) -> dict[str, Any]:
        return _completion(content="the count is 9592")


def test_an_agents_abort_is_the_turns_end_for_the_loop(tmp_path: Path) -> None:
    # Under a graph an agent's failed request ends the run with exit 1 and no root turn/end; loop code cannot catch its way past it.
    # The end comes again from log and from end: neither writes, so the root's open step is never closed.
    code = (
        "def run_turn(ctx):\n"
        "    ctx.model()\n"
        "    try:\n"
        "        ctx.agent('checker')\n"
        "    except BaseException:\n"
        "        try:\n"
        "            ctx.log('swallowed', {})\n"
        "        except BaseException:\n"
        "            ctx.end('completed')\n"
        "    ctx.model()\n"
    )
    model = _CheckerFails()
    try:
        exit_code, events = _run(
            tmp_path,
            model,
            [*_seed_nodes(SEED_TOOLS), CHECKER],
            _loop(code, max_steps=6),
            prompt="how many primes are below 100000?",
        )
    finally:
        model.shutdown()
        model.server_close()
    assert exit_code == 1 and _ends(events) == [] and len(model.requests) == 2
    assert "loop/swallowed" not in {event["type"] for event in events}
    assert events[-1]["type"] == "assistant/message"
    trajectory: list[dict[str, Any]] = []
    for file in sorted((tmp_path / "sessions").rglob("*.jsonl")):
        trajectory.extend(json.loads(line) for line in file.read_text().splitlines() if line.strip())
    path = _stage_path(tuple(trajectory))
    assert path["errored_agent"] == "checker" and path["error"]["code"] == "MODEL_ERROR"


def test_end_ends_with_the_reason_and_refuses_an_unknown_one(tmp_path: Path) -> None:
    code = (
        "def run_turn(ctx):\n"
        "    try:\n"
        "        ctx.end('other')\n"
        "    except ValueError as exc:\n"
        "        ctx.log('refused', {'error': str(exc)})\n"
        "    ctx.end('gave_up')\n"
        "    ctx.log('unreachable', {})\n"
    )
    model = _TextModel()
    try:
        exit_code, events = _run(tmp_path, model, _seed_nodes(SEED_TOOLS), _loop(code))
    finally:
        model.shutdown()
        model.server_close()
    assert exit_code == 0 and not model.requests
    assert [event["type"] for event in events] == [
        "session",
        "turn/start",
        "loop/enter",
        "loop/refused",
        "loop/exit",
        "turn/end",
    ]
    assert "must be one of completed, gave_up" in _typed(events, "loop/refused")[0]["error"]
    assert _typed(events, "loop/exit") == [{"reason": "gave_up"}]
    assert events[-1]["data"]["reason"] == {"kind": "gave_up"}


# -- the serve form ------------------------------------------------------------------------------------------


def _loop_entry(code: str = SEED_LOOP, max_steps: int = 4) -> dict[str, Any]:
    return _entry("main-loop", "native_loop", name="main", code=code, max_steps=max_steps)


def test_a_release_with_a_loop_runs_the_next_turn_and_one_without_returns_to_the_graph(
    tmp_path: Path, reef: _FakeReef
) -> None:
    graph_only = [_tool("shout", "    return 'LOUD'"), _graph(SEED_STAGES, SEED_EDGES)]
    reef.release("r1", graph_only)
    reef.replies = [_reply("one"), _reply(tool_calls=[_call("shout", {}, "c1")]), _reply("two"), _reply("three")]
    with _running(_tree(tmp_path, reef, "r1"), poll_interval_s=0.1) as server:
        first, streamed = _turn(server, "first")
        assert first["text"] == "one" and [e["type"] for e in streamed][:3] == ["session", "turn/start", "stage/enter"]
        assert _typed(streamed, "session")[0]["loop"] is None
        log = server.sessions_dir / serve.SERVE_LOG
        reef.release("r2", [*graph_only, _loop_entry()], parent="r1")
        _wait(lambda: any(m["release_id"] == "r2" for m in _typed(_events(log), "harness/mount")))
        second, streamed = _turn(server, "second")
        assert second["exit"] == 0 and second["text"] == "two"
        assert [e["type"] for e in streamed] == [
            "session",
            "turn/start",
            "loop/enter",
            "step/start",
            "request/header",
            "assistant/message",
            "tool/call",
            "tool/result",
            "step/end",
            "step/start",
            "assistant/message",
            "step/end",
            "loop/exit",
            "turn/end",
        ]
        header = _typed(streamed, "session")[0]
        assert header["loop"] == "main" and header["graph"] == "main" and header["release_id"] == "r2"
        assert _typed(streamed, "loop/enter") == [{"name": "main"}]
        reef.release("r3", graph_only, parent="r2")
        _wait(lambda: any(m["release_id"] == "r3" for m in _typed(_events(log), "harness/mount")))
        third, streamed = _turn(server, "third", session=second["session"])
        assert third["exit"] == 0 and third["text"] == "three" and third["turn"] == 2
        assert (
            "loop/enter" not in {e["type"] for e in streamed}
            and _typed(streamed, "stage/enter")[0]["stage"] == "think"
        )
    events = _events(_session_file(server, second["session"]))
    assert [e["type"] for e in events if e["type"] in ("loop/enter", "loop/exit", "turn/end")] == [
        "loop/enter",
        "loop/exit",
        "turn/end",
        "turn/end",
    ]


def test_harness_try_refuses_a_native_loop_mutation(tmp_path: Path, reef: _FakeReef) -> None:
    reef.release("r1", [_tool("shout", "    return 'LOUD'"), _loop_entry(max_steps=8)])
    other = {
        "op": "create",
        "id": "other",
        "options": {"name": "native_loop", "config": {**_loop_entry()["config"], "name": "other"}},
    }
    update = {"op": "update", "id": "main-loop", "options": {"name": "native_loop", "config": _loop_entry()["config"]}}
    remove = {"op": "remove", "id": "main-loop"}
    whisper = {
        "op": "create",
        "id": "whisper",
        "options": {"name": "native_tool", "config": _tool("whisper")["config"]},
    }
    reef.replies = [
        _reply(tool_calls=[_call("harness_try", {"mutations": [other]}, "c1")]),
        _reply(tool_calls=[_call("harness_try", {"mutations": [whisper, update]}, "c2")]),
        _reply(tool_calls=[_call("harness_try", {"mutations": [remove]}, "c3")]),
        _reply("done"),
    ]
    with _running(_tree(tmp_path, reef, "r1"), self_tools=True) as server:
        result, streamed = _turn(server, "try a loop")
        assert result["exit"] == 0 and result["text"] == "done"
        assert [entry["id"] for entry in server.live_entries()] == ["shout", "main-loop"]
        assert sorted(server.host.tools) == ["harness_inspect", "harness_propose", "harness_try", "shout"]
    assert _typed(streamed, "loop/enter") == [{"name": "main"}]
    results = _typed(streamed, "tool/result")
    reason = "native_loop is reviewed code: propose it, a person promotes it"
    assert [r["error"]["code"] for r in results] == ["TOOL_FAILED"] * 3
    assert all(r["content"] == f"Error: {reason}" for r in results)
    # The refusal is whole: the tool beside the loop mutation did not mount either, and nothing was tried.
    assert [m["source"] for m in _typed(streamed, "harness/mount")] == []
    assert "harness/unmount" not in {event["type"] for event in streamed}


# -- the end is final, the cap, and the failure stage --------------------------------------------------------


def _text_turn(directory: Path, code: str, max_steps: int = 12) -> tuple[int, list[dict[str, Any]]]:
    """One turn of ``code`` against a model that answers in text; the exit status and the root session's events."""
    model = _TextModel()
    try:
        return _run(directory, model, _seed_nodes(SEED_TOOLS), _loop(code, max_steps=max_steps))
    finally:
        model.shutdown()
        model.server_close()


def _ends(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every ``turn/end`` reason of the session, in order."""
    return [event["data"]["reason"] for event in events if event["type"] == "turn/end"]


def test_except_base_exception_cannot_act_past_the_turns_end(tmp_path: Path) -> None:
    code = (
        "def run_turn(ctx):\n"
        "    for _ in range(3):\n"
        "        try:\n"
        "            ctx.model()\n"
        "        except BaseException:\n"
        "            pass\n"
        "    ctx.log('after', {})\n"
    )
    exit_code, events = _text_turn(tmp_path, code, max_steps=1)
    assert exit_code == 0 and _ends(events) == [{"kind": "max-steps", "steps": 1}]
    assert "loop/after" not in {event["type"] for event in events}
    assert [event["type"] for event in events][-2:] == ["loop/exit", "turn/end"]
    assert _stage_path(events) == {"stages": [], "reason": "max-steps"}


def test_a_finally_cannot_replace_an_abort_with_its_own_end(tmp_path: Path) -> None:
    code = (
        "def run_turn(ctx):\n"
        "    try:\n"
        "        while True:\n"
        "            ctx.run_tools()\n"
        "    finally:\n"
        "        ctx.log('late', {})\n"
        "        ctx.end('gave_up')\n"
    )
    exit_code, events = _text_turn(tmp_path, code, max_steps=2)
    error = {"code": "LOOP_ERROR", "message": f"loop 'main' took more than {3 * TRANSITIONS_PER_STEP} transitions"}
    assert exit_code == 1 and _ends(events) == [{"kind": "error", "error": error}]
    assert "loop/late" not in {event["type"] for event in events}
    assert _typed(events, "loop/exit") == [{"reason": "error"}]


def test_a_finally_that_only_ends_gets_the_abort_again(tmp_path: Path) -> None:
    code = "def run_turn(ctx):\n    try:\n        while True:\n            ctx.run_tools()\n    finally:\n        ctx.end('gave_up')\n"
    exit_code, events = _text_turn(tmp_path, code, max_steps=2)
    error = {"code": "LOOP_ERROR", "message": f"loop 'main' took more than {3 * TRANSITIONS_PER_STEP} transitions"}
    assert exit_code == 1 and _ends(events) == [{"kind": "error", "error": error}]
    assert _typed(events, "loop/exit") == [{"reason": "error"}]


def test_an_exception_after_the_end_keeps_the_end_and_is_named_on_stderr(tmp_path: Path, capfd) -> None:
    code = "def run_turn(ctx):\n    try:\n        while True:\n            ctx.model()\n    finally:\n        raise RuntimeError('late')\n"
    exit_code, events = _text_turn(tmp_path, code, max_steps=1)
    assert exit_code == 0 and _ends(events) == [{"kind": "max-steps", "steps": 1}]
    assert "[reef-native] loop 'main' raised after the end: RuntimeError: late" in capfd.readouterr().err


def test_a_return_in_finally_keeps_the_max_steps_end(tmp_path: Path) -> None:
    code = "def run_turn(ctx):\n    try:\n        while True:\n            ctx.model()\n    finally:\n        return\n"
    exit_code, events = _text_turn(tmp_path, code, max_steps=1)
    assert exit_code == 0 and _ends(events) == [{"kind": "max-steps", "steps": 1}]
    assert _stage_path(events) == {"stages": [], "reason": "max-steps"}


def test_a_bare_except_gets_the_end_again_and_writes_nothing(tmp_path: Path) -> None:
    code = (
        "def run_turn(ctx):\n"
        "    for i in range(2):\n"
        "        try:\n"
        "            ctx.model()\n"
        "        except:\n"
        "            ctx.log('swallowed', {'i': i})\n"
    )
    exit_code, events = _text_turn(tmp_path, code, max_steps=1)
    assert exit_code == 0 and _ends(events) == [{"kind": "max-steps", "steps": 1}]
    assert not _typed(events, "loop/swallowed") and len(_typed(events, "step/start")) == 1


def test_say_and_log_count_against_the_transition_cap(tmp_path: Path) -> None:
    limit = 2 * TRANSITIONS_PER_STEP
    error = {"code": "LOOP_ERROR", "message": f"loop 'main' took more than {limit} transitions"}
    logs = "def run_turn(ctx):\n    while True:\n        ctx.log('tick', {})\n"
    exit_code, events = _text_turn(tmp_path / "log", logs, max_steps=1)
    assert exit_code == 1 and len(_typed(events, "loop/tick")) == limit
    assert _ends(events) == [{"kind": "error", "error": error}]
    talks = "def run_turn(ctx):\n    while True:\n        ctx.say('again')\n        ctx.log('tick', {})\n"
    exit_code, events = _text_turn(tmp_path / "say", talks, max_steps=1)
    assert exit_code == 1 and len(_typed(events, "user/message")) == len(_typed(events, "loop/tick")) == limit // 2
    assert _ends(events) == [{"kind": "error", "error": error}]


def test_system_exit_out_of_run_turn_is_a_loop_error(tmp_path: Path) -> None:
    code = "def run_turn(ctx):\n    ctx.model()\n    raise SystemExit(3)\n"
    exit_code, events = _text_turn(tmp_path, code)
    assert exit_code == 1
    assert _ends(events) == [{"kind": "error", "error": {"code": "LOOP_ERROR", "message": "SystemExit: 3"}}]
    assert [event["type"] for event in events][-4:] == ["assistant/message", "step/end", "loop/exit", "turn/end"]
    assert _stage_path(events)["error"]["code"] == "LOOP_ERROR"


def test_log_writes_keys_that_are_not_strings_as_text(tmp_path: Path) -> None:
    code = "def run_turn(ctx):\n    ctx.log('keys', {(1, 2): 'pair', 3: [4], 'nested': {5: 6, (7,): [{None: 8}]}})\n"
    exit_code, events = _text_turn(tmp_path, code)
    assert exit_code == 0 and _typed(events, "loop/keys") == [
        {"(1, 2)": "pair", "3": [4], "nested": {"5": 6, "(7,)": [{"None": 8}]}}
    ]


def _loop_error_trajectory(loop: str | None) -> tuple[dict[str, Any], ...]:
    """A root turn that ended with LOOP_ERROR, read as the reader lays it out: the agents' files before the root's."""
    error = {"code": "LOOP_ERROR", "message": "RuntimeError: boom"}
    return (
        {"type": "session", "data": {"agent": "checker", "parent": "root", "graph": "seed"}},
        {"type": "turn/end", "data": {"turn": 2, "reason": {"kind": "completed"}}},
        {"type": "session", "data": {"agent": "root", "loop": loop, "graph": "main"}},
        {"type": "turn/start", "data": {"turn": 1}},
        {"type": "turn/end", "data": {"turn": 1, "reason": {"kind": "error", "error": error}}},
    )


def test_a_loop_error_failure_is_observed_at_stage_loop(tmp_path: Path) -> None:
    backend = _backend(tmp_path, lambda n, s, m: None)
    observed = {}
    for loop in ("main", None):
        result = EpisodeResult(exit_code=1, stdout="", stderr="", trajectory=_loop_error_trajectory(loop), residue=())
        scored = backend._score_result(result, "task one")
        assert scored.score is None and scored.failure is not None
        observed[loop] = (scored.failure.stage, scored.failure.cause)
    assert observed == {
        "main": ("loop", "LOOP_ERROR: RuntimeError: boom"),
        None: ("graph", "LOOP_ERROR: RuntimeError: boom"),
    }


def test_a_recovered_state_with_a_group_entry_is_refused_before_the_render(tmp_path: Path) -> None:
    # The state gate reads the flat tree rule as admission does, so the step raises the refusal and never renders.
    skill = Mutation("create", "s1", {"name": "skill", "config": {"name": "marker", "text": "# marker"}})
    seed = list(SEED_NODES)
    for options, refusal in (
        (_group(), FLAT_TREE_REFUSAL),
        ({"name": "rules", "config": [{"text": "x"}]}, "entry config must be an object, got list"),
    ):
        backend = _backend(tmp_path, lambda n, s, m: skill)
        state = {**backend.initial_state(), "entries": [*seed, {"id": "g", **options}]}
        with pytest.raises(ValueError, match=f"recovered state entry 'g' rejected: {refusal}"):
            run_backend_step(backend, batch(), state)
    with pytest.raises(ValueError, match=f"seed entry 'g' rejected: {FLAT_TREE_REFUSAL}"):
        _backend(tmp_path, lambda n, s, m: skill, seed=[*seed, {"id": "g", **_group()}])
