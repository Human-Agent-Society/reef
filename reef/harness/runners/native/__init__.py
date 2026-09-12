"""Reef's native coding agent: a headless single prompt loop whose tools and loop events are composition nodes.

One episode is one process and one turn. The rendered composition root
(``REEF_NATIVE_DIR``) holds ``RULES.md``, ``skills/``, ``tools/``, ``hooks/``,
``graphs/``, ``agents/``, ``loops/`` and ``models.json``; the loop reads them once into a
``NativeHost`` (``reef.harness.runners.native.host``), talks to the served model
through the rendered binding, dispatches tool calls to the tool modules
through the capability enforcer ``REEF_NATIVE_ENFORCE`` selects, asks the
hook modules at four events, and appends one JSONL session the
``native-jsonl`` trajectory reader decodes. Everything the model saw is in
that log: the rendered system prompt, the tool declarations, every message,
every call, every result with what was enforced on it, and every hook
decision that changed the loop's course.
"""

from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from reef.harness.episodes.model_binding import ModelBinding, ModelBindingError, usage_of
from reef.harness.runners.native.enforce import Enforcer, InProcessEnforcer, SandboxFailed, ToolFailed, select_enforcer
from reef.harness.tree.nodes import NATIVE_EVENTS, NATIVE_LOOP_DEFAULT_MAX_STEPS, scope_bindings, validate_native_loop

#: Step and tool result budgets; an episode also runs under the executor's wall clock.
MAX_STEPS = 12
MAX_RESULT_CHARS = 20_000
#: A result over the cap is saved whole to this directory under the workspace; the model reads the head, a
#: marker naming the file, and this many characters of tail.
TOOL_OUTPUT_DIR = ".reef/tool-output"
TOOL_OUTPUT_TAIL_CHARS = 2_000
#: Tokens one model call may generate; a local single slot server stalls every other caller behind an unbounded one.
MAX_COMPLETION_TOKENS = 4096
#: Provider attempts one step may spend and the longest wait between them, whatever a request_error hook asks.
MAX_REQUEST_ATTEMPTS = 4
MAX_RETRY_DELAY_MS = 10_000
DEFAULT_SYSTEM_PROMPT = "You are a coding agent. Use the tools to complete the task, then answer."
SESSION_VERSION = 1
#: The entries list beside the rendered files (``files.tree`` of the native descriptor), relative to the root.
TREE_FILE = "tree.json"
#: A tool's declaration as the render writes it after the code: literal constants the loop reads without running it.
TOOL_FIELDS: tuple[str, ...] = ("NAME", "DESCRIPTION", "PARAMETERS", "CAPABILITIES")
_SCALAR_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}
#: What the loop decides at each event when no hook says otherwise.
_DEFAULTS: dict[str, dict[str, Any]] = {
    "pre_step": {"kind": "enter", "messages": []},
    "pre_execute": {"kind": "allow"},
    "request_error": {"kind": "fail"},
    "post_execute": {"kind": "accept", "contexts": []},
}


class LoadError(Exception):
    """A rendered tool or hook module the loop cannot use; the episode ends in error instead of running without it."""


class ToolRunner(Protocol):
    """What a tool module's ``run`` looks like: ``run(args, workdir) -> str``."""

    def __call__(self, args: dict[str, Any], workdir: str, /) -> Any: ...


class Next(Protocol):
    """A hook's ``next``: the decision of the layer below, computed once."""

    def __call__(self) -> dict[str, Any]: ...


class HookListener(Protocol):
    """What a hook module's ``listen`` looks like: ``listen(payload, next) -> decision``."""

    def __call__(self, payload: dict[str, Any], next_: Next, /) -> Any: ...


class ToolModule:
    """One rendered ``native_tool`` node: its declaration for the model and its ``run``."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Mapping[str, Any],
        run: ToolRunner,
        capabilities: Sequence[str] = (),
        path: Path | None = None,
        builtin_tool: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = dict(parameters) or {"type": "object", "properties": {}}
        self.run = run
        self.capabilities = tuple(str(item) for item in capabilities)
        # The module file, imported only where a call runs: afresh in a sandboxing enforcer's child, or at the
        # first in process call; a tool built in code has none.
        self.path = path
        # Reef's own code rather than the tree's (the serve form's self tools): it runs in process whatever
        # enforcer the environment names, since the enforcer confines what a tree entry may do.
        self.builtin_tool = builtin_tool

    def declaration(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    def validate(self, args: Any) -> str | None:
        """The first schema violation, checked before ``run``: required keys and top-level scalar types."""
        if not isinstance(args, dict):
            return "arguments must be a JSON object"
        properties = self.parameters.get("properties") or {}
        for key in self.parameters.get("required") or ():
            if key not in args:
                return f"missing required argument {key!r}"
        for key, value in args.items():
            expected = _SCALAR_TYPES.get(str((properties.get(key) or {}).get("type", "")))
            if expected is not None and (
                not isinstance(value, expected) or (isinstance(value, bool) and expected is not bool)
            ):
                return f"argument {key!r} must be {properties[key]['type']}"
        return None


class HookModule:
    """One rendered ``native_hook`` node: the event it listens at and its ``listen``."""

    def __init__(self, name: str, event: str, listen: HookListener) -> None:
        self.name = name
        self.event = event
        self.listen = listen


@dataclass(frozen=True)
class LoopModule:
    """One rendered ``native_loop`` node: the module whose ``run_turn(ctx)`` replaces the graph for the root turn."""

    name: str
    #: The module file; None for a loop built in code.
    path: Path | None
    max_steps: int
    module: ModuleType

    def run_turn(self, ctx: Any) -> Any:
        return self.module.run_turn(ctx)


def import_module_file(path: Path, prefix: str) -> ModuleType:
    """Import one rendered module outside ``sys.modules``; a failure is a LoadError naming the file.

    The source is compiled directly rather than through the bytecode cache: a
    module rewritten in place within the same second and at the same length
    would otherwise run its previous code."""
    spec = importlib.util.spec_from_file_location(f"{prefix}_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise LoadError(f"{path.name} is not an importable module")
    module = importlib.util.module_from_spec(spec)
    try:
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)  # bytes: the coding cookie and BOM apply
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # a top level that exits the interpreter is the module's failure, not the loop's
        raise LoadError(f"{path.name} failed to import: {type(exc).__name__}: {exc}") from exc
    return module


def _modules(directory: Path, prefix: str) -> Iterator[tuple[Path, ModuleType]]:
    """Import every ``*.py`` in name order (hooks: they run in this process); a module that fails to import fails the episode, so the tree that carries it loses."""
    for path in sorted(directory.glob("*.py")):
        yield path, import_module_file(path, prefix)


def _imported_run(path: Path) -> ToolRunner:
    """The ``run`` a tool module binds once imported; a name the static read saw bound may still be missing (a branch not taken) or not callable."""
    namespace = vars(import_module_file(path, "reef_native_tool"))
    if "run" not in namespace:
        raise LoadError(f"tool {path.name} did not bind run(args, workdir) when imported")
    if not callable(namespace["run"]):
        raise LoadError(f"tool {path.name} binds run but it is not callable")
    run: ToolRunner = namespace["run"]
    return run


class _ModuleRun:
    """A tool module's ``run`` for the in process enforcer: imported at the first call, never at load, and once, so a top level that raised fails every later call the same way without running again."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._run: ToolRunner | None = None
        self._error: LoadError | None = None

    def __call__(self, args: dict[str, Any], workdir: str, /) -> Any:
        run = self._run
        if run is None:
            with self._lock:
                if self._run is None and self._error is None:
                    try:
                        self._run = _imported_run(self._path)
                    except LoadError as exc:
                        self._error = exc
                run = self._run
            if run is None:
                # A fresh instance: raising the kept one again would grow its traceback at every call.
                raise LoadError(str(self._error)) from self._error
        return run(args, workdir)


def _literal(path: Path, name: str, value: ast.expr) -> Any:
    """The constant a declaration assignment carries, evaluated as a literal and never as code."""
    try:
        return ast.literal_eval(value)
    except (ValueError, TypeError) as exc:
        raise LoadError(f"tool {path.name} must declare {name} as a literal") from exc


def _module_scope(nodes: Iterable[ast.AST]) -> Iterator[ast.AST]:
    """Every node at module scope in source order: compound statement bodies and expressions are followed, function, class and lambda bodies are not."""
    # An explicit stack: a long flat operator chain is a tree as deep as it has terms.
    stack = list(nodes)[::-1]
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            stack.extend(list(ast.iter_child_nodes(node))[::-1])


def _target_names(target: ast.expr) -> Iterator[str]:
    """The names an assignment target binds: a name, or every name under a tuple, list or star target."""
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _target_names(element)
    elif isinstance(target, ast.Starred):
        yield from _target_names(target.value)


def _bindings(node: ast.AST) -> Iterator[str]:
    """The names one node binds in the scope it sits in: a def, class or import name; an assignment, for, with, except or walrus target; a match capture."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        yield node.name
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        yield from (alias.asname or alias.name.partition(".")[0] for alias in node.names)
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            yield from _target_names(target)
    elif isinstance(node, (ast.AnnAssign, ast.For, ast.AsyncFor)):
        yield from _target_names(node.target)
    elif isinstance(node, ast.withitem) and node.optional_vars is not None:
        yield from _target_names(node.optional_vars)
    elif isinstance(node, ast.NamedExpr):
        yield node.target.id
    elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name is not None:
        yield node.name
    elif isinstance(node, ast.MatchMapping) and node.rest is not None:
        yield node.rest


def tool_from_source(path: Path) -> ToolModule:
    """The tool a rendered module declares, read from its source without running it: the last module scope assignment to a declaration constant binds (compound statement bodies are followed, function and class bodies are not, and the render writes its constants last), and ``run`` is what the enforcer imports where the call runs."""
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))  # bytes: the coding cookie and BOM apply
    except OSError as exc:
        raise LoadError(f"{path.name} cannot be read: {type(exc).__name__}: {exc}") from exc
    except Exception as exc:  # SyntaxError, a null byte, or a RecursionError from a tree too deep to build
        raise LoadError(f"{path.name} failed to parse: {type(exc).__name__}: {exc}") from exc
    bound: set[str] = set()
    assigned: dict[str, ast.expr] = {}
    for node in _module_scope(tree.body):
        bound.update(_bindings(node))
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(target, ast.Name) and target.id in TOOL_FIELDS:
                    assigned[target.id] = node.value
    if "run" not in bound:
        raise LoadError(f"no top level statement of {path.name} binds run(args, workdir)")
    fields = {name: _literal(path, name, value) for name, value in assigned.items()}
    parameters = fields.get("PARAMETERS", {})
    capabilities = fields.get("CAPABILITIES", ())
    return ToolModule(
        str(fields.get("NAME", path.stem)),
        str(fields.get("DESCRIPTION", "")),
        parameters if isinstance(parameters, dict) else {},
        _ModuleRun(path),
        capabilities if isinstance(capabilities, list) else (),
        path=path,
    )


def hook_from_module(path: Path, module: ModuleType) -> HookModule:
    """The hook a rendered module declares: ``listen`` at the event the render wrote after the code."""
    listen = getattr(module, "listen", None)
    event = str(getattr(module, "EVENT", ""))
    if not callable(listen) or event not in NATIVE_EVENTS:
        raise LoadError(f"hook {path.name} defines no listen(payload, next) at a known event")
    return HookModule(str(getattr(module, "NAME", path.stem)), event, listen)


def loop_from_module(path: Path | None, module: ModuleType, options: Mapping[str, Any]) -> LoopModule:
    """The loop a rendered module declares: ``run_turn`` plus the name and budget of its admitted options."""
    name = str(options["name"])
    if not callable(getattr(module, "run_turn", None)):
        raise LoadError(f"loop {path.name if path is not None else name} defines no run_turn(ctx)")
    return LoopModule(name, path, int(options.get("max_steps", NATIVE_LOOP_DEFAULT_MAX_STEPS)), module)


def _admit_loop(path: Path, options: Mapping[str, Any]) -> Mapping[str, Any]:
    """``options`` admitted as a native_loop node, or the LoadError naming the file that carries them."""
    try:
        return validate_native_loop(options)
    except ValueError as exc:
        raise LoadError(f"loop {path.name} cannot run: {exc}") from exc


def _loop_header(path: Path, code: str) -> dict[str, Any]:
    """The ``NAME`` and ``MAX_STEPS`` literals the render wrote after the code, read without running it.

    Any other binding of the two names at module scope is refused: the import would set what this read
    could not, and the module's top level would have run before admission saw it."""
    header: dict[str, Any] = {"name": path.stem, "code": code}
    try:
        # The bytes the import compiles: a coding cookie or a BOM reads the same here.
        tree = ast.parse(code.encode("utf-8"), path.name)
    except (SyntaxError, ValueError) as exc:
        raise LoadError(f"loop {path.name} cannot run: code does not compile: {exc}") from exc
    written: list[ast.Name] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    written.append(target)
                    if target.id == "NAME":
                        header["name"] = node.value.value
                    elif target.id == "MAX_STEPS":
                        header["max_steps"] = node.value.value
    for name, binding in scope_bindings(tree):
        if name in ("NAME", "MAX_STEPS") and not any(binding is target for target in written):
            raise LoadError(f"loop {path.name} cannot run: the header is not the literals the render wrote")
    return header


def load_loop(loops_dir: Path) -> LoopModule | None:
    """The one ``*.py`` under ``loops/``, admitted again like an agent file; two files are a LoadError, none is None."""
    paths = sorted(loops_dir.glob("*.py")) if loops_dir.is_dir() else []
    if len(paths) > 1:
        raise LoadError(f"one loop per tree: loops/ holds {', '.join(path.name for path in paths)}")
    if not paths:
        return None
    path = paths[0]
    code = path.read_text(encoding="utf-8")
    # The text and its written header meet admission before the import runs the module; what the module binds meets it after.
    _admit_loop(path, _loop_header(path, code))
    module = import_module_file(path, "reef_native_loop")
    options = _admit_loop(
        path,
        {
            "name": str(getattr(module, "NAME", path.stem)),
            "code": code,
            "max_steps": getattr(module, "MAX_STEPS", NATIVE_LOOP_DEFAULT_MAX_STEPS),
        },
    )
    return loop_from_module(path, module, options)


def load_tools(tools_dir: Path) -> dict[str, ToolModule]:
    """Every tool read from its source in name order; no tool module runs in this process at load."""
    tools: dict[str, ToolModule] = {}
    for path in sorted(tools_dir.glob("*.py")):
        tool = tool_from_source(path)
        tools[tool.name] = tool
    return tools


def load_hooks(hooks_dir: Path) -> dict[str, list[HookModule]]:
    """Every hook under its event, in file name order: that order is the waterfall order."""
    hooks: dict[str, list[HookModule]] = {event: [] for event in NATIVE_EVENTS}
    for path, module in _modules(hooks_dir, "reef_native_hook"):
        hook = hook_from_module(path, module)
        hooks[hook.event].append(hook)
    return hooks


def load_agents(agents_dir: Path) -> dict[str, Mapping[str, Any]]:
    """Every ``*.json`` under ``agents/`` by name, admitted again here so a hand edited file cannot run unchecked."""
    from reef.harness.tree.nodes import validate_native_agent

    agents: dict[str, Mapping[str, Any]] = {}
    for path in sorted(agents_dir.glob("*.json")) if agents_dir.is_dir() else []:
        try:
            options = validate_native_agent(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise LoadError(f"agent {path.name} cannot run: {exc}") from exc
        agents[str(options["name"])] = options
    return agents


def binding_from(models_path: Path) -> ModelBinding:
    data = json.loads(models_path.read_text(encoding="utf-8"))
    return ModelBinding(
        base_url=str(data["base_url"]),
        model=str(data["model"]),
        api_key=str(data.get("api_key") or ""),
        api=str(data.get("api") or "openai"),
    )


def context_window_from(models_path: Path) -> int:
    """``context_window`` in models.json (a config node with target ``models`` sets it), else the default."""
    from reef.harness.runners.native.graph import DEFAULT_CONTEXT_WINDOW  # late: graph.py imports this module

    data = json.loads(models_path.read_text(encoding="utf-8"))
    value = data.get("context_window")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return DEFAULT_CONTEXT_WINDOW
    return value


class Session:
    """The trajectory: ``{type, seq, time, data}`` per line, appended and flushed as the loop goes."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")
        self._seq = 0

    def write(self, type_: str, data: Mapping[str, Any]) -> None:
        event = {"type": type_, "seq": self._seq, "time": int(time.time() * 1000), "data": dict(data)}
        self._seq += 1
        self._handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class _Layer:
    """``next`` as one hook sees it: the layer below runs once however often it is called.

    The hook gets a copy; the pristine decision stays here for the comparison
    and the fallback, so an in-place edit is a change like any other."""

    def __init__(
        self,
        hooks: Sequence[HookModule],
        index: int,
        payload: Mapping[str, Any],
        default: Mapping[str, Any],
        trace: list[dict[str, Any]],
    ) -> None:
        self._hooks = hooks
        self._index = index
        self._payload = payload
        self._default = default
        self._trace = trace
        self._decision: dict[str, Any] | None = None
        self._handed: dict[str, Any] | None = None

    @property
    def called(self) -> bool:
        return self._decision is not None

    @property
    def decision(self) -> dict[str, Any]:
        if self._decision is None:
            self._decision = _waterfall(self._hooks, self._index, self._payload, self._default, self._trace)
        return self._decision

    def __call__(self) -> dict[str, Any]:
        if self._handed is None:
            self._handed = copy.deepcopy(self.decision)
        return self._handed


def _plain(decision: dict[str, Any]) -> dict[str, Any]:
    """A decision as the loop and the log carry it: text lists normalized, and proven JSON encodable."""
    plain: dict[str, Any] = dict(decision)
    for key in ("messages", "contexts"):
        if key in plain:
            plain[key] = _texts(plain[key])
    json.dumps(plain, default=str)
    return plain


def _waterfall(
    hooks: Sequence[HookModule],
    index: int,
    payload: Mapping[str, Any],
    default: Mapping[str, Any],
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    """Hook ``index`` decides, seeing the layer below through ``next``; the last layer is the loop's default.

    A hook owns the decision by returning without calling ``next``. A hook
    that raises, or returns anything but a plain object the log can carry,
    is skipped and the layer below stands. ``trace`` receives every hook
    whose decision differs from the one below it, so the log names who
    changed the loop's course."""
    if index == len(hooks):
        return copy.deepcopy(dict(default))
    hook = hooks[index]
    below = _Layer(hooks, index + 1, payload, default, trace)
    try:
        decision = hook.listen(copy.deepcopy(dict(payload)), below)
        if isinstance(decision, dict):
            decision = _plain(decision)
    except Exception as exc:
        trace.append({"hook": hook.name, "error": f"{type(exc).__name__}: {exc}"[:600]})
        return below.decision
    if not isinstance(decision, dict):
        return below.decision
    if not below.called or decision != below.decision:
        trace.append({"hook": hook.name, "owned": not below.called, "decision": copy.deepcopy(decision)})
    return decision


def _decide(
    session: Session, hooks: Sequence[HookModule], event: str, step: int, payload: Mapping[str, Any]
) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []
    decision = _waterfall(hooks, 0, payload, _DEFAULTS[event], trace)
    for entry in trace:
        session.write("hook/error" if "error" in entry else "hook/decision", {"event": event, "step": step, **entry})
    return decision


def _texts(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str) and item] if isinstance(value, list) else []


def _complete(
    binding: ModelBinding, body: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, int] | None]:
    """One provider attempt: the assistant message and the usage it reported, or the closed MODEL_ERROR failure."""
    try:
        response = binding.complete(dict(body))
        return dict(response["choices"][0]["message"]), None, usage_of(response)
    except (ModelBindingError, KeyError, IndexError, TypeError) as exc:
        failure: dict[str, Any] = {"code": "MODEL_ERROR", "message": f"{type(exc).__name__}: {exc}"[:600]}
        if isinstance(exc, ModelBindingError) and exc.status is not None:
            failure["status"] = exc.status
        return None, failure, None


def _request(
    session: Session, binding: ModelBinding, hooks: Sequence[HookModule], body: Mapping[str, Any], step: int
) -> tuple[dict[str, Any] | None, dict[str, int] | None]:
    """The step's model call and the usage it reported, retried while a request_error hook says so;
    ``(None, None)`` once the turn ended in error."""
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        message, failure, usage = _complete(binding, body)
        if failure is None:
            return message, usage
        session.write("request/error", {"step": step, "attempt": attempt, "error": failure})
        action = _decide(session, hooks, "request_error", step, {"step": step, "attempt": attempt, "error": failure})
        if action.get("kind") == "retry" and attempt < MAX_REQUEST_ATTEMPTS:
            delay = action.get("delay_ms")
            time.sleep(
                min(float(delay) if isinstance(delay, (int, float)) and delay > 0 else 0.0, MAX_RETRY_DELAY_MS) / 1000
            )
            continue
        _abort(session, failure, attempts=attempt)
        return None, None
    return None, None


def _abort(session: Session, failure: Mapping[str, Any], turn: int = 1, **detail: Any) -> int:
    """End the turn in error: the closed failure in the log, its message on stderr, exit status 1."""
    session.write("turn/end", {"turn": turn, "reason": {"kind": "error", "error": dict(failure), **detail}})
    print(f"[reef-native] {failure['message']}", file=sys.stderr)
    return 1


def _judged(result: dict[str, Any], verdict: Mapping[str, Any]) -> dict[str, Any]:
    """The result the model sees after post_execute: blocked with feedback, replaced content, or as run."""
    if verdict.get("kind") == "block":
        feedback = str(verdict.get("feedback") or "blocked by a hook")
        return {
            "content": f"Error: {feedback}",
            "is_error": True,
            "error": {"code": "HOOK_BLOCKED", "message": feedback},
            "arguments": result.get("arguments"),
        }
    if isinstance(verdict.get("content"), str):
        return {**result, "content": verdict["content"][:MAX_RESULT_CHARS]}
    return result


def run_loop(prompt: str, root: Path, session_dir: Path, workdir: Path) -> int:
    """One turn: the tree's loop as code when it carries one, else its graph (or the seed graph) walked stage by stage."""
    from reef.harness.runners.native import graph as graphs  # late: graph.py imports this module
    from reef.harness.runners.native.host import NativeHost

    binding = binding_from(root / "models.json")
    session = Session(session_dir / "session.jsonl")
    header = {
        "version": SESSION_VERSION,
        "task": prompt,
        "model": binding.model,
        "base_url": binding.base_url,
        "cwd": str(workdir),
        # Where the composition came from: the entries list a newer render carries, else the rendered files.
        "tree": TREE_FILE if (root / TREE_FILE).is_file() else "files",
    }
    try:
        try:
            # The enforcer is chosen before any module of the tree runs in this process, so the tree cannot choose it.
            enforcer = select_enforcer(os.environ)
            header["enforcement"] = enforcer.mode
            # The session directory is the one writable path under the sandbox, so a tree boot mounts there.
            # One directory per process, cleared on the way out: the wrapper reuses the sessions directory
            # across runs and a mount refuses a module file that is already there.
            mount_dir = session_dir / "mounts" / f"boot-{os.getpid()}"
            shutil.rmtree(mount_dir, ignore_errors=True)
            host = NativeHost.from_root(root, mount_dir)
            graph = host.graph("main")
        except (LoadError, graphs.GraphError, ValueError) as exc:
            session.write("session", {**header, "tools": [], "hooks": {}, "graph": None, "loop": None})
            session.write("turn/start", {"turn": 1})
            return _abort(session, {"code": "LOAD_ERROR", "message": str(exc)[:600]})
        tools, hooks, module = host.tools, host.hooks, host.loop
        session.write(
            "session",
            {
                **header,
                "agent": "root",
                "turn": 1,
                "tools": sorted(tools),
                "capabilities": {name: list(tools[name].capabilities) for name in sorted(tools)},
                "hooks": {hook.name: event for event, listeners in hooks.items() for hook in listeners},
                # The graph stays named beside the loop: it is what the agents the loop calls fall back to.
                "graph": graph.source,
                "loop": None if module is None else module.name,
                "agents": sorted(host.agents),
            },
        )
        session.write("turn/start", {"turn": 1})
        loop = _Loop(session, root, session_dir, header, enforcer=enforcer)
        run = graphs.Run(loop, prompt, binding, host, workdir)
        try:
            if module is not None:
                return graphs.run_loop_module(run, module)
            return graphs.run_graph(run, graph)
        finally:
            host.dispose()
    finally:
        session.close()


def _clip(text: str, workdir: Path, full_output_path: Path | None) -> tuple[str, dict[str, Any]]:
    """What the model reads of a result over the cap: with ``full_output_path``, the whole text lands there and the model gets the head, a marker naming the file, and the tail; without it, the head alone."""
    if len(text) <= MAX_RESULT_CHARS:
        return text, {"truncated": False}
    if full_output_path is None:
        return text[:MAX_RESULT_CHARS], {"truncated": True}
    full_output_path.parent.mkdir(parents=True, exist_ok=True)
    full_output_path.write_text(text, encoding="utf-8")
    relative = full_output_path.relative_to(workdir).as_posix()
    tail = text[-TOOL_OUTPUT_TAIL_CHARS:]
    marker = f"\n... [{len(text) - MAX_RESULT_CHARS} characters omitted; the full result is in {relative}] ...\n"
    head = text[: max(0, MAX_RESULT_CHARS - len(marker) - len(tail))]
    return head + marker + tail, {"truncated": True, "output_file": relative}


def _error(code: str, message: str, arguments: Any) -> dict[str, Any]:
    """A result the model reads as an error, with one closed code."""
    return {
        "content": f"Error: {message}",
        "is_error": True,
        "error": {"code": code, "message": message},
        "arguments": arguments,
    }


def _refused(decision: Mapping[str, Any], arguments: dict[str, Any]) -> dict[str, Any] | None:
    """The result when pre_execute did not allow the call: denied with a reason, or asked with no one here to answer."""
    kind = decision.get("kind")
    if kind == "deny":
        return _error("HOOK_DENIED", str(decision.get("reason") or "denied by a hook"), arguments)
    if kind == "ask":
        reason = str(decision.get("reason") or "this call needs approval")
        return _error(
            "APPROVAL_REQUIRED", f"{reason} (headless run: no one to ask, so the call did not run)", arguments
        )
    return None


def enforcer_for(tool: ToolModule, enforcer: Enforcer | None) -> Enforcer:
    """The enforcer one call runs under: the environment's, except a built-in tool always runs in process."""
    if tool.builtin_tool or enforcer is None:
        return InProcessEnforcer()
    return enforcer


def _invoke(
    tools: Mapping[str, ToolModule],
    name: str,
    raw: str,
    workdir: Path,
    *,
    full_output_path: Path | None = None,
    gate: Callable[[ToolModule, dict[str, Any]], Mapping[str, Any]] | None = None,
    enforcer: Enforcer | None = None,
) -> dict[str, Any]:
    """One result for one call: content, is_error, and a closed error code; run only sees valid arguments the gate allowed, under the enforcer's profile."""
    try:
        arguments = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        arguments = raw

    tool = tools.get(name)
    if tool is None:
        return _error(
            "UNKNOWN_TOOL", f"unknown tool {name!r}; available: {', '.join(sorted(tools)) or 'none'}", arguments
        )
    if not isinstance(arguments, dict):
        return _error("INVALID_ARGS", "arguments must be a JSON object", arguments)
    violation = tool.validate(arguments)
    if violation is not None:
        return _error("INVALID_ARGS", violation, arguments)
    if gate is not None:
        decision = gate(tool, arguments)
        refused = _refused(decision, arguments)
        if refused is not None:
            return refused
        # An allow may rewrite the call; the rewrite meets the schema like the model's own arguments.
        if isinstance(decision.get("arguments"), dict):
            arguments = dict(decision["arguments"])
            violation = tool.validate(arguments)
            if violation is not None:
                return _error("INVALID_ARGS", f"rewritten by a hook: {violation}", arguments)
    started = time.monotonic()
    try:
        result = enforcer_for(tool, enforcer).run(tool, arguments, workdir)
    except SandboxFailed as exc:
        return _error("SANDBOX_FAILED", str(exc), arguments)
    except ToolFailed as exc:
        return _error("TOOL_FAILED", str(exc), arguments)
    except Exception as exc:
        return _error("TOOL_FAILED", f"{type(exc).__name__}: {exc}", arguments)
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    content, clipped = _clip(text, workdir, full_output_path)
    return {
        "content": content,
        "is_error": False,
        "arguments": arguments,
        "meta": {"duration_ms": int((time.monotonic() - started) * 1000), **clipped},
    }


class _Loop:
    """What the stage handlers reach of this module: the session, the root, and the loop's own helpers."""

    TOOL_OUTPUT_DIR = TOOL_OUTPUT_DIR
    MAX_COMPLETION_TOKENS = MAX_COMPLETION_TOKENS

    def __init__(
        self,
        session: Session,
        root: Path,
        session_dir: Path,
        header: Mapping[str, Any] = {},
        enforcer: Enforcer | None = None,
    ) -> None:
        self.session = session
        self.root = root
        self.session_dir = session_dir
        self.header = dict(header)
        self.enforcer = enforcer or InProcessEnforcer()
        self.turns = 1
        self.open: list[Session] = []

    def open_turn(self, agent: str) -> tuple[Session, int]:
        """A session file for one agent turn, numbered in run order under ``agents/``; the root's file sorts last."""
        self.turns += 1
        session = Session(self.session_dir / "agents" / f"{self.turns:03d}-{agent}.jsonl")
        self.open.append(session)
        return session, self.turns

    def before_step(self, run: Any) -> None:
        """Called at the top of every model stage; the episode form has nothing to land between steps."""

    _decide = staticmethod(_decide)
    _complete = staticmethod(_complete)
    _request = staticmethod(_request)
    _invoke = staticmethod(_invoke)
    enforcer_for = staticmethod(enforcer_for)
    _judged = staticmethod(_judged)
    _texts = staticmethod(_texts)
    _abort = staticmethod(_abort)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reef-native", description="Reef's native coding agent, one prompt per run.")
    parser.add_argument("-p", "--prompt", help="the task; the whole problem must be in it")
    parser.add_argument("-V", "--version", action="store_true", help="print the reef version and exit")
    args = parser.parse_args(argv)
    if args.version:
        from reef.core.version import __version__

        print(f"reef-native {__version__}")
        return 0
    if not args.prompt:
        parser.error("-p/--prompt is required")
    root = Path(os.environ.get("REEF_NATIVE_DIR") or "native")
    session_dir = Path(os.environ.get("REEF_NATIVE_SESSION_DIR") or root / "sessions")
    return run_loop(args.prompt, root, session_dir, Path.cwd())
