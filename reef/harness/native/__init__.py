"""Reef's native coding agent: a headless single prompt loop whose tools and loop events are composition nodes.

One episode is one process and one turn. The rendered composition root
(``REEF_NATIVE_DIR``) holds ``RULES.md``, ``skills/``, ``tools/``, ``hooks/``
and ``models.json``; the loop reads them, talks to the served model through
the rendered binding, dispatches tool calls to the tool modules, asks the
hook modules at four events, and appends one JSONL session the
``native-jsonl`` trajectory reader decodes. Everything the model saw is in
that log: the rendered system prompt, the tool declarations, every message,
every call, every result, and every hook decision that changed the loop's
course.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from reef.harness.model_binding import ModelBinding, ModelBindingError
from reef.harness.nodes import NATIVE_EVENTS

#: Step and tool result budgets; an episode also runs under the executor's wall clock.
MAX_STEPS = 12
MAX_RESULT_CHARS = 20_000
#: A result over the cap is spilled whole to this directory under the workspace; the model reads the head, a
#: marker naming the file, and this many characters of tail.
SPILL_DIR = ".reef/spill"
SPILL_TAIL_CHARS = 2_000
#: Provider attempts one step may spend and the longest wait between them, whatever a request_error hook asks.
MAX_REQUEST_ATTEMPTS = 4
MAX_RETRY_DELAY_MS = 10_000
DEFAULT_SYSTEM_PROMPT = "You are a coding agent. Use the tools to complete the task, then answer."
SESSION_VERSION = 1
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
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = dict(parameters) or {"type": "object", "properties": {}}
        self.run = run
        self.capabilities = tuple(str(item) for item in capabilities)

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


def _modules(directory: Path, prefix: str) -> Iterator[tuple[Path, ModuleType]]:
    """Import every ``*.py`` in name order; a module that fails to import fails the episode, so the tree that carries it loses."""
    for path in sorted(directory.glob("*.py")):
        spec = importlib.util.spec_from_file_location(f"{prefix}_{path.stem}", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise LoadError(f"{path.name} failed to import: {type(exc).__name__}: {exc}") from exc
        yield path, module


def load_tools(tools_dir: Path) -> dict[str, ToolModule]:
    tools: dict[str, ToolModule] = {}
    for path, module in _modules(tools_dir, "reef_native_tool"):
        run = getattr(module, "run", None)
        if not callable(run):
            raise LoadError(f"tool {path.name} defines no run(args, workdir)")
        name = str(getattr(module, "NAME", path.stem))
        parameters = getattr(module, "PARAMETERS", {})
        capabilities = getattr(module, "CAPABILITIES", ())
        description = str(getattr(module, "DESCRIPTION", ""))
        tools[name] = ToolModule(
            name, description, parameters, run, capabilities if isinstance(capabilities, list) else ()
        )
    return tools


def load_hooks(hooks_dir: Path) -> dict[str, list[HookModule]]:
    """Every hook under its event, in file name order: that order is the waterfall order."""
    hooks: dict[str, list[HookModule]] = {event: [] for event in NATIVE_EVENTS}
    for path, module in _modules(hooks_dir, "reef_native_hook"):
        listen = getattr(module, "listen", None)
        event = str(getattr(module, "EVENT", ""))
        if not callable(listen) or event not in hooks:
            raise LoadError(f"hook {path.name} defines no listen(payload, next) at a known event")
        hooks[event].append(HookModule(str(getattr(module, "NAME", path.stem)), event, listen))
    return hooks


def system_prompt(root: Path) -> str:
    """Rules first, then every skill, in path order."""
    files = [root / "RULES.md", *sorted((root / "skills").glob("*/SKILL.md"))]
    parts = [path.read_text(encoding="utf-8").strip() for path in files if path.exists()]
    return "\n\n".join(part for part in parts if part) or DEFAULT_SYSTEM_PROMPT


def binding_from(models_path: Path) -> ModelBinding:
    data = json.loads(models_path.read_text(encoding="utf-8"))
    return ModelBinding(
        base_url=str(data["base_url"]),
        model=str(data["model"]),
        api_key=str(data.get("api_key") or ""),
        api=str(data.get("api") or "openai"),
    )


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


def _complete(binding: ModelBinding, body: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """One provider attempt: the assistant message, or the closed MODEL_ERROR failure."""
    try:
        response = binding.complete(dict(body))
        return dict(response["choices"][0]["message"]), None
    except (ModelBindingError, KeyError, IndexError, TypeError) as exc:
        failure: dict[str, Any] = {"code": "MODEL_ERROR", "message": f"{type(exc).__name__}: {exc}"[:600]}
        if isinstance(exc, ModelBindingError) and exc.status is not None:
            failure["status"] = exc.status
        return None, failure


def _request(
    session: Session, binding: ModelBinding, hooks: Sequence[HookModule], body: Mapping[str, Any], step: int
) -> dict[str, Any] | None:
    """The step's model call, retried while a request_error hook says so; None once the turn ended in error."""
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        message, failure = _complete(binding, body)
        if failure is None:
            return message
        session.write("request/error", {"step": step, "attempt": attempt, "error": failure})
        action = _decide(session, hooks, "request_error", step, {"step": step, "attempt": attempt, "error": failure})
        if action.get("kind") == "retry" and attempt < MAX_REQUEST_ATTEMPTS:
            delay = action.get("delay_ms")
            time.sleep(
                min(float(delay) if isinstance(delay, (int, float)) and delay > 0 else 0.0, MAX_RETRY_DELAY_MS) / 1000
            )
            continue
        _abort(session, failure, attempts=attempt)
        return None
    return None


def _abort(session: Session, failure: Mapping[str, Any], **detail: Any) -> int:
    """End the turn in error: the closed failure in the log, its message on stderr, exit status 1."""
    session.write("turn/end", {"turn": 1, "reason": {"kind": "error", "error": dict(failure), **detail}})
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
    """One turn: the tree's graph (or the seed graph) walked stage by stage; each model stage is one step."""
    from reef.harness.native import graph as graphs  # late: graph.py imports this module

    binding = binding_from(root / "models.json")
    session = Session(session_dir / "session.jsonl")
    header = {
        "version": SESSION_VERSION,
        "task": prompt,
        "model": binding.model,
        "base_url": binding.base_url,
        "cwd": str(workdir),
    }
    try:
        try:
            tools = load_tools(root / "tools")
            hooks = load_hooks(root / "hooks")
            graph = graphs.load_graph(root)
        except (LoadError, graphs.GraphError, ValueError) as exc:
            session.write("session", {**header, "tools": [], "hooks": {}, "graph": None})
            session.write("turn/start", {"turn": 1})
            return _abort(session, {"code": "LOAD_ERROR", "message": str(exc)[:600]})
        session.write(
            "session",
            {
                **header,
                "tools": sorted(tools),
                "capabilities": {name: list(tools[name].capabilities) for name in sorted(tools)},
                "hooks": {hook.name: event for event, listeners in hooks.items() for hook in listeners},
                "graph": graph.source,
            },
        )
        session.write("turn/start", {"turn": 1})
        loop = _Loop(session, root)
        return graphs.run_graph(graphs.Run(loop, prompt, binding, tools, hooks, workdir), graph)
    finally:
        session.close()


def _clip(text: str, workdir: Path, spill: Path | None) -> tuple[str, dict[str, Any]]:
    """What the model reads of a result over the cap: with ``spill``, the whole text lands there and the model gets the head, a marker naming the file, and the tail; without it, the head alone."""
    if len(text) <= MAX_RESULT_CHARS:
        return text, {"truncated": False}
    if spill is None:
        return text[:MAX_RESULT_CHARS], {"truncated": True}
    spill.parent.mkdir(parents=True, exist_ok=True)
    spill.write_text(text, encoding="utf-8")
    relative = spill.relative_to(workdir).as_posix()
    tail = text[-SPILL_TAIL_CHARS:]
    marker = f"\n... [{len(text) - MAX_RESULT_CHARS} characters omitted; the full result is in {relative}] ...\n"
    head = text[: max(0, MAX_RESULT_CHARS - len(marker) - len(tail))]
    return head + marker + tail, {"truncated": True, "spill": relative}


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


def _invoke(
    tools: Mapping[str, ToolModule],
    name: str,
    raw: str,
    workdir: Path,
    *,
    spill: Path | None = None,
    gate: Callable[[ToolModule, dict[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """One result for one call: content, is_error, and a closed error code; run only sees valid arguments the gate allowed."""
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
        result = tool.run(arguments, str(workdir))
    except Exception as exc:
        return _error("TOOL_FAILED", f"{type(exc).__name__}: {exc}", arguments)
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    content, clipped = _clip(text, workdir, spill)
    return {
        "content": content,
        "is_error": False,
        "arguments": arguments,
        "meta": {"duration_ms": int((time.monotonic() - started) * 1000), **clipped},
    }


class _Loop:
    """What the stage handlers reach of this module: the session, the root, and the loop's own helpers."""

    SPILL_DIR = SPILL_DIR

    def __init__(self, session: Session, root: Path) -> None:
        self.session = session
        self.root = root

    system_prompt = staticmethod(system_prompt)
    _decide = staticmethod(_decide)
    _request = staticmethod(_request)
    _invoke = staticmethod(_invoke)
    _judged = staticmethod(_judged)
    _texts = staticmethod(_texts)
    _abort = staticmethod(_abort)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reef-native", description="Reef's native coding agent, one prompt per run.")
    parser.add_argument("-p", "--prompt", help="the task; the whole problem must be in it")
    parser.add_argument("-V", "--version", action="store_true", help="print the reef version and exit")
    args = parser.parse_args(argv)
    if args.version:
        from reef import __version__

        print(f"reef-native {__version__}")
        return 0
    if not args.prompt:
        parser.error("-p/--prompt is required")
    root = Path(os.environ.get("REEF_NATIVE_DIR") or "native")
    session_dir = Path(os.environ.get("REEF_NATIVE_SESSION_DIR") or root / "sessions")
    return run_loop(args.prompt, root, session_dir, Path.cwd())
