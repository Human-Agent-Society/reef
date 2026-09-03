"""Reef's native coding agent: a headless single prompt loop whose tools are composition nodes.

One episode is one process and one turn. The rendered composition root
(``REEF_NATIVE_DIR``) holds ``RULES.md``, ``skills/``, ``tools/`` and
``models.json``; the loop reads them, talks to the served model through the
rendered binding, dispatches tool calls to the tool modules, and appends one
JSONL session the ``native-jsonl`` trajectory reader decodes. Everything the
model saw is in that log: the rendered system prompt, the tool declarations,
every message, every call, every result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from reef.harness.model_binding import ModelBinding, ModelBindingError

#: Step and tool result budgets; an episode also runs under the executor's wall clock.
MAX_STEPS = 12
MAX_RESULT_CHARS = 20_000
#: Consecutive identical calls that earn the model a reminder; the loop never vetoes a call.
LOOP_GUARD_THRESHOLDS = (3, 5, 8)
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


class ToolRunner(Protocol):
    """What a tool module's ``run`` looks like: ``run(args, workdir) -> str``."""

    def __call__(self, args: dict[str, Any], workdir: str, /) -> Any: ...


class ToolModule:
    """One rendered ``native_tool`` node: its declaration for the model and its ``run``."""

    def __init__(self, name: str, description: str, parameters: Mapping[str, Any], run: ToolRunner) -> None:
        self.name = name
        self.description = description
        self.parameters = dict(parameters) or {"type": "object", "properties": {}}
        self.run = run

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


def load_tools(tools_dir: Path) -> dict[str, ToolModule]:
    """Import every ``tools/*.py``; a module that fails to import is skipped, never fatal."""
    tools: dict[str, ToolModule] = {}
    for path in sorted(tools_dir.glob("*.py")):
        spec = importlib.util.spec_from_file_location(f"reef_native_tool_{path.stem}", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            print(f"[reef-native] tool {path.name} failed to import: {exc}", file=sys.stderr)
            continue
        run = getattr(module, "run", None)
        if not callable(run):
            print(f"[reef-native] tool {path.name} defines no run(args, workdir)", file=sys.stderr)
            continue
        name = str(getattr(module, "NAME", path.stem))
        parameters = getattr(module, "PARAMETERS", {})
        tools[name] = ToolModule(name, str(getattr(module, "DESCRIPTION", "")), parameters, run)
    return tools


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
        self._handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class _LoopGuard:
    """Counts consecutive identical calls and says so to the model at the thresholds."""

    def __init__(self) -> None:
        self._last: str | None = None
        self._count = 0

    def note(self, name: str, args: Any) -> str | None:
        key = json.dumps([name, args], sort_keys=True, default=str)
        self._count = self._count + 1 if key == self._last else 1
        self._last = key
        if self._count in LOOP_GUARD_THRESHOLDS:
            return f"You have called {name} with the same arguments {self._count} times in a row. Change approach or answer."
        return None


def run_loop(prompt: str, root: Path, session_dir: Path, workdir: Path) -> int:
    binding = binding_from(root / "models.json")
    tools = load_tools(root / "tools")
    session = Session(session_dir / "session.jsonl")
    session.write(
        "session",
        {
            "version": SESSION_VERSION,
            "task": prompt,
            "model": binding.model,
            "base_url": binding.base_url,
            "cwd": str(workdir),
            "tools": sorted(tools),
        },
    )
    session.write("turn/start", {"turn": 1})
    system = system_prompt(root)
    declarations = [tool.declaration() for tool in tools.values()]
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    guard = _LoopGuard()
    try:
        for step in range(1, MAX_STEPS + 1):
            session.write("step/start", {"turn": 1, "step": step})
            if step == 1:
                session.write("request/header", {"model": binding.model, "system": system, "tools": declarations})
            body: dict[str, Any] = {"messages": messages}
            if declarations:
                body["tools"] = declarations
            try:
                response = binding.complete(body)
                message = dict(response["choices"][0]["message"])
            except (ModelBindingError, KeyError, IndexError, TypeError) as exc:
                failure = {"code": "MODEL_ERROR", "message": f"{type(exc).__name__}: {exc}"[:600]}
                session.write("turn/end", {"turn": 1, "reason": {"kind": "error", "error": failure}})
                print(f"[reef-native] model call failed: {exc}", file=sys.stderr)
                return 1
            calls = list(message.get("tool_calls") or [])
            messages.append(message)
            session.write(
                "assistant/message",
                {
                    "step": step,
                    "content": message.get("content"),
                    "tool_calls": calls,
                    "finish": "tool-calls" if calls else "stop",
                },
            )
            if not calls:
                session.write("step/end", {"turn": 1, "step": step})
                session.write("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
                return 0
            for call in calls:
                function = call.get("function") or {}
                name = str(function.get("name", ""))
                raw = function.get("arguments") or "{}"
                call_id = str(call.get("id") or name)
                session.write("tool/call", {"step": step, "call_id": call_id, "name": name, "arguments": raw})
                result = _invoke(tools, name, raw, workdir)
                session.write("tool/result", {"step": step, "call_id": call_id, "name": name, **result})
                messages.append({"role": "tool", "tool_call_id": call_id, "content": result["content"]})
                reminder = guard.note(name, result.get("arguments"))
                if reminder is not None:
                    session.write(
                        "user/message", {"source": {"kind": "plugin", "plugin": "loop-guard"}, "content": reminder}
                    )
                    messages.append({"role": "user", "content": reminder})
            session.write("step/end", {"turn": 1, "step": step})
        session.write("turn/end", {"turn": 1, "reason": {"kind": "max-steps", "steps": MAX_STEPS}})
        return 0
    finally:
        session.close()


def _invoke(tools: Mapping[str, ToolModule], name: str, raw: str, workdir: Path) -> dict[str, Any]:
    """One result for one call: content, is_error, and a closed error code; run only sees valid arguments."""
    try:
        arguments = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        arguments = raw

    def error(code: str, message: str) -> dict[str, Any]:
        return {
            "content": f"Error: {message}",
            "is_error": True,
            "error": {"code": code, "message": message},
            "arguments": arguments,
        }

    tool = tools.get(name)
    if tool is None:
        return error("UNKNOWN_TOOL", f"unknown tool {name!r}; available: {', '.join(sorted(tools)) or 'none'}")
    if not isinstance(arguments, dict):
        return error("INVALID_ARGS", "arguments must be a JSON object")
    violation = tool.validate(arguments)
    if violation is not None:
        return error("INVALID_ARGS", violation)
    started = time.monotonic()
    try:
        result = tool.run(arguments, str(workdir))
    except Exception as exc:
        return error("TOOL_FAILED", f"{type(exc).__name__}: {exc}")
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    truncated = len(text) > MAX_RESULT_CHARS
    return {
        "content": text[:MAX_RESULT_CHARS],
        "is_error": False,
        "arguments": arguments,
        "meta": {"duration_ms": int((time.monotonic() - started) * 1000), "truncated": truncated},
    }


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
