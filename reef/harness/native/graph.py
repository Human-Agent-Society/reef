"""The native loop's interpreter: a graph of stages, run one transition at a time.

The graph is data from the tree (``graphs/main.json``, or the seed graph when
the tree carries none); the stage handlers are fixed code here. The seed
graph reproduces the fixed loop this replaced, event for event, apart from
the ``stage/enter`` and ``stage/exit`` events that now name the path.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.harness.model_binding import ModelBinding
from reef.harness.native.seed import SEED_GRAPH
from reef.harness.nodes import validate_native_graph

#: Transitions one run may take beyond what the step budget implies; admission proves termination, this is the guard.
TRANSITIONS_PER_STEP = 16


class GraphError(Exception):
    """A graph the loop cannot run; the episode ends with LOAD_ERROR like a broken tool."""


class Graph:
    """One validated graph: its stages by name and one target per (stage, outcome)."""

    def __init__(self, config: Mapping[str, Any], *, source: str) -> None:
        options = validate_native_graph(config)
        self.name = str(options["name"])
        self.source = source
        self.start = str(options["start"])
        self.max_steps = int(options.get("max_steps", SEED_GRAPH["max_steps"]))
        self.stages: dict[str, Mapping[str, Any]] = {str(k): v for k, v in options["stages"].items()}
        self.edges: dict[tuple[str, str], str] = {
            (str(e["from"]), str(e["when"])): str(e["to"]) for e in options["edges"]
        }


def load_graph(root: Path) -> Graph:
    """The tree's ``graphs/main.json`` when present, else the seed graph; a bad file is a GraphError."""
    path = root / "graphs" / "main.json"
    if not path.is_file():
        return Graph(SEED_GRAPH, source="seed")
    try:
        import json

        return Graph(json.loads(path.read_text(encoding="utf-8")), source="main")
    except (OSError, ValueError) as exc:
        raise GraphError(f"graphs/main.json cannot run: {exc}") from exc


class _Stop(Exception):
    """The run ended inside a stage; carries the exit status."""

    def __init__(self, exit_code: int) -> None:
        super().__init__(exit_code)
        self.exit_code = exit_code


def _last_assistant_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


class Run:
    """One turn's state, shared by every stage handler: the messages, the step counter, the log."""

    def __init__(
        self,
        loop: Any,
        prompt: str,
        binding: ModelBinding,
        tools: Mapping[str, Any],
        hooks: Mapping[str, list],
        workdir: Path,
    ) -> None:
        self.loop = loop
        self.prompt = prompt
        self.binding = binding
        self.tools = tools
        self.hooks = hooks
        self.workdir = workdir
        self.session = loop.session
        self.system = loop.system_prompt(loop.root)
        self.declarations = [tool.declaration() for tool in tools.values()]
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": prompt},
        ]
        self.step = 0
        self.step_open = False
        self.last: dict[str, Any] = {}

    def say(self, content: str, source: Mapping[str, Any]) -> None:
        self.session.write("user/message", {"step": self.step, "source": dict(source), "content": content})
        self.messages.append({"role": "user", "content": content})

    def close_step(self) -> None:
        if self.step_open:
            self.session.write("step/end", {"turn": 1, "step": self.step})
            self.step_open = False

    # -- stage handlers, one per kind; each returns the outcome its edges name ------------------------

    def model(self, graph: Graph, stage: Mapping[str, Any]) -> str:
        loop = self.loop
        self.close_step()
        if self.step >= graph.max_steps:
            self.session.write("turn/end", {"turn": 1, "reason": {"kind": "max-steps", "steps": graph.max_steps}})
            raise _Stop(0)
        self.step += 1
        step = self.step
        payload = {"step": step, "task": self.prompt, "messages": self.messages}
        entry = loop._decide(self.session, self.hooks["pre_step"], "pre_step", step, payload)
        if entry.get("kind") == "reject":
            reason = {"kind": "rejected", "step": step, "message": str(entry.get("reason") or "")}
            self.session.write("turn/end", {"turn": 1, "reason": reason})
            raise _Stop(0)
        self.session.write("step/start", {"turn": 1, "step": step})
        self.step_open = True
        if step == 1:
            self.session.write(
                "request/header", {"model": self.binding.model, "system": self.system, "tools": self.declarations}
            )
        for content in loop._texts(entry.get("messages")):
            self.say(content, {"kind": "hook", "event": "pre_step"})
        body: dict[str, Any] = {"messages": self.messages}
        if self.declarations:
            body["tools"] = self.declarations
        message = loop._request(self.session, self.binding, self.hooks["request_error"], body, step)
        if message is None:
            raise _Stop(1)
        calls = list(message.get("tool_calls") or [])
        self.messages.append(message)
        self.last = message
        self.session.write(
            "assistant/message",
            {
                "step": step,
                "content": message.get("content"),
                "tool_calls": calls,
                "finish": "tool-calls" if calls else "stop",
            },
        )
        return "tool_calls" if calls else "text"

    def tools_stage(self, graph: Graph, stage: Mapping[str, Any]) -> str:
        loop = self.loop
        allow = stage.get("allow")
        tools = self.tools if not allow else {name: tool for name, tool in self.tools.items() if name in allow}
        step = self.step
        contexts: list[str] = []
        for call in list(self.last.get("tool_calls") or []):
            function = call.get("function") or {}
            name = str(function.get("name", ""))
            raw = function.get("arguments") or "{}"
            call_id = str(call.get("id") or name)
            self.session.write("tool/call", {"step": step, "call_id": call_id, "name": name, "arguments": raw})
            spill = self.workdir / loop.SPILL_DIR / f"{step}-{re.sub(r'[^A-Za-z0-9_-]', '_', call_id)}.txt"

            def gate(tool: Any, arguments: dict[str, Any], call_id: str = call_id) -> dict[str, Any]:
                payload = {
                    "step": step,
                    "call_id": call_id,
                    "name": tool.name,
                    "arguments": arguments,
                    "capabilities": list(tool.capabilities),
                }
                return loop._decide(self.session, self.hooks["pre_execute"], "pre_execute", step, payload)

            result = loop._invoke(tools, name, raw, self.workdir, spill=spill, gate=gate)
            payload = {
                "step": step,
                "call_id": call_id,
                "name": name,
                "arguments": result.get("arguments"),
                "result": result,
            }
            verdict = loop._decide(self.session, self.hooks["post_execute"], "post_execute", step, payload)
            result = loop._judged(result, verdict)
            self.session.write("tool/result", {"step": step, "call_id": call_id, "name": name, **result})
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": result["content"]})
            contexts.extend(loop._texts(verdict.get("contexts")))
        # Contexts land after the batch's results, in call order, so the model reads them as one note.
        for content in contexts:
            self.say(content, {"kind": "hook", "event": "post_execute"})
        return "done"

    def verify(self, graph: Graph, stage: Mapping[str, Any], name: str) -> tuple[str, dict[str, Any]]:
        text = _last_assistant_text(self.messages)
        line = _last_line(text)
        check = stage["check"]
        if check == "last_line_integer":
            passed = re.fullmatch(r"-?\d+", line) is not None
        elif check == "last_line_matches":
            passed = re.search(stage["pattern"], line) is not None
        else:
            passed = bool(line)
        if not passed and isinstance(stage.get("message"), str):
            self.say(stage["message"], {"kind": "stage", "stage": name})
        return ("pass" if passed else "fail"), {"check": check, "last_line": line[:200]}

    def message(self, graph: Graph, stage: Mapping[str, Any], name: str) -> str:
        self.say(str(stage["text"]), {"kind": "stage", "stage": name})
        return "done"

    def end(self, graph: Graph, stage: Mapping[str, Any]) -> None:
        self.close_step()
        self.session.write("turn/end", {"turn": 1, "reason": {"kind": str(stage.get("reason", "completed"))}})
        raise _Stop(0)


def run_graph(run: Run, graph: Graph) -> int:
    """Walk the graph from its start until an end stage, a budget stop, or a failure; the exit status."""
    session = run.session
    name = graph.start
    limit = (graph.max_steps + 1) * TRANSITIONS_PER_STEP
    try:
        for _ in range(limit):
            stage = graph.stages[name]
            kind = str(stage["kind"])
            session.write("stage/enter", {"step": run.step, "stage": name, "kind": kind})
            detail: dict[str, Any] = {}
            if kind == "model":
                outcome = run.model(graph, stage)
            elif kind == "tools":
                outcome = run.tools_stage(graph, stage)
            elif kind == "verify":
                outcome, detail = run.verify(graph, stage, name)
            elif kind == "message":
                outcome = run.message(graph, stage, name)
            else:
                run.end(graph, stage)
                return 0
            target = graph.edges[(name, outcome)]
            session.write("stage/exit", {"step": run.step, "stage": name, "outcome": outcome, "to": target, **detail})
            name = target
    except _Stop as stop:
        return stop.exit_code
    run.close_step()
    failure = {"code": "GRAPH_ERROR", "message": f"graph {graph.name!r} took more than {limit} transitions"}
    return run.loop._abort(session, failure)
