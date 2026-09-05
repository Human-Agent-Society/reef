"""The native loop's interpreter: a graph of stages, run one transition at a time.

The graph is data from the tree (``graphs/main.json``, or the seed graph when
the tree carries none); the stage handlers are fixed code here. The seed
graph reproduces the fixed loop this replaced, event for event, apart from
the ``stage/enter`` and ``stage/exit`` events that now name the path. The
tools, hooks, agents, graphs and prompt come from a ``NativeHost`` read at
each use, so what a step runs on is what the host holds when the step starts.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, Protocol

from reef.harness.model_binding import ModelBinding
from reef.harness.native.seed import SEED_GRAPH
from reef.harness.nodes import NATIVE_MATCH_WINDOW, validate_native_graph

#: Transitions one run may take beyond what the step budget implies; admission proves termination, this is the guard.
TRANSITIONS_PER_STEP = 16
#: The context window a compact stage measures against when models.json names none.
DEFAULT_CONTEXT_WINDOW = 32_768
#: One token per this many characters of a serialized message: the closed estimate the compact stage uses.
CHARS_PER_TOKEN = 4
#: What a compact stage asks the model for over the span it drops.
SUMMARY_PROMPT = (
    "Summarize the work so far for a coding agent that will continue it: what the task asked, what was tried, "
    "what the tools returned, what is settled, and what is still open. Be concrete and short."
)


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


class Host(Protocol):
    """What the interpreter reads at each use; ``reef.harness.native.host.NativeHost`` is the one implementation."""

    @property
    def tools(self) -> Mapping[str, Any]: ...

    @property
    def hooks(self) -> Mapping[str, list]: ...

    @property
    def agents(self) -> Mapping[str, Mapping[str, Any]]: ...

    @property
    def context_window(self) -> int: ...

    def graph(self, name: str = "main") -> Graph: ...

    def system_prompt(self, *, skills: Sequence[str] | None = None, prompt: str | None = None) -> str: ...


class _Stop(Exception):
    """The run ended inside a stage; carries the exit status and, for an agent's turn, its outcome."""

    def __init__(self, exit_code: int, outcome: str = "completed") -> None:
        super().__init__(exit_code)
        self.exit_code = exit_code
        self.outcome = outcome


class _Escalate(Exception):
    """A pre_execute hook asked inside an agent's turn; the parent graph is the one to answer."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _last_assistant_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _tokens(messages: list[dict[str, Any]]) -> int:
    return sum(len(json.dumps(message, ensure_ascii=False, default=str)) // CHARS_PER_TOKEN for message in messages)


def _split(messages: list[dict[str, Any]], keep_tokens: float) -> tuple[list, list, list]:
    """Head (the system prompt and the task), the older span to summarize, and the tail that stays verbatim.

    The tail is the last messages that fit ``keep_tokens``, grown backwards so
    it never opens on a tool result whose call was dropped."""
    head, rest = messages[:2], messages[2:]
    cut = len(rest)
    used = 0
    while cut > 0:
        size = len(json.dumps(rest[cut - 1], ensure_ascii=False, default=str)) // CHARS_PER_TOKEN
        if used + size > keep_tokens:
            break
        used += size
        cut -= 1
    while cut > 0 and cut < len(rest) and rest[cut].get("role") == "tool":
        cut -= 1
    return head, rest[:cut], rest[cut:]


def narrow_allow(parent: Sequence[str] | None, own: Sequence[str] | None) -> tuple[str, ...] | None:
    """An agent sees its own tools list within what its parent sees; no list on either side is no restriction there."""
    if own is None:
        return None if parent is None else tuple(parent)
    return tuple(own) if parent is None else tuple(name for name in own if name in parent)


def _transcript(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = str(message.get("role", ""))
        content = message.get("content")
        calls = [call.get("function", {}) for call in message.get("tool_calls") or []]
        if calls:
            content = f"{content or ''}\n" + "\n".join(f"call {c.get('name')}({c.get('arguments')})" for c in calls)
        lines.append(f"[{role}] {content if isinstance(content, str) else json.dumps(content, default=str)}")
    return "\n".join(lines)


class Run:
    """One turn's state, shared by every stage handler: the messages, the step counter, the log.

    The tools, hooks, agents and prompt are read from the host at each use;
    ``allow`` narrows the tools to the names an agent may see."""

    def __init__(
        self,
        loop: Any,
        prompt: str,
        binding: ModelBinding,
        host: Host,
        workdir: Path,
        *,
        allow: Sequence[str] | None = None,
        parent: Run | None = None,
        agent: str = "root",
        turn: int = 1,
        session: Any = None,
        skills: Sequence[str] | None = None,
        agent_prompt: str | None = None,
        max_tool_calls: int | None = None,
    ) -> None:
        self.loop = loop
        self.prompt = prompt
        self.binding = binding
        self.host = host
        self.workdir = workdir
        self.allow = None if allow is None else tuple(allow)
        self.parent = parent
        self.agent = agent
        self.turn = turn
        self.skills = None if skills is None else tuple(skills)
        self.agent_prompt = agent_prompt
        self.max_tool_calls = max_tool_calls
        self.max_steps = 0
        self.tool_calls = 0
        self.tool_errors = 0
        self.session = session or loop.session
        self.system = host.system_prompt(skills=self.skills, prompt=agent_prompt)
        self.declarations: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": prompt},
        ]
        self.step = 0
        self.step_open = False
        self.last: dict[str, Any] = {}
        # What the last request/header said the model sees; the next step writes a new one when it differs.
        self._header: tuple[str, str] | None = None

    @property
    def tools(self) -> Mapping[str, Any]:
        tools = self.host.tools
        return tools if self.allow is None else {name: tool for name, tool in tools.items() if name in self.allow}

    @property
    def hooks(self) -> Mapping[str, list]:
        return self.host.hooks

    @property
    def agents(self) -> Mapping[str, Mapping[str, Any]]:
        return self.host.agents

    @property
    def context_window(self) -> int:
        return self.host.context_window

    def say(self, content: str, source: Mapping[str, Any]) -> None:
        self.session.write("user/message", {"step": self.step, "source": dict(source), "content": content})
        self.messages.append({"role": "user", "content": content})

    def close_step(self) -> None:
        if self.step_open:
            self.session.write("step/end", {"turn": self.turn, "step": self.step})
            self.step_open = False

    def end_turn(self, reason: Mapping[str, Any], outcome: str) -> NoReturn:
        self.end_turn_quietly(reason)
        raise _Stop(0, outcome)

    def end_turn_quietly(self, reason: Mapping[str, Any]) -> None:
        self.close_step()
        self.session.write("turn/end", {"turn": self.turn, "reason": dict(reason)})

    # -- stage handlers, one per kind; each returns the outcome its edges name ------------------------

    def model(self, graph: Graph, stage: Mapping[str, Any]) -> str:
        loop = self.loop
        self.close_step()
        # Between two steps the serve form lands a queued mount, so what the host holds below is what this step runs on.
        loop.before_step(self)
        if self.step >= graph.max_steps:
            self.end_turn({"kind": "max-steps", "steps": graph.max_steps}, "budget")
        self.step += 1
        step = self.step
        # What the host holds now is what this step runs on; the hooks see the same messages the model will.
        self.system = self.host.system_prompt(skills=self.skills, prompt=self.agent_prompt)
        self.messages[0] = {"role": "system", "content": self.system}
        self.declarations = [tool.declaration() for tool in self.tools.values()]
        payload = {"step": step, "task": self.prompt, "messages": self.messages}
        entry = loop._decide(self.session, self.hooks["pre_step"], "pre_step", step, payload)
        if entry.get("kind") == "reject":
            reason = {"kind": "rejected", "step": step, "message": str(entry.get("reason") or "")}
            self.end_turn(reason, "gave_up")
        self.session.write("step/start", {"turn": self.turn, "step": step})
        self.step_open = True
        # Model visible means logged: the header is written at step 1 and again whenever what the model sees changed.
        seen = (self.system, json.dumps(self.declarations, sort_keys=True, default=str))
        if seen != self._header:
            self._header = seen
            self.session.write(
                "request/header", {"model": self.binding.model, "system": self.system, "tools": self.declarations}
            )
        for content in loop._texts(entry.get("messages")):
            self.say(content, {"kind": "hook", "event": "pre_step"})
        body: dict[str, Any] = {"messages": self.messages, "max_tokens": loop.MAX_COMPLETION_TOKENS}
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
            if self.max_tool_calls is not None and self.tool_calls >= self.max_tool_calls:
                self.end_turn({"kind": "max-tool-calls", "tool_calls": self.max_tool_calls}, "budget")
            self.tool_calls += 1
            self.session.write("tool/call", {"step": step, "call_id": call_id, "name": name, "arguments": raw})
            # An agent turn's spill carries its turn number, so two turns' step 1 do not share a file.
            prefix = f"{step}" if self.parent is None else f"t{self.turn}-{step}"
            spill = self.workdir / loop.SPILL_DIR / f"{prefix}-{re.sub(r'[^A-Za-z0-9_-]', '_', call_id)}.txt"

            def gate(tool: Any, arguments: dict[str, Any], call_id: str = call_id) -> dict[str, Any]:
                payload = {
                    "step": step,
                    "call_id": call_id,
                    "name": tool.name,
                    "arguments": arguments,
                    "capabilities": list(tool.capabilities),
                }
                decision = loop._decide(self.session, self.hooks["pre_execute"], "pre_execute", step, payload)
                if decision.get("kind") == "ask" and self.parent is not None:
                    # An agent's turn has an answerer above it: the parent graph reads ask as an outcome.
                    raise _Escalate(str(decision.get("reason") or f"{tool.name} needs approval"))
                return decision

            result = loop._invoke(tools, name, raw, self.workdir, spill=spill, gate=gate, enforcer=loop.enforcer)
            payload = {
                "step": step,
                "call_id": call_id,
                "name": name,
                "arguments": result.get("arguments"),
                "result": result,
            }
            verdict = loop._decide(self.session, self.hooks["post_execute"], "post_execute", step, payload)
            result = loop._judged(result, verdict)
            # A jail that could not run is the sandbox's failure, not the tool's: it moves no branch on tool errors.
            if result.get("is_error") and (result.get("error") or {}).get("code") != "SANDBOX_FAILED":
                self.tool_errors += 1
            # The log says what was enforced on this tool, whether or not the call reached its run.
            called = tools.get(name)
            # A host plane tool ran in process whatever the enforcer; the log says so.
            enforcer = loop.enforcer if called is None else loop.enforcer_for(called, loop.enforcer)
            enforcement = enforcer.describe(called)
            self.session.write(
                "tool/result",
                {"step": step, "call_id": call_id, "name": name, **result, "enforcement": enforcement},
            )
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

    def branch(self, graph: Graph, stage: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        """The first case that holds names the outcome; none, ``else``."""
        text = _last_assistant_text(self.messages)
        for case in stage["cases"]:
            when, value = str(case["when"]), case["value"]
            if when == "steps_used_at_least":
                hit = self.step >= int(value)
            elif when == "tool_errors_at_least":
                hit = self.tool_errors >= int(value)
            else:
                # Admission refused nested quantifiers; the window bounds what a pattern can still cost.
                hit = re.search(str(value), text[-NATIVE_MATCH_WINDOW:]) is not None
            if hit:
                return str(case["outcome"]), {"case": when, "value": value}
        return "else", {"case": "else"}

    def compact(self, graph: Graph, stage: Mapping[str, Any], name: str) -> tuple[str, dict[str, Any]]:
        """Above ``fire_ratio`` of the window, one model call summarizes the older span; ``keep_ratio`` stays verbatim."""
        loop = self.loop
        policy = {"fire_ratio": float(stage["fire_ratio"]), "keep_ratio": float(stage["keep_ratio"])}
        before = _tokens(self.messages)
        if before <= policy["fire_ratio"] * self.context_window:
            return "done", {"fired": False, "tokens": before}
        head, older, tail = _split(self.messages, policy["keep_ratio"] * self.context_window)
        if not older:
            return "done", {"fired": False, "tokens": before}
        body = {
            "messages": [
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": _transcript(older)},
            ],
            "max_tokens": loop.MAX_COMPLETION_TOKENS,
        }
        message, failure = loop._complete(self.binding, body)
        record: dict[str, Any] = {"step": self.step, "stage": name, "policy": policy, "tokens_before": before}
        if message is None:
            # The span stays as it was: a summary that did not arrive drops nothing the model saw.
            self.session.write("context/compacted", {**record, "fired": False, "error": failure})
            return "done", {"fired": False, "tokens": before, "error": str((failure or {}).get("code", ""))}
        summary = str(message.get("content") or "").strip()
        note = {"role": "user", "content": f"Summary of the earlier steps:\n{summary}"}
        self.messages = [*head, note, *tail]
        after = _tokens(self.messages)
        self.session.write(
            "context/compacted",
            {**record, "fired": True, "tokens_after": after, "dropped": len(older), "summary": summary},
        )
        return "done", {"fired": True, "tokens_before": before, "tokens_after": after, "dropped": len(older)}

    def end(self, graph: Graph, stage: Mapping[str, Any]) -> NoReturn:
        reason = str(stage.get("reason", "completed"))
        self.end_turn({"kind": reason}, reason)

    def subagent(self, graph: Graph, stage: Mapping[str, Any], name: str) -> tuple[str, dict[str, Any]]:
        """Hand the last assistant text (or the task) to an agent, then down its ``then`` pipeline; its text comes back."""
        first = str(stage["agent"])
        text = _last_assistant_text(self.messages) or self.prompt
        outcome = "completed"
        ran: list[str] = []
        queue = [first]
        while queue and outcome == "completed":
            agent = queue.pop(0)
            outcome, text, _ = self.run_agent(agent, text)
            ran.append(agent)
            queue = [*(self.agents[agent].get("then") or ()), *queue]
        content = text or f"{ran[-1]} ended with {outcome}"
        if outcome == "ask":
            content = f"{ran[-1]} asks: {text}"
        self.say(content, {"kind": "agent", "agent": ran[-1], "outcome": outcome})
        return outcome, {"agent": first, "agents": ran, "steps": self.step}

    def run_agent(self, name: str, prompt: str) -> tuple[str, str, int]:
        """One agent's turn in its own session file, on the parent's remaining step budget; (outcome, text, steps)."""
        loop = self.loop
        agent = self.agents[name]
        remaining = self.max_steps - self.step
        if remaining <= 0:
            return "budget", "", 0
        # A copy: the budget below is this turn's, and the host's graph outlives the turn.
        graph = copy.copy(self.host.graph(str(agent.get("graph", "seed"))))
        graph.max_steps = min(int(agent.get("max_steps") or remaining), remaining)
        session, turn = loop.open_turn(name)
        child = Run(
            loop,
            prompt,
            self.binding,
            self.host,
            self.workdir,
            allow=narrow_allow(self.allow, agent.get("tools")),
            parent=self,
            agent=name,
            turn=turn,
            session=session,
            skills=agent.get("skills"),
            agent_prompt=str(agent.get("prompt", "")),
            max_tool_calls=agent.get("max_tool_calls"),
        )
        tools = child.tools
        session.write(
            "session",
            {
                **loop.header,
                "task": prompt,
                "agent": name,
                "turn": turn,
                "parent": self.agent,
                "tools": sorted(tools),
                "capabilities": {n: list(tools[n].capabilities) for n in sorted(tools)},
                "hooks": {hook.name: event for event, listeners in self.hooks.items() for hook in listeners},
                "graph": graph.source,
                "max_steps": graph.max_steps,
            },
        )
        session.write("turn/start", {"turn": turn, "parent": self.agent})
        try:
            outcome, text = _walk(child, graph)
        finally:
            # Steps are drawn from the episode total, so the parent's budget shrinks by what the child spent.
            self.step += child.step
            session.close()
        return outcome, text, child.step


def _walk(run: Run, graph: Graph) -> tuple[str, str]:
    """Walk the graph from its start to an end stage or a budget stop; (outcome, the last assistant text).

    A failure (a model call that ended in error, a graph that exceeded its
    transition bound) raises ``_Stop`` with a nonzero exit status, which the
    root turns into the episode's exit and an agent's turn propagates."""
    session = run.session
    name = graph.start
    limit = (graph.max_steps + 1) * TRANSITIONS_PER_STEP
    run.max_steps = graph.max_steps
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
            elif kind == "branch":
                outcome, detail = run.branch(graph, stage)
            elif kind == "compact":
                outcome, detail = run.compact(graph, stage, name)
            elif kind == "subagent":
                outcome, detail = run.subagent(graph, stage, name)
            else:
                run.end(graph, stage)
            target = graph.edges[(name, outcome)]
            session.write("stage/exit", {"step": run.step, "stage": name, "outcome": outcome, "to": target, **detail})
            name = target
    except _Stop as stop:
        if stop.exit_code != 0:
            raise
        return stop.outcome, _last_assistant_text(run.messages)
    except _Escalate as ask:
        run.end_turn_quietly({"kind": "ask", "reason": ask.reason})
        return "ask", ask.reason
    run.close_step()
    failure = {"code": "GRAPH_ERROR", "message": f"graph {graph.name!r} took more than {limit} transitions"}
    run.loop._abort(session, failure, turn=run.turn)
    raise _Stop(1)


def run_graph(run: Run, graph: Graph) -> int:
    """The root turn: walk the graph and map its end to the episode's exit status."""
    try:
        _walk(run, graph)
    except _Stop as stop:
        return stop.exit_code
    finally:
        for session in run.loop.open:
            session.close()
    return 0
