"""The native variant's change: the loop's own tools, hooks, graph and agents are files the model may change.

``change_prompt`` asks the model for one change over the step's failing
traces: a skill, a native tool (a schema plus a module defining
``run(args, workdir) -> str``), a native hook (a module defining
``listen(payload, next) -> decision`` at one loop event), a native agent (a
helper with its own prompt, tools and budget that a graph's subagent stage
calls; the change then also carries the graph edit that reaches it) or the
main graph that is the loop's own control flow. The model answers in one
JSON shape per kind, which a small model fills more reliably than a whole
file, and ``patch_of`` turns the answer into the file the tree holds.

The prompt (``native_prompt.md``) shows the graph shape as a worked example,
not as placeholders: a 7B model copies a concrete stage and its edges, but
invents check names and drops edges when it sees only ``<stage>`` and
``<outcome>``. ``evaluate`` reads the native-jsonl trajectory. The grader
and the answer table are shared with ``evolution.py``, so ``run.py`` scores
recorded traffic the same way for both variants.
"""

import json
import re
from pathlib import Path
from string import Template

from reef.harness.runners.native.graph import SEED_GRAPH
from reef.harness.tree import HarnessTree, TreePatch

from .evolution import grade_text

KINDS = ("skill", "native_tool", "native_hook", "native_graph", "native_agent")
EVENTS = ("pre_step", "pre_execute", "request_error", "post_execute")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
#: The main graph's file, and the tree's files the prompt shows.
MAIN_GRAPH = "native/graphs/main.json"
SHOWN_PREFIXES = ("skills/", "native/")
#: The prompt: ``$failures`` takes the failing traces and ``$files`` the tree's skill and native files.
PROMPT = Template(Path(__file__).with_name("native_prompt.md").read_text(encoding="utf-8"))


def change_prompt(tree: HarnessTree, failures: str) -> str:
    """The prompt that asks for one change of ``tree``; ``failures`` is the failing traces, fenced as data."""
    current = {path: text for path, text in tree.files.items() if path.startswith(SHOWN_PREFIXES)}
    return PROMPT.substitute(failures=failures, files=json.dumps(current, indent=2))


def patch_of(tree: HarnessTree, reply: str) -> TreePatch | None:
    """The patch of ``tree`` that the model's ``reply`` asks for; ``None`` when the reply holds no usable change."""
    proposal = parse_proposal(reply)
    if proposal is None:
        return None
    kind, name, config = proposal
    changes = dict([file_of(kind, name, config)])
    if kind == "native_agent":
        # An agent alone changes nothing: only a subagent stage runs one. The change carries the graph edit too.
        graph = route_through_agent(tree, name)
        if graph is not None:
            changes[MAIN_GRAPH] = json.dumps(graph, indent=2) + "\n"
    return TreePatch(changes)


def file_of(kind, name, config):
    """The (path, text) of the file one proposal object writes: a JSON file, a skill, or a module with its
    declaration as constants after the code, which the native loop reads without running the module."""
    if kind == "skill":
        return f"skills/{name}/SKILL.md", config["text"]
    if kind == "native_graph":
        return f"native/graphs/{name}.json", json.dumps(config, indent=2) + "\n"
    if kind == "native_agent":
        return f"native/agents/{name}.json", json.dumps(config, indent=2) + "\n"
    if kind == "native_hook":
        fields = {"NAME": name, "EVENT": config["event"]}
        directory = "hooks"
    else:
        fields = {
            "NAME": name,
            "DESCRIPTION": config["description"],
            "PARAMETERS": config["parameters"],
            "CAPABILITIES": config.get("capabilities", []),
        }
        directory = "tools"
    header = "\n".join(f"{key} = {value!r}" for key, value in fields.items())
    return f"native/{directory}/{name}.py", f"{config['code'].rstrip()}\n\n\n{header}\n"


def route_through_agent(tree, name):
    """The main graph that makes a proposed agent reachable, or ``None`` when the main graph has no model text
    edge to route or already runs the agent.

    The main graph's model stage hands its text to a new ``ask-<name>`` subagent stage; the agent's text comes
    back as a user message, so every outcome of that stage goes to a new ``answer-<name>`` model stage whose
    text goes where the model's text went (the grader reads the last assistant message, and only a model stage
    writes one) and whose other outcomes follow the model stage's own edges."""
    graph = json.loads(tree.files[MAIN_GRAPH]) if MAIN_GRAPH in tree.files else dict(SEED_GRAPH)
    stages = {key: dict(value) for key, value in (graph.get("stages") or {}).items()}
    edges = [dict(edge) for edge in graph.get("edges") or ()]
    ask, answer = f"ask-{name}", f"answer-{name}"
    if ask in stages or answer in stages:
        return None
    if any(stage.get("kind") == "subagent" and stage.get("agent") == name for stage in stages.values()):
        return None
    text_edge = next(
        (
            edge
            for edge in edges
            if edge.get("when") == "text" and stages.get(edge.get("from"), {}).get("kind") == "model"
        ),
        None,
    )
    if text_edge is None:
        return None
    model, target = text_edge["from"], text_edge["to"]
    text_edge["to"] = ask
    stages[ask] = {"kind": "subagent", "agent": name}
    stages[answer] = {"kind": "model"}
    edges.extend({"from": ask, "when": outcome, "to": answer} for outcome in ("completed", "gave_up", "budget", "ask"))
    edges.append({"from": answer, "when": "text", "to": target})
    edges.extend(
        {**edge, "from": answer}
        for edge in (graph.get("edges") or ())
        if edge.get("from") == model and edge.get("when") != "text"
    )
    return {**graph, "stages": stages, "edges": edges}


def evaluate(task, result) -> float:
    """Grade the last line of the run's final assistant text against the task's answer, 1.0 exact."""
    return grade_text(task.answer, final_assistant_text(result.trajectory))


def has_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def parse_proposal(reply: str):
    """The strict proposal as (kind, name, config), dug out of the model's text; ``None`` when unusable."""
    try:
        parsed = json.loads(reply[reply.find("{") : reply.rfind("}") + 1])
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    kind, name, config = parsed.get("name"), parsed.get("id"), parsed.get("config")
    if kind not in KINDS and name in KINDS:
        # A small model swaps the kind and the id; the config's own name settles which is which below.
        kind, name = name, kind
    if kind not in KINDS or not isinstance(name, str) or not NAME.fullmatch(name):
        return None
    if not isinstance(config, dict) or config.get("name") != name:
        return None
    if kind == "skill":
        text = config.get("text")
        return (kind, name, {"name": name, "text": text}) if has_text(text) else None
    if kind == "native_agent":
        agent = {"name": name, "prompt": config.get("prompt")}
        agent.update(
            {k: config[k] for k in ("graph", "tools", "skills", "max_steps", "max_tool_calls", "then") if k in config}
        )
        return (kind, name, agent) if has_text(agent["prompt"]) else None
    if kind == "native_graph":
        stages, edges, start = config.get("stages"), config.get("edges"), config.get("start")
        if not isinstance(stages, dict) or not isinstance(edges, list) or not has_text(start):
            return None
        graph = {"name": name, "start": start, "stages": stages, "edges": edges}
        if isinstance(config.get("max_steps"), int):
            graph["max_steps"] = config["max_steps"]
        return kind, name, graph
    code = config.get("code")
    if not has_text(code):
        return None
    if kind == "native_hook":
        event = config.get("event")
        return (kind, name, {"name": name, "event": event, "code": code}) if event in EVENTS else None
    description, parameters = config.get("description"), config.get("parameters")
    if not has_text(description) or not isinstance(parameters, dict):
        return None
    return kind, name, {"name": name, "description": description, "parameters": parameters, "code": code}


def final_assistant_text(trajectory) -> str | None:
    """The final assistant text in a native-jsonl session: the last assistant/message event with content."""
    for event in reversed(trajectory):
        if event.get("type") != "assistant/message":
            continue
        content = (event.get("data") or {}).get("content")
        if isinstance(content, str) and content.strip():
            return content
    return None
