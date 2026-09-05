"""Shared render engine: a composition's nodes to one harness's native files.

Rendering is a pure function of the node sequence and the adapter
descriptor, so a candidate tree can be rendered before and after a mutation
and compared without touching disk. Config nodes deep-merge into their
target in tree order over the descriptor's enforced defaults; rules nodes
concatenate in tree order into the rules file; named kinds render one file
per node through the descriptor's path templates. The descriptor's
``finalize_render`` quirk gets the last word, so adapter traps (opencode's
``autoupdate: false``) are enforced on every rendered tree, not just the
default one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from reef.core.errors import ReefError
from reef.harness.descriptor import AdapterDescriptor


class RenderError(ReefError):
    """The composition cannot be rendered for this adapter."""


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        below = merged.get(key)
        merged[key] = _deep_merge(below, value) if isinstance(below, dict) and isinstance(value, Mapping) else value
    return merged


def _names_of(nodes: Sequence[tuple[str, Any]], kind: str) -> set[str]:
    return {str(config.get("name")) for k, config in nodes if k == kind and isinstance(config, Mapping)}


def _check_native_references(
    nodes: Sequence[tuple[str, Any]], graphs: Sequence[Mapping[str, Any]], agents: Sequence[Mapping[str, Any]]
) -> None:
    """Names a graph or an agent uses exist in the same tree, and agents never call each other in a cycle.

    The checks need every node, so they land here rather than in admission.
    ``main`` is always a graph: the loop runs the seed when the tree carries
    none. The reference graph is an agent's ``then`` list plus the agents its
    graph's subagent stages name; a cycle would run without end, so every
    delegation is a finite tree."""
    tool_names = _names_of(nodes, "native_tool")
    skill_names = _names_of(nodes, "skill")
    # main is always a graph (the loop runs the seed when the tree carries none) and seed is the built in loop.
    graph_names = {str(graph.get("name")) for graph in graphs} | {"main", "seed"}
    agent_names = {str(agent.get("name")) for agent in agents}
    subagents: dict[str, list[str]] = {}
    for graph in graphs:
        called: list[str] = []
        for stage_name, stage in (graph.get("stages") or {}).items():
            if not isinstance(stage, Mapping):
                continue
            missing = sorted(set(stage.get("allow") or ()) - tool_names)
            if missing:
                raise RenderError(
                    f"native_graph stage {stage_name!r} allows tools the tree lacks: {', '.join(missing)}"
                )
            if stage.get("kind") == "subagent":
                if stage.get("agent") not in agent_names:
                    raise RenderError(
                        f"native_graph stage {stage_name!r} calls an agent the tree lacks: {stage.get('agent')}"
                    )
                called.append(str(stage["agent"]))
        subagents[str(graph.get("name"))] = called
    calls: dict[str, list[str]] = {"": subagents.get("main", [])}
    for agent in agents:
        name = str(agent.get("name"))
        for key, names in (("tools", tool_names), ("skills", skill_names), ("then", agent_names)):
            missing = sorted(set(agent.get(key) or ()) - names)
            if missing:
                raise RenderError(f"native_agent {name!r} names {key} the tree lacks: {', '.join(missing)}")
        graph_name = str(agent.get("graph", "seed"))
        if graph_name not in graph_names:
            raise RenderError(f"native_agent {name!r} runs a graph the tree lacks: {graph_name}")
        calls[name] = [*(agent.get("then") or ()), *subagents.get(graph_name, [])]
    state: dict[str, int] = {}

    def visit(name: str) -> None:
        state[name] = 1
        for target in calls.get(name, ()):
            if state.get(target) == 1:
                raise RenderError(f"native_agent {target!r} is called in a cycle; delegation must be a finite tree")
            if state.get(target) is None:
                visit(target)
        state[name] = 2

    for name in calls:
        if state.get(name) is None:
            visit(name)


def render_native_module(kind: str, options: Mapping[str, Any]) -> str:
    """One importable module for a ``native_tool`` or ``native_hook`` node: the code that defines ``run`` or ``listen``, then the declaration as constants, so the node config binds."""
    fields: tuple[tuple[str, Any], ...]
    if kind == "native_hook":
        fields = (("NAME", options.get("name")), ("EVENT", options.get("event")))
    else:
        fields = (
            ("NAME", options.get("name")),
            ("DESCRIPTION", options.get("description", "")),
            ("PARAMETERS", dict(options.get("parameters", {}) or {})),
            ("CAPABILITIES", list(options.get("capabilities", []) or [])),
        )
    header = "\n".join(f"{key} = {value!r}" for key, value in fields)
    return f"{str(options.get('code', '')).rstrip()}\n\n{header}\n"


def render_composition(nodes: Sequence[tuple[str, Any]], descriptor: AdapterDescriptor) -> dict[str, str]:
    """Render ``(kind, config)`` nodes to root-relative file texts."""
    configs = {name: _deep_merge({}, target.defaults) for name, target in descriptor.config_targets.items()}
    rules: list[str] = []
    graphs: list[Mapping[str, Any]] = []
    agents: list[Mapping[str, Any]] = []
    files: dict[str, str] = {}

    def emit(path: str, text: str) -> None:
        if path in files or any(path == target.path for target in descriptor.config_targets.values()):
            raise RenderError(f"two nodes render to the same path {path!r}")
        files[path] = text

    for kind, config in nodes:
        options: Mapping[str, Any] = config if isinstance(config, Mapping) else {}
        if kind == "config":
            target_name = str(options.get("target", "primary"))
            if target_name not in configs:
                raise RenderError(f"adapter {descriptor.name!r} declares no config target {target_name!r}")
            configs[target_name] = _deep_merge(configs[target_name], options.get("data", {}))
        elif kind == "rules":
            rules.append(str(options.get("text", "")).strip())
        elif kind in ("agent_command", "skill", "code_extension"):
            template = descriptor.node_paths.get(kind)
            if template is None:
                raise RenderError(f"adapter {descriptor.name!r} does not render {kind} nodes")
            body = options.get("code", "") if kind == "code_extension" else options.get("text", "")
            emit(template.format(name=options.get("name")), str(body))
        elif kind in ("native_tool", "native_hook"):
            template = descriptor.node_paths.get(kind)
            if template is None:
                raise RenderError(f"adapter {descriptor.name!r} does not render {kind} nodes")
            emit(template.format(name=options.get("name")), render_native_module(kind, options))
        elif kind in ("native_graph", "native_agent"):
            template = descriptor.node_paths.get(kind)
            if template is None:
                raise RenderError(f"adapter {descriptor.name!r} does not render {kind} nodes")
            # Sorted keys, so a proposal's diff against the previous graph is a few lines.
            emit(template.format(name=options.get("name")), json.dumps(options, indent=2, sort_keys=True) + "\n")
            (graphs if kind == "native_graph" else agents).append(options)
        else:
            raise RenderError(f"unknown node kind {kind!r}")

    _check_native_references(nodes, graphs, agents)

    for name, target in descriptor.config_targets.items():
        files[target.path] = json.dumps(configs[name], indent=2, sort_keys=True) + "\n"
    if rules:
        files[descriptor.node_paths["rules"]] = "\n\n".join(rules) + "\n"
    for path, text in files.items():
        if not text.endswith("\n"):
            files[path] = text + "\n"
    if descriptor.finalize_render is not None:
        files = descriptor.finalize_render(files)
        if not isinstance(files, dict):
            raise RenderError(f"adapter {descriptor.name!r} finalize_render must return the files mapping")
    return files
