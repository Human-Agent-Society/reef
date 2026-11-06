"""Validate a harness tree for one adapter, reporting every problem at once.

``validate_tree`` is the one check a tree meets before it is evaluated or
published. It finds every problem in one pass, so a method that calls
``tools.check`` can fix them all before its step ends.
``validate_patch`` checks the tree a patch makes, and refuses an empty patch.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.adapters.parts import Format, TreeLayout, TreeLocation
from reef.harness.tree.checks import (
    reject_inline_secret,
    reject_secret_shaped_text,
    reject_unguarded_console_write,
    validate_native_agent,
    validate_native_graph,
    validate_native_hook,
    validate_native_loop,
    validate_native_tool,
)
from reef.harness.tree.render import RenderError, render_tree
from reef.harness.tree.tree import HarnessTree, TreePatch


@dataclass(frozen=True)
class ValidationResult:
    """The problems validation found in one tree or patch.

    Values: ``problems`` is a tuple of non-empty messages without duplicates,
    in the order the checks found them; each message names the path it is
    about when it is about one path. The tree passed every check if and only
    if ``problems`` is empty.
    Purpose: ``validate_tree`` and ``validate_patch`` make it; the backend
    skips a step whose candidate has problems, ``harness_try`` refuses a
    patch with them, and a method reads them through ``tools.check``.
    """

    problems: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(problem, str) or not problem for problem in self.problems):
            raise ValueError("every validation problem must be a non-empty message")
        if len(set(self.problems)) != len(self.problems):
            raise ValueError("validation problems must not repeat")

    @property
    def passed(self) -> bool:
        return not self.problems

    def message(self) -> str:
        """All problems on one line, for a refusal or a skipped step's reason."""
        return "; ".join(self.problems)


def validate_tree(tree: HarnessTree, descriptor: AdapterDescriptor) -> ValidationResult:
    """Every problem ``tree`` has for ``descriptor``'s adapter.

    Each file must be a path of one of the adapter's parts and pass the checks of the part's format. The native
    files must reference only tools, skills, graphs and agents the tree
    holds, without delegation cycles, and a tree holds at most one native
    loop. A tree with none of those problems must also render. The result
    passes if and only if the tree can be evaluated, published and
    installed.
    """
    layout = descriptor.layout
    problems: list[str] = []
    graphs: dict[str, Mapping[str, Any]] = {}
    agents: dict[str, Mapping[str, Any]] = {}
    for path, text in tree.files.items():
        location = layout.locate(path)
        if location is None:
            taken = ", ".join(part.path for part in layout.parts.values())
            problems.append(f"{path}: not a path of the {descriptor.name} harness tree, which takes {taken}")
            continue
        try:
            parsed = check_file(location, text, descriptor)
        except ValueError as error:
            problems.append(f"{path}: {error}")
            continue
        if location.part.format is Format.NATIVE_GRAPH:
            graphs[location.name] = parsed
        elif location.part.format is Format.NATIVE_AGENT:
            agents[location.name] = parsed
    loops = sorted(path for path in tree.files if format_of(layout, path) is Format.NATIVE_LOOP)
    if len(loops) > 1:
        problems.append(f"one native loop per tree: {', '.join(loops)}")
    problems.extend(native_reference_problems(tree, layout, graphs, agents))
    if not problems:
        try:
            render_tree(tree, descriptor)
        except RenderError as error:
            problems.append(str(error))
    return ValidationResult(tuple(dict.fromkeys(problems)))


def validate_patch(head: HarnessTree, patch: TreePatch, descriptor: AdapterDescriptor) -> ValidationResult:
    """Every problem of ``head.apply(patch)``; an empty patch, or one that changes no text, is a problem too."""
    if not head.diff(head.apply(patch)).changes:
        return ValidationResult(("the patch changes nothing",))
    return validate_tree(head.apply(patch), descriptor)


def check_file(location: TreeLocation, text: str, descriptor: AdapterDescriptor) -> Mapping[str, Any]:
    """Check one file as its part's format; return its parsed JSON object, or an empty mapping for a text format.

    Raises ValueError naming the first problem.
    """
    format = location.part.format
    if format is Format.CONFIG:
        if location.name not in descriptor.config_targets:
            raise ValueError(f"adapter {descriptor.name!r} declares no config target {location.name!r}")
        data = json_object(text)
        reject_inline_secret(data, "config")
        return data
    if not text.strip():
        raise ValueError("the file is empty")
    reject_secret_shaped_text(text, "the file")
    if format is Format.CODE:
        reject_unguarded_console_write(text, "the extension")
    elif format in (Format.NATIVE_TOOL, Format.NATIVE_HOOK):
        # Every NAME the module assigns is the file's, so a file cannot pass for another tool or hook.
        declared_name(text, location.name)
        if format is Format.NATIVE_TOOL:
            validate_native_tool(text, location.name)
        else:
            validate_native_hook(text, location.name)
    elif format is Format.NATIVE_LOOP:
        validate_native_loop({"name": declared_name(text, location.name), "code": text})
    elif format in (Format.NATIVE_GRAPH, Format.NATIVE_AGENT):
        data = json_object(text)
        validator = validate_native_graph if format is Format.NATIVE_GRAPH else validate_native_agent
        validator(data)
        if data.get("name") != location.name:
            raise ValueError(f"the object's name {data.get('name')!r} must be the file name {location.name!r}")
        return data
    return {}


def format_of(layout: TreeLayout, path: str) -> Format | None:
    """The format of the part that takes ``path``, or ``None`` when no part does."""
    location = layout.locate(path)
    return None if location is None else location.part.format


def json_object(text: str) -> Mapping[str, Any]:
    """The JSON object ``text`` holds; ValueError when it is not JSON or not an object."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError(f"must hold a JSON object, got {type(data).__name__}")
    return data


def declared_name(code: str, file_name: str) -> str:
    """The ``NAME`` a native module declares: the file name, which a ``NAME`` literal must repeat if present.

    Raises ValueError when the module does not compile or declares another name.
    """
    try:
        module = ast.parse(code.encode("utf-8"))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"the module does not compile: {exc}") from exc
    for node in module.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        names_module = any(isinstance(target, ast.Name) and target.id == "NAME" for target in node.targets)
        if names_module and node.value.value != file_name:
            raise ValueError(f"NAME {node.value.value!r} must be the file name {file_name!r}")
    return file_name


def native_reference_problems(
    tree: HarnessTree,
    layout: TreeLayout,
    graphs: Mapping[str, Mapping[str, Any]],
    agents: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Names the native graphs and agents use that the tree lacks, and delegation cycles among the agents.

    ``main`` and ``seed`` are always graphs: the loop runs the seed when the
    tree carries no ``main``. An agent delegates to the agents in its
    ``then`` list and to those its graph's subagent stages call; a cycle
    would run without end, so delegation must form a finite tree.
    """
    names: dict[Format, set[str]] = {format: set() for format in Format}
    for path in tree.files:
        location = layout.locate(path)
        if location is not None:
            names[location.part.format].add(location.name)
    graph_names = set(graphs) | {"main", "seed"}
    problems: list[str] = []
    called_by_graph: dict[str, list[str]] = {}
    for graph_name, graph in graphs.items():
        called: list[str] = []
        for stage_name, stage in graph["stages"].items():
            missing = sorted(set(stage.get("allow") or ()) - names[Format.NATIVE_TOOL])
            if missing:
                problems.append(f"native graph {graph_name!r} stage {stage_name!r} allows missing tools: {missing}")
            if stage.get("kind") == "subagent":
                if stage["agent"] not in agents:
                    problems.append(f"native graph {graph_name!r} stage {stage_name!r} calls a missing agent")
                called.append(str(stage["agent"]))
        called_by_graph[graph_name] = called
    calls: dict[str, list[str]] = {"": called_by_graph.get("main", [])}
    for agent_name, agent in agents.items():
        for key, known in (
            ("tools", names[Format.NATIVE_TOOL]),
            ("skills", names[Format.SKILL]),
            ("then", set(agents)),
        ):
            missing = sorted(set(agent.get(key) or ()) - known)
            if missing:
                problems.append(f"native agent {agent_name!r} names missing {key}: {missing}")
        graph_name = str(agent.get("graph", "seed"))
        if graph_name not in graph_names:
            problems.append(f"native agent {agent_name!r} runs a missing graph {graph_name!r}")
        calls[agent_name] = [*(agent.get("then") or ()), *called_by_graph.get(graph_name, [])]
    state: dict[str, str] = {}

    def visit(name: str) -> None:
        state[name] = "open"
        for target in calls.get(name, ()):
            if state.get(target) == "open":
                problems.append(f"native agent {target!r} is called in a cycle; delegation must be a finite tree")
            elif target not in state:
                visit(target)
        state[name] = "done"

    for name in calls:
        if name not in state:
            visit(name)
    return problems


__all__ = ["ValidationResult", "check_file", "validate_patch", "validate_tree"]
