"""Validate a harness tree for one adapter, reporting every problem at once.

``validate_tree`` is the one check a tree meets before it is evaluated or
published. It finds every problem in one pass, so a proposer can fix them all
in one retry: the backend hands the problems back to the proposer, and an
agent proposer's ``harness_check`` tool shows them during its session.
``validate_patch`` checks the tree a patch makes, and refuses an empty patch.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.tree.checks import (
    NATIVE_RESERVED_TOOL_NAMES,
    reject_inline_secret,
    reject_secret_shaped_text,
    reject_unguarded_console_write,
    validate_native_agent,
    validate_native_graph,
    validate_native_loop,
)
from reef.harness.tree.layout import TreeKind, TreeLocation, extension_suffix, locate
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
    skips a step on problems after one retry, the proposal route refuses a
    patch with them, and a proposer reads them to fix its patch.
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

    Each file must sit on the layout, be of a kind the adapter renders, and
    pass its kind's content checks. The native files must reference only
    tools, skills, graphs and agents the tree holds, without delegation
    cycles, and a tree holds at most one native loop. A tree with none of
    those problems must also render. The result passes if and only if the
    tree can be evaluated, published and installed.
    """
    problems: list[str] = []
    graphs: dict[str, Mapping[str, Any]] = {}
    agents: dict[str, Mapping[str, Any]] = {}
    for path, text in tree.files.items():
        location = locate(path)
        if location is None:
            problems.append(f"{path}: not a path of the harness layout (see reef.harness.tree.layout)")
            continue
        try:
            parsed = check_file(location, text, descriptor)
        except ValueError as error:
            problems.append(f"{path}: {error}")
            continue
        if location.kind is TreeKind.NATIVE_GRAPH:
            graphs[location.name] = parsed
        elif location.kind is TreeKind.NATIVE_AGENT:
            agents[location.name] = parsed
    loops = sorted(path for path in tree.files if path.startswith("native/loops/"))
    if len(loops) > 1:
        problems.append(f"one native loop per tree: {', '.join(loops)}")
    problems.extend(native_reference_problems(tree, graphs, agents))
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
    """Check one file of kind ``location.kind``; return its parsed JSON object, or an empty mapping for text kinds.

    Raises ValueError naming the first problem.
    """
    kind = location.kind
    if kind is TreeKind.CONFIG:
        if location.name not in descriptor.config_targets:
            raise ValueError(f"adapter {descriptor.name!r} declares no config target {location.name!r}")
        data = json_object(text)
        reject_inline_secret(data, "config")
        return data
    if kind.value not in descriptor.kind_paths:
        raise ValueError(f"adapter {descriptor.name!r} does not render {kind} files")
    if kind is TreeKind.CODE_EXTENSION and location.suffix != extension_suffix(descriptor.kind_paths[kind.value]):
        raise ValueError(
            f"adapter {descriptor.name!r} loads {extension_suffix(descriptor.kind_paths[kind.value])} extensions"
        )
    if not text.strip():
        raise ValueError("the file is empty")
    reject_secret_shaped_text(text, "the file")
    if kind is TreeKind.CODE_EXTENSION:
        reject_unguarded_console_write(text, "the extension")
    elif kind in (TreeKind.NATIVE_TOOL, TreeKind.NATIVE_HOOK):
        declared = declared_name(text, location.name)
        if kind is TreeKind.NATIVE_TOOL and declared in NATIVE_RESERVED_TOOL_NAMES:
            raise ValueError(f"tool name {declared!r} is reserved for a built-in tool")
    elif kind is TreeKind.NATIVE_LOOP:
        validate_native_loop({"name": declared_name(text, location.name), "code": text})
    elif kind in (TreeKind.NATIVE_GRAPH, TreeKind.NATIVE_AGENT):
        data = json_object(text)
        validator = validate_native_graph if kind is TreeKind.NATIVE_GRAPH else validate_native_agent
        validator(data)
        if data.get("name") != location.name:
            raise ValueError(f"the object's name {data.get('name')!r} must be the file name {location.name!r}")
        return data
    return {}


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
    tree: HarnessTree, graphs: Mapping[str, Mapping[str, Any]], agents: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Names the native graphs and agents use that the tree lacks, and delegation cycles among the agents.

    ``main`` and ``seed`` are always graphs: the loop runs the seed when the
    tree carries no ``main``. An agent delegates to the agents in its
    ``then`` list and to those its graph's subagent stages call; a cycle
    would run without end, so delegation must form a finite tree.
    """
    names: dict[TreeKind, set[str]] = {kind: set() for kind in TreeKind}
    for path in tree.files:
        location = locate(path)
        if location is not None:
            names[location.kind].add(location.name)
    graph_names = set(graphs) | {"main", "seed"}
    problems: list[str] = []
    called_by_graph: dict[str, list[str]] = {}
    for graph_name, graph in graphs.items():
        called: list[str] = []
        for stage_name, stage in graph["stages"].items():
            missing = sorted(set(stage.get("allow") or ()) - names[TreeKind.NATIVE_TOOL])
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
            ("tools", names[TreeKind.NATIVE_TOOL]),
            ("skills", names[TreeKind.SKILL]),
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
