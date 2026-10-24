"""The native host: the registries the loop reads, filled once from a rendered tree.

The interpreter reads a ``NativeHost`` at every use instead of holding the
tools, hooks, agents, graphs, loop and prompt it was built with. The episode
form fills one at boot from the rendered files and never changes it. The
serve form builds a new host for each tree it mounts and swaps it in between
two steps, so the next step runs on the new tree.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reef.harness.runners.native import (
    DEFAULT_SYSTEM_PROMPT,
    HookModule,
    LoadError,
    LoopModule,
    ToolModule,
    context_window_from,
    load_agents,
    load_hooks,
    load_loop,
    load_tools,
)
from reef.harness.runners.native.graph import DEFAULT_CONTEXT_WINDOW, SEED_GRAPH, Graph, GraphError, Host
from reef.harness.tree.checks import NATIVE_EVENTS


class NativeHost(Host):
    """The registries the loop reads.

    Values: tools, hooks, graphs, agents and skills by name, at most one item
    per name in each; at most one loop; the rules text; and the context
    window in tokens. Purpose: ``from_root`` fills one from a rendered tree,
    and the serve form adds its built-in tools; the interpreter reads it at
    every use.
    """

    def __init__(self) -> None:
        self.tool_modules: dict[str, ToolModule] = {}
        self.hook_modules: dict[str, HookModule] = {}
        self.graphs: dict[str, Graph] = {}
        self.agent_options: dict[str, Mapping[str, Any]] = {}
        self.loop_module: LoopModule | None = None
        self.rules = ""
        self.skills: dict[str, str] = {}
        self.window = DEFAULT_CONTEXT_WINDOW

    # -- what the interpreter reads --------------------------------------------------------------------------

    @property
    def tools(self) -> dict[str, ToolModule]:
        """By name, in name order: the order the rendered files load in."""
        return {name: self.tool_modules[name] for name in sorted(self.tool_modules)}

    @property
    def hooks(self) -> dict[str, list[HookModule]]:
        """Every hook under its event in name order, which is the waterfall order and the file name order."""
        hooks: dict[str, list[HookModule]] = {event: [] for event in NATIVE_EVENTS}
        for name in sorted(self.hook_modules):
            hooks[self.hook_modules[name].event].append(self.hook_modules[name])
        return hooks

    @property
    def agents(self) -> dict[str, Mapping[str, Any]]:
        return {name: self.agent_options[name] for name in sorted(self.agent_options)}

    @property
    def loop(self) -> LoopModule | None:
        """The loop code that runs the root turn in place of ``graph("main")``; None when the graph runs."""
        return self.loop_module

    @property
    def context_window(self) -> int:
        return self.window

    def graph(self, name: str = "main") -> Graph:
        """``seed`` is the built in loop and ``main`` falls back to it; any other missing name is a GraphError."""
        if name == "seed" or (name == "main" and "main" not in self.graphs):
            return Graph(SEED_GRAPH, source="seed")
        graph = self.graphs.get(name)
        if graph is None:
            raise GraphError(f"graph {name!r} is missing")
        return graph

    def system_prompt(self, *, skills: Sequence[str] | None = None, prompt: str | None = None) -> str:
        """The rules, then every skill in name order (or the named ones), then an agent's own prompt."""
        parts = [self.rules]
        parts.extend(self.skills[name] for name in sorted(self.skills) if skills is None or name in skills)
        if prompt:
            parts.append(prompt.strip())
        return "\n\n".join(part for part in parts if part) or DEFAULT_SYSTEM_PROMPT

    # -- registration ------------------------------------------------------------------------------------------

    def add_tool(self, tool: ToolModule) -> None:
        if tool.name in self.tool_modules:
            raise LoadError(f"tool {tool.name!r} is already installed; one name, one tool")
        self.tool_modules[tool.name] = tool

    # -- the rendered tree -------------------------------------------------------------------------------------

    @classmethod
    def from_root(cls, root: Path) -> NativeHost:
        """The rendered native directory read once: its tools, hooks, agents, graphs, loop, rules, skills and models.

        A file that cannot load raises LoadError or GraphError. The pinned
        model fields stay in ``models.json``, which the loop reads apart.
        """
        host = cls()
        host.tool_modules = dict(load_tools(root / "tools"))
        for listeners in load_hooks(root / "hooks").values():
            for hook in listeners:
                if hook.name in host.hook_modules:
                    raise LoadError(f"two hook files declare the name {hook.name!r}; one name, one hook")
                host.hook_modules[hook.name] = hook
        host.agent_options = dict(load_agents(root / "agents"))
        graphs_dir = root / "graphs"
        for path in sorted(graphs_dir.glob("*.json")) if graphs_dir.is_dir() else []:
            try:
                host.graphs[path.stem] = Graph(json.loads(path.read_text(encoding="utf-8")), source=path.stem)
            except (OSError, ValueError) as exc:
                raise GraphError(f"graphs/{path.name} cannot run: {exc}") from exc
        host.loop_module = load_loop(root / "loops")
        rules = root / "RULES.md"
        if rules.is_file():
            host.rules = rules.read_text(encoding="utf-8").strip()
        for path in sorted((root / "skills").glob("*/SKILL.md")):
            host.skills[path.parent.name] = path.read_text(encoding="utf-8").strip()
        models = root / "models.json"
        if models.is_file():
            host.window = context_window_from(models)
        return host


__all__ = ["NativeHost"]
