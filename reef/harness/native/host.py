"""The host plane's registries: what the native loop consumes, filled once from files or live by the node plugins.

The interpreter reads a ``NativeHost`` at every use instead of holding the
tools, hooks, agents, graphs and prompt it was built with, so a change to the
registries between two steps is what the next step runs on. The episode form
fills one at boot and never changes it: from the root's ``tree.json`` through
the node plugins in ``reef.harness.native.plugins`` when the render carried
one, else from the rendered files; the plugins fill one entry by entry and
take each entry out again when it leaves (RFC #269).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from reef.harness.native import (
    DEFAULT_SYSTEM_PROMPT,
    TREE_FILE,
    HookModule,
    LoadError,
    ToolModule,
    context_window_from,
    hook_from_module,
    import_module_file,
    load_agents,
    load_hooks,
    load_tools,
    tool_from_module,
)
from reef.harness.native.graph import DEFAULT_CONTEXT_WINDOW, Graph, GraphError
from reef.harness.native.seed import SEED_GRAPH
from reef.harness.nodes import NATIVE_EVENTS
from reef.harness.render import render_native_module

Remover = Callable[[], None]
_MODULE_DIRS = {"native_tool": "tools", "native_hook": "hooks"}


def _graph_from_file(path: Path) -> Graph:
    try:
        return Graph(json.loads(path.read_text(encoding="utf-8")), source=path.stem)
    except (OSError, ValueError) as exc:
        raise GraphError(f"graphs/{path.name} cannot run: {exc}") from exc


class TreeOrder(Protocol):
    """The tree's entry ids in tree order, read when order matters, so a reorder without a config change counts."""

    def ids(self) -> Sequence[str]: ...


class NativeHost:
    """The registries the loop reads; every ``add`` returns the call that takes the item out again."""

    def __init__(self, mount_dir: Path | None = None, order: TreeOrder | None = None) -> None:
        #: Where ``mount_module`` writes tool and hook modules; the episode form imports the rendered files instead.
        self.mount_dir = mount_dir
        #: Where rules and windows take their order from; without one, the order they were added in.
        self.order = order
        #: The compose loader a tree boot reconciled the entries through, so a later mount is one more
        #: ``root.update``; None when the host was filled from the rendered files.
        self.loader: Any = None
        self._tools: dict[str, ToolModule] = {}
        self._hooks: dict[str, HookModule] = {}
        self._graphs: dict[str, Graph] = {}
        self._agents: dict[str, Mapping[str, Any]] = {}
        self._rules: dict[str, str] = {}
        self._skills: dict[str, str] = {}
        self._windows: dict[str, int] = {}

    def _ordered(self, keys: Iterable[str]) -> list[str]:
        """``keys`` in tree order when the host has one, keys the tree does not name last, else as added."""
        added = list(keys)
        if self.order is None:
            return added
        rank: dict[str, int] = {}
        for index, id_ in enumerate(self.order.ids()):
            rank.setdefault(id_, index)  # an id listed twice counts where it first appears
        return sorted(added, key=lambda key: (rank.get(key, len(rank)), added.index(key)))

    # -- what the interpreter reads --------------------------------------------------------------------------

    @property
    def tools(self) -> dict[str, ToolModule]:
        """By name, in name order: the order the rendered files load in."""
        return {name: self._tools[name] for name in sorted(self._tools)}

    @property
    def hooks(self) -> dict[str, list[HookModule]]:
        """Every hook under its event in name order, which is the waterfall order and the file name order."""
        hooks: dict[str, list[HookModule]] = {event: [] for event in NATIVE_EVENTS}
        for name in sorted(self._hooks):
            hooks[self._hooks[name].event].append(self._hooks[name])
        return hooks

    @property
    def agents(self) -> dict[str, Mapping[str, Any]]:
        return {name: self._agents[name] for name in sorted(self._agents)}

    @property
    def context_window(self) -> int:
        """The ``context_window`` of the last models config in tree order, else the default; the render merges the same way."""
        keys = self._ordered(self._windows)
        return self._windows[keys[-1]] if keys else DEFAULT_CONTEXT_WINDOW

    def graph(self, name: str = "main") -> Graph:
        """``seed`` is the built in loop and ``main`` falls back to it; any other missing name is a GraphError."""
        if name == "seed" or (name == "main" and "main" not in self._graphs):
            return Graph(SEED_GRAPH, source="seed")
        graph = self._graphs.get(name)
        if graph is None:
            raise GraphError(f"graph {name!r} is missing")
        return graph

    def system_prompt(self, *, skills: Sequence[str] | None = None, prompt: str | None = None) -> str:
        """Rules in tree order, then every skill in name order (or the named ones), then an agent's own prompt."""
        parts = [self._rules[key] for key in self._ordered(self._rules)]
        parts.extend(self._skills[name] for name in sorted(self._skills) if skills is None or name in skills)
        if prompt:
            parts.append(prompt.strip())
        return "\n\n".join(part for part in parts if part) or DEFAULT_SYSTEM_PROMPT

    # -- registration; each call returns its inverse -----------------------------------------------------------

    def add_tool(self, tool: ToolModule) -> Remover:
        if tool.name in self._tools:
            raise LoadError(f"tool {tool.name!r} is already installed; one name, one tool")
        self._tools[tool.name] = tool

        def remove() -> None:
            if self._tools.get(tool.name) is tool:
                self._tools.pop(tool.name)

        return remove

    def add_hook(self, hook: HookModule) -> Remover:
        if hook.name in self._hooks:
            raise LoadError(f"hook {hook.name!r} is already installed; one name, one hook")
        self._hooks[hook.name] = hook

        def remove() -> None:
            if self._hooks.get(hook.name) is hook:
                self._hooks.pop(hook.name)

        return remove

    def add_graph(self, graph: Graph) -> Remover:
        """Keyed by ``source``: the file stem in the episode form, the node name in the live one."""
        if graph.source in self._graphs:
            raise GraphError(f"graph {graph.source!r} is already installed; one name, one graph")
        self._graphs[graph.source] = graph

        def remove() -> None:
            if self._graphs.get(graph.source) is graph:
                self._graphs.pop(graph.source)

        return remove

    def add_agent(self, options: Mapping[str, Any]) -> Remover:
        name = str(options["name"])
        if name in self._agents:
            raise LoadError(f"agent {name!r} is already installed; one name, one agent")
        self._agents[name] = options

        def remove() -> None:
            if self._agents.get(name) is options:
                self._agents.pop(name)

        return remove

    def add_rule(self, key: str, text: str) -> Remover:
        """One rules section under ``key``, the entry id the tree order names; a later add under the same key replaces it."""
        section = self._rules[key] = text.strip()

        def remove() -> None:
            if self._rules.get(key) is section:
                self._rules.pop(key)

        return remove

    def add_skill(self, name: str, text: str) -> Remover:
        if name in self._skills:
            raise LoadError(f"skill {name!r} is already installed; one name, one skill")
        self._skills[name] = text.strip()

        def remove() -> None:
            self._skills.pop(name, None)

        return remove

    def set_context_window(self, key: str, value: int) -> Remover:
        """The window under ``key``; the last one set wins while it stands."""
        self._windows[key] = value

        def remove() -> None:
            self._windows.pop(key, None)

        return remove

    def mount_module(self, kind: str, options: Mapping[str, Any]) -> Remover:
        """Write a tool or hook node's module under the mount directory, import it and register it.

        A failure at any step leaves nothing behind; the inverse unregisters
        the module and removes the file."""
        if self.mount_dir is None:
            raise LoadError("the native host has no mount directory to write tool and hook modules under")
        name = str(options["name"])
        path = self.mount_dir / _MODULE_DIRS[kind] / f"{name}.py"
        if path.exists():
            # The file is the installed module of the entry that owns the name; a twin fails before touching it.
            raise LoadError(f"{kind} {name!r} is already installed; one name, one module")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_native_module(kind, options), encoding="utf-8")
        try:
            module = import_module_file(path, f"reef_{kind}")
            if kind == "native_tool":
                remove = self.add_tool(tool_from_module(path, module))
            else:
                remove = self.add_hook(hook_from_module(path, module))
        except BaseException:
            path.unlink(missing_ok=True)
            raise

        def uninstall() -> None:
            remove()
            path.unlink(missing_ok=True)
            # The execute seed tool imports its siblings in a subprocess, which caches their bytecode beside them.
            cache = path.parent / "__pycache__"
            for stale in cache.glob(f"{path.stem}.*.pyc"):
                stale.unlink(missing_ok=True)
            for directory in (cache, path.parent):
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()

        return uninstall

    # -- the episode form ----------------------------------------------------------------------------------------

    @classmethod
    def from_root(cls, root: Path, mount_dir: Path | None = None) -> NativeHost:
        """The rendered root read once at boot: its entries list through the node plugins when it carries one, else its files.

        A tree boot needs ``mount_dir`` for the tool and hook modules and
        needs every entry ACTIVE; a file that cannot load raises LoadError or
        GraphError either way. The pinned model fields stay host state in
        ``models.json`` whichever way the host was filled."""
        tree = root / TREE_FILE
        if tree.is_file():
            return cls._from_tree(tree, mount_dir)
        host = cls()
        for tool in load_tools(root / "tools").values():
            host.add_tool(tool)
        for listeners in load_hooks(root / "hooks").values():
            for hook in listeners:
                host.add_hook(hook)
        for options in load_agents(root / "agents").values():
            host.add_agent(options)
        graphs_dir = root / "graphs"
        for path in sorted(graphs_dir.glob("*.json")) if graphs_dir.is_dir() else []:
            host.add_graph(_graph_from_file(path))
        rules = root / "RULES.md"
        if rules.is_file():
            host.add_rule("RULES.md", rules.read_text(encoding="utf-8"))
        for path in sorted((root / "skills").glob("*/SKILL.md")):
            host.add_skill(path.parent.name, path.read_text(encoding="utf-8"))
        models = root / "models.json"
        if models.is_file():
            host.set_context_window("models.json", context_window_from(models))
        return host

    @classmethod
    def _from_tree(cls, tree: Path, mount_dir: Path | None) -> NativeHost:
        """One fresh compose context over the entries list; the loader stays on the host for later mounts."""
        # Late: the training package imports the harness, and the episode form pays for it only on a tree boot.
        from reef.harness.native.plugins import NATIVE_PLUGINS
        from reef.train.cordis_backend.compose import Context
        from reef.train.cordis_backend.compose.loader import Loader

        entries = tree_entries(tree)
        if mount_dir is None:
            raise LoadError(f"{tree.name} needs a mount directory for its tool and hook modules")
        ctx = Context()
        host = cls(mount_dir=mount_dir)
        ctx.provide("native", host)
        loader = Loader(ctx, NATIVE_PLUGINS.get)
        loader.root.update(entries)
        failure = tree_failure(loader)
        if failure is not None:
            # A boot that fails leaves nothing behind: the siblings that did load come out with their modules.
            loader.root.update([])
            raise LoadError(f"{tree.name} entry {failure[0]!r} cannot load: {failure[1]}")
        host.loader = loader
        return host


def tree_entries(path: Path) -> list[dict[str, Any]]:
    """The entries list a ``tree.json`` carries: a JSON array of objects with a string ``id`` and ``name``."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LoadError(f"{path.name} cannot be read: {exc}") from exc
    if not isinstance(data, list) or not all(isinstance(entry, dict) for entry in data):
        raise LoadError(f"{path.name} must be a JSON array of entry objects")
    for entry in data:
        for key in ("id", "name"):
            if not isinstance(entry.get(key), str) or not entry[key]:
                raise LoadError(f"{path.name} entry {entry!r} requires a non-empty string {key!r}")
    return [dict(entry) for entry in data]


def tree_failure(loader: Any) -> tuple[str, str] | None:
    """The first enabled entry that did not end ACTIVE, as (id, what went wrong); None when every entry stands."""
    from reef.train.cordis_backend.compose import FiberState  # late: see _from_tree

    for options in loader.root.data:
        entry = loader.store.get(str(options.get("id")))
        if entry is None or entry.disabled:
            continue
        kind = str(entry.options.get("name"))
        if entry.fiber is None:
            return entry.id, f"no plugin for kind {kind}"
        if entry.fiber.state is not FiberState.ACTIVE:
            return (
                entry.id,
                f"{kind}: {entry.fiber.error if entry.fiber.error is not None else entry.fiber.state.name}",
            )
    return None
