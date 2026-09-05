"""Node plugins with effects: the native tree as a live composition.

``NODE_KINDS`` admits an entry and installs nothing, because the gate reads
the tree back out through the renderer. The plugins here admit an entry the
same way, then register one effect that installs what the loop consumes into
the ``native`` service (a ``NativeHost``) and takes it out again when the
entry leaves or its config changes, so a loader over ``NATIVE_PLUGINS`` is a
composition a resident process reconciles entry by entry (RFC #269). The
kinds the native loop never reads (``agent_command``, ``code_extension``)
have no plugin here: a tree that carries one fails to resolve at the boundary
instead of installing nothing.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

from reef.harness import nodes
from reef.harness.native.graph import Graph
from reef.harness.native.host import NativeHost

#: The models config fields the host pins from the installed binding; a tree entry never sets them.
PINNED_MODEL_FIELDS = ("api", "base_url", "api_key", "model")


class LoaderOrder:
    """The root group's entry ids in tree order, read live, so the host orders rules and windows as the render does.

    The first rules or config plugin a loader loads hands this to its host."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader

    def ids(self) -> Sequence[str]:
        return [str(options.get("id")) for options in self._loader.root.data]


def _key(ctx: Any, host: NativeHost) -> str:
    """The entry's id, the key the tree order names, and the host learns the tree's order from the entry's loader.

    A plugin loaded outside a loader stands alone under its fiber id."""
    from reef.train.cordis_backend.compose.loader import entry_of  # late: the training package imports the harness

    entry = entry_of(ctx)
    if entry is None:
        return f"fiber:{ctx.fiber.uid}"
    if host.order is None:
        host.order = LoaderOrder(entry.loader)
    return entry.id


class NodePlugin:
    """One node kind as a compose plugin: ``apply`` admits the config, then registers the effect that installs it."""

    name = ""
    inject = ("native",)

    def apply(self, ctx: Any, config: Any) -> None:
        raise NotImplementedError


class NativeToolPlugin(NodePlugin):
    name = "native_tool"

    def apply(self, ctx: Any, config: Any) -> None:
        nodes.native_tool_node(ctx, config)
        host: NativeHost = ctx.native
        ctx.effect(lambda: host.mount_module("native_tool", config), f"native_tool {config['name']}")


class NativeHookPlugin(NodePlugin):
    name = "native_hook"

    def apply(self, ctx: Any, config: Any) -> None:
        nodes.native_hook_node(ctx, config)
        host: NativeHost = ctx.native
        ctx.effect(lambda: host.mount_module("native_hook", config), f"native_hook {config['name']}")


class NativeGraphPlugin(NodePlugin):
    name = "native_graph"

    def apply(self, ctx: Any, config: Any) -> None:
        # A copy: the host's graph must not alias the entry's config, which the loader diffs by value.
        options = copy.deepcopy(nodes.validate_native_graph(config))
        graph = Graph(options, source=str(options["name"]))
        host: NativeHost = ctx.native
        ctx.effect(lambda: host.add_graph(graph), f"native_graph {graph.source}")


class NativeAgentPlugin(NodePlugin):
    name = "native_agent"

    def apply(self, ctx: Any, config: Any) -> None:
        options = copy.deepcopy(nodes.validate_native_agent(config))
        host: NativeHost = ctx.native
        ctx.effect(lambda: host.add_agent(options), f"native_agent {options['name']}")


class RulesPlugin(NodePlugin):
    name = "rules"

    def apply(self, ctx: Any, config: Any) -> None:
        nodes.rules_node(ctx, config)
        host: NativeHost = ctx.native
        key = _key(ctx, host)
        ctx.effect(lambda: host.add_rule(key, str(config["text"])), f"rules {key}")


class SkillPlugin(NodePlugin):
    name = "skill"

    def apply(self, ctx: Any, config: Any) -> None:
        nodes.skill_node(ctx, config)
        host: NativeHost = ctx.native
        ctx.effect(lambda: host.add_skill(str(config["name"]), str(config["text"])), f"skill {config['name']}")


class ConfigPlugin(NodePlugin):
    """A models config may set ``context_window``; the pinned binding fields and the other targets are refused."""

    name = "config"

    def apply(self, ctx: Any, config: Any) -> None:
        nodes.config_node(ctx, config)
        target = config.get("target", "primary")
        if target != "models":
            raise ValueError(
                f"config node target {target!r} renders to a file the native loop never reads; the host reads "
                "target 'models' only"
            )
        data = config["data"]
        pinned = sorted(set(data) & set(PINNED_MODEL_FIELDS))
        if pinned:
            raise ValueError(
                f"config node target 'models' cannot set {', '.join(pinned)}: the host pins the binding from the "
                "installed models.json"
            )
        window = data.get("context_window")
        if window is None:
            return
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            raise ValueError("config node 'context_window' must be a positive integer")
        host: NativeHost = ctx.native
        key = _key(ctx, host)
        ctx.effect(lambda: host.set_context_window(key, int(window)), f"config {key}")


NATIVE_PLUGINS: dict[str, NodePlugin] = {
    plugin.name: plugin
    for plugin in (
        ConfigPlugin(),
        RulesPlugin(),
        SkillPlugin(),
        NativeToolPlugin(),
        NativeHookPlugin(),
        NativeGraphPlugin(),
        NativeAgentPlugin(),
    )
}
"""Entry ``name`` to effect registering plugin; the resolver of a live native composition."""

__all__ = ["NATIVE_PLUGINS", "PINNED_MODEL_FIELDS", "LoaderOrder", "NodePlugin"]
