"""Plugin registry: runtimes, instantiation, and the static load-order gate.

A runtime is the shared record of one plugin callback; a fiber is one
instantiation of it. ``plugin`` realizes the paper's Algorithm 4: the load
is a tracked effect of the parent fiber, so unloading a parent retires its
plugins by plain LIFO replay. ``load`` is a reef addition kept from the
static stage: reactivity makes load order irrelevant (a fiber whose provider
is missing parks PENDING until notify wakes it), so this gate is for
orchestrators that want declaration errors up front instead: it reports
cycles and unprovidable keys before instantiating anything, then loads in
dependency order.
"""

from __future__ import annotations

import graphlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol

from .context import Context
from .fiber import Fiber
from .utils import DisposableList


class MissingProviderError(RuntimeError):
    """Raised by ``load`` when a declared inject has no provider anywhere."""


class ObjectPlugin(Protocol):
    """The object plugin shape: anything with an ``apply(ctx, config)``."""

    def apply(self, ctx: Context, config: Any) -> Any: ...


Plugin = Callable[..., Any] | ObjectPlugin
"""A plugin is a function ``apply(ctx, config)``, a class instantiated as
``cls(ctx, config)``, or an object with an ``apply`` method."""

Inject = Sequence[str] | Mapping[str, Any] | None


class CycleError(RuntimeError):
    """Raised when declared provides and injects form a dependency cycle."""


def _resolve_inject(inject: Inject) -> dict[str, Any]:
    """Normalize a declared inject: a list becomes {name: None}, a mapping keeps per-key config."""
    if inject is None:
        return {}
    if isinstance(inject, Mapping):
        return dict(inject)
    return dict.fromkeys(inject)


def _plugin_label(plugin: Plugin) -> str:
    name = getattr(plugin, "name", None) or getattr(plugin, "__name__", None)
    return name if isinstance(name, str) else type(plugin).__name__


@dataclass
class Runtime:
    """The shared record of one plugin callback across its instantiations."""

    name: str | None
    callback: Callable[..., Any]
    fibers: DisposableList
    Config: Any | None


class RegistryService:
    """One Context tree's plugin runtimes, keyed by resolved callback."""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self._counter = 0
        self._runtimes: dict[Any, Runtime] = {}

    def next_uid(self) -> int:
        self._counter += 1
        return self._counter

    def resolve(self, plugin: Plugin) -> Callable[..., Any] | None:
        """The callable a plugin dedups on: itself, or its ``apply``."""
        if callable(plugin):
            return plugin
        apply = getattr(plugin, "apply", None)
        if callable(apply):
            return apply
        return None

    def get(self, plugin: Plugin) -> Runtime | None:
        callback = self.resolve(plugin)
        return None if callback is None else self._runtimes.get(callback)

    def has(self, plugin: Plugin) -> bool:
        return self.get(plugin) is not None

    def values(self) -> Iterator[Runtime]:
        return iter(list(self._runtimes.values()))

    def delete(self, plugin: Plugin) -> Runtime | None:
        """Remove a runtime and dispose every fiber instantiated from it."""
        callback = self.resolve(plugin)
        runtime = None if callback is None else self._runtimes.pop(callback, None)
        if runtime is None:
            return None
        for fiber in runtime.fibers:
            fiber.dispose()
        return runtime

    def plugin(self, plugin: Plugin, config: Any = None, ctx: Context | None = None) -> Fiber:
        """Instantiate a plugin under ``ctx`` (default: this registry's root).

        Returns the new Fiber immediately; a load failure lands on the fiber
        (state FAILED, error stored, effects recovered) and never propagates
        here. ``await fiber`` or ``fiber.wait()`` re-raises it.
        """
        parent = ctx if ctx is not None else self.ctx
        callback = self.resolve(plugin)
        if callback is None:
            raise TypeError(
                'invalid plugin, expect function or object with an "apply" method, received ' + type(plugin).__name__
            )
        parent.fiber.assert_active()
        runtime = self._runtimes.get(callback)
        if runtime is None:
            name = getattr(plugin, "name", None) or getattr(plugin, "__name__", None)
            if not isinstance(name, str) or name == "apply":
                name = None
            runtime = Runtime(
                name=name, callback=callback, fibers=DisposableList(), Config=getattr(plugin, "Config", None)
            )
            self._runtimes[callback] = runtime
        return Fiber(parent, config, _resolve_inject(getattr(plugin, "inject", None)), runtime)

    def inject(self, deps: Inject, callback: Callable[[Context], Any], ctx: Context | None = None) -> Fiber:
        """Sugar: run a one-argument callback as an anonymous plugin with declared deps."""

        def apply(inner: Context, config: Any) -> Any:
            return callback(inner)

        plugin = SimpleNamespace(inject=deps, apply=apply, name=getattr(callback, "__name__", None))
        return self.plugin(plugin, None, ctx=ctx)

    def load(self, components: Sequence[tuple[Plugin, Any]], ctx: Context | None = None) -> list[Fiber]:
        """Instantiate components in dependency order, from declarations alone.

        Raises CycleError when declared provides and injects form a cycle and
        MissingProviderError when a declared inject has neither a declared
        provider in the batch nor an active one already in the store; nothing
        is instantiated on either report. Fibers return in input order.
        """
        provider_of: dict[str, int] = {}
        infos: list[tuple[Plugin, Any, dict[str, Any]]] = []
        for index, (plugin, config) in enumerate(components):
            declared = getattr(plugin, "provide", None)
            provides = [declared] if isinstance(declared, str) else list(declared or [])
            infos.append((plugin, config, _resolve_inject(getattr(plugin, "inject", None))))
            for key in provides:
                provider_of[key] = index
        sorter: graphlib.TopologicalSorter[int] = graphlib.TopologicalSorter()
        for index, (plugin, _, inject) in enumerate(infos):
            dependencies = []
            for key in inject:
                if key in provider_of:
                    if provider_of[key] != index:
                        dependencies.append(provider_of[key])
                elif self.ctx.reflect._get_impl(key, strict=True) is None:
                    raise MissingProviderError(
                        f'no component provides service "{key}" required by plugin "{_plugin_label(plugin)}"'
                    )
            sorter.add(index, *dependencies)
        try:
            order = list(sorter.static_order())
        except graphlib.CycleError as error:
            path = " -> ".join(_plugin_label(infos[node][0]) for node in error.args[1])
            raise CycleError(f"dependency cycle among plugins: {path}") from error
        fibers: dict[int, Fiber] = {}
        for index in order:
            plugin, config, _ = infos[index]
            fibers[index] = self.plugin(plugin, config, ctx=ctx)
        return [fibers[index] for index in range(len(infos))]
