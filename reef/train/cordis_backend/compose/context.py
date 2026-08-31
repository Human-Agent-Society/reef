"""Context: the entry point of the calculus, with property-mediated access.

A Context is a view onto one composition tree: reading an unknown attribute
resolves it as a service through the walk of the paper's Algorithm 6, and
writing one is guarded so only a provider mutates its own service. Python's
attribute protocol is the interposition primitive: ``__getattr__`` fires
only after normal lookup fails, which gives the reference proxy's fast path
for free. ``extend`` derives a child view that shares the tree but carries
its own overrides; recovery of a derived context is discarding it.
``isolate`` remaps one service name to a private realm for the derived
subtree; ``intercept`` overlays per-scope config that services resolve at
use time through ``Service.resolve_config``.

Access control, stated plainly: reading key K on a context succeeds iff
some fiber on its lineage (itself, an ancestor, or root) holds K in its
committed store, because it provides K or declared and committed K at load,
and the lineage does not cross a realm boundary for K on the way. A
declared but uncommitted K is the inactive error; a K nowhere on the
reachable lineage is the undeclared error, enforced at the point of use.

The whole package assumes one thread and, when asyncio is used, one event
loop: composition state interleaves only at iteration boundaries, so no
locks are taken. Free-threaded Python and multi-loop use are out of scope.
"""

from __future__ import annotations

import logging
from collections import ChainMap
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from typing import Any


class ServiceAccessError(AttributeError):
    """Undeclared, inactive, or foreign service access through a Context."""


class RealmKey:
    """One realm's identity for one service name (the reference's realm symbol)."""

    __slots__ = ("label",)

    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:
        return f"<realm {self.label}>"


class Context:
    """One view onto a composition tree; the root view owns the services."""

    fiber: Fiber
    root: Context
    events: EventsService
    reflect: ReflectService
    registry: RegistryService
    logger: logging.Logger

    _parent: Context | None
    _overrides: dict[str, Any]
    _isolate: ChainMap[str, Any]
    _intercept: ChainMap[str, Any]

    def __init__(self) -> None:
        # Lazy imports: these sibling modules import ``Context`` for type
        # annotations, so importing them at module level here would create a
        # circular dependency.  Deferring to call time breaks the cycle —
        # and ``Context()`` runs only once (for the root), so the overhead
        # is negligible.
        from .events import EventsService
        from .fiber import Fiber
        from .reflect import ReflectService
        from .registry import RegistryService

        self._parent = None
        self._overrides = {}
        # The root layers of the two overlay tables: realm defaults land in
        # _isolate at first provide, and every derived view chains onto them.
        self._isolate = ChainMap()
        self._intercept = ChainMap()
        own = self._overrides
        own["root"] = self
        fiber = Fiber(self, None, {}, None)
        own["fiber"] = fiber
        own["reflect"] = ReflectService(self)
        own["registry"] = RegistryService(self)
        own["events"] = EventsService(self)
        own["logger"] = logging.getLogger("reef.train.cordis_backend.compose")
        # Builtin registrations are not revertible on the root (context.ts:47).
        fiber._disposables.clear()

    def extend(self, **meta: Any) -> Context:
        """Derive a child view carrying ``meta`` as its own overrides."""
        derived = object.__new__(type(self))
        object.__setattr__(derived, "_parent", self)
        object.__setattr__(derived, "_overrides", dict(meta))
        object.__setattr__(derived, "_isolate", self._isolate.new_child())
        object.__setattr__(derived, "_intercept", self._intercept.new_child())
        return derived

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        node: Context | None = self
        while node is not None:
            overrides = node.__dict__.get("_overrides")
            if overrides and name in overrides:
                return overrides[name]
            node = node.__dict__.get("_parent")
        if self.fiber.runtime is not None and self.events._hooks.get("internal/get"):
            # The interposition seam: listeners may serve or veto the read
            # before the walk runs (reflect.ts:80). Root reads bypass it,
            # as upstream.
            return self.events.waterfall("internal/get", self, name, lambda *_: self._service_walk(name))
        return self._service_walk(name)

    def _service_walk(self, name: str) -> Any:
        """Algorithm 6: resolve ``name`` against the fiber lineage's
        committed stores, stopping at a realm boundary (reflect.ts:81-93)."""
        key = self.root._isolate.get(name)
        fiber = self.fiber
        while True:
            store = fiber.store
            if store is not None and name in store:
                return store[name].value
            if name in fiber.inject:
                raise ServiceAccessError(f'cannot get required service "{name}" in inactive context')
            if fiber.runtime is None:
                raise ServiceAccessError(f'cannot get property "{name}" without inject')
            if fiber.parent._isolate.get(name) is not key:
                raise ServiceAccessError(f'cannot get property "{name}" without inject')
            fiber = fiber.parent.fiber

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if name in self.reflect.props:
            if self.events._hooks.get("internal/set"):
                self.events.waterfall(
                    "internal/set", self, name, value, lambda *_: self.reflect.set(name, value, ctx=self)
                )
                return
            self.reflect.set(name, value, ctx=self)
            return
        if self.fiber.runtime is None:
            # Plain sets pass through on the root, and derived views see them
            # via the override walk, as prototype inheritance did upstream.
            self._overrides[name] = value
            return
        raise ServiceAccessError(f'cannot set property "{name}" without provide')

    def isolate(self, name: str, label: RealmKey | None = None) -> Context:
        """Derive a view where ``name`` lives in a private realm: provides
        and reads under it cannot collide with, or leak to, the rest of the
        tree. Passing the same ``label`` to two isolates joins their realms.
        """
        realm = label if label is not None else RealmKey(name)
        derived = self.extend()
        object.__setattr__(derived, "_isolate", self._isolate.new_child({name: realm}))
        return derived

    def intercept(self, name: str, config: Any) -> Context:
        """Derive a view overlaying per-scope config for the service
        ``name``, resolved at use time by ``Service.resolve_config``."""
        derived = self.extend()
        object.__setattr__(derived, "_intercept", self._intercept.new_child({name: config}))
        return derived

    def __repr__(self) -> str:
        return f"Context <{self.fiber.name}>"

    def effect(self, run: Callable[[], EffectResult], label: str = "anonymous") -> EffectHandle:
        """Track one effect on this context's fiber."""
        return self.fiber.effect(run, label)

    def enter(
        self,
        cm: AbstractContextManager[Any] | AbstractAsyncContextManager[Any],
        label: str = "ctx.enter()",
    ) -> EffectHandle:
        """Enter a (a)sync context manager and collect its exit as the disposer."""
        if isinstance(cm, AbstractAsyncContextManager):

            async def acquire() -> Callable[[], Any]:
                await cm.__aenter__()
                return lambda: cm.__aexit__(None, None, None)

            return self.effect(lambda: acquire(), label)

        def acquire_sync() -> Callable[[], None]:
            cm.__enter__()

            def release() -> None:
                cm.__exit__(None, None, None)

            return release

        return self.effect(acquire_sync, label)

    def get(self, name: str, strict: bool = True) -> Any:
        return self.reflect.get(name, strict, ctx=self)

    def set(self, name: str, value: Any) -> None:
        self.reflect.set(name, value, ctx=self)

    def provide(self, name: str, value: Any = None, check: Callable[[Context], Any] | None = None) -> EffectHandle:
        return self.reflect.provide(name, value, check, ctx=self)

    def plugin(self, plugin: Plugin, config: Any = None) -> Fiber:
        return self.registry.plugin(plugin, config, ctx=self)

    def inject(self, deps: Inject, callback: Callable[..., Any]) -> Fiber:
        return self.registry.inject(deps, callback, ctx=self)

    def on(
        self, name: str, listener: Callable[..., Any], *, prepend: bool = False, glob: bool = False
    ) -> EffectHandle:
        return self.events.on(name, listener, prepend=prepend, glob=glob, ctx=self)

    def once(self, name: str, listener: Callable[..., Any], *, prepend: bool = False) -> EffectHandle:
        return self.events.once(name, listener, prepend=prepend, ctx=self)

    def emit(self, *args: Any) -> None:
        self.events.emit(*args)

    async def parallel(self, *args: Any) -> None:
        await self.events.parallel(*args)

    async def serial(self, *args: Any) -> Any:
        return await self.events.serial(*args)

    def bail(self, *args: Any) -> Any:
        return self.events.bail(*args)

    def waterfall(self, *args: Any) -> Any:
        return self.events.waterfall(*args)


# Type-only imports for annotations.  Placed after the ``Context`` class
# definition so that sibling modules can import ``Context`` at their top
# without triggering a circular import.  These names appear only in
# annotations (made lazy by ``from __future__ import annotations``); the
# runtime references in method bodies use lazy imports inside ``__init__``
# or access services through ``self.<name>`` which resolves at call time.
from .events import EventsService
from .fiber import EffectHandle, EffectResult, Fiber
from .reflect import ReflectService
from .registry import Inject, Plugin, RegistryService
