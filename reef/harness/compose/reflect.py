"""Service registry half of the calculus: provide, strict read, guarded write.

Every key has at most one provider per realm (the paper's Definition 45);
providing is an ordinary tracked effect of the provider's fiber, so a service
withdraws exactly when its provider unloads. Reads resolve strictly: a
binding counts only while its provider fiber is ACTIVE and its check
predicate, if any, passes. Undeclared and inactive access raise through
ServiceAccessError, which subclasses AttributeError so ``hasattr`` degrades
correctly on Context.

The store is keyed by realm (Definitions 28-29): every name defaults to one
root realm, and ``Context.isolate`` remaps a name to a private realm for a
subtree, so the same key can have a different provider on each side of the
boundary. ``notify`` is the reactive edge (the paper's Algorithm 3): every
registration, withdrawal, and provider state crossing re-resolves the fibers
that declared the name, and their epoch machinery restarts whoever's
provider actually changed.

The acting scope is an explicit ``ctx`` parameter throughout: the reference
rebinds ``this.ctx`` through its traceable proxy machinery, which is
permanently omitted (see UPSTREAM.md), so Context's delegating methods pass
themselves instead.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .context import Context, RealmKey, ServiceAccessError
from .fiber import EffectHandle, Fiber, FiberState


@dataclass
class Impl:
    """One service binding: name, providing fiber, current value, and the
    optional check predicate consulted at resolution time with the
    resolving scope as its argument."""

    name: str
    fiber: Fiber
    value: Any
    check: Callable[[Context], Any] | None = None


class ReflectService:
    """The shared service store and the property model behind Context access."""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.store: dict[RealmKey, Impl] = {}
        self.props: dict[str, str] = {}

    def get(self, name: str, strict: bool = True, ctx: Context | None = None) -> Any:
        """Read a service value; None when absent or, strictly, inactive."""
        impl = self._get_impl(name, strict, ctx=ctx)
        return None if impl is None else impl.value

    def _get_impl(self, name: str, strict: bool = True, ctx: Context | None = None) -> Impl | None:
        scope = ctx if ctx is not None else self.ctx
        key = scope._isolate.get(name)
        impl = None if key is None else self.store.get(key)
        if impl is None:
            return None
        if strict and impl.fiber.state is not FiberState.ACTIVE:
            return None
        return impl

    def set(self, name: str, value: Any, ctx: Context | None = None) -> None:
        """Overwrite a service value in place; only the provider fiber may.

        An in-place overwrite is deliberately not a notify edge: dependents
        keep the binding, and only identity changes (provide, withdraw,
        provider state crossings) restart them.
        """
        scope = ctx if ctx is not None else self.ctx
        key = scope._isolate.get(name)
        impl = None if key is None else self.store.get(key)
        if impl is None:
            raise ServiceAccessError(f'cannot set property "{name}" without provide')
        if impl.fiber is not scope.fiber:
            raise ServiceAccessError(f'cannot set property "{name}" in multiple fibers')
        impl.value = value

    def provide(
        self,
        name: str,
        value: Any = None,
        check: Callable[[Context], Any] | None = None,
        ctx: Context | None = None,
    ) -> EffectHandle:
        """Register a service as a tracked effect of the providing fiber.

        The forward edge notifies dependents when the provider is already
        ACTIVE (a provide during load is announced by the LOADING to ACTIVE
        crossing instead). The inverse withdraws the binding, notifies, and
        waits for the woken dependents to finish unloading before releasing
        the provider's own committed entry, so a dependent's disposers can
        still read the service they depended on (the Theorem 63 ordering).
        """
        scope = ctx if ctx is not None else self.ctx
        owner = scope.fiber

        def forward() -> Any:
            self.props[name] = "service"
            root_realm = scope.root._isolate
            root_realm.setdefault(name, RealmKey(name))
            key = scope._isolate[name]
            existing = self.store.get(key)
            if existing is not None:
                raise ServiceAccessError(f'service "{name}" has been registered at <{existing.fiber.name}>')
            impl = Impl(name=name, fiber=owner, value=value, check=check)
            self.store[key] = impl
            if owner.store is None:
                raise RuntimeError("service owner has no provider store")
            owner.store[name] = impl
            if owner.state is FiberState.ACTIVE:
                self.notify([name], ctx=scope)

            def inverse() -> Any:
                self.store.pop(key, None)
                dependents = self.notify([name], ctx=scope)

                def release() -> None:
                    # The provider's own committed entry goes last: self
                    # access stays valid while dependents tear down.
                    if owner.store is not None:
                        owner.store.pop(name, None)

                pending = [fiber for fiber in dependents if fiber._inertia is not None]
                if not pending:
                    release()
                    return None

                async def drain() -> None:
                    for fiber in pending:
                        with contextlib.suppress(BaseException):  # allSettled: a dependent's failure is its own
                            await fiber.wait()
                    release()

                return drain()

            return inverse

        return owner.effect(forward, f'ctx.provide("{name}")')

    def notify(
        self,
        names: list[str],
        ctx: Context | None = None,
        filter: Callable[[Context, str], bool] | None = None,
    ) -> list[Fiber]:
        """Re-resolve every fiber that declared one of ``names`` and let its
        epoch machinery react; returns the fibers whose declarations matched.

        ``ctx`` is the acting scope whose realm view defines the default
        filter: only fibers seeing the name at the same realm are touched.
        Also emits 'internal/service' per name through a receiver carrying
        the filter, so listeners are scoped the same way (reflect.ts:205-227).
        """
        scope = ctx if ctx is not None else self.ctx
        matches = filter
        if matches is None:

            def matches(target: Context, name: str) -> bool:
                return target._isolate.get(name) is scope._isolate.get(name)

        fibers: list[Fiber] = []
        for runtime in list(self.ctx.registry.values()):
            for fiber in list(runtime.fibers):
                has_update = False
                for name in names:
                    if name not in fiber.inject:
                        continue
                    if not matches(fiber.ctx, name):
                        continue
                    has_update = True
                    fiber._check_impl(name)
                if not has_update:
                    continue
                fiber._refresh()
                fibers.append(fiber)
        for name in names:
            receiver = scope.extend()
            object.__setattr__(receiver, "__event_filter__", lambda target, name=name: matches(target, name))
            self.ctx.events.emit(receiver, "internal/service", name, self.get(name, strict=False, ctx=scope))
        return fibers
