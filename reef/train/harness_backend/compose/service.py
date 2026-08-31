"""Service base class: constructing one provides it under its name.

Construction registers self as the value of the named service, as a tracked
effect of the constructing context's fiber; the name may come from the
``provide`` class attribute instead of the argument. A ``__compose_check__``
method becomes the binding's check predicate, consulted every time a
dependent resolves the service. ``resolve_config`` reads the layered
intercept tables, so a scope can reconfigure a service without touching it.
``extend`` derives a live view with overriding attributes, and
``__event_filter__`` scopes service events to contexts sharing the
service's realm.

Callable services need no machinery here: the reference builds a callable
proxy because JS objects are not callable, while a Python subclass simply
defines ``__call__``. Init hooks and the init method run at plugin
instantiation; see ``Fiber._instantiate``.
"""

from __future__ import annotations

from typing import Any

from .context import Context


class Service:
    """Provide ``self`` as the service ``name`` on construction."""

    provide: str | None = None
    """Class-level default for the service name (service.ts:19)."""

    def __init__(self, ctx: Context, name: str | None = None) -> None:
        if name is None:
            declared = type(self).provide
            if not isinstance(declared, str):
                raise TypeError("Service needs a name argument or a str `provide` class attribute")
            name = declared
        self.ctx = ctx
        self.name = name
        check = getattr(self, "__compose_check__", None)
        ctx.provide(name, self, check)

    def __getattr__(self, name: str) -> Any:
        # Attribute fallback for extend() views; see extend below.
        base = self.__dict__.get("_extend_base_")
        if base is not None:
            return getattr(base, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __event_filter__(self, ctx: Context) -> bool:
        """Scope service events to contexts that see this service's realm
        (service.ts:37-39)."""
        return ctx._isolate.get(self.name) is self.ctx._isolate.get(self.name)

    def extend(self, **props: Any) -> Any:
        """A live view of this service with ``props`` overriding: reads fall
        back to the base, so later base mutations show through, as the
        reference's prototype-chained extend does (service.ts:41-49)."""
        view = object.__new__(type(self))
        view.__dict__.update(props)
        view.__dict__["_extend_base_"] = self
        return view

    def resolve_config(self, base: Any = None, head: Any = None, ctx: Context | None = None) -> Any:
        """Merge this service's config across the acting scope's intercept
        layers: outermost layer first, nearest layer last, ``base`` before
        all and ``head`` after all, later entries winning (service.ts:51-67).

        ``ctx`` is the acting scope; it defaults to the service's own, and a
        consumer passes its context to see its own overlays (the reference
        reaches the consumer's scope through traceable rebinding instead).
        """
        scope = ctx if ctx is not None else self.ctx
        configs: list[Any] = []
        for layer in scope._intercept.maps:
            if self.name in layer:
                configs.insert(0, layer[self.name])
        if base is not None:
            configs.insert(0, base)
        if head is not None:
            configs.append(head)
        config_cls = getattr(type(self), "Config", None)
        merge = getattr(config_cls, "merge", None)
        if callable(merge):
            return merge(*configs)
        merged: dict[str, Any] = {}
        for config in configs:
            merged.update(config)
        return merged
