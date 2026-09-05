"""packages/core/tests/reflect.spec.ts -- describe('Reflect')."""

from __future__ import annotations

import pytest

from reef.train.cordis_backend.compose import Context, ServiceAccessError

from .utils import anonymous, apply, upstream

SPEC = "core/tests/reflect.spec.ts"


@upstream(SPEC, "access check")
def test_access_check() -> None:
    root = Context()

    def probe_unprovided(ctx: Context, config: object = None) -> None:
        # The reference probes 'prototype' and 'constructor' to show the trap
        # does not intercept host properties. Python's __getattr__ fires only
        # after normal lookup fails, so the equivalent probe is a real
        # attribute (UPSTREAM.md, context.py <- context.ts).
        assert ctx.__class__ is Context
        with pytest.raises(ServiceAccessError, match='cannot get property "bar" without inject'):
            _ = ctx.bar
        with pytest.raises(ServiceAccessError, match='cannot set property "bar" without provide'):
            ctx.bar = 0

    apply(root, anonymous(probe_unprovided))

    def probe_provide(ctx: Context, config: object = None) -> None:
        with pytest.raises(ServiceAccessError, match='cannot set property "foo" without provide'):
            ctx.foo = 0
        ctx.provide("foo")
        with pytest.raises(ServiceAccessError, match='service "foo" has been registered at <root>'):
            ctx.provide("foo")
        ctx.foo = 0

    apply(root, anonymous(probe_provide))


@upstream(SPEC, "service injection")
def test_service_injection() -> None:
    # The reference also asserts that a mixin name reads as undefined through
    # ctx.get and that no warning is logged; both belong to the mixin and
    # traceable machinery, permanent omissions (UPSTREAM.md), so this mirrors
    # the service/property halves only.
    root = Context()
    root.provide("foo")
    root.set("foo", {"bar": 1})

    # foo is a service
    assert root.get("foo")
    # root is a property, not a service
    assert root.get("root", strict=False) is None

    reached: list[int] = []

    def injected(ctx: Context, config: object = None) -> None:
        def nested(ctx: Context, config: object = None) -> None:
            assert ctx.baz == 2
            reached.append(2)

        apply(ctx.extend(baz=2), nested)

    root.inject(["foo"], injected)
    assert reached == [2]


@upstream(SPEC, "service inject leak")
def test_service_inject_leak() -> None:
    root = Context()
    root.provide("foo")
    root.set("foo", {"bar": 1})
    fiber = root.inject(["foo"], lambda ctx: None)
    assert fiber.ctx.foo
    fiber.dispose()
    with pytest.raises(ServiceAccessError, match='cannot get required service "foo" in inactive context'):
        _ = fiber.ctx.foo
