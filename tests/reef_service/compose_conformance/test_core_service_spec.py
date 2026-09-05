"""packages/core/tests/service.spec.ts -- describe('Service')."""

from __future__ import annotations

import asyncio

from reef.train.cordis_backend.compose import Context, Service

from .utils import EVENT, Mock, get_hook_snapshot, run, sleep, upstream

SPEC = "core/tests/service.spec.ts"


@upstream(SPEC, "pending inject")
def test_pending_inject() -> None:
    async def body() -> None:
        class Foo(Service):
            def __init__(self, ctx: Context, config: object = None) -> None:
                super().__init__(ctx, "foo")

            async def __compose_init__(self) -> None:
                waiter = asyncio.get_running_loop().create_future()
                self.ctx.on(EVENT, lambda *_: waiter.done() or waiter.set_result(None))
                await waiter

        root = Context()

        callback = Mock()
        root.inject(["foo"], callback)
        assert len(callback.calls) == 0

        # inject should be blocked by the init hook
        root.plugin(Foo)
        await sleep()
        assert len(callback.calls) == 0

        root.emit(EVENT)
        await sleep()
        assert len(callback.calls) == 1

    run(body)


@upstream(SPEC, "compare snapshot")
def test_compare_snapshot() -> None:
    async def body() -> None:
        class Test(Service):
            def __init__(self, ctx: Context, config: object = None) -> None:
                super().__init__(ctx, "test")
                ctx.inject(["test"], lambda ctx: None)

        root = Context()
        before = get_hook_snapshot(root)
        root.plugin(Test)
        after = get_hook_snapshot(root)
        root.registry.delete(Test)
        await sleep()
        assert before == get_hook_snapshot(root)
        root.plugin(Test)
        assert after == get_hook_snapshot(root)

    run(body)


@upstream(SPEC, "multiple injects")
def test_multiple_injects() -> None:
    async def body() -> None:
        foo = Mock()
        bar = Mock()
        qux = Mock()

        class Foo(Service):
            inject = ["qux"]
            __compose_init__ = foo

            def __init__(self, ctx: Context, config: object = None) -> None:
                super().__init__(ctx, "foo")

        class Bar(Service):
            inject = ["foo", "qux"]
            __compose_init__ = bar

            def __init__(self, ctx: Context, config: object = None) -> None:
                super().__init__(ctx, "bar")

        class Qux(Service):
            __compose_init__ = qux

            def __init__(self, ctx: Context, config: object = None) -> None:
                super().__init__(ctx, "qux")

        root = Context()
        root.plugin(Foo)
        root.plugin(Bar)
        root.plugin(Qux)
        await sleep()
        assert len(foo.calls) == 1
        assert len(bar.calls) == 1
        assert len(qux.calls) == 1

    run(body)
