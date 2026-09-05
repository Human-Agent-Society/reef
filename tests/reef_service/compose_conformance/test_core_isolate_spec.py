"""packages/core/tests/isolate.spec.ts -- describe('Isolation')."""

from __future__ import annotations

from reef.train.cordis_backend.compose import Context, RealmKey, Service

from .utils import EVENT, Mock, run, sleep, upstream

SPEC = "core/tests/isolate.spec.ts"


def _injector(callback: Mock, disposer: Mock) -> object:
    """The reference's ``{ inject: ['foo'], apply }`` object plugin."""

    class Plugin:
        inject = ["foo"]

        @staticmethod
        def apply(ctx: Context, config: object = None):
            callback()
            return disposer

    return Plugin()


@upstream(SPEC, "isolated context")
def test_isolated_context() -> None:
    async def body() -> None:
        root = Context()
        callback = Mock()
        disposer = Mock()
        plugin = _injector(callback, disposer)

        root.plugin(plugin)
        ctx1 = root.isolate("foo")
        ctx1.plugin(plugin)
        ctx2 = root.isolate("foo")
        ctx2.plugin(plugin)

        dispose0 = root.provide("foo", {"bar": 100})
        assert root.foo["bar"] == 100
        assert ctx1.get("foo", strict=False) is None
        assert ctx2.get("foo", strict=False) is None
        await sleep()
        assert len(callback.calls) == 1
        assert len(disposer.calls) == 0

        ctx1.provide("foo", {"bar": 200})
        assert root.foo["bar"] == 100
        assert ctx1.foo["bar"] == 200
        assert ctx2.get("foo", strict=False) is None
        await sleep()
        assert len(callback.calls) == 2
        assert len(disposer.calls) == 0

        dispose0()
        assert root.get("foo", strict=False) is None
        assert ctx1.foo["bar"] == 200
        assert ctx2.get("foo", strict=False) is None
        await sleep()
        assert len(callback.calls) == 2
        assert len(disposer.calls) == 1

        ctx2.provide("foo", {"bar": 300})
        assert root.get("foo", strict=False) is None
        assert ctx1.foo["bar"] == 200
        assert ctx2.foo["bar"] == 300
        await sleep()
        assert len(callback.calls) == 3
        assert len(disposer.calls) == 1

    run(body)


@upstream(SPEC, "shared label")
def test_shared_label() -> None:
    async def body() -> None:
        root = Context()
        callback = Mock()
        disposer = Mock()
        plugin = _injector(callback, disposer)

        # the port of the reference's realm symbol (context.py <- context.ts)
        label = RealmKey("test")
        root.plugin(plugin)
        ctx1 = root.isolate("foo", label)
        ctx1.plugin(plugin)
        ctx2 = root.isolate("foo", label)
        ctx2.plugin(plugin)
        await sleep()
        assert len(callback.calls) == 0

        root.provide("foo", {"bar": 100})
        assert root.foo["bar"] == 100
        assert ctx1.get("foo", strict=False) is None
        assert ctx2.get("foo", strict=False) is None
        await sleep()
        assert len(callback.calls) == 1
        assert len(disposer.calls) == 0

        dispose12 = ctx1.provide("foo", {"bar": 200})
        assert root.foo["bar"] == 100
        assert ctx1.foo["bar"] == 200
        assert ctx2.foo["bar"] == 200
        await sleep()
        assert len(callback.calls) == 3
        assert len(disposer.calls) == 0

        dispose12()
        assert root.foo["bar"] == 100
        assert ctx1.get("foo", strict=False) is None
        assert ctx2.get("foo", strict=False) is None
        await sleep()
        assert len(callback.calls) == 3
        assert len(disposer.calls) == 2

    run(body)


@upstream(SPEC, "isolated event")
def test_isolated_event() -> None:
    class Foo(Service):
        def __init__(self, ctx: Context, config: object = None) -> None:
            super().__init__(ctx, "foo")
            self.ctx.emit(self, EVENT)

    root = Context()
    ctx = root.isolate("foo")
    outer = Mock()
    inner = Mock()
    root.on(EVENT, outer)
    ctx.on(EVENT, inner)
    ctx.plugin(Foo)

    assert len(outer.calls) == 0
    assert len(inner.calls) == 1
