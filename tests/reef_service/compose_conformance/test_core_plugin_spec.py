"""packages/core/tests/plugin.spec.ts -- describe('Plugin')."""

from __future__ import annotations

import pytest

from reef.train.cordis_backend.compose import Context

from .utils import EVENT, Mock, anonymous, apply, get_hook_snapshot, run, sleep, upstream

SPEC = "core/tests/plugin.spec.ts"


@upstream(SPEC, "apply functional plugin")
def test_apply_functional_plugin() -> None:
    root = Context()
    callback = Mock()
    options = {"foo": "bar"}
    root.plugin(callback, options)

    assert len(callback.calls) == 1
    assert callback.calls[0][1] == options


@upstream(SPEC, "apply object plugin")
def test_apply_object_plugin() -> None:
    root = Context()
    callback = Mock()
    options = {"bar": "foo"}

    class Plugin:
        apply = staticmethod(callback)

    root.plugin(Plugin(), options)

    assert len(callback.calls) == 1
    assert callback.calls[0][1] == options


@upstream(SPEC, "apply invalid plugin")
def test_apply_invalid_plugin() -> None:
    root = Context()

    class NoApply:
        pass

    class ApplyNotCallable:
        apply = {}

    with pytest.raises(TypeError):
        root.plugin(None)
    with pytest.raises(TypeError):
        root.plugin(NoApply())
    with pytest.raises(TypeError):
        root.plugin(ApplyNotCallable())


@upstream(SPEC, "inactive context")
def test_inactive_context() -> None:
    root = Context()
    callback = Mock()
    seen: list[str] = []

    def plugin(ctx: Context, config: object = None):
        def teardown() -> None:
            for run_ in (
                lambda: ctx.plugin(callback),
                lambda: ctx.effect(lambda: (lambda: None)),
                lambda: ctx.on(EVENT, lambda: None),
            ):
                with pytest.raises(Exception, match="inactive context"):
                    run_()
                seen.append("raised")

        return teardown

    fiber = root.plugin(plugin)
    fiber.dispose()
    assert seen == ["raised"] * 3
    assert len(callback.calls) == 0


@upstream(SPEC, "context inspect")
def test_context_inspect() -> None:
    root = Context()

    assert repr(root) == "Context <root>"

    def arrow(ctx: Context, config: object = None) -> None:
        # the reference's arrow plugin is nameless and inherits <root>
        assert repr(ctx) == "Context <root>"

    apply(root, anonymous(arrow))

    def foo(ctx: Context, config: object = None) -> None:
        assert repr(ctx) == "Context <foo>"

    apply(root, foo)

    class Bar:
        name = "bar"

        @staticmethod
        def apply(ctx: Context, config: object = None) -> None:
            assert repr(ctx) == "Context <bar>"

    apply(root, Bar())

    class Qux:
        def __init__(self, ctx: Context, config: object = None) -> None:
            assert repr(ctx) == "Context <Qux>"

    apply(root, Qux)


@upstream(SPEC, "ctx.registry")
def test_ctx_registry() -> None:
    # make coverage happy. The reference's Map surface is keys()/entries()/
    # forEach(); the port exposes the walk notify needs (registry.py:100).
    root = Context()
    list(root.registry.values())
    assert root.registry.get(lambda ctx, config: None) is None
    assert root.registry.has(lambda ctx, config: None) is False


@upstream(SPEC, "nested plugins")
def test_nested_plugins() -> None:
    callback = Mock()

    def plugin(ctx: Context, config: object = None) -> None:
        ctx.on(EVENT, callback)

        def middle(ctx: Context, config: object = None) -> None:
            ctx.on(EVENT, callback)

            def leaf(ctx: Context, config: object = None) -> None:
                ctx.on(EVENT, callback)

            ctx.plugin(leaf)

        ctx.plugin(middle)

    root = Context()
    root.on(EVENT, callback)
    fiber = root.plugin(plugin)

    # 4 handlers by now
    assert len(callback.calls) == 0
    assert len(list(root.registry.values())) == 3
    root.emit(EVENT)
    assert len(callback.calls) == 4

    # only 1 handler left
    callback.reset_calls()
    fiber.dispose()
    assert len(list(root.registry.values())) == 0
    root.emit(EVENT)
    assert len(callback.calls) == 1

    # subsequent calls should be noop
    callback.reset_calls()
    fiber.dispose()
    assert len(list(root.registry.values())) == 0
    root.emit(EVENT)
    assert len(callback.calls) == 1


@upstream(SPEC, "compare snapshot")
def test_compare_snapshot() -> None:
    async def body() -> None:
        def plugin(ctx: Context, config: object = None) -> None:
            ctx.on(EVENT, lambda: None)

            def middle(ctx: Context, config: object = None) -> None:
                ctx.on(EVENT, lambda: None)

                def leaf(ctx: Context, config: object = None) -> None:
                    ctx.on(EVENT, lambda: None)

                ctx.plugin(leaf)

            ctx.plugin(middle)

        root = Context()
        before = get_hook_snapshot(root)
        root.plugin(plugin)
        after = get_hook_snapshot(root)
        root.registry.delete(plugin)
        await sleep()
        assert before == get_hook_snapshot(root)
        root.plugin(plugin)
        assert after == get_hook_snapshot(root)

    run(body)


@upstream(SPEC, "root dispose")
def test_root_dispose() -> None:
    root = Context()
    disposer = Mock()
    fiber = root.plugin(lambda ctx, config: disposer)
    assert root.fiber.uid == 0
    assert fiber.uid == 1
    assert len(disposer.calls) == 0
    assert len(root.fiber._disposables) == 1
    root.fiber.dispose()
    assert root.fiber.uid == 0
    assert fiber.uid is None
    assert len(disposer.calls) == 1
    assert len(root.fiber._disposables) == 0
    root.fiber.dispose()
    assert root.fiber.uid == 0
    assert fiber.uid is None
    assert len(disposer.calls) == 1
    assert len(root.fiber._disposables) == 0


@upstream(SPEC, "Service.init")
def test_service_init() -> None:
    start = Mock()
    stop = Mock()

    class Foo:
        def __init__(self, ctx: Context, config: object = None) -> None:
            pass

        def __compose_init__(self):
            start()
            return stop

    root = Context()
    fiber = root.plugin(Foo)
    assert len(start.calls) == 1
    assert len(stop.calls) == 0
    fiber.dispose()
    assert len(start.calls) == 1
    assert len(stop.calls) == 1
