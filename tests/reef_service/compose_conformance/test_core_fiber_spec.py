"""packages/core/tests/fiber.spec.ts -- describe('Fiber')."""

from __future__ import annotations

import asyncio
import logging

import pytest

from reef.train.cordis_backend.compose import Context, FiberState, Service

from .utils import EVENT, Mock, anonymous, maybe_await, run, sleep, upstream

SPEC = "core/tests/fiber.spec.ts"

LOGGER = "reef.train.cordis_backend.compose"


def _slow_consumer(ctx: Context):
    """The reference's ``async () => { await sleep(1000); return () => sleep(1000) }``."""

    async def setup():
        await asyncio.sleep(1.0)

        async def teardown() -> None:
            await asyncio.sleep(1.0)

        return teardown

    return setup()


@upstream(SPEC, "inertia lock 1")
def test_inertia_lock_1() -> None:
    async def body(root: Context, clock) -> None:
        handle = root.provide("foo", 1)
        fiber = root.inject(["foo"], _slow_consumer)
        await clock.advance(400)  # 400
        assert fiber.state is FiberState.LOADING
        handle()
        await clock.advance(400)  # 800
        assert fiber.state is FiberState.LOADING
        await clock.advance(400)  # 1200
        assert fiber.state is FiberState.UNLOADING
        root.provide("foo", 1)
        await clock.advance(1000)  # 2200
        assert fiber.state is FiberState.LOADING
        await clock.advance(1000)  # 3200
        assert fiber.state is FiberState.ACTIVE

    _with_timers(body)


@upstream(SPEC, "inertia lock 2")
def test_inertia_lock_2() -> None:
    async def body(root: Context, clock) -> None:
        handle = root.provide("foo", 1)
        fiber = root.inject(["foo"], _slow_consumer)
        await clock.advance(400)  # 400
        assert fiber.state is FiberState.LOADING
        handle()
        await clock.advance(400)  # 800
        assert fiber.state is FiberState.LOADING
        root.provide("foo", 2)
        await clock.advance(400)  # 1200
        assert fiber.state is FiberState.ACTIVE

    _with_timers(body)


@upstream(SPEC, "inertia lock 3")
def test_inertia_lock_3() -> None:
    async def body(root: Context, clock) -> None:
        class Foo(Service):
            def __init__(self, ctx: Context, config: object = None) -> None:
                super().__init__(ctx, "foo")

        provider = root.plugin(Foo)
        await provider
        fiber = root.inject(["foo"], _slow_consumer)
        await clock.advance(400)  # 400
        assert fiber.state is FiberState.LOADING
        await clock.run_all()  # 1000
        assert fiber.state is FiberState.ACTIVE
        pending = provider.dispose()
        if pending is None:
            await clock.run_all()  # 2000
        else:
            await asyncio.gather(pending, clock.run_all())
        assert fiber.state is FiberState.PENDING

    _with_timers(body)


@upstream(SPEC, "plugin error")
def test_plugin_error(caplog: pytest.LogCaptureFixture) -> None:
    async def body() -> None:
        root = Context()
        callback = Mock()

        def impl(ctx: Context, config: dict | None = None) -> None:
            ctx.on(EVENT, callback)
            if not (config or {}).get("foo"):
                raise RuntimeError("plugin error")

        plugin = Mock(anonymous(impl))

        with caplog.at_level(logging.ERROR, logger=LOGGER):
            fiber1 = root.plugin(plugin)
            fiber2 = root.plugin(plugin, {"foo": True})
            await sleep()
            assert fiber1.state is FiberState.FAILED
            assert fiber2.state is FiberState.ACTIVE
            assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 1

        root.emit(EVENT)
        assert len(callback.calls) == 1

    run(body)


@upstream(SPEC, "failed fiber does not re-enter on dependency refresh")
def test_failed_fiber_does_not_re_enter_on_dependency_refresh(caplog: pytest.LogCaptureFixture) -> None:
    async def body() -> None:
        root = Context()

        def boom(ctx: Context) -> None:
            raise RuntimeError("boom")

        apply = Mock(boom)
        with caplog.at_level(logging.ERROR, logger=LOGGER):
            handle = root.provide("foo", 1)
            fiber = root.inject(["foo"], apply)
            await sleep()
            assert fiber.state is FiberState.FAILED
            await maybe_await(handle())
            root.provide("foo", 2)
            await sleep()
            assert len(apply.calls) == 1
            assert fiber.state is FiberState.FAILED

    run(body)


@upstream(SPEC, "update recovers a failed fiber")
def test_update_recovers_a_failed_fiber(caplog: pytest.LogCaptureFixture) -> None:
    async def body() -> None:
        root = Context()

        def boom(ctx: Context) -> None:
            raise RuntimeError("boom")

        apply = Mock(boom)
        with caplog.at_level(logging.ERROR, logger=LOGGER):
            root.provide("foo", 1)
            fiber = root.inject(["foo"], apply)
            await sleep()
            assert fiber.state is FiberState.FAILED
            apply.set_impl_once(lambda ctx: None)
            # the reference's bare update() keeps the current config
            fiber.update(fiber.config)
            await fiber
            assert len(apply.calls) == 2
            assert fiber.state is FiberState.ACTIVE

    run(body)


@upstream(SPEC, "update surfaces a failed reload to its caller")
def test_update_surfaces_a_failed_reload_to_its_caller(caplog: pytest.LogCaptureFixture) -> None:
    """Conforms modulo the recorded sync-first divergence.

    The reference's update() always returns a promise, so a failed reload
    reaches the caller through it. This port completes a transition inline
    unless the fiber holds genuinely async work (UPSTREAM.md, fiber.py <-
    fiber.ts): a fully synchronous plugin leaves nothing to await, and the
    failure is read off the fiber instead -- `await fiber` re-raises it.
    Give the fiber an async disposer, the reference's precondition, and the
    awaited update raises as upstream.
    """

    def scenario(async_teardown: bool) -> tuple[bool, FiberState]:
        raised = False

        async def body() -> None:
            nonlocal raised
            root = Context()

            def base(ctx: Context, config: object = None) -> None:
                if async_teardown:

                    async def teardown() -> None:
                        await asyncio.sleep(0)

                    ctx.effect(lambda: teardown)

            apply = Mock(base)
            fiber = root.plugin(apply)
            await fiber

            def boom(ctx: Context, config: object = None) -> None:
                raise RuntimeError("boom")

            apply.set_impl_once(boom)
            try:
                await maybe_await(fiber.update({}))
            except RuntimeError as error:
                assert str(error) == "boom"
                raised = True
            assert fiber.state is FiberState.FAILED
            with pytest.raises(RuntimeError, match="boom"):
                await fiber
            scenario.state = fiber.state  # type: ignore[attr-defined]

        with caplog.at_level(logging.ERROR, logger=LOGGER):
            run(body)
        return raised, scenario.state  # type: ignore[attr-defined]

    # the reference's precondition: the fiber holds async work
    assert scenario(async_teardown=True) == (True, FiberState.FAILED)
    # sync-first: nothing to await, the failure is on the fiber
    assert scenario(async_teardown=False) == (False, FiberState.FAILED)


@upstream(SPEC, "update does not leak a dropped failure")
def test_update_does_not_leak_a_dropped_failure(caplog: pytest.LogCaptureFixture) -> None:
    async def body() -> None:
        root = Context()
        apply = Mock(lambda ctx, config=None: None)
        with caplog.at_level(logging.ERROR, logger=LOGGER):
            fiber = root.plugin(apply)
            await fiber

            def boom(ctx: Context, config: object = None) -> None:
                raise RuntimeError("boom")

            apply.set_impl_once(boom)
            fiber.update({})
            await sleep()
            assert fiber.state is FiberState.FAILED

    run(body)


@upstream(SPEC, "dispose error")
def test_dispose_error(caplog: pytest.LogCaptureFixture) -> None:
    async def body() -> None:
        root = Context()

        def disposer() -> None:
            raise RuntimeError("test")

        recorded = Mock(disposer)

        def plugin(ctx: Context, config: object = None):
            return recorded

        with caplog.at_level(logging.ERROR, logger=LOGGER):
            fiber = root.plugin(anonymous(plugin))
            assert len(recorded.calls) == 0
            result = fiber.dispose()
            if result is not None:
                assert await result is None
            await sleep()
            assert len(recorded.calls) == 1
            assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 1

    run(body)


@upstream(SPEC, "update config on wrapped fiber")
def test_update_config_on_wrapped_fiber() -> None:
    async def body() -> None:
        root = Context()
        callback = Mock()

        fiber = root.plugin(callback, {"msg": "hello"})
        await fiber
        assert len(callback.calls) == 1
        assert callback.calls[0][1] == {"msg": "hello"}

        fiber.update({"msg": "world"})
        await fiber
        assert len(callback.calls) == 2
        assert callback.calls[1][1] == {"msg": "world"}

        fiber.update({"msg": "!!!"})
        await fiber
        assert len(callback.calls) == 3
        assert callback.calls[2][1] == {"msg": "!!!"}

    run(body)


@upstream(SPEC, "restart wrapped fiber")
def test_restart_wrapped_fiber() -> None:
    # The reference also asserts that `state` and `inertia` are not own
    # properties: its wrapped fiber shares them through a prototype. The port
    # has no prototype wrapper, so those two assertions have no counterpart.
    async def body() -> None:
        root = Context()
        callback = Mock()
        fiber = root.plugin(callback)

        await fiber
        pending = fiber.restart()
        if pending is not None:
            await pending

        assert len(callback.calls) == 2
        assert fiber.state is FiberState.ACTIVE

    run(body)


@upstream(SPEC, "update config while injected service reloads")
def test_update_config_while_injected_service_reloads() -> None:
    """Conforms modulo the recorded sync-first divergence.

    The reference is promise-based throughout, so the provider's reload is
    still queued when the consumer's own update lands, and the consumer
    applies once, with both new values. This port completes a transition
    inline unless an effect form or disposer is genuinely async (UPSTREAM.md,
    fiber.py <- fiber.ts), so a fully synchronous provider finishes its
    reload -- restarting the consumer with its old config -- before
    ``consumer.update`` runs. The second scenario below gives the provider an
    async disposer, restoring the reference's precondition, and the port then
    coalesces exactly as upstream does.

    The reference's remaining assertions are about `config`/`state` not being
    own properties of its prototype-wrapped fiber; the port has no such
    wrapper, so they have no counterpart.
    """

    def scenario(async_teardown: bool) -> list[tuple[int, str]]:
        applied: list[tuple[int, str]] = []

        class Provider(Service):
            def __init__(self, ctx: Context, config: dict) -> None:
                super().__init__(ctx, "provider")
                self.value = config["value"]
                if async_teardown:

                    async def teardown() -> None:
                        await asyncio.sleep(0)

                    ctx.effect(lambda: teardown)

        class Consumer:
            inject = ["provider"]

            @staticmethod
            def apply(ctx: Context, config: dict) -> None:
                applied.append((ctx.provider.value, config["mode"]))

        async def body() -> None:
            root = Context()
            provider = root.plugin(Provider, {"value": 1})
            consumer = root.plugin(Consumer(), {"mode": "old"})

            await provider
            await consumer

            provider.update({"value": 2})
            consumer.update({"mode": "new"})

            await asyncio.gather(provider.wait(), consumer.wait())
            assert consumer.state is FiberState.ACTIVE

        run(body)
        return applied

    # the reference's precondition: the provider's teardown is genuinely async
    assert scenario(async_teardown=True) == [(1, "old"), (2, "new")]
    # sync-first: the provider's reload lands before the consumer's update
    assert scenario(async_teardown=False) == [(1, "old"), (2, "old"), (2, "new")]


def _with_timers(body) -> None:
    from .utils import with_timers

    with_timers(body)
