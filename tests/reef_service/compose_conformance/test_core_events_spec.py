"""packages/core/tests/events.spec.ts -- describe('Events')."""

from __future__ import annotations

import asyncio
import sys

import pytest

from reef.train.cordis_backend.compose import Context, EventName

from .utils import EVENT, Filter, Mock, Session, maybe_await, run, upstream

if sys.version_info >= (3, 11):
    from builtins import ExceptionGroup
else:
    from reef.train.cordis_backend.compose.events import _ExceptionGroup as ExceptionGroup

SPEC = "core/tests/events.spec.ts"

WATERFALL = "test/waterfall"
ASYNC_WATERFALL = "test/async-waterfall"


@upstream(SPEC, "supports symbol event names across dispatch modes")
def test_supports_symbol_event_names_across_dispatch_modes() -> None:
    async def body() -> None:
        root = Context()
        # EventName is the port's stand-in for the reference's symbol events
        name = EventName("event")
        callback = Mock(lambda value: value)
        handle = root.on(name, callback)

        root.emit(name, 1)
        assert root.bail(name, 2) == 2
        assert await root.serial(name, 3) == 3
        await root.parallel(name, 4)
        assert [call[0] for call in callback.calls] == [1, 2, 3, 4]

        handle()
        root.emit(name, 5)
        assert len(callback.calls) == 4

    run(body)


@upstream(SPEC, "supports symbol event names with ctx.once()")
def test_supports_symbol_event_names_with_ctx_once() -> None:
    root = Context()
    name = EventName("once")
    callback = Mock()
    root.once(name, callback)

    root.emit(name)
    root.emit(name)
    assert len(callback.calls) == 1


@upstream(SPEC, "treats prototype property names as ordinary events")
def test_treats_prototype_property_names_as_ordinary_events() -> None:
    root = Context()

    # dispatching an unregistered name must not reach the type's own attributes
    root.emit("__str__")

    for name in ("__proto__", "__str__", "__class__"):
        callback = Mock()
        handle = root.on(name, callback)

        root.emit(name)
        assert len(callback.calls) == 1, name
        handle()
        root.emit(name)
        assert len(callback.calls) == 1, name
        assert name not in root.events._hooks, name


@upstream(SPEC, "removes empty event buckets after disposal")
def test_removes_empty_event_buckets_after_disposal() -> None:
    root = Context()
    name = EventName("temporary")
    first = root.on(name, lambda: None)
    second = root.on(name, lambda: None)

    first()
    assert name in root.events._hooks
    second()
    assert name not in root.events._hooks


@upstream(SPEC, "ctx.on()")
def test_ctx_on() -> None:
    root = Context()
    callback = Mock()
    handle = root.on(EVENT, callback)
    root.emit(EVENT)
    assert len(callback.calls) == 1
    root.emit(EVENT)
    assert len(callback.calls) == 2
    handle()
    root.emit(EVENT)
    assert len(callback.calls) == 2


@upstream(SPEC, "ctx.once()")
def test_ctx_once() -> None:
    root = Context()
    callback = Mock()
    handle = root.once(EVENT, callback)
    root.emit(EVENT)
    assert len(callback.calls) == 1
    root.emit(EVENT)
    assert len(callback.calls) == 1
    handle()
    root.emit(EVENT)
    assert len(callback.calls) == 1


@upstream(SPEC, "ctx.parallel()")
def test_ctx_parallel() -> None:
    async def body() -> None:
        root = Context()
        await root.parallel(EVENT)
        callback = Mock()
        root.extend(filter=Filter(True).filter).on(EVENT, callback)

        await root.parallel(EVENT)
        assert len(callback.calls) == 1
        await root.parallel(Session(False), EVENT)
        assert len(callback.calls) == 1
        await root.parallel(Session(True), EVENT)
        assert len(callback.calls) == 2

        # a rejecting listener must not short-circuit the others
        settled = False

        async def rejecting(*args: object) -> None:
            nonlocal settled
            await asyncio.sleep(0)
            settled = True
            raise RuntimeError("async")

        handle = root.on(EVENT, rejecting)

        def boom(*args: object) -> None:
            raise RuntimeError("test")

        callback.set_impl(boom)
        with pytest.raises(ExceptionGroup) as caught:
            await root.parallel(EVENT)
        assert sorted(str(error) for error in caught.value.exceptions) == ["async", "test"]
        assert settled is True
        handle()

    run(body)


@upstream(SPEC, "ctx.emit()")
def test_ctx_emit() -> None:
    root = Context()
    root.emit(EVENT)
    callback = Mock()
    root.extend(filter=Filter(True).filter).on(EVENT, callback)

    root.emit(EVENT)
    assert len(callback.calls) == 1
    root.emit(Session(False), EVENT)
    assert len(callback.calls) == 1
    root.emit(Session(True), EVENT)
    assert len(callback.calls) == 2

    def boom(*args: object) -> None:
        raise RuntimeError("test")

    callback.set_impl(boom)
    with pytest.raises(RuntimeError, match="test"):
        root.emit(EVENT)


@upstream(SPEC, "ctx.serial()")
def test_ctx_serial() -> None:
    async def body() -> None:
        root = Context()
        await root.serial(EVENT)
        callback = Mock()
        root.extend(filter=Filter(True).filter).on(EVENT, callback)

        await root.serial(EVENT)
        assert len(callback.calls) == 1
        await root.serial(Session(False), EVENT)
        assert len(callback.calls) == 1
        await root.serial(Session(True), EVENT)
        assert len(callback.calls) == 2

        def boom(*args: object) -> None:
            raise RuntimeError("message")

        callback.set_impl(boom)
        with pytest.raises(RuntimeError, match="message"):
            await root.serial(EVENT)

    run(body)


@upstream(SPEC, "ctx.bail()")
def test_ctx_bail() -> None:
    root = Context()
    root.bail(EVENT)
    callback = Mock()
    root.extend(filter=Filter(True).filter).on(EVENT, callback)

    root.bail(EVENT)
    assert len(callback.calls) == 1
    root.bail(Session(False), EVENT)
    assert len(callback.calls) == 1
    root.bail(Session(True), EVENT)
    assert len(callback.calls) == 2

    def boom(*args: object) -> None:
        raise RuntimeError("message")

    callback.set_impl(boom)
    with pytest.raises(RuntimeError, match="message"):
        root.bail(EVENT)


@upstream(SPEC, "ctx.waterfall()")
def test_ctx_waterfall() -> None:
    root = Context()
    cb1 = Mock(lambda value, next_: value + next_())
    root.on(WATERFALL, cb1)
    cb2 = Mock(lambda value, next_: value + next_())
    root.on(WATERFALL, cb2)

    assert root.waterfall(WATERFALL, 1, lambda: 2) == 4
    assert len(cb1.calls) == 1
    assert len(cb2.calls) == 1
    cb1.reset_calls()
    cb2.reset_calls()

    cb3 = Mock(lambda value, next_: value)
    root.on(WATERFALL, cb3)
    cb4 = Mock(lambda value, next_: value + next_())
    root.on(WATERFALL, cb4)
    assert root.waterfall(WATERFALL, 1, lambda: 2) == 3
    assert len(cb1.calls) == 1
    assert len(cb2.calls) == 1
    assert len(cb3.calls) == 1
    assert len(cb4.calls) == 0


@upstream(SPEC, "ctx.waterfall() rejects duplicate next()")
def test_ctx_waterfall_rejects_duplicate_next() -> None:
    root = Context()
    terminal = Mock(lambda: 2)

    def twice(value, next_):
        next_()
        return next_()

    callback = Mock(twice)
    root.on(WATERFALL, callback)

    with pytest.raises(RuntimeError, match="next\\(\\) called multiple times"):
        root.waterfall(WATERFALL, 1, terminal)
    assert len(callback.calls) == 1
    assert len(terminal.calls) == 1


@upstream(SPEC, "ctx.waterfall() rejects continuations from outer frames")
def test_ctx_waterfall_rejects_continuations_from_outer_frames() -> None:
    root = Context()
    calls: list[str] = []
    outer: list = []

    def first(value, next_):
        outer.append(next_)
        calls.append("first")
        return next_()

    def second(value, next_):
        calls.append("second")
        return next_()

    root.on(WATERFALL, first)
    root.on(WATERFALL, second)

    def terminal():
        calls.append("terminal")
        return outer[0]()

    with pytest.raises(RuntimeError, match="next\\(\\) called multiple times"):
        root.waterfall(WATERFALL, 1, terminal)
    assert calls == ["first", "second", "terminal"]


@upstream(SPEC, "ctx.waterfall() rejects duplicate next() after awaiting")
def test_ctx_waterfall_rejects_duplicate_next_after_awaiting() -> None:
    async def body() -> None:
        root = Context()
        terminal = Mock(lambda: 2)

        async def listener(value, next_):
            result = await maybe_await(next_())
            with pytest.raises(RuntimeError, match="next\\(\\) called multiple times"):
                next_()
            return value + result

        root.on(ASYNC_WATERFALL, listener)

        assert await maybe_await(root.waterfall(ASYNC_WATERFALL, 1, terminal)) == 3
        assert len(terminal.calls) == 1

    run(body)


@upstream(SPEC, "ctx.waterfall() supports nested async calls")
def test_ctx_waterfall_supports_nested_async_calls() -> None:
    async def body() -> None:
        root = Context()
        terminal = Mock(lambda: 2)

        async def impl(value, next_):
            result = await maybe_await(next_())
            if value == 1:
                return result + await maybe_await(root.waterfall(ASYNC_WATERFALL, 2, terminal))
            return result

        callback = Mock(impl)
        root.on(ASYNC_WATERFALL, callback)

        assert await maybe_await(root.waterfall(ASYNC_WATERFALL, 1, terminal)) == 4
        assert len(callback.calls) == 2
        assert len(terminal.calls) == 2

    run(body)
