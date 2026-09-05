"""packages/core/tests/dispose.spec.ts -- describe('Effects')."""

from __future__ import annotations

import asyncio

import pytest

from reef.train.cordis_backend.compose import Context
from reef.train.cordis_backend.compose.fiber import EffectMeta

from .utils import EVENT, Mock, dispose, run, upstream, with_timers

SPEC = "core/tests/dispose.spec.ts"


@upstream(SPEC, "dispose by plugin")
def test_dispose_by_plugin() -> None:
    root = Context()
    disposer = Mock()

    def plugin(ctx: Context, config: object = None) -> None:
        ctx.effect(lambda: disposer, "test")

    fiber = root.plugin(plugin)
    assert fiber.get_effects() == [EffectMeta(label="test", children=[])]
    assert len(disposer.calls) == 0
    fiber.dispose()
    assert len(disposer.calls) == 1
    fiber.dispose()
    assert len(disposer.calls) == 1


@upstream(SPEC, "dispose manually")
def test_dispose_manually() -> None:
    root = Context()
    disposer = Mock()
    handle = root.effect(lambda: disposer)
    assert root.fiber.get_effects() == [EffectMeta(label="anonymous", children=[])]
    assert len(disposer.calls) == 0
    handle()
    assert len(disposer.calls) == 1
    handle()
    assert len(disposer.calls) == 1


@upstream(SPEC, "yield dispose")
def test_yield_dispose() -> None:
    root = Context()
    seq: list[int] = []

    def outer():
        yield lambda: seq.append(1)
        yield root.on(EVENT, lambda: None)
        yield lambda: seq.append(2)

        def inner():
            yield root.on(EVENT, lambda: None)
            yield lambda: seq.append(3)

        yield root.effect(inner)

    handle = root.effect(outer)
    root.on(EVENT, lambda: None)
    assert root.fiber.get_effects() == [
        EffectMeta(
            label="anonymous",
            children=[
                # only root level anonymous effects are included
                EffectMeta(label=f'ctx.on("{EVENT}")', children=[]),
                EffectMeta(
                    label="anonymous",
                    children=[EffectMeta(label=f'ctx.on("{EVENT}")', children=[])],
                ),
            ],
        ),
        EffectMeta(label=f'ctx.on("{EVENT}")', children=[]),
    ]
    assert seq == []
    handle()
    assert seq == [3, 2, 1]
    handle()
    assert seq == [3, 2, 1]


@upstream(SPEC, "async return 1")
def test_async_return_1() -> None:
    async def body(root: Context, clock) -> None:
        seq: list[int] = []

        async def setup():
            await asyncio.sleep(0.1)
            seq.append(1)
            return lambda: seq.append(2)

        handle = root.effect(setup)
        assert seq == []
        await clock.advance(100)
        assert seq == [1]
        await dispose(handle)
        assert seq == [1, 2]

    with_timers(body)


@upstream(SPEC, "async return 2")
def test_async_return_2() -> None:
    async def body(root: Context, clock) -> None:
        seq: list[int] = []

        async def setup():
            await asyncio.sleep(0.1)
            seq.append(1)
            return lambda: seq.append(2)

        handle = root.effect(setup)
        handle()
        assert seq == []
        await clock.advance(100)
        assert seq == [1, 2]

    with_timers(body)


def _stepped(seq: list[int]):
    """The reference's three-step async generator (dispose.spec.ts:101-112)."""

    async def setup():
        await asyncio.sleep(0.1)
        seq.append(1)
        yield lambda: seq.append(2)
        await asyncio.sleep(0.1)
        seq.append(3)
        yield lambda: seq.append(4)
        await asyncio.sleep(0.1)
        seq.append(5)
        yield lambda: seq.append(6)

    return setup


@upstream(SPEC, "async yield 1")
def test_async_yield_1() -> None:
    async def body(root: Context, clock) -> None:
        seq: list[int] = []
        handle = root.effect(_stepped(seq))
        assert seq == []
        await clock.advance(300)
        assert seq == [1, 3, 5]
        await dispose(handle)
        assert seq == [1, 3, 5, 6, 4, 2]

    with_timers(body)


@upstream(SPEC, "async yield 2 (aborted)")
def test_async_yield_2_aborted() -> None:
    async def body(root: Context, clock) -> None:
        seq: list[int] = []
        handle = root.effect(_stepped(seq))
        await clock.advance(50)
        handle()
        assert seq == []
        await clock.advance(300)
        assert seq == [1, 2]

    with_timers(body)


@upstream(SPEC, "async yield 3 (aborted)")
def test_async_yield_3_aborted() -> None:
    async def body(root: Context, clock) -> None:
        seq: list[int] = []
        handle = root.effect(_stepped(seq))
        assert seq == []
        await clock.advance(100)
        assert seq == [1]
        handle()
        assert seq == [1]
        await clock.advance(200)
        assert seq == [1, 3, 4, 2]

    with_timers(body)


@upstream(SPEC, "async yield 4 (await dispose)")
def test_async_yield_4_await_dispose() -> None:
    async def body(root: Context, clock) -> None:
        seq: list[int] = []
        handle = root.effect(_stepped(seq))
        assert seq == []
        handle2, _ = await asyncio.gather(handle, clock.advance(300))
        assert seq == [1, 3, 5]
        await dispose(handle2)
        assert seq == [1, 3, 5, 6, 4, 2]

    with_timers(body)


@upstream(SPEC, "return with error")
def test_return_with_error() -> None:
    root = Context()
    seq: list[int] = []

    def setup():
        raise RuntimeError("test")

    with pytest.raises(RuntimeError, match="test"):
        root.effect(setup)
    assert seq == []


@upstream(SPEC, "yield with error")
def test_yield_with_error() -> None:
    root = Context()
    seq: list[int] = []

    def setup():
        yield lambda: seq.append(1)
        raise RuntimeError("test")
        yield lambda: seq.append(2)  # unreachable, as upstream

    with pytest.raises(RuntimeError, match="test"):
        root.effect(setup)
    assert seq == [1]


@upstream(SPEC, "async return with error")
def test_async_return_with_error() -> None:
    async def body() -> None:
        root = Context()
        seq: list[int] = []

        async def setup():
            raise RuntimeError("test")

        handle = root.effect(setup)
        assert seq == []
        with pytest.raises(RuntimeError):
            await handle
        assert seq == []

    run(body)


@upstream(SPEC, "async yield with error")
def test_async_yield_with_error() -> None:
    async def body() -> None:
        root = Context()
        seq: list[int] = []

        async def setup():
            yield lambda: seq.append(1)
            raise RuntimeError("test")

        handle = root.effect(setup)
        assert seq == []
        caught: BaseException | None = None
        try:
            await handle
        except BaseException as error:  # mirrors the reference's bare catch
            caught = error
        assert isinstance(caught, Exception)
        assert seq == [1]

    run(body)
