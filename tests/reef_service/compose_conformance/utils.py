"""Mirror of the reference suite's test helpers.

``packages/core/tests/utils.ts`` and ``packages/loader/tests/utils.ts``,
plus the two things the reference gets from its host and Python does not:
``node:test``'s ``mock.fn`` and vitest's fake timers.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import Callable, Iterator
from typing import Any

from reef.train.cordis_backend.compose import Context
from reef.train.cordis_backend.compose.context import chain_layers
from reef.train.cordis_backend.compose.loader import EntryOptions, Loader

# utils.ts:22 -- the event name every dispatch test registers on.
EVENT = "custom-event"

# The reference resolves the group plugin through its module name; this port
# has no module resolution and routes group entries by their ``group`` flag
# (UPSTREAM.md, permanent omissions), so a mirrored entry carries both.
GROUP = "@cordisjs/plugin-group"


def upstream(spec: str, name: str) -> Callable[[Any], Any]:
    """Bind a mirrored test to the reference ``it()`` it conforms to.

    ``spec`` is the path under ``third_party/cordis/packages``; ``name`` is
    the ``it()`` string verbatim. ``test_conformance_coverage`` reads these
    back and fails when the reference grows a case nothing mirrors.
    """

    def decorate(func: Any) -> Any:
        func.__upstream__ = (spec, name)
        return func

    return decorate


# --- mock.fn -------------------------------------------------------------


class Mock:
    """``node:test``'s ``mock.fn`` for a function-shaped plugin or listener."""

    def __init__(self, impl: Callable[..., Any] | None = None, name: str | None = None) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.impl = impl
        self.name = name or getattr(impl, "__name__", None)
        self.__name__ = self.name or "mock"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(args)
        if self.impl is None:
            return None
        return self.impl(*args, **kwargs)

    def set_impl(self, impl: Callable[..., Any]) -> None:
        self.impl = impl

    def set_impl_once(self, impl: Callable[..., Any]) -> None:
        """``mock.mockImplementationOnce``: override the next call only."""
        previous = self.impl

        def once(*args: Any, **kwargs: Any) -> Any:
            self.impl = previous
            return impl(*args, **kwargs)

        self.impl = once

    def reset_calls(self) -> None:
        self.calls.clear()


def mock_class(cls: type, name: str | None = None) -> Any:
    """``mock.fn`` over a class plugin.

    A ``Mock`` instance cannot stand in for a class plugin: the port routes
    the ``__compose_init__`` protocol on ``inspect.isclass(callback)``
    (fiber.py:613), so the recorder has to be a class itself.
    """
    calls: list[tuple[Any, ...]] = []

    class Recorded(cls):  # type: ignore[misc, valid-type]
        def __init__(self, ctx: Context, config: Any = None) -> None:
            calls.append((ctx, config))
            super().__init__(ctx, config)

    Recorded.__name__ = name or cls.__name__
    Recorded.__qualname__ = Recorded.__name__
    Recorded.calls = calls  # type: ignore[attr-defined]
    Recorded.reset_calls = calls.clear  # type: ignore[attr-defined]
    return Recorded


# --- the receiver filter (utils.ts:24-39) --------------------------------


class Session:
    """A dispatch receiver whose filter consults the listening context."""

    def __init__(self, flag: bool) -> None:
        self.flag = flag

    def __event_filter__(self, ctx: Context) -> bool:
        # utils.ts:27 reads a possibly-absent own property. ServiceAccessError
        # subclasses AttributeError, so getattr's default is the loose read.
        predicate = getattr(ctx, "filter", None)
        if not predicate:
            return True
        return bool(predicate(self))


class Filter:
    """Carried as a context override; ``Session`` reads it back out."""

    def __init__(self, flag: bool) -> None:
        self._flag = flag

    def filter(self, session: Session) -> bool:
        return session.flag == self._flag


def get_hook_snapshot(ctx: Context) -> dict[str, int]:
    """utils.ts:79-85: the live listener count per event name."""
    return {name: len(hooks) for name, hooks in ctx.events._hooks.items() if hooks}


# --- fake timers ---------------------------------------------------------
#
# vitest's ``vi.useFakeTimers`` / ``advanceTimersByTimeAsync`` have no stdlib
# counterpart. An event loop whose ``time()`` the test advances by hand gives
# the same guarantee -- timers fire only when the test says so -- and keeps
# the inertia and async-effect cases deterministic instead of racing a real
# clock.

_SELECT_CAP = 0.005
_DEADLOCK_AFTER = 10.0


class _ClampedSelector:
    """Never block on a deadline only virtual time can reach."""

    def __init__(self, selector: Any, clock: VirtualClock) -> None:
        self._selector = selector
        self._clock = clock

    def select(self, timeout: float | None = None) -> Any:
        if timeout is None or timeout > _SELECT_CAP:
            timeout = _SELECT_CAP
        if time.monotonic() - self._clock.progress > _DEADLOCK_AFTER:
            raise RuntimeError("virtual clock deadlock: the test is awaiting a timer that no advance() call reaches")
        return self._selector.select(timeout)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._selector, name)


class VirtualClock:
    """A hand-advanced clock for one event loop."""

    def __init__(self) -> None:
        self.now = 0.0
        self.progress = time.monotonic()
        self._loop: Any = None

    def _build_loop(self) -> Any:
        loop = asyncio.SelectorEventLoop()
        loop.time = lambda: self.now  # type: ignore[method-assign]
        loop._selector = _ClampedSelector(loop._selector, self)
        self._loop = loop
        return loop

    async def settle(self) -> None:
        """Run everything already runnable, without moving the clock.

        The mirror of the reference's bare ``await sleep()``.
        """
        self.progress = time.monotonic()
        for _ in range(8):
            await asyncio.sleep(0)

    def _next_deadline(self, target: float) -> float | None:
        best: float | None = None
        for handle in self._loop._scheduled:
            if handle._cancelled:
                continue
            if handle._when <= target and (best is None or handle._when < best):
                best = handle._when
        return best

    async def run_all(self, limit: int = 1000) -> None:
        """``vi.runAllTimersAsync``: jump to each pending timer until none is left."""
        await self.settle()
        for _ in range(limit):
            deadline: float | None = None
            for handle in self._loop._scheduled:
                if handle._cancelled:
                    continue
                if deadline is None or handle._when < deadline:
                    deadline = handle._when
            if deadline is None:
                return
            self.now = max(self.now, deadline)
            await self.settle()
        raise RuntimeError("run_all: timers kept rescheduling past the iteration limit")

    async def advance(self, ms: float) -> None:
        """``vi.advanceTimersByTimeAsync``: fire every timer due within ``ms``."""
        target = self.now + ms / 1000
        await self.settle()
        while True:
            deadline = self._next_deadline(target)
            if deadline is None:
                break
            self.now = max(self.now, deadline)
            await self.settle()
        self.now = target
        await self.settle()


def with_timers(body: Callable[[Context, VirtualClock], Any]) -> None:
    """utils.ts:5-15: run ``body`` against a fresh root under fake timers."""
    clock = VirtualClock()
    loop = clock._build_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(body(Context(), clock))
    finally:
        for task in asyncio.all_tasks(loop):
            task.cancel()
        loop.run_until_complete(asyncio.sleep(0))
        asyncio.set_event_loop(None)
        loop.close()


def run(body: Callable[[], Any]) -> Any:
    """Run one async mirror case on a plain loop (no fake timers)."""
    return asyncio.run(body())


async def sleep() -> None:
    """The reference's ``await sleep()``: let queued work run."""
    for _ in range(8):
        await asyncio.sleep(0)


# --- MockLoader (packages/loader/tests/utils.ts) -------------------------


class MockLoader(Loader):
    """The reference's test loader.

    Divergences from ``utils.ts``, both from permanent omissions: ``import``
    is a synchronous resolver over the registered mocks, because the port
    takes a resolver callable instead of module resolution; and ``read``
    drives the root group directly, because there is no config file to
    parse. ``write`` captures what the reference writes to disk.
    """

    def __init__(self, ctx: Context) -> None:
        self.data: list[EntryOptions] = []
        self.modules: dict[str, Any] = {}
        super().__init__(ctx, self.modules.get)

    def write(self) -> None:
        self.data = self.root.data

    async def read(self, data: list[EntryOptions]) -> None:
        self.data = data
        self.root.update(data)
        await self.wait()

    def mock(self, name: str, plugin: Any) -> Any:
        if inspect.isclass(plugin):
            recorded = mock_class(plugin, name)
        else:
            recorded = Mock(plugin, name)
        self.modules[name] = recorded
        return recorded

    def expect_enable(self, plugin: Any) -> None:
        assert self.ctx.registry.get(plugin) is not None

    def expect_disable(self, plugin: Any) -> None:
        assert self.ctx.registry.get(plugin) is None

    def expect_fiber(self, id_: str) -> Any:
        fiber = self.resolve(id_).fiber
        assert fiber is not None
        return fiber


async def dispose(handle: Any) -> None:
    """``await dispose()``: call the handle and settle any async disposer."""
    result = handle()
    if result is not None:
        await result


def apply(ctx: Context, plugin: Any, config: Any = None) -> Any:
    """``ctx.plugin()`` that surfaces a failure raised inside the plugin.

    The port parks a load failure on the fiber rather than propagating it
    (fiber.py:666-683), so a mirrored assertion inside a plugin body would
    pass vacuously. Every mirror that asserts inside a plugin goes through
    here.
    """
    fiber = ctx.plugin(plugin, config)
    if fiber.error is not None:
        raise fiber.error
    return fiber


def anonymous(func: Any) -> Any:
    """Mark a plugin as nameless, the way the reference's arrow plugins are.

    A named plugin gives its fiber a name, which shows up in ``repr`` and in
    the service-registration error; the reference's ``(ctx) => {}`` plugins
    do not, and several assertions depend on it.
    """
    func.__name__ = ""
    return func


class Harness:
    """One ``describe`` block's shared root, loader and event loop.

    The loader specs mutate one tree across their cases (the reference's
    ``beforeAll`` plus ordered ``it``s), and the port's async work has to
    settle on the same loop throughout, so the loop outlives each case.
    """

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.root = Context()
        self.loader = MockLoader(self.root)

    def run(self, body: Callable[[], Any]) -> Any:
        return self.loop.run_until_complete(body())

    def close(self) -> None:
        asyncio.set_event_loop(None)
        self.loop.close()


@contextlib.contextmanager
def fresh_harness() -> Iterator[Harness]:
    """A private root and loader, for the reference's self-contained cases."""
    built = Harness()
    try:
        yield built
    finally:
        built.close()


def intercept_layers(ctx: Context, name: str) -> list[Any]:
    """Every intercept config for ``name`` along the chain, nearest first.

    The reference reads ``ctx[Context.intercept]`` and walks its prototypes;
    ``chain_layers`` is the port's equivalent walk.
    """
    return [layer[name] for layer in chain_layers(ctx._intercept) if name in layer]


async def maybe_await(value: Any) -> Any:
    """JS `await` accepts a non-thenable; Python's does not.

    A reference listener writes `await next()` whether or not the frame
    below it is async, so a mirrored listener goes through here.
    """
    if inspect.isawaitable(value):
        return await value
    return value
