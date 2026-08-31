"""Event bus: five dispatch modes over listeners that die with their fiber.

Listener registration is an ordinary tracked effect of the registering
fiber, so a listener unregisters exactly when its scope disposes, at its
LIFO position, with no separate unsubscribe path. Dispatch modes: ``emit``
fires synchronously in order; ``parallel`` awaits all listeners and
aggregates every failure; ``serial`` awaits one at a time until a result
bails; ``bail`` is its sync twin; ``waterfall`` threads a ``next``
continuation through the listeners middleware-style. Bailed means not None
and not False, so 0 and the empty string bail.

Receiver convention: a dispatch may name a receiver before the event name
(the reference host's thisArg). Python has no implicit receiver, so when a
dispatch carries one, every listener receives it as its leading argument;
the receiver's ``__event_filter__``, when present, filters which hooks run.

Internal events: 'internal/plugin' (fiber) on create and retire;
'internal/status' (fiber, old_state) on state change; 'internal/service'
(ctx, name, value) when a service binding appears, withdraws, or changes
provider; 'internal/update' (fiber, config, no_save, next), the reconfigure
waterfall; 'internal/dispatch' (mode, name, args, receiver) before any
non-internal dispatch that has listeners; 'internal/listener' (ctx, name,
listener, options), a bail seam consulted before a listener registers;
'internal/get' and 'internal/set' (ctx, name, ..., next), the service
access interposition seams.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .context import Context
from .fiber import EffectHandle, Fiber

if sys.version_info >= (3, 11):
    from builtins import ExceptionGroup as _ExceptionGroup
else:

    class _ExceptionGroup(Exception):
        """Python 3.10 stand-in carrying aggregated errors like 3.11's ExceptionGroup."""

        def __init__(self, message: str, exceptions: Sequence[BaseException]) -> None:
            super().__init__(message)
            self.exceptions = tuple(exceptions)


def is_bailed(value: Any) -> bool:
    """Whether a listener result stops a bailing dispatch."""
    return value is not None and value is not False


@dataclass(frozen=True)
class EventOptions:
    """How a listener registers: prepended, and exempt from receiver filters."""

    prepend: bool = False
    glob: bool = False


@dataclass
class Hook:
    """One registered listener with the context that registered it."""

    ctx: Context
    callback: Callable[..., Any]
    options: EventOptions


def _route_update_listener(ctx: Context, name: str, listener: Callable[..., Any], options: EventOptions) -> Any:
    """Bail seam hook: divert non-global 'internal/update' listeners onto the
    registering fiber as reconfigure middleware (events.ts:54-60)."""
    if name != "internal/update" or options.glob:
        return None
    hooks = ctx.fiber._hooks.setdefault("internal/update", [])

    def forward() -> Callable[[], None]:
        if options.prepend:
            hooks.insert(0, listener)
        else:
            hooks.append(listener)

        def inverse() -> None:
            if listener in hooks:
                hooks.remove(listener)

        return inverse

    # Registered as a tracked effect, so the middleware dies with its fiber
    # like any other listener (the reference stores it on the fiber object,
    # which amounts to the same lifetime).
    return ctx.fiber.effect(forward, 'ctx.on("internal/update")')


def _chain_update_hooks(fiber: Fiber, config: Any, no_save: bool, next_: Callable[[], Any]) -> Any:
    """Global prepended 'internal/update' listener: thread the receiving
    fiber's own middleware into the waterfall (events.ts:62-69)."""
    pending = list(fiber._hooks.get("internal/update", []))

    def step() -> Any:
        if pending:
            return pending.pop(0)(fiber, config, no_save, step)
        return next_()

    return step()


class EventsService:
    """One Context tree's listener table and dispatch modes."""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self._hooks: dict[str, list[Hook]] = {}
        # A non-global 'internal/update' listener is per-fiber reconfigure
        # middleware: it rides on the registering fiber, and the global
        # prepended chain below threads it into every update waterfall
        # (events.ts:54-69).
        self.on("internal/listener", _route_update_listener, ctx=ctx)
        self.on("internal/update", _chain_update_hooks, prepend=True, glob=True, ctx=ctx)

    def _resolve(self, mode: str, args: Sequence[Any]) -> tuple[Any, list[Callable[..., Any]], list[Any]]:
        """Split off the optional leading receiver, fire the dispatch
        meta-event, and filter hooks through the receiver's event filter.

        The returned argument list keeps the receiver at its head, so
        listeners of a receiver-carrying dispatch see it as their first
        argument (the module docstring's receiver convention)."""
        rest = list(args)
        receiver = rest.pop(0) if rest and not isinstance(rest[0], str) else None
        name = rest.pop(0)
        if not name.startswith("internal/") and self._hooks.get("internal/dispatch"):
            self.emit("internal/dispatch", mode, name, list(rest), receiver)
        event_filter = getattr(receiver, "__event_filter__", None) if receiver is not None else None
        hooks = list(self._hooks.get(name, ()))
        callbacks = [
            hook.callback for hook in hooks if hook.options.glob or event_filter is None or event_filter(hook.ctx)
        ]
        if receiver is not None:
            rest.insert(0, receiver)
        return receiver, callbacks, rest

    def emit(self, *args: Any) -> None:
        """Fire synchronously in order; a listener exception aborts the loop."""
        _, callbacks, rest = self._resolve("emit", args)
        for callback in callbacks:
            callback(*rest)

    async def parallel(self, *args: Any) -> None:
        """Await every listener concurrently; aggregate all failures without cancelling peers."""
        _, callbacks, rest = self._resolve("emit", args)

        async def call(callback: Callable[..., Any]) -> Any:
            result = callback(*rest)
            return await result if inspect.isawaitable(result) else result

        results = await asyncio.gather(*(call(callback) for callback in callbacks), return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if not failures:
            return
        hard = next((failure for failure in failures if not isinstance(failure, Exception)), None)
        if hard is not None:
            # A KeyboardInterrupt-class failure must surface as itself, not
            # inside an Exception subclass; peers stay visible on the chain.
            peers = [failure for failure in failures if failure is not hard]
            if peers:
                raise hard from _ExceptionGroup("parallel event dispatch failed alongside", peers)
            raise hard
        raise _ExceptionGroup("parallel event dispatch failed", failures)

    async def serial(self, *args: Any) -> Any:
        """Await listeners one at a time; the first bailed result wins."""
        _, callbacks, rest = self._resolve("serial", args)
        for callback in callbacks:
            result = callback(*rest)
            if inspect.isawaitable(result):
                result = await result
            if is_bailed(result):
                return result
        return None

    def bail(self, *args: Any) -> Any:
        """Call listeners synchronously; the first bailed result wins."""
        _, callbacks, rest = self._resolve("bail", args)
        for callback in callbacks:
            result = callback(*rest)
            self._reject_awaitable("bail", result)
            if is_bailed(result):
                return result
        return None

    def waterfall(self, *args: Any) -> Any:
        """Thread a ``next`` continuation through the listeners.

        The last positional argument is the caller's innermost callback;
        every listener (and the innermost callback) receives the remaining
        arguments plus ``next``, and decides whether to call through.
        """
        _, callbacks, rest = self._resolve("waterfall", args)
        inner = rest.pop()
        queue = list(callbacks)
        # Awaitables minted by the caller's own innermost callback may pass
        # back through the chain; awaitables minted by a listener are the
        # silent never-awaited bug the sync-mode guard exists for.
        passthrough: list[Any] = []

        def next_() -> Any:
            if queue:
                result = queue.pop(0)(*rest)
                if not any(result is allowed for allowed in passthrough):
                    self._reject_awaitable("waterfall", result)
                return result
            result = inner(*rest)
            if inspect.isawaitable(result):
                passthrough.append(result)
            return result

        rest.append(next_)
        return next_()

    def _reject_awaitable(self, mode: str, result: Any) -> None:
        if not inspect.isawaitable(result) or isinstance(result, EffectHandle):
            # An effect handle awaits only to join async setup; as a listener
            # result it is an ordinary sync value, not an unawaited coroutine.
            return
        if inspect.iscoroutine(result):
            result.close()
        raise TypeError(f'sync dispatch mode "{mode}" cannot use an awaitable listener result; use serial or parallel')

    def on(
        self,
        name: str,
        listener: Callable[..., Any],
        *,
        prepend: bool = False,
        glob: bool = False,
        ctx: Context | None = None,
    ) -> EffectHandle:
        """Register a listener as a tracked effect of ``ctx``'s fiber.

        ``ctx`` names the registering scope explicitly because the traceable
        machinery that rebinds a service's context is deferred; Context's
        delegating method supplies itself.
        """
        owner = ctx if ctx is not None else self.ctx
        owner.fiber.assert_active()
        options = EventOptions(prepend=prepend, glob=glob)
        seam = self.bail(owner, "internal/listener", name, listener, options)
        if seam:
            return seam
        hooks = self._hooks.setdefault(name, [])

        def forward() -> Callable[[], None]:
            hook = Hook(ctx=owner, callback=listener, options=options)
            if prepend:
                hooks.insert(0, hook)
            else:
                hooks.append(hook)

            def inverse() -> None:
                for index, existing in enumerate(hooks):
                    if existing.callback is listener:
                        hooks.pop(index)
                        return

            return inverse

        return owner.fiber.effect(forward, f'ctx.on("{name}")')

    def once(
        self,
        name: str,
        listener: Callable[..., Any],
        *,
        prepend: bool = False,
        glob: bool = False,
        ctx: Context | None = None,
    ) -> EffectHandle:
        """Register a listener that disposes itself on its first fire."""

        def wrapper(*args: Any) -> Any:
            handle()
            return listener(*args)

        handle = self.on(name, wrapper, prepend=prepend, glob=glob, ctx=ctx)
        return handle
