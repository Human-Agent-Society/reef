"""Fiber: one plugin instantiation's state, and the effect engine under it.

A fiber owns everything one plugin did to shared state, as disposers held in
collection order. ``effect`` realizes the paper's Algorithm 1: run a setup
callback, collect every disposer it produces, and register the composite
inverse with the owning scope, so a child effect is itself an effect of its
parent and disposal replays inverses LIFO, at most once. Instantiation
(Algorithm 4) is an ordinary tracked effect of the parent fiber, which makes
parent teardown cascade to children through plain LIFO replay. A failed
activation recovers what it collected and lands FAILED without touching the
parent (the paper's L-Raise rule).

The lifecycle is reactive (the paper's Algorithm 5): a fiber's epoch is the
identity string of its resolved providers, and every provision change that
reaches the fiber through ``ReflectService.notify`` re-resolves and recomputes
it. An epoch that appears loads the fiber; one that changes or vanishes
unloads it, replaying its inverses, and reloads when a provider set is again
complete. PENDING is therefore a live waiting state: a fiber whose declared
dependency has no provider parks until one appears. The engine is sync-first:
transitions complete inline unless an effect form or disposer is genuinely
asynchronous, in which case ``_inertia`` carries the in-flight work.

The runtime never verifies that a disposer actually inverts its effect; that
witness is the author's obligation.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import inspect
import logging
from collections.abc import AsyncIterable, Awaitable, Callable, Coroutine, Iterable
from dataclasses import dataclass, field
from typing import Any

from .context import Context
from .utils import DisposableList

Disposer = Callable[[], Any]
EffectResult = Any
"""What a setup callback may return: a disposer, None, an awaitable of a
disposer, an iterable of disposers, or an async iterable of disposers."""

INACTIVE = "__INACTIVE__"
"""The epoch of a fiber whose provider set is incomplete (fiber.ts:101)."""


class InactiveEffectError(RuntimeError):
    """Raised when an effect is created on a disposed fiber."""


class ValidationError(TypeError):
    """Raised when a plugin config fails its declared Config validation."""

    def __init__(self, issues: list[tuple[str, str | None]]) -> None:
        lines = [f"  - {message} (at {path})" if path else f"  - {message}" for message, path in issues]
        super().__init__("invalid config:\n" + "\n".join(lines))


class FiberState(enum.Enum):
    """The lifecycle states of a fiber. PENDING is a live waiting state: the
    fiber has no complete provider set yet and loads when notify delivers
    one. LOADING and UNLOADING cover transitions with async work in flight;
    the sync-first engine passes through them inline otherwise."""

    PENDING = enum.auto()
    LOADING = enum.auto()
    ACTIVE = enum.auto()
    FAILED = enum.auto()
    DISPOSED = enum.auto()
    UNLOADING = enum.auto()


def resolve_config(runtime: Runtime, config: Any) -> Any:
    """Validate a raw config against the plugin's declared Config, if any.

    A pydantic model class validates through ``model_validate``; any other
    callable is applied to the raw config. Async validation is unsupported,
    as in the reference implementation.
    """
    if runtime.Config is None:
        return config
    validate = getattr(runtime.Config, "model_validate", None)
    target = validate if callable(validate) else runtime.Config
    try:
        result = target(config)
    except Exception as error:
        raise ValidationError(_issues_from(error)) from error
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise TypeError("Async config validation is not supported")
    return result


def _issues_from(error: Exception) -> list[tuple[str, str | None]]:
    details = getattr(error, "errors", None)
    if callable(details):
        with contextlib.suppress(Exception):
            found = details()
            if isinstance(found, list) and all(isinstance(item, dict) for item in found):
                return [
                    (str(item.get("msg", item)), ".".join(str(part) for part in item.get("loc", ())) or None)
                    for item in found
                ]
    return [(str(error), None)]


@dataclass
class EffectMeta:
    """One node of the inspectable effect tree: a label and collected children."""

    label: str
    children: list[EffectMeta] = field(default_factory=list)


def _spawn(coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    """Schedule a coroutine eagerly, as the reference host's promises are."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        coro.close()
        raise RuntimeError("async effect forms require a running asyncio event loop") from None
    return loop.create_task(coro)


def _retrieve(task: asyncio.Task[Any]) -> None:
    # Mark the exception retrieved so a recovered setup failure does not warn
    # at interpreter exit; awaiting the handle still re-raises it.
    if not task.cancelled():
        task.exception()


def _drive(
    result: EffectResult, collect: Callable[[Disposer], None], live: Callable[[], bool]
) -> Coroutine[Any, Any, None] | None:
    """Classify a setup callback's return value and collect its disposers.

    Sync forms are driven to completion here; async forms return a coroutine
    for the caller to schedule. Anything else raises ``Invalid effect``.
    """

    def safe_collect(value: Any) -> None:
        if callable(value):
            collect(value)
        elif value is not None:
            raise TypeError("Invalid effect")

    if callable(result):
        collect(result)
        return None
    if result is None:
        return None
    if isinstance(result, (str, bytes)):  # the reference host rejects primitives before probing iteration
        raise TypeError("Invalid effect")
    if inspect.isawaitable(result):

        async def drive_awaitable() -> None:
            safe_collect(await result)

        return drive_awaitable()
    if isinstance(result, Iterable):
        iterator = iter(result)
        while True:
            try:
                value = next(iterator)
            except StopIteration as stop:  # a generator's return value is a disposer too, as upstream
                safe_collect(stop.value)
                return None
            safe_collect(value)
    if isinstance(result, AsyncIterable):

        async def drive_async_iterable() -> None:
            iterator = result.__aiter__()
            while True:
                if not live():  # disposed mid-setup: keep only the collected prefix
                    return
                try:
                    value = await iterator.__anext__()
                except StopAsyncIteration:
                    return
                safe_collect(value)

        return drive_async_iterable()
    raise TypeError("Invalid effect")


def _chain(disposers: list[Disposer]) -> Coroutine[Any, Any, None] | None:
    """Run disposers in list order; a failure propagates and stops the chain.

    Sync disposers run inline; from the first async result on, the rest
    chain sequentially inside the returned coroutine.
    """
    pending = list(disposers)
    while pending:
        result = pending.pop(0)()
        if inspect.isawaitable(result):
            return _finish_chain(result, pending)
    return None


async def _finish_chain(first: Awaitable[Any], rest: list[Disposer]) -> None:
    await first
    while rest:
        result = rest.pop(0)()
        if inspect.isawaitable(result):
            await result


def _dispose_logged(disposer: Disposer, logger: logging.Logger) -> Awaitable[Any] | None:
    try:
        result = disposer()
    except Exception:
        logger.exception("effect disposer failed; teardown continues")
        return None
    return result if inspect.isawaitable(result) else None


async def _await_logged(awaitable: Awaitable[Any], logger: logging.Logger) -> None:
    try:
        await awaitable
    except Exception:
        logger.exception("effect disposer failed; teardown continues")


def _drain_logged(disposers: list[Disposer], logger: logging.Logger) -> Coroutine[Any, Any, None] | None:
    """Run a fiber's collected disposers; log each failure and keep going."""
    pending = list(disposers)
    while pending:
        result = _dispose_logged(pending.pop(0), logger)
        if result is not None:
            return _finish_drain(result, pending, logger)
    return None


async def _finish_drain(first: Awaitable[Any], rest: list[Disposer], logger: logging.Logger) -> None:
    await _await_logged(first, logger)
    while rest:
        result = _dispose_logged(rest.pop(0), logger)
        if result is not None:
            await _await_logged(result, logger)


class EffectHandle:
    """The live handle of one tracked effect.

    Calling the handle disposes the effect, at most once; the returned
    awaitable, if any, settles when async disposers finish. Awaiting the
    handle joins async setup, re-raises its failure, and evaluates to the
    handle itself so the result is again the disposer.
    """

    def __init__(self, fiber: Fiber, label: str) -> None:
        self.meta = EffectMeta(label)
        self._fiber = fiber
        self._collected: list[Disposer] = []
        self._armed = True
        self._live = True
        self._task: asyncio.Task[Any] | None = None

    def _collect(self, disposer: Disposer) -> None:
        self._collected.append(disposer)
        # Re-parent a nested handle: it leaves the fiber's top level and
        # disposes at its LIFO position inside this effect instead.
        self._fiber._disposables.delete(disposer)
        if isinstance(disposer, EffectHandle):
            self.meta.children.append(disposer.meta)

    def _dispose_collected(self) -> Coroutine[Any, Any, None] | None:
        pending = self._collected[:]
        self._collected.clear()
        pending.reverse()
        return _chain(pending)

    def __call__(self) -> Awaitable[None] | None:
        if not self._armed:
            return None
        self._armed = False
        self._live = False
        task = self._task
        if task is not None and not task.done():

            async def dispose_after_setup() -> None:
                with contextlib.suppress(BaseException):
                    await task
                tail = self._dispose_collected()
                if tail is not None:
                    await tail

            return _spawn(dispose_after_setup())
        chain = self._dispose_collected()
        return None if chain is None else _spawn(chain)

    async def _join(self) -> EffectHandle:
        if self._task is not None:
            await self._task
        return self

    def __await__(self) -> Any:
        return self._join().__await__()


class Fiber:
    """A node of the composition tree: root, or one plugin instantiation.

    ``_resolved`` is the live resolution of the declared injects, kept
    current by ``_check_impl`` under notify; ``store`` is the committed
    snapshot of it taken when a load begins, and survives until that load's
    teardown finishes, so a disposer can still read the services its fiber
    depended on. The epoch string over ``_resolved`` drives every
    transition; see the module docstring.
    """

    def __init__(self, parent: Context, config: Any, inject: dict[str, Any], runtime: Runtime | None) -> None:
        self.parent = parent
        self.config = config
        self.inject = inject
        self.runtime = runtime
        self.entry: Any = None
        self.error: BaseException | None = None
        self.store: dict[str, Impl] | None = None
        self._resolved: dict[str, Impl] = {}
        self._epoch = INACTIVE
        self._disposables = DisposableList()
        self._hooks: dict[str, list[Callable[..., Any]]] = {}
        self._inertia: asyncio.Task[Any] | None = None
        self._handle: EffectHandle | None = None
        self._dispose_requested = False
        self._remove_from_runtime: Callable[[], None] | None = None
        if runtime is not None:
            self.uid: int | None = parent.registry.next_uid()
            self.ctx: Context = parent.extend(fiber=self)
            # Per-key inject configs overlay the plugin's own intercept
            # layer, resolved by Service.resolve_config (fiber.ts:137-144).
            for name, dep_config in inject.items():
                if dep_config is not None:
                    self.ctx._intercept[name] = dep_config
            self.state = FiberState.PENDING
            self.ctx.events.emit("internal/plugin", self)
            for name in inject:
                self._check_impl(name)
            self._handle = parent.fiber.effect(self._load, "ctx.plugin()")
            if self._dispose_requested:
                self._handle()
        else:
            self.uid = 0
            self.ctx = parent
            self.state = FiberState.ACTIVE
            self.store = {}
            self._epoch = ""

    @property
    def name(self) -> str:
        """The nearest named runtime up the parent chain, or ``root``."""
        fiber: Fiber = self
        while True:
            if fiber.runtime is not None and fiber.runtime.name:
                return fiber.runtime.name
            parent_fiber = fiber.parent.fiber
            if parent_fiber is fiber:
                return "root"
            fiber = parent_fiber

    def assert_active(self) -> None:
        if self.uid is None:
            raise InactiveEffectError("cannot create effect on inactive context")

    def effect(self, run: Callable[[], EffectResult], label: str = "anonymous") -> EffectHandle:
        """Track one effect on this fiber; see EffectHandle for the contract.

        A synchronous setup failure recovers the collected prefix and
        re-raises; an async one recovers, logs, and re-raises on await.
        """
        self.assert_active()
        handle = EffectHandle(self, label)
        try:
            coro = _drive(run(), handle._collect, lambda: handle._live)
        except BaseException:
            recovery = handle._dispose_collected()
            if recovery is not None:  # best effort: an async disposer without a loop cannot be replayed
                with contextlib.suppress(RuntimeError):
                    _spawn(recovery)
            raise
        if coro is not None:

            async def setup() -> None:
                try:
                    await coro
                except BaseException as error:
                    handle._armed = False
                    handle._live = False
                    self.ctx.logger.error("async effect %s failed during setup: %r", label, error)
                    tail = handle._dispose_collected()
                    if tail is not None:
                        await _await_logged(tail, self.ctx.logger)
                    raise

            wrapped = setup()
            try:
                handle._task = _spawn(wrapped)
            except RuntimeError:
                coro.close()  # _spawn closed only the wrapper; drop the driver too
                raise
            handle._task.add_done_callback(_retrieve)
        # Disposing the effect also unregisters it from the fiber: the removal
        # token rides in the handle's own list (upstream fiber.ts:338).
        handle._collected.append(self._disposables.push(handle))
        return handle

    def get_effects(self) -> list[EffectMeta]:
        """The labeled tree of this fiber's live top-level effects."""
        return [handle.meta for handle in self._disposables if isinstance(handle, EffectHandle)]

    def dispose(self) -> Awaitable[Any] | None:
        """Retire this fiber, replaying its disposers LIFO.

        On the root there is no owning effect to invert: disposal is a
        restart, draining the accumulated effects and reactivating (the
        reference's fiber.ts:211).
        """
        if self.runtime is None:
            return self.restart()
        handle = self._handle
        if handle is None:
            # The synchronous load cascade is still inside the parent's
            # effect() call; the reference defers plugin bodies to a
            # microtask (fiber.ts:419) so its dispose always exists first.
            # Honor the self dispose as soon as the owning handle lands.
            self._dispose_requested = True
            return None
        return handle()

    def restart(self) -> Awaitable[Any] | None:
        """Retire and re-apply this fiber: unload, replay inverses, reload.

        Returns None when the round trip completed synchronously, else an
        awaitable that joins it and re-raises the stored error if any.
        """
        self.assert_active()
        self._set_epoch(INACTIVE)
        self._refresh()
        if self._inertia is None:
            return None
        task = _spawn(self.wait())
        task.add_done_callback(_retrieve)
        return task

    def update(self, config: Any, no_save: bool = False) -> Any:
        """Reconfigure: validate, thread the 'internal/update' waterfall,
        then restart with the new config (fiber.ts:476-485).

        A non-global 'internal/update' listener on this fiber intercepts the
        chain and may swallow the restart; ``no_save`` tells persistence
        middleware not to write the change back to its source of truth.
        """
        self.assert_active()
        runtime = self.runtime
        if runtime is None:
            raise RuntimeError("the root fiber has no config to update")
        config = resolve_config(runtime, config)

        def inner(*_: Any) -> Any:
            self.config = config
            self.error = None
            return self.restart()

        return self.ctx.waterfall(self, "internal/update", config, no_save, inner)

    async def wait(self) -> Fiber:
        """Join outstanding async work, then re-raise the stored error if any."""
        while self._inertia is not None:
            await self._inertia
        if self.error is not None:
            raise self.error
        return self

    def __await__(self) -> Any:
        return self.wait().__await__()

    # -- the epoch engine (fiber.ts:348-458) --------------------------------

    def _get_state(self) -> FiberState:
        if self.uid is None:
            return FiberState.DISPOSED
        if self.error is not None:
            return FiberState.FAILED
        if self._epoch != INACTIVE:
            return FiberState.ACTIVE
        return FiberState.PENDING

    def _update_state(self, callback: Callable[[], FiberState | None] | None = None) -> None:
        old = self.state
        state = callback() if callback is not None else None
        self.state = state if state is not None else self._get_state()
        if old is self.state:
            return
        self.ctx.events.emit("internal/status", self, old)
        # Only a crossing between ACTIVE and non-ACTIVE changes what this
        # fiber provides, so only that crossing notifies (fiber.ts:362-368).
        if old is not FiberState.ACTIVE and self.state is not FiberState.ACTIVE:
            return
        reflect = self.ctx.reflect
        for impl in list(reflect.store.values()):
            if impl.fiber is self:
                reflect.notify([impl.name], ctx=self.ctx)

    def _check_impl(self, name: str) -> None:
        """Re-resolve one declared dependency into ``_resolved`` (fiber.ts:371-383).

        A binding whose check predicate fails or raises does not count; the
        raise is logged against the provider, whose predicate it is. The
        predicate receives the resolving scope, so it can read the intercept
        overlays of the fiber asking (the reference reaches them through
        traceable ``this`` rebinding instead).
        """
        impl = self.ctx.reflect._get_impl(name, strict=True, ctx=self.ctx)
        if impl is None:
            self._resolved.pop(name, None)
            return
        if impl.check is not None:
            try:
                passed = bool(impl.check(self.ctx))
            except Exception:
                impl.fiber.ctx.logger.exception('check predicate of service "%s" raised', name)
                self._resolved.pop(name, None)
                return
            if not passed:
                self._resolved.pop(name, None)
                return
        self._resolved[name] = impl

    def _refresh(self) -> None:
        """Recompute the epoch over the resolved providers (fiber.ts:385-397)."""
        epoch = ""
        for name in self.inject:
            impl = self._resolved.get(name)
            if impl is None:
                epoch = INACTIVE
                break
            epoch += ":" + str(impl.fiber.uid)
        self._set_epoch(epoch)

    def _set_epoch(self, epoch: str) -> None:
        old = self._epoch
        if epoch == old:
            return
        # A failed fiber recovers only through update(), which clears the error (fiber.ts:402, cordis 10194de).
        if self.error is not None:
            return
        self._epoch = epoch
        if self._inertia is not None:
            return  # the running chain reads the new epoch at its next decision point
        if old == INACTIVE:
            self._update_state(lambda: FiberState.LOADING)
            self._reload()
        else:
            self._update_state(lambda: FiberState.UNLOADING)
            self._unload()

    def _reload(self) -> None:
        """Commit the resolved view and run the plugin callback (fiber.ts:415-435).

        Any failure is contained: logged, stored on the fiber, its partial
        effects recovered by the chained unload; parent and siblings are
        unaffected (the paper's L-Raise rule).
        """
        self.store = dict(self._resolved)
        old_epoch = self._epoch
        coro: Coroutine[Any, Any, None] | None = None
        try:
            coro = _drive(self._instantiate(), self._collect, lambda: self._epoch == old_epoch)
        except BaseException as error:
            self._record_failure(error)
        if coro is None:
            self._after_reload(old_epoch)
            return

        async def settle() -> None:
            try:
                await coro
            except BaseException as error:
                self._record_failure(error)
            finally:
                if self._inertia is asyncio.current_task():
                    self._inertia = None
            self._after_reload(old_epoch)

        wrapped = settle()
        try:
            self._inertia = _spawn(wrapped)
        except RuntimeError as error:
            coro.close()  # _spawn closed only the wrapper; drop the driver too
            self._record_failure(error)
            self._after_reload(old_epoch)

    def _after_reload(self, old_epoch: str) -> None:
        if self._epoch == old_epoch:
            self._inertia = None
            self._update_state()
            return
        # The epoch moved while loading (a provider changed, or the fiber
        # was retired): what this load installed must come back out.
        self._update_state(lambda: FiberState.UNLOADING)
        self._unload()

    def _instantiate(self) -> EffectResult:
        runtime = self.runtime
        if runtime is None:
            return None
        callback = runtime.callback
        result = callback(self.ctx, self.config)
        if inspect.isclass(callback):
            # A class plugin's teardown comes from effects registered by its
            # constructor, its init hooks, or its init method's iteration
            # (fiber.ts:150-156), never from the constructor's return value.
            for hook in getattr(result, "__compose_init_hooks__", ()) or ():
                hook()
            init = getattr(result, "__compose_init__", None)
            return init() if callable(init) else None
        return result

    def _collect(self, disposer: Disposer) -> None:
        self._disposables.push(disposer)

    def _record_failure(self, error: BaseException) -> None:
        self.ctx.logger.error("plugin fiber <%s> failed to load: %r", self.name, error)
        self.error = error
        self._epoch = INACTIVE

    def _unload(self) -> None:
        """Replay the collected inverses LIFO, then decide the next phase."""
        chain = _drain_logged(self._disposables.clear(), self.ctx.logger)
        if chain is None:
            self._after_unload()
            return

        async def settle() -> None:
            try:
                await chain
            finally:
                if self._inertia is asyncio.current_task():
                    self._inertia = None
            self._after_unload()

        wrapped = settle()
        try:
            self._inertia = _spawn(wrapped)
        except RuntimeError:
            chain.close()
            self.ctx.logger.error("async teardown needs a running event loop; effects were not fully recovered")
            self._after_unload()

    def _after_unload(self) -> None:
        # The committed view outlives the disposers (they may still read
        # dependencies); it is released only here, after the replay.
        self.store = None
        if self._epoch == INACTIVE:
            self._inertia = None
            self._update_state()
            return
        # A complete provider set reappeared while unloading: load again.
        self._update_state(lambda: FiberState.LOADING)
        self._reload()

    # -- instantiation as a tracked effect of the parent (fiber.ts:170-199) --

    def _load(self) -> Disposer:
        """Forward direction of the ``ctx.plugin()`` effect on the parent."""
        runtime = self.runtime
        if runtime is None:
            raise RuntimeError("a child fiber cannot load without a runtime")
        self._remove_from_runtime = runtime.fibers.push(self)
        try:
            self.config = resolve_config(runtime, self.config)
            self._refresh()
        except BaseException as error:
            # A config error parks the fiber with the error stored; nothing
            # was installed yet, so there is nothing to recover.
            self.ctx.logger.error("plugin fiber <%s> failed to load: %r", self.name, error)
            self.error = error
        return self._retire

    def _retire(self) -> Awaitable[Any] | None:
        """Inverse of the load effect: retire, unregister, replay disposers."""
        runtime = self.runtime
        if runtime is None:
            raise RuntimeError("a child fiber cannot retire without a runtime")
        self.uid = None
        self.ctx.events.emit("internal/plugin", self)
        registry = self.ctx.registry
        if registry.has(runtime.callback):
            remove = self._remove_from_runtime
            if remove is not None:
                remove()
            if not len(runtime.fibers):
                registry.delete(runtime.callback)
        self._set_epoch(INACTIVE)
        if self._inertia is not None:

            async def finish() -> None:
                while self._inertia is not None:
                    await self._inertia
                self._update_state()

            return _spawn(finish())
        # A fiber retired while PENDING or FAILED had no transition to run;
        # reconcile the state field so the corpse reads DISPOSED.
        self._update_state()
        return None


# Type-only imports placed after all definitions: reflect and registry import
# Fiber (and EffectHandle, FiberState) at module level for runtime use, so
# importing them at the top of this module would create a cycle before Fiber
# is defined.  These two names appear only in annotations (made lazy by
# ``from __future__ import annotations``), so deferring the import to here
# satisfies mypy without triggering the cycle at runtime.
from .reflect import Impl
from .registry import Runtime
