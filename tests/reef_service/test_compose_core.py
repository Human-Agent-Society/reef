"""Guarantees of reef.train.cordis_backend.compose, one test per ported guarantee."""

from __future__ import annotations

import asyncio
import logging
import sys

import pytest

from reef.train.cordis_backend import compose
from reef.train.cordis_backend.compose import FiberState

if sys.version_info >= (3, 11):
    from builtins import ExceptionGroup
else:
    from reef.train.cordis_backend.compose.events import _ExceptionGroup as ExceptionGroup


def test_effect_disposers_run_lifo() -> None:
    root = compose.Context()
    log: list[str] = []

    def apply(ctx, config):
        yield lambda: log.append("first")
        yield lambda: log.append("second")

    fiber = root.plugin(apply)
    assert fiber.state is FiberState.ACTIVE
    fiber.dispose()
    assert log == ["second", "first"]
    assert fiber.state is FiberState.DISPOSED
    assert fiber.dispose() is None  # at most once: a second call is a no-op
    assert log == ["second", "first"]


def test_generator_effect_partial_rollback() -> None:
    root = compose.Context()
    log: list[str] = []

    def setup():
        yield lambda: log.append("a")
        yield lambda: log.append("b")
        raise RuntimeError("mid-activation")

    with pytest.raises(RuntimeError, match="mid-activation"):
        root.effect(setup)
    assert log == ["b", "a"]


def test_disposer_failure_logged_and_teardown_continues(caplog: pytest.LogCaptureFixture) -> None:
    root = compose.Context()
    log: list[str] = []

    def apply(ctx, config):
        yield lambda: log.append("early")

        def bad() -> None:
            raise RuntimeError("disposer boom")

        yield bad

    fiber = root.plugin(apply)
    with caplog.at_level(logging.ERROR, logger="reef.train.cordis_backend.compose"):
        fiber.dispose()
    assert log == ["early"]
    assert any("teardown continues" in record.message for record in caplog.records)
    assert fiber.state is FiberState.DISPOSED


def test_listener_dies_with_its_fiber() -> None:
    root = compose.Context()
    hits: list[tuple] = []

    def apply(ctx, config):
        ctx.on("ping", lambda *args: hits.append(args))

    fiber = root.plugin(apply)
    root.emit("ping", 1)
    assert hits == [(1,)]
    fiber.dispose()
    root.emit("ping", 2)
    assert hits == [(1,)]


def test_waterfall_chains_through_next() -> None:
    root = compose.Context()
    order: list[str] = []

    def outer(seed, next_):
        order.append("outer-pre")
        result = next_()
        order.append("outer-post")
        return result + ":outer"

    def mid(seed, next_):
        order.append("mid")
        return next_() + ":mid"

    root.on("chain", outer)
    root.on("chain", mid)
    result = root.waterfall("chain", "seed", lambda seed, next_: seed + ":base")
    assert result == "seed:base:mid:outer"
    assert order == ["outer-pre", "mid", "outer-post"]


def test_bail_short_circuits() -> None:
    root = compose.Context()
    calls: list[str] = []

    root.on("evt", lambda: calls.append("a"))  # returns None: not bailed
    root.on("evt", lambda: "hit")
    root.on("evt", lambda: calls.append("never"))
    assert root.bail("evt") == "hit"
    assert calls == ["a"]


def test_parallel_aggregates_listener_errors() -> None:
    root = compose.Context()
    calls: list[str] = []

    async def bad_one() -> None:
        raise ValueError("one")

    async def bad_two() -> None:
        raise KeyError("two")

    root.on("evt", bad_one)
    root.on("evt", bad_two)
    root.on("evt", lambda: calls.append("peer"))
    with pytest.raises(ExceptionGroup) as caught:
        asyncio.run(root.parallel("evt"))
    assert {type(error) for error in caught.value.exceptions} == {ValueError, KeyError}
    assert calls == ["peer"]  # peers are not cancelled by failures


def test_serial_bails_on_async_listener() -> None:
    root = compose.Context()
    order: list[str] = []

    async def first() -> None:
        order.append("first")

    async def second() -> str:
        order.append("second")
        return "stop"

    root.on("evt", first)
    root.on("evt", second)
    root.on("evt", lambda: order.append("never"))
    assert asyncio.run(root.serial("evt")) == "stop"
    assert order == ["first", "second"]


def test_double_provide_names_the_provider() -> None:
    root = compose.Context()

    def alpha(ctx, config):
        ctx.provide("dup", 1)

    def beta(ctx, config):
        ctx.provide("dup", 2)

    root.plugin(alpha)
    fiber = root.plugin(beta)
    assert fiber.state is FiberState.FAILED
    assert 'service "dup" has been registered at <alpha>' in str(fiber.error)
    assert root.get("dup") == 1  # the first provider is untouched


def test_undeclared_access_raises_targeted_error() -> None:
    root = compose.Context()
    grabbed: dict[str, compose.Context] = {}

    def apply(ctx, config):
        grabbed["ctx"] = ctx

    root.plugin(apply)
    with pytest.raises(compose.ServiceAccessError, match='cannot get property "mystery" without inject'):
        _ = grabbed["ctx"].mystery
    with pytest.raises(compose.ServiceAccessError, match='cannot get property "mystery" without inject'):
        _ = root.mystery
    assert not hasattr(root, "mystery")  # ServiceAccessError is an AttributeError


def test_cycle_detected_from_declarations_alone() -> None:
    root = compose.Context()
    ran: list[str] = []

    def plugin_a(ctx, config):
        ran.append("a")

    plugin_a.provide = "service_a"
    plugin_a.inject = ["service_b"]

    def plugin_b(ctx, config):
        ran.append("b")

    plugin_b.provide = "service_b"
    plugin_b.inject = ["service_a"]

    with pytest.raises(compose.CycleError, match="dependency cycle among plugins"):
        root.registry.load([(plugin_a, None), (plugin_b, None)])
    assert ran == []  # detected from declarations, before any apply runs


def test_missing_provider_parks_consumer_pending_not_crashed() -> None:
    root = compose.Context()
    ran: list[str] = []

    def needy(ctx, config):
        ran.append("needy")

    needy.inject = ["ghost"]

    fiber = root.plugin(needy)  # does not raise
    assert fiber.state is FiberState.PENDING  # a live waiting state, not a failure
    assert fiber.error is None
    assert ran == []
    with pytest.raises(compose.ServiceAccessError, match='cannot get required service "ghost" in inactive context'):
        _ = fiber.ctx.ghost
    assert root.fiber.state is FiberState.ACTIVE  # the parent is unaffected

    def bystander(ctx, config):
        ran.append("bystander")

    root.plugin(bystander)
    assert ran == ["bystander"]

    with pytest.raises(compose.MissingProviderError, match='"ghost"'):
        root.registry.load([(needy, None)])  # the batch gate still reports up front


def test_committed_view_readable_during_teardown() -> None:
    root = compose.Context()
    seen: list[str] = []

    def provider(ctx, config):
        ctx.provide("svc", "payload")

    provider.provide = "svc"

    def consumer(ctx, config):
        ctx.effect(lambda: lambda: seen.append(ctx.svc))

    consumer.inject = ["svc"]

    fibers = root.registry.load([(consumer, None), (provider, None)])
    assert [fiber.state for fiber in fibers] == [FiberState.ACTIVE, FiberState.ACTIVE]
    root.fiber.dispose()  # LIFO: the consumer unloads first, provider after
    assert seen == ["payload"]


def test_pending_dependent_activates_when_provider_appears() -> None:
    root = compose.Context()
    log: list[str] = []

    def consumer(ctx, config):
        log.append(f"load:{ctx.db}")
        yield lambda: log.append("unload")

    consumer.inject = ["db"]

    fiber = root.plugin(consumer)
    assert fiber.state is FiberState.PENDING
    assert log == []

    def provider(ctx, config):
        ctx.provide("db", "v1")

    root.plugin(provider)  # notify wakes the parked dependent
    assert fiber.state is FiberState.ACTIVE
    assert log == ["load:v1"]


def test_provider_withdrawal_unloads_dependent_with_committed_view() -> None:
    root = compose.Context()
    log: list[str] = []

    def provider(ctx, config):
        ctx.provide("db", "v1")

    def consumer(ctx, config):
        yield lambda: log.append(f"unload:{ctx.db}")  # reads the committed view during teardown

    consumer.inject = ["db"]

    provider_fiber = root.plugin(provider)
    consumer_fiber = root.plugin(consumer)
    provider_fiber.dispose()
    assert log == ["unload:v1"]
    assert consumer_fiber.state is FiberState.PENDING  # parked again, not failed
    assert provider_fiber.state is FiberState.DISPOSED


def test_provider_replacement_restarts_dependent() -> None:
    root = compose.Context()
    log: list[str] = []

    def consumer(ctx, config):
        log.append(f"load:{ctx.db}")
        yield lambda: log.append(f"unload:{ctx.db}")

    consumer.inject = ["db"]

    def provider_one(ctx, config):
        ctx.provide("db", "v1")

    def provider_two(ctx, config):
        ctx.provide("db", "v2")

    fiber = root.plugin(consumer)
    root.plugin(provider_one).dispose()
    root.plugin(provider_two)
    assert fiber.state is FiberState.ACTIVE
    assert log == ["load:v1", "unload:v1", "load:v2"]  # a full restart, not a value swap


def test_in_place_set_is_not_a_notify_edge() -> None:
    root = compose.Context()
    loads: list[str] = []

    def provider(ctx, config):
        ctx.provide("db", "v1")

    def consumer(ctx, config):
        loads.append(ctx.db)

    consumer.inject = ["db"]

    provider_fiber = root.plugin(provider)
    root.plugin(consumer)
    provider_fiber.ctx.db = "v2"  # provider overwrites in place
    assert loads == ["v1"]  # no restart: only identity changes notify
    assert root.get("db") == "v2"  # but reads see the new value


def test_internal_service_event_announces_bindings() -> None:
    root = compose.Context()
    seen: list[tuple[str, object]] = []

    root.on("internal/service", lambda ctx, name, value: seen.append((name, value)))

    def provider(ctx, config):
        ctx.provide("db", "v1")

    fiber = root.plugin(provider)
    assert ("db", "v1") in seen
    seen.clear()
    fiber.dispose()
    assert ("db", None) in seen  # withdrawal announces the absent binding


def test_update_reconfigures_through_the_waterfall() -> None:
    root = compose.Context()
    log: list[object] = []

    def apply(ctx, config):
        log.append(("load", config))
        yield lambda: log.append(("unload", config))

    fiber = root.plugin(apply, {"n": 1})
    saves: list[object] = []
    root.on(
        "internal/update",
        lambda who, config, no_save, next_: (saves.append((config, no_save)), next_())[1],
        glob=True,
    )
    fiber.update({"n": 2})
    assert log == [("load", {"n": 1}), ("unload", {"n": 1}), ("load", {"n": 2})]
    assert saves == [({"n": 2}, False)]
    assert fiber.config == {"n": 2}


def test_fiber_scoped_update_hook_can_swallow_the_restart() -> None:
    root = compose.Context()
    log: list[object] = []

    def apply(ctx, config):
        # A non-global 'internal/update' listener is per-fiber middleware; not
        # calling through means the fiber itself does not restart.
        ctx.on("internal/update", lambda fiber, config, no_save, next_: log.append(("intercepted", config)))
        log.append("load")
        yield lambda: log.append("unload")

    fiber = root.plugin(apply)
    fiber.update({"n": 2})
    assert log == ["load", ("intercepted", {"n": 2})]
    assert fiber.config is None  # the swallowed chain never committed the config


def test_restart_replays_and_reapplies() -> None:
    root = compose.Context()
    log: list[str] = []

    def apply(ctx, config):
        log.append("load")
        yield lambda: log.append("unload")

    fiber = root.plugin(apply)
    assert fiber.restart() is None  # fully synchronous round trip
    assert log == ["load", "unload", "load"]
    assert fiber.state is FiberState.ACTIVE


def test_isolate_gives_a_name_a_private_realm() -> None:
    root = compose.Context()

    def provider(ctx, config):
        ctx.provide("db", config)

    def reader(ctx, config):
        config.append(ctx.db)

    reader.inject = ["db"]

    iso = root.isolate("db")
    assert root.plugin(provider, "outer").state is FiberState.ACTIVE
    assert iso.plugin(provider, "inner").state is FiberState.ACTIVE  # no double-provide across realms
    outer_seen: list[str] = []
    inner_seen: list[str] = []
    root.plugin(reader, outer_seen)
    iso.plugin(reader, inner_seen)
    assert outer_seen == ["outer"]
    assert inner_seen == ["inner"]

    leaked: list[object] = []

    def undeclared(ctx, config):
        leaked.append(getattr(ctx, "db", "blocked"))

    iso.plugin(undeclared)
    assert leaked == ["blocked"]  # the realm boundary cuts the ambient walk

    shared = compose.RealmKey("db")
    left = root.isolate("db", shared)
    right = root.isolate("db", shared)
    assert left.plugin(provider, "shared").state is FiberState.ACTIVE
    seen: list[str] = []
    right.plugin(reader, seen)
    assert seen == ["shared"]  # the same label joins realms


def test_intercept_layers_resolve_config() -> None:
    root = compose.Context()

    class Cache(compose.Service):
        provide = "cache"

        def __init__(self, ctx, config):
            super().__init__(ctx)

    root.plugin(Cache)
    service = root.get("cache")
    scoped = root.intercept("cache", {"ttl": 60}).intercept("cache", {"ttl": 30, "depth": 2})
    assert service.resolve_config(ctx=scoped) == {"ttl": 30, "depth": 2}  # nearest layer wins
    assert service.resolve_config(base={"ttl": 90, "size": 1}, ctx=scoped) == {"ttl": 30, "size": 1, "depth": 2}
    assert service.resolve_config(head={"ttl": 5}, ctx=scoped) == {"ttl": 5, "depth": 2}

    seen: list[object] = []

    def consumer(ctx, config):
        seen.append(ctx.cache.resolve_config(ctx=ctx))

    consumer.inject = {"cache": {"ttl": 10}}  # per-key inject config becomes an intercept layer
    root.plugin(consumer)
    assert seen == [{"ttl": 10}]


def test_service_check_predicate_gates_resolution() -> None:
    root = compose.Context()
    gate = {"open": False}

    class Gated(compose.Service):
        provide = "gated"

        def __init__(self, ctx, config):
            super().__init__(ctx)

        def __compose_check__(self, ctx):
            return gate["open"]

    root.plugin(Gated)
    loads: list[object] = []

    def consumer(ctx, config):
        loads.append(ctx.gated)

    consumer.inject = ["gated"]
    fiber = root.plugin(consumer)
    assert fiber.state is FiberState.PENDING  # provided but not resolvable
    gate["open"] = True
    root.reflect.notify(["gated"])
    assert fiber.state is FiberState.ACTIVE
    assert len(loads) == 1


def test_service_base_completion() -> None:
    root = compose.Context()

    class Cache(compose.Service):
        provide = "cache"  # class-level name (service.ts:19)

        def __init__(self, ctx, config):
            super().__init__(ctx)
            self.hits = 0

        def __call__(self, key):  # callable services are native in Python
            return f"hit:{key}"

    root.plugin(Cache)
    service = root.get("cache")
    assert service("k") == "hit:k"
    view = service.extend(hits=99)
    assert view.hits == 99
    assert service.hits == 0
    service.label = "base"  # a live view: later base mutations show through
    assert view.label == "base"


def test_class_plugin_init_hooks_and_init_generator() -> None:
    root = compose.Context()
    log: list[str] = []

    class Plugin:
        def __init__(self, ctx, config):
            self.__compose_init_hooks__ = [lambda: log.append("hook")]

        def __compose_init__(self):
            yield lambda: log.append("dispose")
            log.append("init")

    fiber = root.plugin(Plugin)
    assert log == ["hook", "init"]
    fiber.dispose()
    assert log == ["hook", "init", "dispose"]  # the init generator's yield is a tracked effect


def test_get_and_set_seams_interpose() -> None:
    root = compose.Context()
    root.on("internal/get", lambda ctx, name, next_: "served" if name == "phantom" else next_())
    reads: list[object] = []

    def reader(ctx, config):
        reads.append(ctx.phantom)

    root.plugin(reader)
    assert reads == ["served"]

    writes: list[object] = []
    root.on("internal/set", lambda ctx, name, value, next_: (writes.append((name, value)), next_())[1])

    def provider(ctx, config):
        ctx.provide("db", "v1")
        ctx.db = "v2"

    root.plugin(provider)
    assert writes == [("db", "v2")]
    assert root.get("db") == "v2"


def test_async_withdrawal_waits_for_dependent_drain() -> None:
    order: list[str] = []

    async def main() -> None:
        root = compose.Context()

        def provider(ctx, config):
            ctx.provide("db", "v1")

        async def slow_teardown(ctx):
            await asyncio.sleep(0)
            order.append(f"drained:{ctx.db}")  # the committed view survives the async drain

        def consumer(ctx, config):
            async def setup():
                return lambda: slow_teardown(ctx)

            ctx.effect(setup)

        consumer.inject = ["db"]
        provider_fiber = root.plugin(provider)
        consumer_fiber = root.plugin(consumer)
        await consumer_fiber.wait()
        pending = provider_fiber.dispose()
        assert pending is not None  # the provide inverse waits for the dependent
        await pending
        order.append("provider-released")

    asyncio.run(main())
    assert order == ["drained:v1", "provider-released"]


def test_two_context_trees_share_no_state() -> None:
    one = compose.Context()
    two = compose.Context()

    def provide_one(ctx, config):
        ctx.provide("svc", "one")

    def provide_two(ctx, config):
        ctx.provide("svc", "two")

    assert one.plugin(provide_one).state is FiberState.ACTIVE
    assert two.plugin(provide_two).state is FiberState.ACTIVE  # no cross-tree double-provide
    assert one.get("svc") == "one"
    assert two.get("svc") == "two"

    hits: list[str] = []
    one.on("evt", lambda: hits.append("one"))
    two.emit("evt")
    assert hits == []
    one.emit("evt")
    assert hits == ["one"]


def test_self_dispose_inside_the_sync_load_cascade_retires_cleanly() -> None:
    # The reference defers plugin bodies to a microtask so fiber.dispose
    # always exists before user code runs (fiber.ts:419); the sync-first
    # engine must honor a self dispose requested before the handle lands.
    root = compose.Context()

    def quitter(ctx, config):
        assert ctx.fiber.dispose() is None

    fiber = root.plugin(quitter)
    assert fiber.state is FiberState.DISPOSED


def test_dependent_disposing_its_provider_during_notify_retires_the_provider() -> None:
    root = compose.Context()
    seen: list[str] = []

    def follower(ctx, config):
        seen.append(ctx.svc)
        ctx.get("svc_fiber").dispose()

    follower.inject = ["svc"]
    root.plugin(follower)

    def provider(ctx, config):
        ctx.provide("svc", "value")
        ctx.provide("svc_fiber", ctx.fiber)

    fiber = root.plugin(provider)
    assert seen == ["value"]
    assert fiber.state is FiberState.DISPOSED
