"""Guarantees of reef.train.cordis_backend.compose.loader, one test per ported guarantee."""

from __future__ import annotations

import asyncio
import gc
import logging

import pytest

from reef.train.cordis_backend import compose
from reef.train.cordis_backend.compose import FiberState
from reef.train.cordis_backend.compose.loader import Loader


def build(log):
    """A root context with a loader over a small plugin catalog."""

    def alpha(ctx, config):
        log.append(("alpha", config))
        yield lambda: log.append(("alpha-out", config))

    def db(ctx, config):
        ctx.provide("db", config)

    def beta(ctx, config):
        log.append(("beta", ctx.db))
        yield lambda: log.append(("beta-out",))

    beta.inject = ["db"]

    root = compose.Context()
    loader = Loader(root, {"alpha": alpha, "db": db, "beta": beta}.get)
    return root, loader


def test_tree_builds_and_loads_entries() -> None:
    log: list = []
    _root, loader = build(log)
    loader.create({"id": "a", "name": "alpha", "config": {"n": 1}})
    loader.create({"id": "d", "name": "db", "config": "v1"})
    loader.create({"id": "b", "name": "beta"})
    assert log == [("alpha", {"n": 1}), ("beta", "v1")]
    assert [(e.options["id"], e.fiber.state.name) for e in loader.entries()] == [
        ("a", "ACTIVE"),
        ("d", "ACTIVE"),
        ("b", "ACTIVE"),
    ]


def test_update_config_restarts_only_that_entry() -> None:
    log: list = []
    _root, loader = build(log)
    loader.create({"id": "a", "name": "alpha", "config": {"n": 1}})
    loader.create({"id": "d", "name": "db", "config": "v1"})
    loader.create({"id": "b", "name": "beta"})
    log.clear()
    loader.update("a", {"config": {"n": 2}})
    assert log == [("alpha-out", {"n": 1}), ("alpha", {"n": 2})]  # beta untouched: minimal mutation


def test_noop_update_applies_no_mutation() -> None:
    log: list = []
    _root, loader = build(log)
    loader.create({"id": "a", "name": "alpha", "config": {"n": 1}})
    log.clear()
    loader.resolve("a").update({"config": {"n": 1}})  # deep-equal options: no diff
    assert log == []


def test_disable_and_reenable_entry() -> None:
    log: list = []
    _root, loader = build(log)
    loader.create({"id": "d", "name": "db", "config": "v1"})
    loader.create({"id": "b", "name": "beta"})
    log.clear()
    loader.update("d", {"disabled": True})
    assert log == [("beta-out",)]  # the dependent unloads reactively
    assert loader.resolve("b").fiber.state is FiberState.PENDING
    log.clear()
    loader.update("d", {"disabled": None})  # None deletes the key: re-enable
    assert log == [("beta", "v1")]
    assert loader.resolve("b").fiber.state is FiberState.ACTIVE


def test_remove_entry_disposes_fiber() -> None:
    log: list = []
    _root, loader = build(log)
    loader.create({"id": "a", "name": "alpha", "config": {"n": 1}})
    log.clear()
    loader.remove("a")
    assert log == [("alpha-out", {"n": 1})]
    with pytest.raises(LookupError, match="cannot resolve entry a"):
        loader.resolve("a")


def test_dependent_entry_waits_for_provider_entry() -> None:
    log: list = []
    _root, loader = build(log)
    loader.create({"id": "b", "name": "beta"})
    assert loader.resolve("b").fiber.state is FiberState.PENDING
    loader.create({"id": "d", "name": "db", "config": "v1"})
    assert loader.resolve("b").fiber.state is FiberState.ACTIVE
    assert log == [("beta", "v1")]


def test_group_reconciles_children_without_restarting_itself() -> None:
    log: list = []
    _root, loader = build(log)
    loader.create({"id": "g", "group": True, "config": [{"id": "g1", "name": "alpha", "config": {"g": 1}}]})
    group_fiber = loader.resolve("g").fiber
    assert log == [("alpha", {"g": 1})]
    log.clear()
    loader.update("g", {"config": [{"id": "g2", "name": "alpha", "config": {"g": 2}}]})
    assert log == [("alpha-out", {"g": 1}), ("alpha", {"g": 2})]  # g1 removed, g2 created
    assert loader.resolve("g").fiber is group_fiber  # the group fiber itself never restarted
    with pytest.raises(LookupError):
        loader.resolve("g1")


def test_disabling_group_disables_children() -> None:
    log: list = []
    _root, loader = build(log)
    loader.create({"id": "g", "group": True, "config": [{"id": "g1", "name": "alpha", "config": {"g": 1}}]})
    log.clear()
    loader.update("g", {"disabled": True})
    assert log == [("alpha-out", {"g": 1})]  # children inherit the disable through the entry chain
    log.clear()
    loader.update("g", {"disabled": None})
    assert log == [("alpha", {"g": 1})]


def test_config_writeback_on_fiber_update() -> None:
    log: list = []
    _root, loader = build(log)
    loader.create({"id": "a", "name": "alpha", "config": {"n": 1}})
    entry = loader.resolve("a")
    entry.fiber.update({"n": 3})
    assert entry.options["config"] == {"n": 3}  # the live tree writes back to the desired tree


def test_self_dispose_marks_entry_disabled() -> None:
    log: list = []
    _root, loader = build(log)
    loader.create({"id": "a", "name": "alpha", "config": {"n": 1}})
    entry = loader.resolve("a")
    entry.fiber.dispose()
    assert entry.options.get("disabled") is True


def test_tree_teardown_does_not_mark_entries_disabled() -> None:
    log: list = []
    root, loader = build(log)
    loader.create({"id": "d", "name": "db", "config": "v1"})
    loader.create({"id": "b", "name": "beta"})
    log.clear()
    root.fiber.dispose()
    assert log == [("beta-out",)]
    assert all(e.options.get("disabled") is None for e in loader.entries())


def test_entry_isolate_realms() -> None:
    log: list = []
    root, loader = build(log)
    loader.create({"id": "d", "name": "db", "config": "root-v"})
    loader.create({"id": "iso-d", "name": "db", "config": "iso-v", "isolate": {"db": "realmA"}})
    loader.create({"id": "iso-b", "name": "beta", "isolate": {"db": "realmA"}})
    loader.create({"id": "b", "name": "beta"})
    assert ("beta", "iso-v") in log  # the labeled realm pairs provider and consumer
    assert ("beta", "root-v") in log
    assert root.get("db") == "root-v"


def test_entry_inject_merges_into_fiber() -> None:
    log: list = []
    _root, loader = build(log)
    loader.create({"id": "a", "name": "alpha", "config": {"n": 1}, "inject": ["db"]})
    entry = loader.resolve("a")
    assert entry.fiber.state is FiberState.PENDING  # the entry-level inject gates the load
    assert log == []
    loader.create({"id": "d", "name": "db", "config": "v1"})
    assert entry.fiber.state is FiberState.ACTIVE
    assert log == [("alpha", {"n": 1})]


def test_move_entry_between_groups() -> None:
    log: list = []
    _root, loader = build(log)
    loader.create({"id": "g", "group": True, "config": []})
    loader.create({"id": "a", "name": "alpha", "config": {"n": 1}})
    loader.update("a", {}, parent="g", move=True)
    entry = loader.resolve("a")
    assert entry.parent is loader.resolve("g").subgroup
    assert entry.fiber.state is FiberState.ACTIVE


def test_unresolvable_name_leaves_entry_without_fiber() -> None:
    log: list = []
    _root, loader = build(log)
    loader.create({"id": "x", "name": "no-such-plugin"})
    entry = loader.resolve("x")
    assert entry.fiber is None  # logged, not raised, as an import failure upstream
    assert log == []


def test_loader_await_intercept_waits_for_entry_work() -> None:
    async def main() -> None:
        log: list = []

        async def slow(ctx, config):
            await asyncio.sleep(0)
            yield lambda: None

        def watcher(ctx, config):
            log.append("watcher")

        watcher.inject = {"loader": {"await": True}}

        root = compose.Context()
        loader = Loader(root, {"slow": slow, "watcher": watcher}.get)
        loader.create({"id": "s", "name": "slow"})
        loader.create({"id": "w", "name": "watcher"})
        entry = loader.resolve("w")
        assert entry.fiber.state is FiberState.PENDING  # parked while entry work is in flight
        assert log == []
        await loader.wait()
        assert entry.fiber.state is FiberState.ACTIVE
        assert log == ["watcher"]

    asyncio.run(main())


def test_reconcile_after_self_reconfigure_is_a_noop() -> None:
    # Write-back unparses through Config.simplify or model_dump
    # (index.ts:76-77): storing the validated model itself would diff
    # against the raw dict on the next reconcile and spuriously reload.
    log: list = []
    root = compose.Context()

    class VCfg:
        def __init__(self, n: int = 0) -> None:
            self.n = n

        @classmethod
        def model_validate(cls, raw):
            return cls(int(raw.get("n", 0)))

        def model_dump(self):
            return {"n": self.n}

    def tracked(ctx, config):
        log.append(("load", config.n))
        yield lambda: log.append(("unload", config.n))

    tracked.Config = VCfg
    loader = Loader(root, {"tracked": tracked}.get)
    loader.create({"id": "t", "name": "tracked", "config": {"n": 1}})
    loader.resolve("t").fiber.update({"n": 5})
    assert loader.resolve("t").options["config"] == {"n": 5}
    log.clear()
    loader.resolve("t").update({"config": {"n": 5}})
    assert log == []


def test_a_failing_entry_lands_failed_beside_active_siblings_on_load_and_on_update(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An async plugin body that raises is reported by its fiber: the entry is FAILED, its siblings
    ACTIVE, the loader's join of the failed fiber leaves no unretrieved task exception, and a config
    update of the bad entry fails the same way (cordis 2ceea23)."""
    log: list = []
    unhandled: list[object] = []

    async def bad(ctx, config):
        await asyncio.sleep(0)
        raise RuntimeError("boom")

    async def alpha(ctx, config):
        await asyncio.sleep(0)
        log.append(("alpha", config))

    async def main() -> None:
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: unhandled.append(context))
        loader = Loader(compose.Context(), {"bad": bad, "alpha": alpha}.get)
        with caplog.at_level(logging.ERROR):
            loader.root.update([{"id": "1", "name": "bad"}, {"id": "2", "name": "alpha"}])
            await loader.wait()
        states = {e.options["id"]: e.fiber.state for e in loader.entries()}
        assert states == {"1": FiberState.FAILED, "2": FiberState.ACTIVE}
        with caplog.at_level(logging.ERROR):
            loader.root.update([{"id": "1", "name": "bad", "config": {"a": 1}}, {"id": "2", "name": "alpha"}])
            await loader.wait()
        states = {e.options["id"]: e.fiber.state for e in loader.entries()}
        assert states == {"1": FiberState.FAILED, "2": FiberState.ACTIVE}
        assert log == [("alpha", None)]

    asyncio.run(main())
    gc.collect()
    assert unhandled == []
