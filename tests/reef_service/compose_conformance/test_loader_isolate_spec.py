"""packages/loader/tests/isolate.spec.ts."""

from __future__ import annotations

import pytest

from reef.train.cordis_backend.compose import Context, FiberState, Service

from .utils import GROUP, Harness, Mock, fresh_harness, sleep, upstream

SPEC = "loader/tests/isolate.spec.ts"


def _group(**options: object) -> dict:
    return {"name": GROUP, "group": True, **options}


class _Bar(Service):
    def __init__(self, ctx: Context, config: object = None) -> None:
        super().__init__(ctx, "bar")


def _bar_provider(ctx: Context, config: object = None) -> None:
    ctx.provide("bar", config if config is not None else {})


def _register(loader, bar_is_service: bool) -> dict:
    """beforeAll: a ``foo`` injecting ``bar``, and a ``bar`` providing it."""
    disposer = Mock()
    foo = loader.mock("foo", lambda ctx, config=None: disposer)
    foo.inject = ["bar"]
    bar = loader.mock("bar", _Bar if bar_is_service else _bar_provider)
    return {"foo": foo, "bar": bar, "dispose": disposer}


def _reset(state: dict) -> None:
    state["foo"].reset_calls()
    state["dispose"].reset_calls()
    reset = getattr(state["bar"], "reset_calls", None)
    if reset is not None:
        reset()


class TestServiceIsolationBasic:
    """describe('Service Isolation: basic')."""

    @staticmethod
    @pytest.fixture(autouse=True, scope="class")
    def state(harness: Harness) -> dict:
        return _register(harness.loader, bar_is_service=True)

    @upstream(SPEC, "initiate")
    def test_initiate(self, harness: Harness, state: dict) -> None:
        _reset(state)

        async def body() -> None:
            state["provider"] = harness.loader.create({"name": "bar"})
            state["injector"] = harness.loader.create({"name": "foo"})

            await sleep()
            assert len(state["foo"].calls) == 1
            assert len(state["dispose"].calls) == 0

        harness.run(body)

    @upstream(SPEC, "add isolate on injector (relevant)")
    def test_add_isolate_on_injector_relevant(self, harness: Harness, state: dict) -> None:
        _reset(state)

        async def body() -> None:
            harness.loader.update(state["injector"], {"isolate": {"bar": True}})
            await sleep()
            assert len(state["foo"].calls) == 0
            assert len(state["dispose"].calls) == 1

        harness.run(body)

    @upstream(SPEC, "add isolate on injector (irrelevant)")
    def test_add_isolate_on_injector_irrelevant(self, harness: Harness, state: dict) -> None:
        _reset(state)

        async def body() -> None:
            harness.loader.update(state["injector"], {"isolate": {"bar": True, "qux": True}})
            await sleep()
            assert len(state["foo"].calls) == 0
            assert len(state["dispose"].calls) == 0

        harness.run(body)

    @upstream(SPEC, "remove isolate on injector (relevant)")
    def test_remove_isolate_on_injector_relevant(self, harness: Harness, state: dict) -> None:
        _reset(state)

        async def body() -> None:
            harness.loader.update(state["injector"], {"isolate": {"qux": True}})
            await sleep()
            assert len(state["foo"].calls) == 1
            assert len(state["dispose"].calls) == 0

        harness.run(body)

    @upstream(SPEC, "remove isolate on injector (irrelevant)")
    def test_remove_isolate_on_injector_irrelevant(self, harness: Harness, state: dict) -> None:
        _reset(state)

        async def body() -> None:
            harness.loader.update(state["injector"], {"isolate": None})
            await sleep()
            assert len(state["foo"].calls) == 0
            assert len(state["dispose"].calls) == 0

        harness.run(body)

    @upstream(SPEC, "add isolate on provider (relevant)")
    def test_add_isolate_on_provider_relevant(self, harness: Harness, state: dict) -> None:
        _reset(state)

        async def body() -> None:
            harness.loader.update(state["provider"], {"isolate": {"bar": True}})
            await sleep()
            assert len(state["foo"].calls) == 0
            assert len(state["dispose"].calls) == 1

        harness.run(body)

    @upstream(SPEC, "add isolate on provider (irrelevant)")
    def test_add_isolate_on_provider_irrelevant(self, harness: Harness, state: dict) -> None:
        _reset(state)

        async def body() -> None:
            harness.loader.update(state["provider"], {"isolate": {"bar": True, "qux": True}})
            await sleep()
            assert len(state["foo"].calls) == 0
            assert len(state["dispose"].calls) == 0

        harness.run(body)

    @upstream(SPEC, "remove isolate on provider (relevant)")
    def test_remove_isolate_on_provider_relevant(self, harness: Harness, state: dict) -> None:
        _reset(state)

        async def body() -> None:
            harness.loader.update(state["provider"], {"isolate": {"qux": True}})
            await sleep()
            assert len(state["foo"].calls) == 1
            assert len(state["dispose"].calls) == 0

        harness.run(body)

    @upstream(SPEC, "remove isolate on provider (irrelevant)")
    def test_remove_isolate_on_provider_irrelevant(self, harness: Harness, state: dict) -> None:
        _reset(state)

        async def body() -> None:
            harness.loader.update(state["provider"], {"isolate": None})
            await sleep()
            assert len(state["foo"].calls) == 0
            assert len(state["dispose"].calls) == 0

        harness.run(body)


class TestServiceIsolationRealm:
    """describe('Service Isolation: realm')."""

    @staticmethod
    @pytest.fixture(autouse=True, scope="class")
    def state(harness: Harness) -> dict:
        return _register(harness.loader, bar_is_service=False)

    @upstream(SPEC, "add isolate group")
    def test_add_isolate_group(self, harness: Harness, state: dict) -> None:
        _reset(state)

        async def body() -> None:
            state["alpha"] = harness.loader.create(
                _group(
                    isolate={"bar": True},
                    config=[{"name": "bar", "config": {"value": "alpha"}}],
                )
            )
            state["beta"] = harness.loader.create(
                _group(
                    isolate={"bar": "beta"},
                    config=[{"name": "bar", "config": {"value": "beta"}}],
                )
            )

            await sleep()
            assert len(harness.root.registry.get(state["bar"]).fibers) == 2
            assert len(state["foo"].calls) == 0
            assert len(state["dispose"].calls) == 0

        harness.run(body)

    @upstream(SPEC, "update isolate group (no change)")
    def test_update_isolate_group_no_change(self, harness: Harness, state: dict) -> None:
        _reset(state)

        async def body() -> None:
            harness.loader.update(state["alpha"], {"isolate": {"bar": True}})
            await sleep()
            assert len(harness.root.registry.get(state["bar"]).fibers) == 2
            assert len(state["foo"].calls) == 0
            assert len(state["dispose"].calls) == 0

        harness.run(body)

    @upstream(SPEC, "realm reference")
    def test_realm_reference(self, harness: Harness, state: dict) -> None:
        _reset(state)

        async def body() -> None:
            nested1 = harness.loader.create({"name": "foo"}, state["alpha"])
            nested2 = harness.loader.create({"name": "foo", "isolate": {"bar": "beta"}}, state["alpha"])
            nested3 = harness.loader.create({"name": "foo", "isolate": {"bar": True}}, state["alpha"])

            await sleep()
            assert len(state["foo"].calls) == 2
            assert len(state["dispose"].calls) == 0
            fiber1 = harness.loader.expect_fiber(nested1)
            assert fiber1.ctx.get("bar")["value"] == "alpha"
            assert fiber1.state is FiberState.ACTIVE
            fiber2 = harness.loader.expect_fiber(nested2)
            assert fiber2.ctx.get("bar")["value"] == "beta"
            assert fiber2.state is FiberState.ACTIVE
            fiber3 = harness.loader.expect_fiber(nested3)
            assert fiber3.ctx.get("bar") is None
            assert fiber3.state is FiberState.PENDING

        harness.run(body)

    @upstream(SPEC, "special case: nested realms")
    def test_special_case_nested_realms(self) -> None:
        with fresh_harness() as own:
            state = _register(own.loader, bar_is_service=False)

            async def body() -> None:
                outer = own.loader.create(_group(config=[]))
                inner = own.loader.create(_group(isolate={"bar": "custom"}, config=[]), outer)
                own.loader.create({"name": "bar", "config": {"value": "custom"}}, inner)

                alpha = own.loader.create({"name": "foo", "isolate": {"bar": "custom"}})
                beta = own.loader.create({"name": "foo"}, inner)

                await sleep()
                assert own.loader.expect_fiber(alpha).ctx.get("bar")["value"] == "custom"
                assert own.loader.expect_fiber(beta).ctx.get("bar")["value"] == "custom"

                _reset(state)
                own.loader.update(outer, {"isolate": {"bar": "custom"}})
                await sleep()
                assert len(state["foo"].calls) == 0
                assert len(state["dispose"].calls) == 0

                _reset(state)
                own.loader.update(outer, {"isolate": {}})
                await sleep()
                assert len(state["foo"].calls) == 0
                assert len(state["dispose"].calls) == 0

            own.run(body)

    @upstream(SPEC, "special case: change provider")
    def test_special_case_change_provider(self) -> None:
        with fresh_harness() as own:
            state = _register(own.loader, bar_is_service=False)

            async def body() -> None:
                own.loader.create({"name": "bar", "isolate": {"bar": "alpha"}, "config": {"value": "alpha"}})
                own.loader.create({"name": "bar", "isolate": {"bar": "beta"}, "config": {"value": "beta"}})
                group = own.loader.create(_group(isolate={"bar": "alpha"}, config=[]))
                id_ = own.loader.create({"name": "foo"}, group)

                await sleep()
                assert len(state["foo"].calls) == 1
                assert len(state["dispose"].calls) == 0
                fiber = own.loader.expect_fiber(id_)
                assert fiber.ctx.get("bar")["value"] == "alpha"

                _reset(state)
                own.loader.update(group, {"isolate": {"bar": "beta"}})
                await sleep()
                assert len(state["foo"].calls) == 1
                assert len(state["dispose"].calls) == 1
                assert fiber.ctx.get("bar")["value"] == "beta"

            own.run(body)

    @upstream(SPEC, "special case: change injector")
    def test_special_case_change_injector(self) -> None:
        with fresh_harness() as own:
            state = _register(own.loader, bar_is_service=False)

            async def body() -> None:
                alpha = own.loader.create({"name": "foo", "isolate": {"bar": "alpha"}})
                beta = own.loader.create({"name": "foo", "isolate": {"bar": "beta"}})
                group = own.loader.create(_group(isolate={"bar": "alpha"}, config=[]))
                own.loader.create({"name": "bar"}, group)

                await sleep()
                assert len(state["foo"].calls) == 1
                assert len(state["dispose"].calls) == 0
                fiber1 = own.loader.expect_fiber(alpha)
                assert fiber1.ctx.get("bar") is not None
                fiber2 = own.loader.expect_fiber(beta)
                assert fiber2.ctx.get("bar") is None

                _reset(state)
                own.loader.update(group, {"isolate": {"bar": "beta"}})
                await sleep()
                assert len(state["foo"].calls) == 1
                assert len(state["dispose"].calls) == 1
                assert fiber1.ctx.get("bar") is None
                assert fiber2.ctx.get("bar") is not None

            own.run(body)


class TestServiceIsolationTransfer:
    """describe('Service Isolation: transfer')."""

    @staticmethod
    @pytest.fixture(autouse=True, scope="class")
    def state(harness: Harness) -> dict:
        return _register(harness.loader, bar_is_service=True)

    @upstream(SPEC, "initiate")
    def test_initiate(self, harness: Harness, state: dict) -> None:
        _reset(state)

        async def body() -> None:
            state["group"] = harness.loader.create(_group(isolate={"bar": True}, config=[]))
            state["provider"] = harness.loader.create({"name": "bar"})
            state["injector"] = harness.loader.create({"name": "foo"})

            await sleep()
            assert len(state["foo"].calls) == 1
            assert len(state["dispose"].calls) == 0

        harness.run(body)

    @upstream(SPEC, "transfer injector into group")
    def test_transfer_injector_into_group(self, harness: Harness, state: dict) -> None:
        _reset(state)

        async def body() -> None:
            harness.loader.update(state["injector"], {}, state["group"], move=True)
            await sleep()
            assert len(state["foo"].calls) == 0
            assert len(state["dispose"].calls) == 1

        harness.run(body)

    @upstream(SPEC, "transfer provider into group")
    def test_transfer_provider_into_group(self, harness: Harness, state: dict) -> None:
        _reset(state)

        async def body() -> None:
            harness.loader.update(state["provider"], {}, state["group"], move=True)
            await sleep()
            assert len(state["foo"].calls) == 1
            assert len(state["dispose"].calls) == 0

        harness.run(body)

    @upstream(SPEC, "transfer injector out of group")
    def test_transfer_injector_out_of_group(self, harness: Harness, state: dict) -> None:
        _reset(state)

        async def body() -> None:
            harness.loader.update(state["injector"], {}, None, move=True)
            await sleep()
            assert len(state["foo"].calls) == 0
            assert len(state["dispose"].calls) == 1

        harness.run(body)

    @upstream(SPEC, "transfer provider out of group")
    def test_transfer_provider_out_of_group(self, harness: Harness, state: dict) -> None:
        _reset(state)

        async def body() -> None:
            harness.loader.update(state["provider"], {}, None, move=True)
            await sleep()
            assert len(state["foo"].calls) == 1
            assert len(state["dispose"].calls) == 0

        harness.run(body)
