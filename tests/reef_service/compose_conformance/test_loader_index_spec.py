"""packages/loader/tests/index.spec.ts."""

from __future__ import annotations

import pytest

from reef.train.cordis_backend.compose import Context, FiberState

from .utils import GROUP, Harness, sleep, upstream

SPEC = "loader/tests/index.spec.ts"


class TestLoaderBasicSupport:
    """describe('Loader: basic support')."""

    @staticmethod
    @pytest.fixture(autouse=True, scope="class")
    def mocks(harness: Harness) -> dict:
        def listener(ctx: Context, config: object = None) -> None:
            ctx.on("internal/update", lambda *_: None)

        return {name: harness.loader.mock(name, listener) for name in ("foo", "bar", "qux")}

    @upstream(SPEC, "loader initiate")
    def test_loader_initiate(self, harness: Harness, mocks: dict) -> None:
        async def body() -> None:
            await harness.loader.read(
                [
                    {"id": "1", "name": "foo"},
                    {
                        "id": "2",
                        "name": GROUP,
                        "group": True,
                        "config": [
                            {"id": "3", "name": "bar", "config": {"a": 1}},
                            {"id": "4", "name": "qux", "disabled": True},
                        ],
                    },
                ]
            )

            harness.loader.expect_enable(mocks["foo"])
            harness.loader.expect_enable(mocks["bar"])
            harness.loader.expect_disable(mocks["qux"])
            assert len(mocks["foo"].calls) == 1
            assert len(mocks["bar"].calls) == 1
            assert len(mocks["qux"].calls) == 0

        harness.run(body)

    @upstream(SPEC, "loader update")
    def test_loader_update(self, harness: Harness, mocks: dict) -> None:
        async def body() -> None:
            mocks["foo"].reset_calls()
            mocks["bar"].reset_calls()
            await harness.loader.read(
                [
                    {"id": "1", "name": "foo"},
                    {"id": "4", "name": "qux"},
                ]
            )

            harness.loader.expect_enable(mocks["foo"])
            harness.loader.expect_disable(mocks["bar"])
            harness.loader.expect_enable(mocks["qux"])
            assert len(mocks["foo"].calls) == 0
            assert len(mocks["bar"].calls) == 0
            assert len(mocks["qux"].calls) == 1

        harness.run(body)

    @upstream(SPEC, "plugin self-update")
    def test_plugin_self_update(self, harness: Harness, mocks: dict) -> None:
        async def body() -> None:
            harness.loader.expect_fiber("1").update({"a": 3})
            await sleep()
            assert harness.loader.data == [
                {"id": "1", "name": "foo", "config": {"a": 3}},
                {"id": "4", "name": "qux"},
            ]

        harness.run(body)

    @upstream(SPEC, "plugin self-dispose")
    def test_plugin_self_dispose(self, harness: Harness, mocks: dict) -> None:
        async def body() -> None:
            harness.loader.expect_fiber("1").dispose()
            await sleep()
            assert harness.loader.data == [
                {"id": "1", "name": "foo", "disabled": True, "config": {"a": 3}},
                {"id": "4", "name": "qux"},
            ]

        harness.run(body)


class TestLoaderInterceptConfig:
    """describe('Loader: intercept config')."""

    @staticmethod
    @pytest.fixture(autouse=True, scope="class")
    def mocks(harness: Harness) -> dict:
        state: dict = {}
        state["waiter"] = harness.loop.create_future()

        def foo(ctx: Context, config: object = None):
            return state["waiter"]

        def bar(ctx: Context, config: object = None) -> None:
            ctx.on("internal/update", lambda *_: True)

        harness.loader.mock("foo", foo)
        harness.loader.mock("bar", bar).inject = ["never"]
        harness.loader.mock("qux", lambda ctx, config=None: None)
        return state

    @upstream(SPEC, "pending")
    def test_pending(self, harness: Harness, mocks: dict) -> None:
        async def body() -> None:
            mocks["foo_id"] = harness.loader.create({"name": "foo"})
            mocks["bar_id"] = harness.loader.create({"name": "bar"})
            mocks["qux_id"] = harness.loader.create(
                {
                    "name": "qux",
                    "inject": {"loader": True},
                    "intercept": {"loader": {"await": True}},
                }
            )

            await sleep()
            assert harness.loader.expect_fiber(mocks["foo_id"]).state is FiberState.LOADING
            assert harness.loader.expect_fiber(mocks["bar_id"]).state is FiberState.PENDING
            assert harness.loader.expect_fiber(mocks["qux_id"]).state is FiberState.PENDING

        harness.run(body)

    @upstream(SPEC, "resolved")
    def test_resolved(self, harness: Harness, mocks: dict) -> None:
        async def body() -> None:
            mocks["waiter"].set_result(None)
            await sleep()
            assert harness.loader.expect_fiber(mocks["foo_id"]).state is FiberState.ACTIVE
            assert harness.loader.expect_fiber(mocks["bar_id"]).state is FiberState.PENDING
            assert harness.loader.expect_fiber(mocks["qux_id"]).state is FiberState.ACTIVE

        harness.run(body)


# a failing entry is reported by the fiber itself; it must not escape as an
# unhandled rejection, which would take the whole process down with it
class TestLoaderEntryFailure:
    """describe('Loader: entry failure')."""

    @staticmethod
    @pytest.fixture(autouse=True, scope="class")
    def mocks(harness: Harness) -> dict:
        def bad(ctx: Context, config: object = None) -> None:
            raise RuntimeError("boom")

        harness.loader.mock("bad", bad)
        harness.loader.mock("good", lambda ctx, config=None: None)
        return {}

    @upstream(SPEC, "on load")
    def test_on_load(self, harness: Harness, mocks: dict) -> None:
        async def body() -> None:
            await harness.loader.read(
                [
                    {"id": "1", "name": "bad"},
                    {"id": "2", "name": "good"},
                ]
            )

            assert harness.loader.expect_fiber("1").state is FiberState.FAILED
            assert harness.loader.expect_fiber("2").state is FiberState.ACTIVE

        harness.run(body)

    @upstream(SPEC, "on config update")
    def test_on_config_update(self, harness: Harness, mocks: dict) -> None:
        async def body() -> None:
            await harness.loader.read(
                [
                    {"id": "1", "name": "bad", "config": {"a": 1}},
                    {"id": "2", "name": "good"},
                ]
            )

            assert harness.loader.expect_fiber("1").state is FiberState.FAILED
            assert harness.loader.expect_fiber("2").state is FiberState.ACTIVE

        harness.run(body)
