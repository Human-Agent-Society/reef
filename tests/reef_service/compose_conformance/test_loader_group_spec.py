"""packages/loader/tests/group.spec.ts."""

from __future__ import annotations

import pytest

from reef.train.cordis_backend.compose import Context

from .utils import GROUP, Harness, Mock, intercept_layers, sleep, upstream

SPEC = "loader/tests/group.spec.ts"


def _group(**options: object) -> dict:
    """A group entry. The reference names the group module; this port routes
    group entries by their ``group`` flag (UPSTREAM.md, permanent omissions)."""
    return {"name": GROUP, "group": True, **options}


class TestGroupBasicSupport:
    """describe('Group: basic support')."""

    @staticmethod
    @pytest.fixture(autouse=True, scope="class")
    def state(harness: Harness) -> dict:
        disposer = Mock()
        foo = harness.loader.mock("foo", lambda ctx, config=None: disposer)
        return {"foo": foo, "dispose": disposer}

    @staticmethod
    def _each(state: dict) -> None:
        state["foo"].reset_calls()
        state["dispose"].reset_calls()

    @upstream(SPEC, "initialize")
    def test_initialize(self, harness: Harness, state: dict) -> None:
        self._each(state)

        async def body() -> None:
            state["outer"] = harness.loader.create(_group(config=[{"name": "foo"}]))
            state["inner"] = harness.loader.create(_group(config=[{"name": "foo"}]), state["outer"])

            await sleep()
            harness.loader.expect_fiber(state["outer"])
            harness.loader.expect_fiber(state["inner"])
            assert len(state["foo"].calls) == 2
            assert len(state["dispose"].calls) == 0
            assert len(list(harness.loader.entries())) == 4

        harness.run(body)

    @upstream(SPEC, "disable inner")
    def test_disable_inner(self, harness: Harness, state: dict) -> None:
        self._each(state)

        async def body() -> None:
            harness.loader.update(state["inner"], {"disabled": True})

            assert len(state["foo"].calls) == 0
            assert len(state["dispose"].calls) == 1
            assert len(list(harness.loader.entries())) == 4

        harness.run(body)

    @upstream(SPEC, "disable outer")
    def test_disable_outer(self, harness: Harness, state: dict) -> None:
        self._each(state)

        async def body() -> None:
            harness.loader.update(state["outer"], {"disabled": True})

            assert len(state["foo"].calls) == 0
            assert len(state["dispose"].calls) == 1
            assert len(list(harness.loader.entries())) == 4

        harness.run(body)

    @upstream(SPEC, "enable inner")
    def test_enable_inner(self, harness: Harness, state: dict) -> None:
        self._each(state)

        async def body() -> None:
            harness.loader.update(state["inner"], {"disabled": None})

            assert len(state["foo"].calls) == 0  # outer is still disabled
            assert len(state["dispose"].calls) == 0
            assert len(list(harness.loader.entries())) == 4

        harness.run(body)

    @upstream(SPEC, "enable outer")
    def test_enable_outer(self, harness: Harness, state: dict) -> None:
        self._each(state)

        async def body() -> None:
            harness.loader.update(state["outer"], {"disabled": None})

            await sleep()
            assert len(state["foo"].calls) == 2
            assert len(state["dispose"].calls) == 0
            assert len(list(harness.loader.entries())) == 4

        harness.run(body)


class TestGroupTransfer:
    """describe('Group: transfer')."""

    @staticmethod
    @pytest.fixture(autouse=True, scope="class")
    def state(harness: Harness) -> dict:
        disposer = Mock()
        foo = harness.loader.mock("foo", lambda ctx, config=None: disposer)
        return {"foo": foo, "dispose": disposer}

    @staticmethod
    def _each(state: dict) -> None:
        state["foo"].reset_calls()
        state["dispose"].reset_calls()

    @upstream(SPEC, "initialize")
    def test_initialize(self, harness: Harness, state: dict) -> None:
        self._each(state)

        async def body() -> None:
            state["id"] = harness.loader.create({"name": "foo"})
            state["alpha"] = harness.loader.create(_group(config=[]))
            state["beta"] = harness.loader.create(_group(disabled=True, config=[]), state["alpha"])
            state["gamma"] = harness.loader.create(_group(config=[]), state["beta"])

            assert len(state["foo"].calls) == 1
            assert len(state["dispose"].calls) == 0
            assert len(list(harness.loader.entries())) == 4

        harness.run(body)

    @upstream(SPEC, "enabled -> enabled")
    def test_enabled_to_enabled(self, harness: Harness, state: dict) -> None:
        self._each(state)

        async def body() -> None:
            harness.loader.update(state["id"], {}, state["alpha"], move=True)

            assert len(state["foo"].calls) == 0
            assert len(state["dispose"].calls) == 0
            assert len(list(harness.loader.entries())) == 4

        harness.run(body)

    @upstream(SPEC, "enabled -> disabled")
    def test_enabled_to_disabled(self, harness: Harness, state: dict) -> None:
        self._each(state)

        async def body() -> None:
            harness.loader.update(state["id"], {}, state["beta"], move=True)

            assert len(state["foo"].calls) == 0
            assert len(state["dispose"].calls) == 1
            assert len(list(harness.loader.entries())) == 4

        harness.run(body)

    @upstream(SPEC, "disabled -> disabled")
    def test_disabled_to_disabled(self, harness: Harness, state: dict) -> None:
        self._each(state)

        async def body() -> None:
            harness.loader.update(state["id"], {}, state["gamma"], move=True)

            assert len(state["foo"].calls) == 0  # outer is still disabled
            assert len(state["dispose"].calls) == 0
            assert len(list(harness.loader.entries())) == 4

        harness.run(body)

    @upstream(SPEC, "disabled -> enabled")
    def test_disabled_to_enabled(self, harness: Harness, state: dict) -> None:
        self._each(state)

        async def body() -> None:
            harness.loader.update(state["id"], {}, None, move=True)

            assert len(state["foo"].calls) == 1
            assert len(state["dispose"].calls) == 0
            assert len(list(harness.loader.entries())) == 4

        harness.run(body)


class TestGroupIntercept:
    """describe('Group: intercept')."""

    @staticmethod
    @pytest.fixture(autouse=True, scope="class")
    def state(harness: Harness) -> dict:
        seen: list[list[dict]] = []

        def foo(ctx: Context, config: object = None) -> None:
            # The reference reads ctx[Context.intercept] and walks its
            # prototype chain for `foo`; intercept_layers is that walk.
            seen.append(intercept_layers(ctx, "foo"))

        harness.loader.mock("foo", foo)
        return {"seen": seen}

    @upstream(SPEC, "initialize")
    def test_initialize(self, harness: Harness, state: dict) -> None:
        state["seen"].clear()

        async def body() -> None:
            outer = harness.loader.create(_group(intercept={"foo": {"a": 1}}, config=[]))
            inner = harness.loader.create(_group(intercept={"foo": {"b": 2}}, config=[]), outer)
            harness.loader.create({"name": "foo", "intercept": {"foo": {"c": 3}}}, inner)

            assert len(state["seen"]) == 1
            assert state["seen"][0] == [{"c": 3}, {"b": 2}, {"a": 1}]

        harness.run(body)
