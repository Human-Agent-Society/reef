"""A scenario's traffic split, the weights above 0 of its ``served`` mapping: which served artifact answers a request."""

from __future__ import annotations

import asyncio
from collections import Counter

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.reef_service.test_served_artifacts import ArtifactEcho, two_artifacts
from tests.support.serving import wait_for_serving

from reef.dispatcher import Dispatcher
from reef.inference.interfaces import FixedEngineClients
from reef.service.app import create_app
from reef.service.inference_service import BUCKET_COUNT, TrafficWeights, build_inference_app, conversation_bucket
from reef.storage.sqlite import SQLiteScenarioStorage

HEADERS = {"x-reef-scenario": "math"}


def chat(user: str = "hi", *later: str) -> dict[str, object]:
    turns = [{"role": "user", "content": user}]
    for text in later:
        turns += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": text}]
    return {"model": "active", "messages": turns}


def serving_both(value: Dispatcher, active: str, beside: str) -> None:
    """Serve ``beside`` with weight 0 beside the ``active`` artifact, and wait until its route is published."""
    value.set_served("math", {active: 100, beside: 0})
    wait_for_serving(value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        {"a": 90},
        {"a": 60, "b": 60},
        {"a": 101, "b": -1},
        {"a": 50.0, "b": 50},
        {"a": True, "b": 99},
        {"": 100},
        [1],
        "a",
    ],
)
def test_invalid_weights_are_refused(value: object) -> None:
    with pytest.raises(ValueError, match=r"traffic_weights|traffic weight"):
        TrafficWeights.from_json(value)


@pytest.mark.unit
def test_a_changed_ratio_moves_only_the_buckets_that_go_to_another_artifact() -> None:
    before = TrafficWeights({"a": 90, "b": 10})
    after = TrafficWeights({"a": 80, "b": 20})
    moved = [bucket for bucket in range(BUCKET_COUNT) if before.artifact_for(bucket) != after.artifact_for(bucket)]
    assert len(moved) == 10
    assert all(before.artifact_for(bucket) == "a" and after.artifact_for(bucket) == "b" for bucket in moved)
    assert TrafficWeights({"a": 0, "b": 100}).artifact_for(0) == "b"


@pytest.mark.unit
def test_every_turn_of_a_conversation_gets_one_bucket_whatever_the_protocol() -> None:
    first = conversation_bucket("math", chat("hi"))
    assert first == 95  # A fixed vector: a change moves the conversations of every deployment.
    assert conversation_bucket("math", chat("hi", "more", "and more")) == first
    # A client moves its cache marks between turns.
    marked = {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}
    assert conversation_bucket("math", {"messages": [marked]}) == first
    assert conversation_bucket("math", {"input": "hi"}) == first
    assert conversation_bucket("math", {"input": [{"role": "user", "content": "hi"}]}) == first

    system = conversation_bucket("math", {"system": "be brief", "messages": [{"role": "user", "content": "hi"}]})
    assert system != first
    assert system == conversation_bucket("math", {"instructions": "be brief", "input": "hi"})
    assert conversation_bucket("other", chat("hi")) != first
    assert conversation_bucket("math", {"messages": []}) is None
    assert conversation_bucket("math", {"model": "active"}) is None


@pytest.mark.unit
def test_a_session_key_keeps_a_conversation_on_one_bucket() -> None:
    keyed = conversation_bucket("math", chat("first"), "session-1")
    assert conversation_bucket("math", chat("rewritten history"), "session-1") == keyed
    assert conversation_bucket("math", {"input": "next", "previous_response_id": "r-1"}, "session-1") == keyed
    assert conversation_bucket("other", chat("first"), "session-1") != keyed
    # Without a key, a later turn holds only its new input, so Reef cannot identify the conversation.
    assert conversation_bucket("math", {"input": "next", "previous_response_id": "r-1"}) is None
    assert conversation_bucket("math", chat("first"), "") == conversation_bucket("math", chat("first"))


@pytest.mark.unit
def test_conversations_split_near_the_weights() -> None:
    weights = TrafficWeights({"a": 90, "b": 10})
    buckets = [conversation_bucket("math", chat(f"user {n}")) for n in range(5000)]
    counts = Counter(weights.artifact_for(bucket) for bucket in buckets if bucket is not None)
    assert sum(counts.values()) == 5000
    assert 0.08 < counts["b"] / 5000 < 0.12


@pytest.mark.unit
def test_weights_split_requests_and_each_record_names_the_artifact_that_answered(tmp_path) -> None:
    async def run() -> None:
        value, _, _, newest, older = two_artifacts(tmp_path)
        handler = ArtifactEcho()
        client = TestClient(TestServer(create_app(value, inference_handler=handler)))
        await client.start_server()
        try:
            # The larger weight keeps the newest artifact active; the mapping starts the older one.
            weights = {newest["artifact_id"]: 60, older["artifact_id"]: 40}
            update = await client.post("/reef/scenarios/math/update", json={"served": weights})
            assert update.status == 200
            assert (await update.json())["served"] == weights
            settings = await (await client.get("/reef/scenarios/math/settings")).json()
            assert settings["served"] == {"weights": weights}
            wait_for_serving(value)

            answers = []
            for n in range(60):
                answer = await client.post("/v1/chat/completions", json=chat(f"user {n}"), headers=HEADERS)
                assert answer.status == 200
                answers.append(answer.headers["x-reef-agent-record-id"])
            assert set(handler.artifact_ids) == set(weights)

            # Each record names the artifact that answered.
            scenario = value.get_or_create_scenario("math")
            stored = [scenario.records.read("math", record_id) for record_id in answers]
            assert [item.artifact_id for item in stored] == handler.artifact_ids
            assert stored[0].payload["metadata"]["traffic"]["weights"] == weights

            # The later turns of a conversation stay on one artifact.
            handler.artifact_ids.clear()
            for later in (chat("user 0"), chat("user 0", "more"), chat("user 0", "more", "again")):
                await client.post("/v1/chat/completions", json=later, headers=HEADERS)
            assert len(set(handler.artifact_ids)) == 1

            # The session key decides the artifact, whatever the messages say.
            handler.artifact_ids.clear()
            keyed = {**HEADERS, "x-reef-session-key": "session-1"}
            for text in ("one", "two", "three", "four"):
                await client.post("/v1/chat/completions", json=chat(text), headers=keyed)
            assert len(set(handler.artifact_ids)) == 1

            # A request with previous_response_id and no key goes to the active artifact.
            handler.artifact_ids.clear()
            for n in range(20):
                later = {"model": "active", "input": f"turn {n}", "previous_response_id": "r-1"}
                await client.post("/v1/responses", json=later, headers=HEADERS)
            assert set(handler.artifact_ids) == {newest["artifact_id"]}

            # A request that names an artifact, and a rollout call, do not use the weights.
            handler.artifact_ids.clear()
            pinned = {"model": older["artifact_id"], "messages": chat()["messages"]}
            await client.post("/v1/chat/completions", json=pinned, headers=HEADERS)
            for n in range(20):
                await client.post(
                    "/v1/chat/completions", json=chat(f"rollout {n}"), headers={**HEADERS, "x-reef-rollout": "1"}
                )
            assert handler.artifact_ids[0] == older["artifact_id"]
            assert set(handler.artifact_ids[1:]) == {newest["artifact_id"]}

            # One weight above 0 ends the split: the active artifact answers everything.
            handler.artifact_ids.clear()
            whole = {newest["artifact_id"]: 100, older["artifact_id"]: 0}
            cleared = await client.post("/reef/scenarios/math/update", json={"served": whole})
            assert cleared.status == 200
            assert value.coordination.served_models.read_traffic_weights("math") == {}
            for n in range(20):
                await client.post("/v1/chat/completions", json=chat(f"user {n}"), headers=HEADERS)
            assert set(handler.artifact_ids) == {newest["artifact_id"]}
        finally:
            await client.close()
            value.close()

    asyncio.run(run())


@pytest.mark.unit
def test_a_separate_inference_service_follows_a_ratio_change_without_a_restart(tmp_path) -> None:
    async def run() -> None:
        value, _, _, newest, older = two_artifacts(tmp_path)
        handler = ArtifactEcho()
        storage = SQLiteScenarioStorage(tmp_path / "agent-record")
        app = build_inference_app(value.coordination.served_models, storage, FixedEngineClients(handler), tokens=None)
        client = TestClient(TestServer(app))
        await client.start_server()

        async def answered_by() -> list[str]:
            handler.artifact_ids.clear()
            for n in range(40):
                answer = await client.post("/v1/chat/completions", json=chat(f"user {n}"), headers=HEADERS)
                assert answer.status == 200
            return list(handler.artifact_ids)

        def expected(weights: dict[str, int]) -> list[str]:
            split = TrafficWeights(weights)
            return [split.artifact_for(conversation_bucket("math", chat(f"user {n}"))) for n in range(40)]

        try:
            mostly_newest = {newest["artifact_id"]: 90, older["artifact_id"]: 10}
            mostly_older = {newest["artifact_id"]: 10, older["artifact_id"]: 90}
            assert expected(mostly_newest) != expected(mostly_older)
            for weights in (mostly_newest, mostly_older):
                value.set_served("math", weights)
                wait_for_serving(value)
                assert await answered_by() == expected(weights)
        finally:
            await client.close()
            storage.close()
            value.close()

    asyncio.run(run())


@pytest.mark.unit
def test_an_artifact_gets_its_weight_before_its_route_and_the_active_artifact_answers_until_then(tmp_path) -> None:
    async def run() -> None:
        value, _, _, newest, older = two_artifacts(tmp_path)
        wait_for_serving(value)
        handler = ArtifactEcho()
        client = TestClient(TestServer(create_app(value, inference_handler=handler)))
        await client.start_server()
        table = value.coordination.served_models

        async def answered_by() -> set[str]:
            handler.artifact_ids.clear()
            for n in range(20):
                answer = await client.post("/v1/chat/completions", json=chat(f"user {n}"), headers=HEADERS)
                assert answer.status == 200
            return set(handler.artifact_ids)

        try:
            # The state while an artifact starts: the split names it, and it has no route. No request fails.
            waiting = {newest["artifact_id"]: 0, older["artifact_id"]: 100}
            value.publisher.write_traffic_weights("math", waiting)
            assert table.read_traffic_weights("math") == waiting
            assert older["artifact_id"] not in table.read_scenario("math")
            assert await answered_by() == {newest["artifact_id"]}

            # Weights that do not sum to 100 are refused, and the split stays.
            refused = await client.post("/reef/scenarios/math/update", json={"served": {newest["artifact_id"]: 50}})
            assert refused.status == 400
            assert table.read_traffic_weights("math") == waiting
            assert not value.get_or_create_scenario("math").served_beside()

            # A mapping gives an artifact without a route its weight, and starts the artifact.
            weights = {newest["artifact_id"]: 50, older["artifact_id"]: 50}
            accepted = await client.post("/reef/scenarios/math/update", json={"served": weights})
            assert accepted.status == 200, await accepted.text()
            assert table.read_traffic_weights("math") == weights
            assert value.scenario_settings("math")["served"] == {"weights": weights}
            assert set(value.get_or_create_scenario("math").served_beside()) == set(weights) - {min(weights)}
            wait_for_serving(value)
            assert await answered_by() == set(weights)
        finally:
            await client.close()
            value.close()

    asyncio.run(run())


@pytest.mark.unit
def test_an_artifact_that_stops_serving_sends_its_requests_to_the_active_artifact(tmp_path) -> None:
    async def run() -> None:
        value, _, _, newest, older = two_artifacts(tmp_path)
        weights = {newest["artifact_id"]: 90, older["artifact_id"]: 10}
        value.set_served("math", weights)
        # An activation moves the active artifact alone: the newest artifact stops serving, and the split stays.
        value.activate("math", older["artifact_id"])
        wait_for_serving(value)
        table = value.coordination.served_models
        assert table.read_traffic_weights("math") == weights
        assert newest["artifact_id"] not in table.read_scenario("math")
        handler = ArtifactEcho()
        client = TestClient(TestServer(create_app(value, inference_handler=handler)))
        await client.start_server()
        try:
            for n in range(20):
                answer = await client.post("/v1/chat/completions", json=chat(f"user {n}"), headers=HEADERS)
                assert answer.status == 200
            assert set(handler.artifact_ids) == {older["artifact_id"]}
            # The settings show the stopped artifact's weight on the active artifact, which answers its requests.
            settings = await (await client.get("/reef/scenarios/math/settings")).json()
            assert settings["served"] == {"weights": {older["artifact_id"]: 100}}
        finally:
            await client.close()
            value.close()

    asyncio.run(run())


@pytest.mark.unit
def test_an_activation_keeps_the_split_and_a_deletion_removes_it(tmp_path) -> None:
    value, _, _, newest, older = two_artifacts(tmp_path)
    try:
        weights = {newest["artifact_id"]: 10, older["artifact_id"]: 90}
        value.set_served("math", weights)
        wait_for_serving(value)
        # The newest artifact is served beside the active one, so it keeps its route and its weight.
        value.activate("math", newest["artifact_id"])
        wait_for_serving(value)
        table = value.coordination.served_models
        assert table.read_traffic_weights("math") == weights
        assert value.scenario_settings("math")["served"] == {"weights": {newest["artifact_id"]: 100}}
        value.delete_scenario("math")
        assert table.read_traffic_weights("math") == {}
    finally:
        value.close()


@pytest.mark.unit
def test_the_trainer_learns_which_artifacts_serve_beside_from_the_commit_history(tmp_path) -> None:
    value, _, _, newest, older = two_artifacts(tmp_path)
    try:
        serving_both(value, newest["artifact_id"], older["artifact_id"])
        scenario = value.get_or_create_scenario("math")
        assert older["artifact_id"] in scenario.served_beside()
        scenario.prepare_job(None, scenario.serving_artifact_id())
        served = scenario.trainers[0].trainer.pipeline.served_artifacts
        assert older["artifact_id"] in served.beside_artifact_ids
    finally:
        value.close()
