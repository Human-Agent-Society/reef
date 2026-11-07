"""A scenario's traffic split: which served artifact answers a request."""

from __future__ import annotations

import asyncio
from collections import Counter

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.reef_service.routing.routing_fakes import routing_dispatcher
from tests.reef_service.test_served_artifacts import ArtifactEcho, two_artifacts, wait_for_serving

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


def serving_both(value: Dispatcher, older: str) -> None:
    """Serve ``older`` beside the active artifact, and wait until its route is published."""
    value.serve("math", older)
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
        serving_both(value, older["artifact_id"])
        handler = ArtifactEcho()
        client = TestClient(TestServer(create_app(value, inference_handler=handler)))
        await client.start_server()
        try:
            weights = {newest["artifact_id"]: 50, older["artifact_id"]: 50}
            update = await client.post("/reef/scenarios/math/update", json={"traffic_weights": weights})
            assert update.status == 200
            assert (await update.json())["traffic_weights"] == weights
            settings = await (await client.get("/reef/scenarios/math/settings")).json()
            assert settings["traffic_weights"] == weights

            answers = []
            for n in range(60):
                answer = await client.post("/v1/chat/completions", json=chat(f"user {n}"), headers=HEADERS)
                assert answer.status == 200
                answers.append(answer.headers["x-reef-agent-record-id"])
            assert set(handler.artifact_ids) == set(weights)

            # Each record names the artifact that answered.
            scenario = value.get_or_create_scenario("math")
            stored = [scenario.records.get("math", record_id) for record_id in answers]
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

            # null ends the split: the active artifact answers everything.
            handler.artifact_ids.clear()
            cleared = await client.post("/reef/scenarios/math/update", json={"traffic_weights": None})
            assert cleared.status == 200
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
        serving_both(value, older["artifact_id"])
        handler = ArtifactEcho()
        storage = SQLiteScenarioStorage(tmp_path / "agent-record")
        app = build_inference_app(value.coordination.served_models, storage, FixedEngineClients(handler), tokens=None)
        client = TestClient(TestServer(app))
        await client.start_server()

        async def answered_by() -> set[str]:
            handler.artifact_ids.clear()
            for n in range(40):
                answer = await client.post("/v1/chat/completions", json=chat(f"user {n}"), headers=HEADERS)
                assert answer.status == 200
            return set(handler.artifact_ids)

        try:
            value.set_traffic_weights("math", {newest["artifact_id"]: 100, older["artifact_id"]: 0})
            assert await answered_by() == {newest["artifact_id"]}
            value.set_traffic_weights("math", {newest["artifact_id"]: 0, older["artifact_id"]: 100})
            assert await answered_by() == {older["artifact_id"]}
        finally:
            await client.close()
            storage.close()
            value.close()

    asyncio.run(run())


@pytest.mark.unit
def test_an_artifact_without_a_route_cannot_get_a_weight_over_http(tmp_path) -> None:
    async def run() -> None:
        value, _, _, newest, older = two_artifacts(tmp_path)
        wait_for_serving(value)
        client = TestClient(TestServer(create_app(value, inference_handler=ArtifactEcho())))
        await client.start_server()
        try:
            for weights in ({newest["artifact_id"]: 50}, {newest["artifact_id"]: 50, older["artifact_id"]: 50}):
                refused = await client.post("/reef/scenarios/math/update", json={"traffic_weights": weights})
                assert refused.status == 400
            assert value.scenario_settings("math")["traffic_weights"] == {}
            # A split does not start or stop an artifact.
            assert not value.get_or_create_scenario("math").served_beside()
            unknown = await client.post("/reef/scenarios/other/update", json={"traffic_weights": None})
            assert unknown.status == 404
        finally:
            await client.close()
            value.close()

    asyncio.run(run())


@pytest.mark.unit
def test_an_artifact_that_stops_serving_sends_its_requests_to_the_active_artifact(tmp_path) -> None:
    async def run() -> None:
        value, _, _, newest, older = two_artifacts(tmp_path)
        serving_both(value, older["artifact_id"])
        value.set_traffic_weights("math", {newest["artifact_id"]: 0, older["artifact_id"]: 100})
        value.stop_serving("math", older["artifact_id"])
        wait_for_serving(value)
        handler = ArtifactEcho()
        client = TestClient(TestServer(create_app(value, inference_handler=handler)))
        await client.start_server()
        try:
            answer = await client.post("/v1/chat/completions", json=chat(), headers=HEADERS)
            assert answer.status == 200
            assert handler.artifact_ids == [newest["artifact_id"]]
        finally:
            await client.close()
            value.close()

    asyncio.run(run())


@pytest.mark.unit
def test_an_activation_keeps_the_split_and_a_deletion_removes_it(tmp_path) -> None:
    value, _, _, newest, older = two_artifacts(tmp_path)
    try:
        serving_both(value, older["artifact_id"])
        weights = {newest["artifact_id"]: 90, older["artifact_id"]: 10}
        value.set_traffic_weights("math", weights)
        value.activate("math", older["artifact_id"])
        wait_for_serving(value)
        table = value.coordination.served_models
        assert table.read_traffic_weights("math") == weights
        value.delete_scenario("math")
        assert table.read_traffic_weights("math") == {}
    finally:
        value.close()


@pytest.mark.unit
def test_a_scenario_without_routes_cannot_split(tmp_path) -> None:
    value = routing_dispatcher(tmp_path)
    try:
        value.get_or_create_scenario("math")
        with pytest.raises(ValueError, match="serve these artifacts"):
            value.set_traffic_weights("math", {"artifact": 100})
        assert value.set_traffic_weights("math", {}) == {"traffic_weights": {}}
    finally:
        value.close()


@pytest.mark.unit
def test_the_trainer_learns_which_artifacts_serve_beside_from_the_commit_history(tmp_path) -> None:
    value, _, _, _, older = two_artifacts(tmp_path)
    try:
        serving_both(value, older["artifact_id"])
        scenario = value.get_or_create_scenario("math")
        assert older["artifact_id"] in scenario.served_beside()
        scenario.prepare_job(None, scenario.serving_artifact_id())
        served = scenario.trainers[0].trainer.pipeline.served_artifacts
        assert older["artifact_id"] in served.beside_artifact_ids
    finally:
        value.close()
