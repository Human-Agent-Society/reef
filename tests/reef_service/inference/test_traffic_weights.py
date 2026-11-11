"""A scenario's traffic split, the weights above 0 of its ``served`` mapping, and the retry of a request: which served
artifact answers it."""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from tests.reef_service.inference.served_artifacts import ArtifactEcho, two_artifacts
from tests.support.serving import wait_for_serving

from reef.core.records_types import AgentRecord
from reef.core.served_models import ACTIVE, ScenarioModel, ServedArtifact
from reef.dispatcher import Dispatcher
from reef.inference.interfaces import FixedEngineClients, full_weight_name
from reef.service.app import create_app
from reef.service.engine_clients import DeploymentEngineClients
from reef.service.inference_service import (
    ARTIFACT_HEADER,
    BUCKET_COUNT,
    RECORD_HEADER,
    TrafficWeights,
    build_inference_app,
    conversation_bucket,
)
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage

HEADERS = {"x-reef-scenario": "math"}
#: The model name of each row of the retry tests, by (engine, artifact): the full-weight name of the runtime load that
#: serves the artifact there. An artifact that moves to the new engine has a new load there.
ROW_NAMES = {
    ("old", "artifact-a"): full_weight_name("artifact-a", "old:1"),
    ("old", "artifact-x"): full_weight_name("artifact-x", "old:2"),
    ("new", "artifact-a"): full_weight_name("artifact-a", "new:1"),
    ("new", "artifact-b"): full_weight_name("artifact-b", "new:2"),
    ("new", "artifact-x"): full_weight_name("artifact-x", "new:3"),
}
#: For each selection of a retried request: its route and its body's ``model``, then the (engine, artifact) rows of its
#: first and of its retried engine call. ``body_artifact`` names X in the body; ``body_active`` names A, which only
#: the ``active`` row serves, as in a scenario without durable records.
RETRY_CALLS = {
    "active": ("/v1/chat/completions", ACTIVE, ("old", "artifact-a"), ("new", "artifact-b")),
    "split": ("/v1/chat/completions", ACTIVE, ("old", "artifact-x"), ("new", "artifact-x")),
    "named": (
        "/reef/scenarios/math/artifacts/artifact-x/v1/chat/completions",
        ACTIVE,
        ("old", "artifact-x"),
        ("new", "artifact-x"),
    ),
    "body_artifact": ("/v1/chat/completions", "artifact-x", ("old", "artifact-x"), ("new", "artifact-x")),
    "body_active": ("/v1/chat/completions", "artifact-a", ("old", "artifact-a"), ("new", "artifact-a")),
}


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


def expected_calls(*rows: tuple[str, str]) -> list[tuple[str, str, str, str]]:
    """The engine calls that reach ``rows``, each an (engine, artifact) key of ``ROW_NAMES``: the engine, the
    ``x-reef-artifact-id`` header, and the row's name as the body's ``model`` and as the ``x-reef-served-model`` pin.
    """
    return [
        (engine, artifact_id, ROW_NAMES[engine, artifact_id], ROW_NAMES[engine, artifact_id])
        for engine, artifact_id in rows
    ]


def retried_request(
    tmp_path: Path, *, selection: str, stream: bool, cause: str, after: str
) -> tuple[int, dict[str, str], bytes, list[tuple[str, str | None, str, str | None]], list[AgentRecord]]:
    """Send one request to a separate inference service while a promotion changes the rows of its first engine call.

    Before the promotion, ``active`` is A on the old engine and ``artifact-x`` is X there, each under its name in
    ``ROW_NAMES``; the ``split`` selection sends every conversation to X. At the first call the old engine refuses the
    model name (``cause`` "refuse": HTTP 421) or aborts, and the promotion makes ``active`` B on the new engine; with
    ``after`` "reloaded", it is A again there, under a new load. X moves to the new engine, and ``after`` "refused"
    also makes the new engine refuse; "removed" stops serving X instead. Return the answer's status, headers (lowercase
    names) and body, the engine calls in the form of ``expected_calls``, and the stored records.
    """
    coordination = memory_coordination()
    table = coordination.served_models
    lease = coordination.control_lease.acquire("control").number
    storage = SQLiteScenarioStorage(tmp_path / "records")
    calls: list[tuple[str, str | None, str, str | None]] = []
    urls: dict[str, str] = {}

    def serve(name: str, artifact_id: str, engine: str) -> None:
        served = ServedArtifact(artifact_id, urls[engine], ROW_NAMES[engine, artifact_id])
        table.write(ScenarioModel("math", name), served, control_lease_number=lease)

    def engine_app(engine: str) -> web.Application:
        async def chat_completions(request: web.Request) -> web.Response:
            payload = await request.json()
            artifact_header = request.headers.get("x-reef-artifact-id")
            calls.append((engine, artifact_header, payload["model"], request.headers.get("x-reef-served-model")))
            if engine == "old":
                serve(ACTIVE, "artifact-a" if after == "reloaded" else "artifact-b", "new")
                if after == "removed":
                    table.delete(ScenarioModel("math", "artifact-x"), control_lease_number=lease)
                else:
                    serve("artifact-x", "artifact-x", "new")
                if cause == "abort":
                    return web.json_response({"choices": [{"message": {"content": ""}, "finish_reason": "abort"}]})
            if engine == "old" or after == "refused":
                return web.json_response({"error": {"message": "the engine does not serve this model"}}, status=421)
            if payload.get("stream") is True:
                return web.Response(
                    body=b'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
                    content_type="text/event-stream",
                )
            return web.json_response({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat_completions)
        return app

    async def run() -> tuple[int, dict[str, str], bytes]:
        async with TestServer(engine_app("old")) as old, TestServer(engine_app("new")) as new:
            urls.update(old=str(old.make_url("")).rstrip("/"), new=str(new.make_url("")).rstrip("/"))
            serve(ACTIVE, "artifact-a", "old")
            serve("artifact-x", "artifact-x", "old")
            if selection == "split":
                table.write_traffic_weights("math", {"artifact-x": 100}, control_lease_number=lease)
            # No runtime: each row's URL gets its own proxy handler, as for independently configured endpoints.
            engines = DeploymentEngineClients(None, provider_key=None, timeout_seconds=30.0)
            app = build_inference_app(table, storage, engines, tokens=None)
            route, model, _, _ = RETRY_CALLS[selection]
            async with TestClient(TestServer(app)) as client:
                answer = await client.post(route, json={**chat(), "model": model, "stream": stream}, headers=HEADERS)
                headers = {name.lower(): value for name, value in answer.headers.items()}
                return answer.status, headers, await answer.read()

    try:
        status, headers, body = asyncio.run(run())
        store = storage.open("math")
        try:
            records = list(store.records.replay("math"))
        finally:
            store.close()
    finally:
        storage.close()
        coordination.close()
    return status, headers, body, calls, records


@pytest.mark.unit
@pytest.mark.parametrize(
    ("selection", "after"),
    [
        ("active", "moved"),
        ("split", "moved"),
        ("named", "moved"),
        ("body_artifact", "moved"),
        # Only ``active`` serves A: its retry needs no row named A, only a row that still serves A.
        ("body_active", "reloaded"),
    ],
)
@pytest.mark.parametrize(("cause", "stream"), [("refuse", False), ("refuse", True), ("abort", False)])
def test_a_retry_reads_the_row_of_its_selection_again(tmp_path, selection, after, cause, stream) -> None:
    status, headers, body, calls, records = retried_request(
        tmp_path, selection=selection, stream=stream, cause=cause, after=after
    )
    _, _, first, retried = RETRY_CALLS[selection]
    assert status == 200
    # The retry reaches the row's new load on the new engine: the body's model and the pin header both name it. A
    # request for the active artifact reaches B. A request that selected its artifact keeps it, though B now serves
    # on the same engine.
    assert calls == expected_calls(first, retried)
    [stored] = records
    assert headers[ARTIFACT_HEADER] == stored.artifact_id == retried[1]
    if stream:
        assert stored.agent_record_id.encode() in body
    else:
        assert headers[RECORD_HEADER] == stored.agent_record_id
    split = {"bucket": conversation_bucket("math", chat()), "weights": {"artifact-x": 100}}
    assert stored.payload.get("metadata", {}).get("traffic") == (split if selection == "split" else None)


@pytest.mark.unit
@pytest.mark.parametrize("selection", ["split", "named", "body_artifact", "body_active"])
@pytest.mark.parametrize(("cause", "stream"), [("refuse", False), ("refuse", True), ("abort", False)])
def test_a_retry_whose_artifact_stopped_serving_answers_503_instead_of_moving(
    tmp_path, selection, cause, stream
) -> None:
    status, headers, _, calls, records = retried_request(
        tmp_path, selection=selection, stream=stream, cause=cause, after="removed"
    )
    # The selected artifact has no row any more, and the active row now serves B: no call reaches B.
    assert (status, headers.get("retry-after")) == (503, "1")
    assert calls == expected_calls(RETRY_CALLS[selection][2])
    assert records == []


@pytest.mark.unit
@pytest.mark.parametrize("stream", [False, True])
def test_a_second_refusal_answers_503_without_a_record(tmp_path, stream) -> None:
    status, headers, _, calls, records = retried_request(
        tmp_path, selection="split", stream=stream, cause="refuse", after="refused"
    )
    _, _, first, retried = RETRY_CALLS["split"]
    assert (status, headers.get("retry-after")) == (503, "1")
    assert calls == expected_calls(first, retried)
    assert records == []


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
