"""What a scenario serves: the ``served`` mapping of artifact IDs to weights, and each artifact's own URL (RFCs #997, #1325)."""

from __future__ import annotations

import asyncio
import time

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.reef_service.inference.served_artifacts import (
    ArtifactEcho,
    NamedComponent,
    answering_each_artifact,
    serving_dispatcher,
    two_artifacts,
)
from tests.support.artifact_versions import ImportingComponent, dispatcher, train
from tests.support.harness_jobs import COMPONENT, MODEL, harness_recipe
from tests.support.policy_recipe import TestPolicyRecipe
from tests.support.runtime_stubs import live_objects
from tests.support.serving import wait_for_serving

from reef.artifact import InMemoryRepositoryBackend
from reef.artifact.artifact import ArtifactNotFoundError
from reef.artifact.sources import DirectorySource
from reef.core.errors import ReefError
from reef.core.served_models import ScheduledStop, ServedArtifact
from reef.dispatcher import Dispatcher
from reef.dispatcher.serving import ServingWorker
from reef.inference.interfaces import FixedEngineClients
from reef.service.app import create_app
from reef.service.inference_service import ARTIFACT_HEADER, RECORD_HEADER, build_inference_app
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.trainer import Trainer

CHAT = {"model": "reef", "messages": [{"role": "user", "content": "hi"}]}


@pytest.mark.unit
def test_a_served_artifact_answers_at_its_url_while_the_default_url_answers_from_the_active_release(
    tmp_path,
) -> None:
    async def run() -> None:
        value, _, _, newest, older = two_artifacts(tmp_path)
        mapping = {newest["artifact_id"]: 100, older["artifact_id"]: 0}
        assert value.set_served("math", mapping) == {"served": mapping}
        wait_for_serving(value)
        # The artifact with weight 0 has its own row, under the runtime's model name for it.
        rows = value.coordination.served_models.read_scenario("math")
        assert rows[older["artifact_id"]] == ServedArtifact(
            older["artifact_id"], "http://trainer", f"sampler:{older['artifact_id']}"
        )
        assert rows["active"].model_name == "stub-model"
        handler = ArtifactEcho()
        client = TestClient(TestServer(create_app(value, inference_handler=handler)))
        await client.start_server()
        try:
            headers = {"x-reef-scenario": "math"}
            assert (await client.post("/v1/chat/completions", json=CHAT, headers=headers)).status == 200
            beside = await client.post(
                f"/reef/scenarios/math/artifacts/{older['artifact_id']}/v1/chat/completions", json=CHAT
            )
            assert beside.status == 200
            # The active artifact answers at its artifact's URL too.
            active = await client.post(
                f"/reef/scenarios/math/artifacts/{newest['artifact_id']}/v1/chat/completions", json=CHAT
            )
            assert active.status == 200
            assert handler.artifact_ids == [newest["artifact_id"], older["artifact_id"], newest["artifact_id"]]
            # The record names the snapshot that answered.
            record_id = beside.headers["x-reef-agent-record-id"]
            record = value.get_or_create_scenario("math").records.read("math", record_id)
            assert record.artifact_id == older["artifact_id"]

            missing = await client.post("/reef/scenarios/math/artifacts/missing/v1/chat/completions", json=CHAT)
            assert missing.status == 404
            unknown = await client.post(
                f"/reef/scenarios/other/artifacts/{older['artifact_id']}/v1/chat/completions", json=CHAT
            )
            assert unknown.status == 404
            assert not value.has_scenario("other")
            route = await client.post(f"/reef/scenarios/math/artifacts/{older['artifact_id']}/v1/models", json=CHAT)
            assert route.status == 404
        finally:
            await client.close()
            value.close()

    asyncio.run(run())


@pytest.mark.unit
def test_a_harness_artifact_served_beside_the_active_one_answers_under_its_own_id(tmp_path) -> None:
    """A harness scenario loads no weights, so one endpoint model answers every route. The route of an artifact that
    serves beside the active one still names that artifact, at its own URL and in a traffic split."""
    recipe, live = harness_recipe(tmp_path, lambda workspace, services: None)
    backends = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    coordination = memory_coordination()
    storage = SQLiteScenarioStorage(tmp_path / "agent-record")
    control = Dispatcher(
        recipe,
        backends,
        live=live,
        scenario_storage=storage,
        coordination=coordination,
        components=frozenset({"control"}),
        local_artifact_dir=tmp_path / "staged",
    )
    worker = ServingWorker(
        recipe, live, backends, coordination, worker_id="inference", local_artifact_dir=tmp_path / "inference"
    )
    tree, imported = tmp_path / "tree", tmp_path / "imported"
    tree.mkdir()
    imported.mkdir()
    (tree / "rules.md").write_text("Harness two.\n")
    handler = ArtifactEcho()
    app = build_inference_app(coordination.served_models, storage, FixedEngineClients(handler), tokens=None)

    async def answers(beside: str) -> list[tuple[int, str, str]]:
        """The status, artifact header and record ID of a request to the artifact's URL, then of a split request."""
        async with TestClient(TestServer(app)) as client:
            named = await client.post(f"/reef/scenarios/math/artifacts/{beside}/v1/chat/completions", json=CHAT)
            split = await client.post("/v1/chat/completions", json=CHAT, headers={"x-reef-scenario": "math"})
            return [
                (answer.status, answer.headers.get(ARTIFACT_HEADER, ""), answer.headers.get(RECORD_HEADER, ""))
                for answer in (named, split)
            ]

    try:
        scenario = control.create_scenario("math")[0]
        active = scenario.serving_artifact_id()
        beside = scenario.import_artifact(
            DirectorySource(tree, source_version="", link=False),
            COMPONENT,
            source={"uri": tree.as_uri(), "version": ""},
            directory=imported,
        ).artifact_id
        control.set_served("math", {active: 100, beside: 0})
        while worker.run_once():
            pass
        rows = coordination.served_models.read_scenario("math")
        assert active != beside and rows["active"] == ServedArtifact(active, MODEL.base_url, MODEL.model)
        assert rows[beside] == ServedArtifact(beside, MODEL.base_url, MODEL.model)
        # A split that gives every conversation to the artifact beside the active one.
        control.publisher.write_traffic_weights("math", {active: 0, beside: 100})

        answered = asyncio.run(answers(beside))
        assert [(status, artifact_id) for status, artifact_id, record_id in answered] == [(200, beside)] * 2
        assert handler.artifact_ids == [beside] * 2
        records = scenario.records
        assert [records.read("math", record_id).artifact_id for status, artifact_id, record_id in answered] == [
            beside
        ] * 2

        # The scenario serves nothing, and the stops of its withdrawn routes come due. No engine holds weights for
        # these routes, so the same mapping activates nothing: the endpoint model answers both artifacts again.
        routes = coordination.served_models
        assert control.set_served("math", {}) == {"served": {}} and routes.read_scenario("math") == {}
        for served in set(rows.values()):
            routes.schedule_stop(ScheduledStop(served, 0), control_lease_number=control.publisher.lease_number)
        while worker.run_once():
            pass
        assert routes.due_stops(time.time()) == ()
        control.set_served("math", {active: 100, beside: 0})
        while worker.run_once():
            pass
        assert routes.read_scenario("math") == rows and worker.activated == {}
    finally:
        worker.stop()
        control.close()


@pytest.mark.unit
def test_a_mapping_serves_exactly_its_artifacts_and_the_commit_history_records_each_change(tmp_path) -> None:
    value, _, _, newest, older = two_artifacts(tmp_path)
    scenario = value.get_or_create_scenario("math")
    new_id, old_id = newest["artifact_id"], older["artifact_id"]
    # The older artifact becomes active, and the artifact that was active stays loaded with weight 0.
    assert value.set_served("math", {old_id: 100, new_id: 0}) == {"served": {old_id: 100, new_id: 0}}
    assert scenario.serving_artifact_id() == old_id
    assert scenario.served_beside()[new_id].selection.components == {"weights": new_id}
    rows = {row["artifact_id"]: row for row in value.list_artifacts("math")}
    assert (rows[new_id]["served"], rows[new_id]["active"]) == (True, False)
    assert (rows[old_id]["served"], rows[old_id]["active"]) == (False, True)
    # The same mapping again changes nothing.
    steps = len(scenario.store.history())
    assert value.set_served("math", {old_id: 100, new_id: 0}) == {"served": {old_id: 100, new_id: 0}}
    assert len(scenario.store.history()) == steps

    # A mapping without the artifact stops it.
    assert value.set_served("math", {old_id: 100}) == {"served": {old_id: 100}}
    assert [record.operation for record in scenario.store.history()[-3:]] == ["serve", "activate", "stop_serving"]
    assert new_id not in scenario.served_beside()
    with pytest.raises(ArtifactNotFoundError):
        scenario.stop_serving(new_id)
    # The active artifact still answers at its artifact's URL; the stopped artifact has no row.
    wait_for_serving(value)
    rows = value.coordination.served_models.read_scenario("math")
    assert old_id in rows and new_id not in rows
    value.close()


@pytest.mark.unit
def test_a_mapping_of_one_artifact_makes_it_active_and_stops_the_artifacts_served_beside(tmp_path) -> None:
    value, _, _, newest, older = two_artifacts(tmp_path)
    scenario = value.get_or_create_scenario("math")
    new_id, old_id = newest["artifact_id"], older["artifact_id"]
    value.set_served("math", {new_id: 100, old_id: 0})
    assert set(scenario.served_beside()) == {old_id}
    # The artifact that was served beside becomes the only served artifact.
    assert value.set_served("math", {old_id: 100}) == {"served": {old_id: 100}}
    assert scenario.serving_artifact_id() == old_id
    assert not scenario.served_beside()
    assert value.scenario_settings("math")["served"] == {"weights": {old_id: 100}}
    wait_for_serving(value)
    rows = value.coordination.served_models.read_scenario("math")
    assert old_id in rows and new_id not in rows
    value.close()


@pytest.mark.unit
def test_the_largest_weight_is_active_and_the_first_in_sorted_order_among_equals(tmp_path) -> None:
    value, _, _, newest, older = two_artifacts(tmp_path)
    scenario = value.get_or_create_scenario("math")
    first, last = sorted([newest["artifact_id"], older["artifact_id"]])
    value.set_served("math", {first: 50, last: 50})
    assert scenario.serving_artifact_id() == first
    assert value.scenario_settings("math")["served"] == {"weights": {first: 50, last: 50}}
    value.set_served("math", {first: 40, last: 60})
    assert scenario.serving_artifact_id() == last
    assert set(scenario.served_beside()) == {first}
    assert value.scenario_settings("math")["served"] == {"weights": {first: 40, last: 60}}
    value.close()


@pytest.mark.unit
def test_a_mapping_that_names_an_unknown_artifact_changes_nothing(tmp_path) -> None:
    value, _, _, newest, older = two_artifacts(tmp_path)
    scenario = value.get_or_create_scenario("math")
    new_id, old_id = newest["artifact_id"], older["artifact_id"]
    steps = len(scenario.store.history())
    for mapping in ({"missing": 100}, {new_id: 60, "missing": 40}, {old_id: 100, "missing": 0}):
        with pytest.raises(ArtifactNotFoundError):
            value.set_served("math", mapping)
    with pytest.raises(ValueError, match="no component 'other'"):
        value.set_served("math", {old_id: 100}, "other")
    assert len(scenario.store.history()) == steps
    assert scenario.serving_artifact_id() == new_id and not scenario.served_beside()
    assert value.coordination.served_models.read_traffic_weights("math") == {}
    assert value.scenario_settings("math")["served"] == {"weights": {new_id: 100}}
    value.close()


@pytest.mark.unit
def test_an_empty_mapping_makes_the_scenario_serve_nothing_until_a_mapping_names_an_artifact(tmp_path) -> None:
    async def run() -> None:
        value, _, _, newest, older = two_artifacts(tmp_path)
        new_id, old_id = newest["artifact_id"], older["artifact_id"]
        value.set_served("math", {new_id: 60, old_id: 40})
        wait_for_serving(value)
        client = TestClient(TestServer(create_app(value, inference_handler=ArtifactEcho())))
        await client.start_server()
        headers = {"x-reef-scenario": "math"}
        try:
            assert (await client.post("/v1/chat/completions", json=CHAT, headers=headers)).status == 200
            with pytest.raises(ValueError, match="leave component out"):
                value.set_served("math", {}, older["component"])

            stopped = await client.post("/reef/scenarios/math/update", json={"served": {}})
            assert (stopped.status, await stopped.json()) == (200, {"scenario": "math", "served": {}})
            scenario = value.get_or_create_scenario("math")
            # The routes, the artifacts beside and the split go; the active selection stays.
            assert value.coordination.served_models.read_scenario("math") == {}
            assert value.coordination.served_models.read_traffic_weights("math") == {}
            assert not scenario.served_beside()
            assert scenario.serving_artifact_id() == new_id
            assert value.scenario_settings("math")["served"] == {"weights": {}}
            assert (await client.post("/v1/chat/completions", json=CHAT, headers=headers)).status == 404
            beside = await client.post(f"/reef/scenarios/math/artifacts/{new_id}/v1/chat/completions", json=CHAT)
            assert beside.status == 404
            # Training still commits, and its activation starts no route.
            train(value, 3)
            assert value.publisher.desired("math") == {}

            assert value.set_served("math", {old_id: 100}) == {"served": {old_id: 100}}
            wait_for_serving(value)
            assert value.coordination.served_models.read_scenario("math")["active"].artifact_id == old_id
            assert (await client.post("/v1/chat/completions", json=CHAT, headers=headers)).status == 200
        finally:
            await client.close()
            value.close()

    asyncio.run(run())


@pytest.mark.unit
def test_a_scenario_that_serves_nothing_serves_nothing_after_a_restart(tmp_path) -> None:
    value, _, backend_factory, _, _ = two_artifacts(tmp_path)
    value.set_served("math", {})
    value.close()

    component = NamedComponent(tmp_path / "restarted-export")
    restarted = Dispatcher(
        TestPolicyRecipe(),
        backend_factory,
        live=live_objects(component, component.inference),
        local_artifact_dir=tmp_path / "restarted-staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record"),
        coordination=memory_coordination(),
    )
    assert not restarted.get_or_create_scenario("math").serves
    assert restarted.publisher.desired("math") == {}
    assert restarted.scenario_settings("math")["served"] == {"weights": {}}
    restarted.close()


@pytest.mark.unit
def test_a_scenario_created_with_serves_false_serves_nothing_until_an_update_names_an_artifact(tmp_path) -> None:
    async def run() -> None:
        value, _, _ = serving_dispatcher(tmp_path)
        client = TestClient(TestServer(create_app(value, inference_handler=ArtifactEcho())))
        await client.start_server()
        headers = {"x-reef-scenario": "math"}
        try:
            refused = await client.post("/reef/scenarios", json={"name": "math", "serves": "no"})
            assert refused.status == 400
            created = await client.post("/reef/scenarios", json={"name": "math", "serves": False})
            assert created.status == 201
            start = (await created.json())["artifact_id"]
            settings = await (await client.get("/reef/scenarios/math/settings")).json()
            assert settings["served"] == {"weights": {}}
            assert value.coordination.served_models.current_request("math").payload == {}
            assert value.coordination.served_models.read_scenario("math") == {}
            assert (await client.post("/v1/chat/completions", json=CHAT, headers=headers)).status == 404
            # A second create changes nothing.
            assert (await client.post("/reef/scenarios", json={"name": "math", "serves": True})).status == 200
            assert value.publisher.desired("math") == {}

            served = await client.post("/reef/scenarios/math/update", json={"served": {start: 100}})
            assert (served.status, await served.json()) == (200, {"scenario": "math", "served": {start: 100}})
            wait_for_serving(value)
            assert (await client.post("/v1/chat/completions", json=CHAT, headers=headers)).status == 200
        finally:
            await client.close()
            value.close()

    asyncio.run(run())


@pytest.mark.unit
def test_the_served_artifacts_survive_a_restart(tmp_path) -> None:
    value, _, backend_factory, newest, older = two_artifacts(tmp_path)
    value.set_served("math", {newest["artifact_id"]: 100, older["artifact_id"]: 0})
    value.close()

    component = ImportingComponent(tmp_path / "restarted-export")
    answering_each_artifact(component)
    restarted = Dispatcher(
        TestPolicyRecipe(),
        backend_factory,
        live=live_objects(component, component.inference),
        local_artifact_dir=tmp_path / "restarted-staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record"),
        coordination=memory_coordination(),
    )
    scenario = restarted.get_or_create_scenario("math")
    assert scenario.served_beside()[older["artifact_id"]].selection.components == {"weights": older["artifact_id"]}
    assert [row["artifact_id"] for row in restarted.list_artifacts("math") if row["served"]] == [older["artifact_id"]]
    restarted.close()


@pytest.mark.unit
def test_a_runtime_that_answers_only_from_its_loaded_release_refuses_a_second_served_artifact(tmp_path) -> None:
    value, _, _ = dispatcher(tmp_path)
    train(value, 1)
    active, older = value.get_or_create_scenario("math").artifacts()
    scenario = value.get_or_create_scenario("math")
    steps = len(scenario.store.history())
    for mapping in (
        {active["artifact_id"]: 100, older["artifact_id"]: 0},
        {active["artifact_id"]: 0, older["artifact_id"]: 100},
    ):
        with pytest.raises(ReefError, match="answers only from the snapshot it has loaded"):
            value.set_served("math", mapping)
    # The scenario itself refuses the artifact too, for a caller that does not pass a mapping.
    with pytest.raises(ReefError, match="answers only from the snapshot it has loaded"):
        scenario.serve(older["artifact_id"], older["component"])
    assert len(scenario.store.history()) == steps
    assert scenario.serving_artifact_id() == active["artifact_id"]
    assert not scenario.served_beside()
    value.close()


@pytest.mark.unit
def test_a_runtime_with_two_checkpoint_slots_serves_two_artifacts_and_refuses_a_third_over_http(tmp_path) -> None:
    async def run() -> None:
        trained = two_artifacts(tmp_path, checkpoint_slots=True)
        value, newest, older = trained[0], trained[3], trained[4]
        scenario = value.get_or_create_scenario("math")
        new_id, old_id = newest["artifact_id"], older["artifact_id"]
        # The artifact that the scenario started with is the third one.
        root_id = scenario.artifacts()[-1]["artifact_id"]
        served_models = value.coordination.served_models
        client = TestClient(TestServer(create_app(value)))
        await client.start_server()

        async def update(mapping: dict[str, int]):
            body = {"served": mapping, "component": older["component"]}
            return await client.post("/reef/scenarios/math/update", json=body)

        async def served() -> dict[str, dict[str, int]]:
            return (await (await client.get("/reef/scenarios/math/settings")).json())["served"]

        def state() -> tuple[object, ...]:
            request = served_models.current_request("math")
            return (
                [record.operation for record in scenario.store.history()],
                scenario.serving_artifact_id(),
                sorted(scenario.served_beside()),
                served_models.read_scenario("math"),
                served_models.read_traffic_weights("math"),
                (request.id, request.state),
            )

        try:
            # The older artifact loads beside the active one with weight 0, takes a share of the traffic, becomes the
            # active one, and steps back. Each mapping names two artifacts, as many as the runtime keeps loaded.
            for mapping, active_id in (
                ({new_id: 100, old_id: 0}, new_id),
                ({new_id: 70, old_id: 30}, new_id),
                ({new_id: 0, old_id: 100}, old_id),
                ({new_id: 100, old_id: 0}, new_id),
            ):
                accepted = await update(mapping)
                assert accepted.status == 200, (mapping, await accepted.text())
                assert await accepted.json() == {"scenario": "math", "served": mapping}
                assert await served() == {"weights": mapping}
                assert scenario.serving_artifact_id() == active_id
                assert set(scenario.served_beside()) == {new_id, old_id} - {active_id}
                split = mapping if min(mapping.values()) > 0 else {}
                assert served_models.read_traffic_weights("math") == split
            # Each of the two artifacts has its own route: the runtime loaded the older one under its own model name.
            wait_for_serving(value)
            rows = served_models.read_scenario("math")
            assert rows["active"] == rows[new_id] == ServedArtifact(new_id, "http://trainer", "stub-model")
            assert rows[old_id] == ServedArtifact(old_id, "http://trainer", f"sampler:{old_id}")
            assert root_id not in rows

            # A third artifact is one more than the runtime keeps loaded, whatever its weight. The refusal comes
            # before every change: the commit history, the selection, the routes, the traffic split and the request
            # to inference stay as they were.
            before = state()
            for mapping in (
                {new_id: 100, old_id: 0, root_id: 0},
                {new_id: 50, old_id: 30, root_id: 20},
                {new_id: 0, old_id: 0, root_id: 100},
            ):
                refused = await update(mapping)
                assert refused.status == 400, mapping
                reason = await refused.text()
                assert f"keeps at most 2 artifacts of component {older['component']!r} loaded at once" in reason
                assert "cannot serve the 3 artifacts" in reason
            assert state() == before
            assert await served() == {"weights": {new_id: 100, old_id: 0}}

            # The runtime serves the artifacts that a mapping deploys. It still evaluates no candidate.
            refusal = scenario.evaluation_refusal(older["component"])
            assert refusal is not None and "serves only its current artifact" in refusal

            # A scenario that serves nothing gets the same refusal, also before every change: it still serves
            # nothing. A mapping of two artifacts then serves both again under the routes from before.
            stopped = await client.post("/reef/scenarios/math/update", json={"served": {}})
            assert (stopped.status, await served()) == (200, {"weights": {}})
            before = state()
            refused = await update({new_id: 50, old_id: 30, root_id: 20})
            assert refused.status == 400 and "cannot serve the 3 artifacts" in await refused.text()
            assert state() == before and not scenario.serves and value.publisher.desired("math") == {}
            assert await served() == {"weights": {}}
            restored = await update({new_id: 100, old_id: 0})
            assert (restored.status, await served()) == (200, {"weights": {new_id: 100, old_id: 0}})
            wait_for_serving(value)
            assert served_models.read_scenario("math") == rows
        finally:
            await client.close()
            value.close()

    asyncio.run(run())


@pytest.mark.unit
def test_a_mapping_that_replaces_one_of_two_loaded_artifacts_counts_only_the_artifacts_that_it_names(tmp_path) -> None:
    trained = two_artifacts(tmp_path, checkpoint_slots=True)
    value, newest, older = trained[0], trained[3], trained[4]
    scenario = value.get_or_create_scenario("math")
    new_id, old_id = newest["artifact_id"], older["artifact_id"]
    root_id = scenario.artifacts()[-1]["artifact_id"]
    value.set_served("math", {new_id: 100, old_id: 0})
    # The mapping keeps the older artifact and replaces the newest one. The test covers only how control accepts the
    # final mapping, which names two artifacts, as many as the runtime keeps loaded. ``TwoSlotInference`` is a stub
    # without slots: the test does not show that a real runtime with both slots in use can make this change before
    # the stopped artifact retires. ``CheckpointSlotsRuntime`` refuses the added artifact until then.
    mapping = {old_id: 100, root_id: 0}
    assert value.set_served("math", mapping) == {"served": mapping}
    assert [record.operation for record in scenario.store.history()[-3:]] == ["serve", "activate", "stop_serving"]
    assert scenario.serving_artifact_id() == old_id
    assert set(scenario.served_beside()) == {root_id}
    assert value.scenario_settings("math")["served"] == {"weights": mapping}
    rows = {row["artifact_id"]: row for row in value.list_artifacts("math")}
    assert (rows[new_id]["served"], rows[new_id]["active"]) == (False, False)
    value.close()


@pytest.mark.unit
def test_served_http_api(tmp_path, monkeypatch) -> None:
    async def run() -> None:
        value, _, _, newest, older = two_artifacts(tmp_path)
        new_id, old_id = newest["artifact_id"], older["artifact_id"]
        client = TestClient(TestServer(create_app(value)))
        await client.start_server()

        async def update(body: dict[str, object], scenario: str = "math"):
            return await client.post(f"/reef/scenarios/{scenario}/update", json=body)

        async def served() -> dict[str, dict[str, int]]:
            return (await (await client.get("/reef/scenarios/math/settings")).json())["served"]

        try:
            assert await served() == {"weights": {new_id: 100}}
            # Weight 0 loads an artifact beside the active one.
            beside = {new_id: 100, old_id: 0}
            loaded = await update({"served": beside, "component": older["component"]})
            assert loaded.status == 200
            assert await loaded.json() == {"scenario": "math", "served": beside}
            listed = await (await client.get("/reef/scenarios/math/artifacts")).json()
            assert [row["artifact_id"] for row in listed["artifacts"] if row["served"]] == [old_id]
            assert await served() == {"weights": beside}

            # Each refused body leaves what the scenario serves.
            for body in (
                {"component": older["component"]},
                {"served": beside, "component": ""},
                {"served": beside, "component": 1},
                {"served": beside, "component": "other"},
                {"served": {}, "component": older["component"]},
                {"served": None},
                {"served": [new_id]},
                {"served": {new_id: 90}},
                {"served": {new_id: 60, old_id: 60}},
                {"served": {new_id: 100.0}},
                {"served": {new_id: 101, old_id: -1}},
                {"served": {old_id: 100}, "traffic_weights": None},
            ):
                refused = await update(body)
                assert refused.status == 400, (body, await refused.text())
            for body in ({"served": {"missing": 100}}, {"served": {new_id: 100, "missing": 0}}):
                assert (await update(body)).status == 404, body
            assert (await update({"served": beside}, scenario="other")).status == 404
            assert await served() == {"weights": beside}

            # A mapping without the artifact stops it.
            stopped = await update({"served": {new_id: 100}})
            assert stopped.status == 200
            assert await stopped.json() == {"scenario": "math", "served": {new_id: 100}}
            listed = await (await client.get("/reef/scenarios/math/artifacts")).json()
            assert not [row for row in listed["artifacts"] if row["served"]]

            # The three operations that ``served`` replaced have no route.
            for path in ("activate", "serve", "stop-serving"):
                removed = await client.post(f"/reef/scenarios/math/{path}", json={"artifact_id": old_id})
                assert removed.status in (404, 405), path
            assert await served() == {"weights": {new_id: 100}}

            monkeypatch.setattr(Trainer, "has_pending_commit", property(lambda self: True))
            waiting = await update({"served": beside})
            assert waiting.status == 409
        finally:
            await client.close()
            value.close()

    asyncio.run(run())
