"""Artifacts served beside the active release, each at its own URL (RFC #997)."""

from __future__ import annotations

import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support._policy_recipe import TestPolicyRecipe
from tests.support.artifact_versions import ImportingComponent, ImportingInference, dispatcher, train
from tests.support.runtime_stubs import runtime_bindings

from reef.artifact import InMemoryRepositoryBackend, RepositoryBackendFactory
from reef.artifact.artifact import ArtifactNotFound
from reef.core.errors import ReefError
from reef.core.served_models import ServedArtifact
from reef.dispatcher import Dispatcher
from reef.runtime.interfaces import InferenceHandler
from reef.service.app import create_app
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.trainer import Trainer

CHAT = {"model": "reef", "messages": [{"role": "user", "content": "hi"}]}


class ReleaseEcho(InferenceHandler):
    """Answer each call with the release it was given, so a test sees which release answered."""

    def __init__(self) -> None:
        self.release_ids: list[str] = []

    async def inference(self, artifact, path, payload):
        del path, payload
        self.release_ids.append(artifact.ref.release_id)
        return {"choices": [{"message": {"content": "ok"}}]}


class NamedInference(ImportingInference):
    """An ``ImportingInference`` whose engine serves every scenario under one model name, so control writes the
    scenario's ``served_models`` rows and the inference routes answer."""

    def served_model_name(self, scenario: str) -> str:
        return "stub-model"


class NamedComponent(ImportingComponent):
    """An ``ImportingComponent`` whose checkpoints a ``NamedInference`` serves."""

    def __init__(self, checkpoint_dir) -> None:
        super().__init__(checkpoint_dir)
        self.inference = NamedInference(self.restore_order)


def serving_dispatcher(tmp_path) -> tuple[Dispatcher, NamedComponent, RepositoryBackendFactory]:
    """The dispatcher of ``tests.support.artifact_versions.dispatcher`` whose runtime serves under a model name."""
    initial = tmp_path / "initial"
    initial.mkdir(exist_ok=True)
    (initial / "model.txt").write_text("w0")
    component = NamedComponent(tmp_path / "exported")
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    value = Dispatcher(
        TestPolicyRecipe(**runtime_bindings(component, component.inference)),
        backend_factory,
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record"),
    )
    return value, component, backend_factory


def answering_each_release(runtime) -> None:
    """Make the stub runtime answer each call from the release that the call names, as Tinker does."""
    runtime.inference.answers_each_release = True


def two_artifacts(tmp_path):
    """A scenario that trained twice: the newest artifact is active, and ``older`` is the one before it."""
    value, runtime, backend_factory = serving_dispatcher(tmp_path)
    answering_each_release(runtime)
    train(value, 1)
    train(value, 2)
    newest, older, _ = value.list_artifacts("math")
    return value, runtime, backend_factory, newest, older


@pytest.mark.unit
def test_a_served_artifact_answers_at_its_url_while_the_default_url_answers_from_the_active_release(
    tmp_path,
) -> None:
    async def run() -> None:
        value, _, _, newest, older = two_artifacts(tmp_path)
        served = value.serve("math", older["artifact_id"])
        assert (served.artifact_id, served.release_id) == (older["artifact_id"], older["release_id"])
        # The artifact served beside the active one has its own row, under the runtime's model name for it.
        rows = value.coordination.served_models.read_scenario("math")
        assert rows[older["artifact_id"]] == ServedArtifact(
            older["artifact_id"], "http://trainer", f"sampler:{older['artifact_id']}"
        )
        assert rows["active"].model_name == "stub-model"
        handler = ReleaseEcho()
        client = TestClient(TestServer(create_app(value, inference_handler=handler)))
        await client.start_server()
        try:
            headers = {"x-reef-scenario": "math"}
            assert (await client.post("/v1/chat/completions", json=CHAT, headers=headers)).status == 200
            beside = await client.post(
                f"/reef/scenarios/math/artifacts/{older['artifact_id']}/v1/chat/completions", json=CHAT
            )
            assert beside.status == 200
            # The active release answers at its artifact's URL without a serve.
            active = await client.post(
                f"/reef/scenarios/math/artifacts/{newest['artifact_id']}/v1/chat/completions", json=CHAT
            )
            assert active.status == 200
            assert handler.release_ids == [newest["release_id"], older["release_id"], newest["release_id"]]
            # The record names the release that answered.
            record_id = beside.headers["x-reef-agent-record-id"]
            record = value.get_or_create_scenario("math").records.get("math", record_id)
            assert record.artifact_ref.release_id == older["release_id"]

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
def test_an_activation_leaves_the_served_artifacts_and_stop_serving_ends_one(tmp_path) -> None:
    value, _, _, newest, older = two_artifacts(tmp_path)
    scenario = value.get_or_create_scenario("math")
    # Serving the active artifact keeps it serving after another activation.
    value.serve("math", newest["artifact_id"])
    value.activate("math", older["artifact_id"])
    assert scenario.current_artifact_ref().release_id == older["release_id"]
    assert scenario.served_release(newest["artifact_id"]).release_id == newest["release_id"]
    rows = {row["artifact_id"]: row for row in value.list_artifacts("math")}
    assert (rows[newest["artifact_id"]]["served"], rows[newest["artifact_id"]]["active"]) == (True, False)
    assert (rows[older["artifact_id"]]["served"], rows[older["artifact_id"]]["active"]) == (False, True)
    # A second serve of a served artifact changes nothing.
    steps = len(scenario.store.history())
    value.serve("math", newest["artifact_id"])
    assert len(scenario.store.history()) == steps

    stopped = value.stop_serving("math", newest["artifact_id"])
    assert (stopped.artifact_id, stopped.release.release_id) == (newest["artifact_id"], newest["release_id"])
    assert [record.operation for record in scenario.store.history()[-3:]] == ["serve", "activate", "stop_serving"]
    with pytest.raises(ArtifactNotFound):
        scenario.served_release(newest["artifact_id"])
    with pytest.raises(ArtifactNotFound):
        value.stop_serving("math", newest["artifact_id"])
    # The active release still answers at its artifact's URL.
    assert scenario.served_release(older["artifact_id"]).release_id == older["release_id"]
    value.close()


@pytest.mark.unit
def test_the_served_artifacts_survive_a_restart(tmp_path) -> None:
    value, _, backend_factory, _, older = two_artifacts(tmp_path)
    value.serve("math", older["artifact_id"])
    value.close()

    component = ImportingComponent(tmp_path / "restarted-export")
    answering_each_release(component)
    restarted = Dispatcher(
        TestPolicyRecipe(**runtime_bindings(component, component.inference)),
        backend_factory,
        local_artifact_dir=tmp_path / "restarted-staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record"),
    )
    scenario = restarted.get_or_create_scenario("math")
    assert scenario.served_release(older["artifact_id"]).release_id == older["release_id"]
    assert [row["artifact_id"] for row in restarted.list_artifacts("math") if row["served"]] == [older["artifact_id"]]
    restarted.close()


@pytest.mark.unit
def test_a_runtime_that_answers_only_from_its_loaded_release_refuses_a_serve(tmp_path) -> None:
    value, _, _ = dispatcher(tmp_path)
    train(value, 1)
    older = value.list_artifacts("math")[-1]
    scenario = value.get_or_create_scenario("math")
    steps = len(scenario.store.history())
    with pytest.raises(ReefError, match="answers only from the release it has loaded"):
        value.serve("math", older["artifact_id"])
    assert len(scenario.store.history()) == steps
    value.close()


@pytest.mark.unit
def test_serve_and_stop_serving_http_api(tmp_path, monkeypatch) -> None:
    async def run() -> None:
        value, _, _, _, older = two_artifacts(tmp_path)
        client = TestClient(TestServer(create_app(value)))
        await client.start_server()
        try:
            body = {"artifact_id": older["artifact_id"]}
            served = await client.post("/reef/scenarios/math/serve", json=body)
            assert served.status == 200
            assert await served.json() == {
                "scenario": "math",
                "component": older["component"],
                "artifact_id": older["artifact_id"],
                "release_id": older["release_id"],
                "content_id": older["content_id"],
            }
            listed = await (await client.get("/reef/scenarios/math/artifacts")).json()
            assert [row["artifact_id"] for row in listed["artifacts"] if row["served"]] == [older["artifact_id"]]
            refused = await client.post(
                "/reef/scenarios/math/stop-serving", json={**body, "component": older["component"]}
            )
            assert refused.status == 400
            stopped = await client.post("/reef/scenarios/math/stop-serving", json=body)
            assert stopped.status == 200
            assert await stopped.json() == {
                "scenario": "math",
                "component": older["component"],
                "artifact_id": older["artifact_id"],
                "release_id": older["release_id"],
            }
            for path in ("serve", "stop-serving"):
                missing = await client.post(f"/reef/scenarios/math/{path}", json={"artifact_id": "missing"})
                assert missing.status == 404
                unknown = await client.post(f"/reef/scenarios/other/{path}", json=body)
                assert unknown.status == 404
                empty = await client.post(f"/reef/scenarios/math/{path}", json={"artifact_id": " "})
                assert empty.status == 400

            monkeypatch.setattr(Trainer, "has_pending_commit", property(lambda self: True))
            waiting = await client.post("/reef/scenarios/math/serve", json=body)
            assert waiting.status == 409
        finally:
            await client.close()
            value.close()

    asyncio.run(run())
