"""The harness service (``reef serve --component harness``): it serves a harness client, and control commits imports.

Control and the harness service run as two dispatchers over one record store, one artifact repository and one
coordination store, as two processes on one host do.
"""

from __future__ import annotations

import asyncio
import dataclasses
from contextlib import closing
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support.harness_service import harness_repository
from tests.support.harness_steps import PI_SEED, harness_recipe

import reef.service.harness_imports as harness_imports
from reef.dispatcher import Dispatcher
from reef.execution.workers import SERVICE_COMPONENTS
from reef.service.app import create_harness_app
from reef.service.harness_imports import HarnessSyncWorker
from reef.service.request_service import RequestService
from reef.storage.sql_coordination import open_coordination
from reef.storage.sqlite import SQLiteScenarioStorage

HEADERS = {"x-reef-scenario": "demo"}
SKILL = "---\nname: fix\ndescription: Fix the failing test first.\n---\n\nRun the tests, then fix.\n"
MINE = {"plugin/rules.md": "Mine.\n", "plugin/skills/fix/SKILL.md": SKILL}
INFERENCE_URL = "http://inference.example:8902"


def open_service(tmp_path: Path, repository, components: frozenset[str]) -> Dispatcher:
    """One process's dispatcher over the shared stores and ``repository``; each process builds its own recipe."""
    recipe = harness_recipe(tmp_path, lambda workspace, services: None)
    return Dispatcher(
        dataclasses.replace(recipe),
        repository,
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        coordination=open_coordination("sqlite", agent_record_dir=str(tmp_path), record_database_url=None),
        components=components,
    )


@pytest.mark.unit
def test_the_harness_service_serves_the_tree_and_control_commits_its_imports(tmp_path: Path) -> None:
    repository = harness_repository(tmp_path, PI_SEED.files)
    control = open_service(tmp_path, repository, frozenset(SERVICE_COMPONENTS) - {"harness"})
    control.get_or_create_scenario("demo")
    # The assembly of a control process without the harness service starts this worker.
    control.workers.add(HarnessSyncWorker(control.queue, RequestService(control), worker_id="harness-sync-test"))
    harness = open_service(tmp_path, repository, frozenset({"harness"}))

    async def run() -> None:
        app = create_harness_app(harness, inference_url=INFERENCE_URL)
        async with TestClient(TestServer(app)) as client:
            served = await (await client.get("/reef/harness", params={"form": "source"}, headers=HEADERS)).json()
            assert served["files"] == dict(PI_SEED.files)
            base = served["release_id"]
            assert base == control.get_or_create_scenario("demo").current_artifact_ref().release_id
            # A client's layer sends its model calls to the inference service that the deployment names.
            layer = await client.get(
                "/reef/harness",
                params={"form": "client", "service_url": "http://harness.example:8907", "root": "/home/me/.reef"},
                headers=HEADERS,
            )
            text = "".join((await layer.json())["files"].values())
            assert f"{INFERENCE_URL}/reef/scenarios/demo" in text and "harness.example" not in text
            # An import goes to control, which commits it as the active release.
            response = await client.post(
                "/reef/harness/import", headers=HEADERS, json={"files": MINE, "base_release_id": base}
            )
            assert response.status == 200, await response.text()
            imported = await response.json()
            assert imported["changed"] is True and imported["parent_release_id"] == base
            current = control.get_or_create_scenario("demo").current_artifact_ref()
            assert current.release_id == imported["release_id"] == response.headers["x-reef-release-id"]
            # The harness service reads the release that control committed.
            source = await (await client.get("/reef/harness", params={"form": "source"}, headers=HEADERS)).json()
            assert source["release_id"] == imported["release_id"] and source["files"]["rules.md"] == "Mine.\n"
            releases = await (await client.get("/reef/harness/releases", headers=HEADERS)).json()
            assert releases["releases"][-1]["release_id"] == imported["release_id"]
            # Control's refusals keep their status: a file that both sides changed, an import without its base,
            # and an unknown scenario.
            conflicting = {**MINE, "plugin/rules.md": "Theirs.\n"}
            refusals = (
                (HEADERS, {"files": conflicting, "base_release_id": base}, 409),
                (HEADERS, {"files": MINE}, 400),
                ({"x-reef-scenario": "missing"}, {"files": MINE, "base_release_id": base}, 404),
            )
            for headers, body, status in refusals:
                refused = await client.post("/reef/harness/import", headers=headers, json=body)
                assert refused.status == status, await refused.text()
            # The pages and the writes stay with control.
            assert (await client.get("/reef/harness/releases/0/page", headers=HEADERS)).status == 404
            assert (await client.get("/reef/status")).status == 404

    try:
        asyncio.run(run())
    finally:
        control.close()


@pytest.mark.unit
def test_an_import_that_control_does_not_take_answers_503_and_stays_queued(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(harness_imports, "SYNC_WAIT_SECONDS", 0.2)
    repository = harness_repository(tmp_path, PI_SEED.files)
    control = open_service(tmp_path, repository, frozenset(SERVICE_COMPONENTS) - {"harness"})
    control.get_or_create_scenario("demo")
    control.close()
    harness = open_service(tmp_path, repository, frozenset({"harness"}))

    async def run() -> None:
        async with TestClient(TestServer(create_harness_app(harness))) as client:
            source = await (await client.get("/reef/harness", params={"form": "source"}, headers=HEADERS)).json()
            body = {"files": MINE, "base_release_id": source["release_id"]}
            response = await client.post("/reef/harness/import", headers=HEADERS, json=body)
            assert response.status == 503 and "send it again" in await response.text()

    asyncio.run(run())
    # The app closed the harness service's dispatcher; the shared queue still holds the import for control.
    with closing(open_coordination("sqlite", agent_record_dir=str(tmp_path), record_database_url=None)) as shared:
        (item,) = shared.work_queue.read_changes(harness_imports.HARNESS_SYNC_QUEUE, 0)
    assert item.state == "queued" and item.scenario == "demo"


@pytest.mark.unit
def test_the_harness_app_runs_the_harness_service_without_control(tmp_path: Path) -> None:
    control = open_service(tmp_path, harness_repository(tmp_path, PI_SEED.files), frozenset(SERVICE_COMPONENTS))
    try:
        with pytest.raises(ValueError, match="without control"):
            create_harness_app(control)
    finally:
        control.close()
