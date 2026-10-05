"""Serving-head snapshots and formal selection through the real publication path."""

import asyncio
import json
from dataclasses import replace

import pytest
from aiohttp.test_utils import TestClient, TestServer
from reef_service.test_harness_proposals import _dispatcher
from reef_service.test_harness_recipe import MODEL, make_binary, runtime

from reef.core.training_request import TrainingRequest
from reef.harness.adapters import get_adapter
from reef.harness.tree.mutations import Mutation
from reef.recipe.reefine import ReefineRecipe
from reef.service.app import create_app
from reef.train.cordis_backend.contracts import ServedComposition
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer
from reef.train.reefine.backend import ReefineBackend
from reef.train.types import TrainingBatch


def propose(nodes, samples, model, *, requests=(), **kwargs):
    return Mutation("create", "workflow", {"name": "rules", "config": {"text": "Use the requested workflow."}})


def score(task, result):
    return 1.0


def test_pending_algorithm_entries_are_not_the_current_serving_tree(tmp_path):
    backend = ReefineBackend(
        descriptor=get_adapter("pi"),
        propose=resolve_proposer(propose),
        score_episode=resolve_episode_scorer(score),
        tasks=("health",),
        models=MODEL,
        binary=str(make_binary(tmp_path)),
    )
    serving = {"id": "served", "name": "rules", "config": {"text": "Published rules."}}
    pending = {"id": "pending", "name": "rules", "config": {"text": "Unpublished rules."}}
    backend.set_served_composition(ServedComposition("head-1", (serving,)))
    request = TrainingRequest(
        "use a workflow", "session", "head-1", "request-1", requires=({"name": "curl", "kind": "binary"},)
    )
    prepared = backend.prepare_step(TrainingBatch("batch", (), request=request), {"entries": [pending]}, 1)
    candidate = prepared.candidate
    assert candidate.current_entries == (serving,)
    assert candidate.requires == ({"name": "curl", "kind": "binary"},)
    assert "Unpublished" not in json.dumps(candidate.candidate_files)
    new_entry = {"id": "other", "name": "rules", "config": {"text": "Newly published rules."}}
    backend.set_served_composition(ServedComposition("head-2", (serving, new_entry)))
    refreshed = backend.prepare_reevaluation(prepared).candidate
    assert refreshed.context.current.release_id == "head-2"
    assert refreshed.current_entries == (serving, new_entry)
    assert "Newly published" in json.dumps(refreshed.current_files)
    assert any(entry["id"] == "workflow" for entry in refreshed.candidate_entries)
    assert candidate.context.current.release_id == "head-1"


@pytest.mark.parametrize("case", ["success", "request_failure", "protected_regression", "invalid_plan"])
def test_formal_checks_control_publication_and_progress_route(tmp_path, monkeypatch, case):
    recipe = ReefineRecipe.from_environment(
        {},
        runtime=runtime(),
        config={
            "evolution": {
                "propose": propose,
                "proposer_agent": None,
                "evaluate": score,
                "tasks": ["health"],
                "binary": str(make_binary(tmp_path)),
                "proposals_dir": str(tmp_path / "proposals"),
                "evaluation": {"protected_tasks": ["protected task"]},
            }
        },
    )
    replies = [
        {"prompt": "request task", "checks": ["requested behavior"]},
        {"passed": case != "request_failure", "reason": "request behavior checked"},
        {"passed": True, "reason": "protected behavior checked"},
        {"passed": case != "protected_regression", "reason": "protected behavior checked"},
        {"passed": True, "reason": "change reviewed independently"},
    ]
    if case == "invalid_plan":
        replies[0] = {"prompt": "", "checks": []}
        replies.pop(1)
    responses = iter(replies)

    def chat(self, messages, **params):
        return json.dumps(next(responses))

    monkeypatch.setattr(type(MODEL), "chat", chat)
    dispatcher = _dispatcher(tmp_path, replace(recipe, training_mode="manual"))
    scenario = dispatcher.get_or_create_scenario("agents")
    head = scenario.current_artifact_ref().release_id
    headers = {"x-reef-scenario": "agents"}

    async def run():
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            response = await client.post(
                "/reef/train",
                headers=headers,
                json={
                    "text": "use a workflow",
                    "session": "session",
                    "release_id": head,
                },
            )
            assert response.status == 200
            request_id = (await response.json())["agent_record_id"]
            for attempt in range(200):
                progress = await (
                    await client.get(f"/reef/harness/requests/{request_id}/progress", headers=headers)
                ).json()
                if progress["settled"]:
                    break
                if attempt == 199:
                    pytest.fail("request did not settle within the test deadline")
                await asyncio.sleep(0.02)
            assert progress["settled"]
            assert progress["state"] == ("selected" if case == "success" else "rejected")
            report = progress["evaluation"]
            assert report["current_release_id"] == head and report["request_id"] == request_id
            assert progress["checks"] == report["checks"]
            assert (scenario.current_artifact_ref().release_id != head) is (case == "success")
            page = await (
                await client.get(f"/reef/harness/requests/{request_id}/page", params={"scenario": "agents"})
            ).text()
            assert "Independent evaluation" in page and "request-behavior" in page
        finally:
            await client.close()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
