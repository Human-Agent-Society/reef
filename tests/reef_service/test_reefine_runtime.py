"""Serving-head snapshots and formal selection through the real publication path."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support._strategies import resolve_episode_scorer, resolve_proposer

from reef.artifact import InMemoryRepositoryBackend
from reef.core.training_request import TrainingRequest
from reef.dispatcher import Dispatcher
from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.tree.mutations import Mutation
from reef.inference.http import InferenceProxyRuntime
from reef.recipe.reefine import ReefineRecipe
from reef.service.app import create_app
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.cordis_backend.contracts import ServedComposition
from reef.train.reefine.backend import ReefineBackend
from reef.train.types import TrainingBatch

PI_FAKE = """\
#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

prompt = sys.argv[sys.argv.index("-p") + 1]
agent_dir = Path(os.environ["PI_CODING_AGENT_DIR"])
session_dir = Path(os.environ["PI_CODING_AGENT_SESSION_DIR"])
session_dir.mkdir(parents=True, exist_ok=True)
rules_path = agent_dir / "AGENTS.md"
event = {"type": "agent_end", "rules": rules_path.read_text() if rules_path.exists() else ""}
(session_dir / "session.jsonl").write_text(json.dumps(event) + "\\n")
"""

MODEL = ModelBinding(base_url="http://localhost:8000", model="qwen3-8b", api_key="dummy")


def make_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "fake-pi"
    binary.write_text(PI_FAKE)
    binary.chmod(0o755)
    return binary


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
    assert candidate.context.current.release_id == "head-1"


@pytest.mark.parametrize(
    "case", ["success", "request_failure", "protected_regression", "invalid_plan", "harness_limit"]
)
def test_formal_checks_control_publication_and_progress_route(tmp_path, monkeypatch, case):
    recipe = ReefineRecipe.from_environment(
        {},
        runtime=InferenceProxyRuntime(model_path=MODEL.model, base_url=MODEL.base_url, api_key=MODEL.api_key),
        config={
            "evolution": {
                "propose": resolve_proposer(propose),
                "proposer_agent": None,
                "evaluate": resolve_episode_scorer(score),
                "tasks": ["health"],
                "binary": str(make_binary(tmp_path)),
                "proposals_dir": str(tmp_path / "proposals"),
                "evaluation": {"protected_tasks": ["protected task"]},
            }
        },
    )
    replies = [
        {"tasks": [{"messages": ["request task"], "checks": ["requested behavior"]}]},
        {"passed": case != "request_failure", "reason": "request behavior checked"},
        {"passed": True, "reason": "protected behavior checked"},
        {"passed": case != "protected_regression", "reason": "protected behavior checked"},
        {"passed": True, "reason": "change reviewed independently"},
    ]
    if case == "invalid_plan":
        # The plan is asked for once more and stays invalid, so no request episode is judged.
        replies[0:2] = [{"prompt": "", "checks": []}, {"prompt": "", "checks": []}]
    if case == "harness_limit":
        replies[0]["limits"] = [{"requirement": "refuse other tools", "reason": "not possible on this harness"}]
    responses = iter(replies)

    def chat(self, messages, **params):
        return json.dumps(next(responses))

    monkeypatch.setattr(type(MODEL), "chat", chat)
    initial = tmp_path / "initial"
    initial.mkdir()
    factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    dispatcher = Dispatcher(
        replace(recipe, training_mode="manual"),
        factory,
        scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record"),
    )
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
            # A limit holds a passing release for the person's review: it is published but not served.
            assert progress["state"] == {"success": "selected", "harness_limit": "pending"}.get(case, "rejected")
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
