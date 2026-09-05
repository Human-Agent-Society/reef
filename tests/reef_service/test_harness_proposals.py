"""The proposals route and inbox: an agent's proposed tree change is admitted at the route, waits as a file, and the next evolve step takes it into the gate."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from reef_service.test_harness_recipe import SEED_MODELS, SEED_SETTINGS, _report_once, evaluate, make_binary, runtime

from reef.artifact import InMemoryRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.runtime.inference import InferenceBackend
from reef.service.app import create_app
from reef.surface import Surface, create_harness_surface
from reef.train.cordis_backend import CordisRecipe
from reef.train.cordis_backend.proposals import ProposalInbox
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer

CREATE_RULES = {"op": "create", "id": "r1", "options": {"name": "rules", "config": {"text": "marker rules"}}}


def _recipe(tmp_path: Path, propose, *, max_pending: int = 8) -> CordisRecipe:
    return CordisRecipe(
        resolve_proposer(propose),
        resolve_episode_scorer(evaluate),
        ("task one",),
        binary=str(make_binary(tmp_path)),
        seed=(SEED_MODELS, SEED_SETTINGS),
        runtime=runtime(),
        proposals_dir=str(tmp_path / "inbox"),
        max_pending_proposals=max_pending,
    )


def _dispatcher(tmp_path: Path, recipe: Recipe) -> Dispatcher:
    initial = tmp_path / "initial"
    initial.mkdir(parents=True, exist_ok=True)
    factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    return Dispatcher(recipe, factory, agent_record_dir=tmp_path / "agent-record")


def _proposal(*mutations: dict, session: str = "3f1c2a9d0b7e", release_id: str = "rel-0", reason: str = "because"):
    return {"mutations": list(mutations), "reason": reason, "session": session, "release_id": release_id}


async def _post(client: TestClient, body: dict, scenario: str = "agents"):
    return await client.post("/reef/harness/proposals", headers={"x-reef-scenario": scenario}, json=body)


class _EchoBackend(InferenceBackend):
    async def inference(self, artifact, path, payload):
        del artifact, path, payload
        return {"choices": [{"message": {"content": "ok"}}]}


class _FilesOnly(Recipe):
    """A pull delivery recipe with no training backend: files to serve, nothing to take a proposal."""

    def build_surface(self, scenario: str) -> Surface:
        return create_harness_surface()


def test_the_route_admits_against_the_head_and_stores_the_proposal(tmp_path: Path) -> None:
    dispatcher = _dispatcher(tmp_path, _recipe(tmp_path, lambda n, s, m: None))

    async def run() -> None:
        scenario = dispatcher.get_or_create_scenario("agents")
        assert scenario is not None
        head = scenario.repository.require_current_artifact().release_id
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            response = await _post(client, _proposal(CREATE_RULES))
            assert response.status == 200
            answer = await response.json()
            assert answer["admitted"] is True and answer["reason"] is None and answer["release_id"] == head
            inbox = tmp_path / "inbox" / "agents"
            stored = json.loads((inbox / f"{answer['proposal_id']}.json").read_text(encoding="utf-8"))
            assert stored["mutations"] == [CREATE_RULES] and stored["reason"] == "because"
            assert stored["session"] == "3f1c2a9d0b7e" and stored["release_id"] == "rel-0"
            assert stored["proposal_id"] == answer["proposal_id"] and stored["head_release_id"] == head
            assert isinstance(stored["received_at"], float)
            assert sorted(path.name for path in inbox.iterdir()) == [f"{answer['proposal_id']}.json"]

            # The rules every mutation meets, answered as the refusal, with nothing stored.
            tool = {"name": "t", "description": "d", "parameters": {}, "code": "def run(a, w):\n    return 1\n"}
            cases = [
                ({"op": "update", "id": "x", "options": {"config": {"text": "y"}}}, "cannot resolve entry x"),
                (
                    {"op": "update", "id": "models", "options": {"name": "rules", "config": {"text": "y"}}},
                    "cannot change the entry's kind from 'config' to 'rules'",
                ),
                (
                    {"op": "create", "id": "t", "options": {"name": "native_tool", "config": tool}},
                    "adapter 'pi' does not render native_tool nodes",
                ),
                ({"op": "rename", "id": "r1"}, "mutation op must be one of"),
                ({"op": "create", "id": "s", "options": {"name": "skill", "config": {"name": "s"}}}, "'text'"),
                ({"op": "remove", "id": "r1", "options": {}}, "remove mutation takes no options"),
            ]
            for mutation, rule in cases:
                answer = await (await _post(client, _proposal(mutation))).json()
                assert answer["admitted"] is False and rule in answer["reason"], answer
                assert answer["release_id"] == head
            assert len(list(inbox.glob("*.json"))) == 1
        finally:
            await client.close()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_the_inbox_holds_max_pending_proposals_and_refuses_the_rest(tmp_path: Path) -> None:
    dispatcher = _dispatcher(tmp_path, _recipe(tmp_path, lambda n, s, m: None, max_pending=1))

    async def run() -> None:
        dispatcher.get_or_create_scenario("agents")
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            first = await (await _post(client, _proposal(CREATE_RULES))).json()
            second = await (await _post(client, _proposal(CREATE_RULES))).json()
            assert first["admitted"] is True
            assert second == {**second, "admitted": False, "reason": "inbox full"}
            assert second["proposal_id"] != first["proposal_id"]
            assert [path.stem for path in (tmp_path / "inbox" / "agents").glob("*.json")] == [first["proposal_id"]]
        finally:
            await client.close()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_a_malformed_body_is_400_and_a_scenario_that_takes_no_proposals_is_404(tmp_path: Path) -> None:
    dispatcher = _dispatcher(tmp_path, _recipe(tmp_path, lambda n, s, m: None))
    plain = _dispatcher(tmp_path / "plain", Recipe())
    files_only = _dispatcher(tmp_path / "files", _FilesOnly())

    async def run() -> None:
        dispatcher.get_or_create_scenario("agents")
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            for body, message in (
                ({"mutations": [], "reason": "", "session": "s", "release_id": "r"}, "non-empty list"),
                ({"mutations": [CREATE_RULES], "reason": "", "release_id": "r"}, "session must be a string"),
                (
                    {"mutations": [{"op": "create"}], "reason": "", "session": "s", "release_id": "r"},
                    "string op and id",
                ),
                (
                    {"mutations": [{**CREATE_RULES, "options": 1}], "reason": "", "session": "s", "release_id": "r"},
                    "options must be an object",
                ),
                ([CREATE_RULES], "must be an object"),
            ):
                response = await _post(client, body)
                assert response.status == 400 and message in await response.text()
            response = await _post(client, _proposal(CREATE_RULES), scenario="unknown")
            assert response.status == 404
            assert list((tmp_path / "inbox").rglob("*.json")) == []
        finally:
            await client.close()
        for other, message in ((plain, "carries no harness surface"), (files_only, "takes no proposals")):
            other.get_or_create_scenario("agents")
            client = TestClient(TestServer(create_app(other, inference_backend=_EchoBackend())))
            await client.start_server()
            try:
                response = await _post(client, _proposal(CREATE_RULES))
                assert response.status == 404 and message in await response.text()
            finally:
                await client.close()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
        plain.close()
        files_only.close()


def test_inference_responses_carry_no_release_header_on_a_scenario_that_serves_no_files(tmp_path: Path) -> None:
    plain = _dispatcher(tmp_path, Recipe())

    async def run() -> None:
        client = TestClient(TestServer(create_app(plain, inference_backend=_EchoBackend())))
        await client.start_server()
        try:
            for stream in (False, True):
                response = await client.post(
                    "/v1/chat/completions",
                    headers={"x-reef-scenario": "records"},
                    json={"messages": [{"role": "user", "content": "hi"}], "stream": stream},
                )
                assert response.status == 200
                await response.read()
                assert response.headers["x-reef-agent-record-id"] and "x-reef-release-id" not in response.headers
        finally:
            await client.close()

    try:
        asyncio.run(run())
    finally:
        plain.close()


def test_the_next_step_takes_the_oldest_proposal_first_and_the_verdict_settles_it(tmp_path: Path) -> None:
    """Two proposals admitted against one head: the first wins the gate and publishes; the second meets the
    moved head at the step's own admission and is refused; the method is asked only once the inbox is empty."""
    asked: list[int] = []
    dispatcher = _dispatcher(tmp_path, _recipe(tmp_path, lambda nodes, s, m: asked.append(len(nodes)) or None))
    scenario = dispatcher.get_or_create_scenario("agents")
    assert scenario is not None

    async def run() -> tuple[dict, dict]:
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            first = await (await _post(client, _proposal(CREATE_RULES, session="s1"))).json()
            second = await (await _post(client, _proposal(CREATE_RULES, session="s2"))).json()
        finally:
            await client.close()
        return first, second

    inbox = tmp_path / "inbox" / "agents"
    try:
        first, second = asyncio.run(run())
        assert first["admitted"] and second["admitted"] and first["proposal_id"] < second["proposal_id"]

        _report_once(scenario, "agents", "1")
        result = scenario.prepare_training_step()
        assert result is not None
        scenario.commit(result)
        assert result.metrics["proposal"] == {"id": first["proposal_id"], "session": "s1", "release_id": "rel-0"}
        assert result.metrics["published"] is True
        assert [entry["id"] for entry in result.state["entries"]] == ["models", "settings", "r1"]
        settled = json.loads((inbox / "settled" / f"{first['proposal_id']}.json").read_text(encoding="utf-8"))
        assert settled["session"] == "s1"
        assert settled["verdict"] == {"step": 1, "selected": True, "reason": result.metrics["selection"]["reason"]}

        _report_once(scenario, "agents", "2")
        result = scenario.prepare_training_step()
        assert result is not None
        scenario.commit(result)
        assert result.metrics["proposal"]["id"] == second["proposal_id"]
        assert result.metrics["skipped"] == "entry 'r1' already exists" and "published" not in result.metrics
        refused = json.loads((inbox / "refused" / f"{second['proposal_id']}.json").read_text(encoding="utf-8"))
        assert refused["session"] == "s2" and refused["refused"] == "entry 'r1' already exists"
        assert asked == []

        _report_once(scenario, "agents", "3")
        result = scenario.prepare_training_step()
        assert result is not None
        scenario.commit(result)
        assert asked == [3] and result.metrics["skipped"] == "no proposal" and "proposal" not in result.metrics
        assert sorted(path.name for path in inbox.iterdir()) == ["claimed", "refused", "settled"]
        assert list((inbox / "claimed").iterdir()) == [] and list(inbox.glob("*.json")) == []

        # The commit log names the proposal on the steps that took one.
        by_step = {
            row["metrics"]["steps"]: row["metrics"] for row in scenario.releases() if row["operation"] == "training"
        }
        assert by_step[1]["proposal"]["session"] == "s1" and by_step[2]["proposal"]["session"] == "s2"
        assert "proposal" not in by_step[3]
    finally:
        dispatcher.close()


def test_the_inbox_claims_oldest_first_and_keeps_a_claimed_file_out_of_the_queue(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_pending_proposals must be an integer of at least 1"):
        ProposalInbox(tmp_path, 0)
    inbox = ProposalInbox(tmp_path / "inbox", 2)
    assert inbox.pending() == [] and inbox.claim() is None and not inbox.directory.exists()
    older, newer = ProposalInbox.new_id(), ProposalInbox.new_id()
    assert older < newer
    assert inbox.submit(newer, _proposal(CREATE_RULES, session="b")) is None
    assert inbox.submit(older, _proposal(CREATE_RULES, session="a")) is None
    assert inbox.submit(ProposalInbox.new_id(), _proposal(CREATE_RULES)) == "inbox full"
    assert inbox.pending() == [older, newer]
    claimed = inbox.claim()
    assert claimed is not None and claimed.id == older and claimed.session == "a"
    assert claimed.mutations == (CREATE_RULES,) and claimed.reason == "because" and claimed.release_id == "rel-0"
    assert inbox.pending() == [newer] and (inbox.directory / "claimed" / f"{older}.json").is_file()
    inbox.settle(older, {"step": 1, "selected": False, "reason": "lost"})
    assert json.loads((inbox.directory / "settled" / f"{older}.json").read_text(encoding="utf-8"))["verdict"] == {
        "step": 1,
        "selected": False,
        "reason": "lost",
    }
    second = inbox.claim()
    assert second is not None and second.id == newer and inbox.claim() is None
    inbox.refuse(newer, "entry 'r1' already exists")
    assert json.loads((inbox.directory / "refused" / f"{newer}.json").read_text(encoding="utf-8"))["refused"] == (
        "entry 'r1' already exists"
    )
    assert sorted(path.name for path in inbox.directory.iterdir()) == ["claimed", "refused", "settled"]
    assert list((inbox.directory / "claimed").iterdir()) == []
