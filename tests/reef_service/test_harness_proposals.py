"""The proposals route and inbox: a client's patch is validated at the route, waits as a file, and the next automatic step evaluates it."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support.harness_service import EchoInference, harness_dispatcher, run_step
from tests.support.harness_steps import harness_recipe

from reef.harness.adapters import get_adapter
from reef.harness.delivery import create_harness_surface
from reef.recipe import Recipe
from reef.service.app import create_app
from reef.surface import Surface
from reef.train.harness_backend.inbox import ProposalInbox

MARKER_RULES = {"rules.md": "marker rules\n"}


def proposal(patch: dict, *, session: str = "3f1c2a9d0b7e", release_id: str = "rel-0", reason: str = "because"):
    return {"patch": patch, "reason": reason, "session": session, "release_id": release_id}


async def post(client: TestClient, body: object, scenario: str = "agents"):
    return await client.post("/reef/harness/proposals", headers={"x-reef-scenario": scenario}, json=body)


def answers(dispatcher, *bodies: object) -> list[tuple[int, dict | str]]:
    """Each body's status and answer, posted in order to one app over ``dispatcher``."""

    async def run() -> list[tuple[int, dict | str]]:
        client = TestClient(TestServer(create_app(dispatcher, inference_handler=EchoInference())))
        await client.start_server()
        try:
            results: list[tuple[int, dict | str]] = []
            for body in bodies:
                response = await post(client, body)
                results.append(
                    (response.status, await response.json() if response.status == 200 else await response.text())
                )
            return results
        finally:
            await client.close()

    return asyncio.run(run())


class FilesOnly(Recipe):
    """A delivery recipe with no training backend: files to serve, nothing to take a proposal."""

    def build_surface(self, scenario: str) -> Surface:
        return create_harness_surface(get_adapter("pi"))


def test_the_route_validates_the_patch_against_the_head_and_stores_it(tmp_path: Path) -> None:
    dispatcher = harness_dispatcher(tmp_path, harness_recipe(tmp_path, lambda context: None))
    scenario = dispatcher.get_or_create_scenario("agents")
    assert scenario is not None
    head = scenario.repository.require_current_artifact().release_id
    refused = {
        "notes.txt": "notes.txt: not a path of the harness layout",
        "rules.md": "the patch changes nothing",
        "config/other.json": "adapter 'pi' declares no config target 'other'",
        "native/tools/t.py": "adapter 'pi' does not render native_tool files",
        "config/primary.json": "config field 'config.apiKey' carries an inline credential",
    }
    texts = {"rules.md": "Answer briefly.\n", "config/primary.json": json.dumps({"apiKey": "abcd"})}
    try:
        results = answers(
            dispatcher,
            proposal(MARKER_RULES),
            *(proposal({path: texts.get(path, "x\n")}) for path in refused),
        )
        (status, admitted), *refusals = results
        assert status == 200
        assert admitted["admitted"] is True and admitted["reason"] is None and admitted["release_id"] == head
        inbox = tmp_path / "proposals" / "agents"
        stored = json.loads((inbox / f"{admitted['proposal_id']}.json").read_text(encoding="utf-8"))
        assert stored["patch"] == MARKER_RULES and stored["reason"] == "because"
        assert stored["session"] == "3f1c2a9d0b7e" and stored["release_id"] == "rel-0"
        assert stored["proposal_id"] == admitted["proposal_id"] and stored["head_release_id"] == head
        assert isinstance(stored["received_at"], float)
        for (status, answer), problem in zip(refusals, refused.values(), strict=True):
            assert status == 200 and answer["admitted"] is False and problem in answer["reason"], answer
            assert answer["release_id"] == head
        assert sorted(path.name for path in inbox.iterdir()) == [f"{admitted['proposal_id']}.json"]
    finally:
        dispatcher.close()


def test_the_inbox_holds_max_pending_proposals_and_refuses_the_rest(tmp_path: Path) -> None:
    dispatcher = harness_dispatcher(tmp_path, harness_recipe(tmp_path, lambda context: None, max_pending_proposals=1))
    dispatcher.get_or_create_scenario("agents")
    try:
        (_, first), (_, second) = answers(dispatcher, proposal(MARKER_RULES), proposal(MARKER_RULES))
        assert first["admitted"] is True
        assert second == {**second, "admitted": False, "reason": "inbox full"}
        assert second["proposal_id"] != first["proposal_id"]
        assert [path.stem for path in (tmp_path / "proposals" / "agents").glob("*.json")] == [first["proposal_id"]]
    finally:
        dispatcher.close()


def test_a_manual_scenario_takes_instructions_only(tmp_path: Path) -> None:
    recipe = harness_recipe(tmp_path, lambda context: None, reads_requests=True, training_mode="manual")
    dispatcher = harness_dispatcher(tmp_path, recipe)
    dispatcher.get_or_create_scenario("agents")
    try:
        [(status, answer)] = answers(dispatcher, proposal(MARKER_RULES))
        assert status == 200
        assert answer["admitted"] is False and answer["reason"] == "manual mode takes instructions only"
        assert not (tmp_path / "proposals" / "agents").exists()
    finally:
        dispatcher.close()


def test_a_malformed_body_is_400_and_a_scenario_that_takes_no_proposals_is_404(tmp_path: Path) -> None:
    dispatcher = harness_dispatcher(tmp_path, harness_recipe(tmp_path, lambda context: None))
    dispatcher.get_or_create_scenario("agents")
    plain = harness_dispatcher(tmp_path / "plain", Recipe())
    files_only = harness_dispatcher(tmp_path / "files", FilesOnly())
    try:
        malformed = [
            ({"patch": {}, "reason": "", "session": "s", "release_id": "r"}, "patch must change at least one path"),
            ({"patch": ["rules.md"], "reason": "", "session": "s", "release_id": "r"}, "must be a JSON object"),
            ({"patch": {"rules.md": 1}, "reason": "", "session": "s", "release_id": "r"}, "must hold text"),
            ({"patch": {"/etc/passwd": "x"}, "reason": "", "session": "s", "release_id": "r"}, "must be relative"),
            ({"patch": MARKER_RULES, "reason": "", "release_id": "r"}, "session must be a string"),
            ([MARKER_RULES], "must be an object"),
        ]
        results = answers(dispatcher, *(body for body, _ in malformed))
        for (status, text), (_, message) in zip(results, malformed, strict=True):
            assert status == 400 and message in text, text

        async def unknown_scenario() -> int:
            client = TestClient(TestServer(create_app(dispatcher)))
            await client.start_server()
            try:
                return (await post(client, proposal(MARKER_RULES), scenario="unknown")).status
            finally:
                await client.close()

        assert asyncio.run(unknown_scenario()) == 404
        assert list((tmp_path / "proposals").rglob("*.json")) == []
        for other, message in ((plain, "carries no harness surface"), (files_only, "takes no proposals")):
            other.get_or_create_scenario("agents")
            [(status, text)] = answers(other, proposal(MARKER_RULES))
            assert status == 404 and message in text
    finally:
        dispatcher.close()
        plain.close()
        files_only.close()


def test_inference_responses_carry_no_release_header_on_a_scenario_that_serves_no_files(tmp_path: Path) -> None:
    plain = harness_dispatcher(tmp_path, Recipe())

    async def run() -> None:
        client = TestClient(TestServer(create_app(plain, inference_handler=EchoInference())))
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


def test_the_next_step_takes_the_oldest_proposal_first_and_the_result_settles_it(tmp_path: Path) -> None:
    """Two proposals validated against one head: the first wins and publishes; the second meets the moved head at
    the step's own validation and is refused; the method is asked only once the inbox is empty."""
    asked: list[int] = []
    dispatcher = harness_dispatcher(tmp_path, harness_recipe(tmp_path, lambda context: asked.append(context.step)))
    scenario = dispatcher.get_or_create_scenario("agents")
    assert scenario is not None
    inbox = tmp_path / "proposals" / "agents"
    try:
        (_, first), (_, second) = answers(
            dispatcher, proposal(MARKER_RULES, session="s1"), proposal(MARKER_RULES, session="s2")
        )
        assert first["admitted"] and second["admitted"] and first["proposal_id"] < second["proposal_id"]

        result = run_step(scenario, "1")
        proposal_metrics = result.metrics["proposal"]
        assert (proposal_metrics["id"], proposal_metrics["session"], proposal_metrics["reason"]) == (
            first["proposal_id"],
            "s1",
            "because",
        )
        assert result.metrics["selected"] is True
        head = scenario.repository.require_current_artifact()
        assert result.artifact is not None and head.release_id != first["release_id"]
        settled = json.loads((inbox / "settled" / f"{first['proposal_id']}.json").read_text(encoding="utf-8"))
        assert settled["session"] == "s1"
        assert settled["result"] == {"step": 1, "selected": True, "reason": result.metrics["selection"]["reason"]}
        # The route validates against the head the step published: the same patch now changes nothing.
        [(_, again)] = answers(dispatcher, proposal(MARKER_RULES))
        assert again["admitted"] is False and again["reason"] == "the patch changes nothing"
        assert again["release_id"] == head.release_id

        result = run_step(scenario, "2")
        assert result.metrics["proposal"]["id"] == second["proposal_id"]
        assert result.metrics["skipped"] == "the patch changes nothing" and "selected" not in result.metrics
        refused = json.loads((inbox / "refused" / f"{second['proposal_id']}.json").read_text(encoding="utf-8"))
        assert refused["session"] == "s2" and refused["refused"] == "the patch changes nothing"
        assert asked == []

        result = run_step(scenario, "3")
        assert asked == [3] and result.metrics["skipped"] == "no proposal" and "proposal" not in result.metrics
        assert sorted(path.name for path in inbox.iterdir()) == ["claimed", "refused", "settled"]
        assert list((inbox / "claimed").iterdir()) == [] and list(inbox.glob("*.json")) == []

        # The commit history names the proposal on the steps that took one.
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
    assert inbox.submit(newer, proposal(MARKER_RULES, session="b")) is None
    assert inbox.submit(older, proposal(MARKER_RULES, session="a")) is None
    assert inbox.submit(ProposalInbox.new_id(), proposal(MARKER_RULES)) == "inbox full"
    assert inbox.pending() == [older, newer]
    claimed = inbox.claim()
    assert claimed is not None and claimed.id == older and claimed.session == "a"
    assert claimed.patch.to_json() == MARKER_RULES and claimed.reason == "because" and claimed.release_id == "rel-0"
    assert inbox.pending() == [newer] and (inbox.directory / "claimed" / f"{older}.json").is_file()
    inbox.settle(older, {"step": 1, "selected": False, "reason": "lost"})
    assert json.loads((inbox.directory / "settled" / f"{older}.json").read_text(encoding="utf-8"))["result"] == {
        "step": 1,
        "selected": False,
        "reason": "lost",
    }
    second = inbox.claim()
    assert second is not None and second.id == newer and inbox.claim() is None
    inbox.refuse(newer, "the patch changes nothing")
    refused = json.loads((inbox.directory / "refused" / f"{newer}.json").read_text(encoding="utf-8"))
    assert refused["refused"] == "the patch changes nothing"
    assert sorted(path.name for path in inbox.directory.iterdir()) == ["claimed", "refused", "settled"]
    assert list((inbox.directory / "claimed").iterdir()) == []
    # A claimed file an operator removed by hand still gets its result filed, and no staging file stays behind.
    third = ProposalInbox.new_id()
    assert inbox.submit(third, proposal(MARKER_RULES, session="c")) is None
    claimed = inbox.claim()
    assert claimed is not None and claimed.id == third
    (inbox.directory / "claimed" / f"{third}.json").unlink()
    inbox.settle(third, {"step": 2, "selected": False, "reason": "lost"})
    assert json.loads((inbox.directory / "settled" / f"{third}.json").read_text(encoding="utf-8")) == {
        "proposal_id": third,
        "result": {"step": 2, "selected": False, "reason": "lost"},
    }
    assert list(inbox.directory.rglob(".*.part")) == []


def test_an_aborted_step_files_its_claimed_proposal_as_refused(tmp_path: Path, monkeypatch) -> None:
    dispatcher = harness_dispatcher(tmp_path, harness_recipe(tmp_path, lambda context: None))
    scenario = dispatcher.get_or_create_scenario("agents")
    assert scenario is not None

    def die(self, prepared, decision):
        raise RuntimeError("evaluator died")

    try:
        [(_, admitted)] = answers(dispatcher, proposal(MARKER_RULES))
        monkeypatch.setattr(type(scenario.trainer.candidate_backend), "settle_step", die)
        with pytest.raises(RuntimeError, match="evaluator died"):
            run_step(scenario, "1")
        inbox = tmp_path / "proposals" / "agents"
        refused = json.loads((inbox / "refused" / f"{admitted['proposal_id']}.json").read_text(encoding="utf-8"))
        assert refused["refused"] == "step aborted before a result" and refused["session"] == "3f1c2a9d0b7e"
        assert list((inbox / "claimed").iterdir()) == [] and list(inbox.glob("*.json")) == []
    finally:
        dispatcher.close()
