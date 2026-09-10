"""Read-only dashboard contract: retirement is not committed participation."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from aiohttp.test_utils import TestClient, TestServer

from reef.core import AgentRecord, RequestType
from reef.dispatcher import build_default_dispatcher
from reef.scenario.commit_log import CommitLog, CommitRecord
from reef.service.app import create_app


def test_learning_reads_preserve_compacted_bodies_and_isolate_scenarios(tmp_path):
    async def run():
        dispatcher = build_default_dispatcher()
        scenario = dispatcher.get_or_create_scenario("one")
        other = dispatcher.get_or_create_scenario("two")
        for name in ("consumed", "retired", "waiting"):
            scenario.records.append(
                AgentRecord.create(
                    scenario="one",
                    request_type=RequestType.INFERENCE,
                    agent_record_id=name,
                    payload={"messages": [{"role": "user", "content": "test trace"}]},
                )
            )
        other.records.append(
            AgentRecord.create(
                scenario="two",
                request_type=RequestType.INFERENCE,
                agent_record_id="other",
                payload={"secret": "other scenario"},
            )
        )
        log = CommitLog(tmp_path / "commits.jsonl")
        scenario._commit_protocol._commit_log = log
        commit = CommitRecord(
            scenario="one",
            step=1,
            artifact_ref=scenario.current_artifact_ref(),
            checkpoint=False,
            algorithm_state=None,
            high_water_sequence=3,
            high_water_offset=3,
            consumed_ids=frozenset({"consumed"}),
            compacted_ids=frozenset({"consumed", "retired"}),
            metrics={"selected": False, "selection": {"candidate_id": "candidate-one", "reason": "regressed"}},
        )
        log.append(commit)
        log.append(replace(commit, step=2, consumed_ids=frozenset(), compacted_ids=frozenset({"waiting"})))
        scenario.records.compact("one", frozenset({"consumed", "retired"}))
        before = scenario.records.count("one")
        client = TestClient(TestServer(create_app(dispatcher, tokens="inspection-test-token")))
        await client.start_server()
        headers = {"authorization": "Bearer inspection-test-token"}
        try:
            response = await client.get("/reef/scenarios/one/learning")
            assert response.status == 401
            response = await client.get("/reef/scenarios/one/learning?limit=2", headers=headers)
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store"
            page = await response.json()
            assert len(page["records"]) == 2
            assert [r["learning_state"] for r in page["records"]] == ["consumed", "unknown"]
            assert page["records"][0]["learning_steps"][0]["candidate_id"] == "candidate-one"
            assert page["records"][0]["learning_steps"][0]["selected"] is False
            assert "payload" not in page["records"][0]
            response = await client.get(
                f'/reef/scenarios/one/learning?after_sequence={page["next_after_sequence"]}', headers=headers
            )
            tail = await response.json()
            assert [r["agent_record_id"] for r in tail["records"]] == ["waiting"]
            assert tail["next_after_sequence"] is None
            assert tail["records"][0]["learning_state"] == "awaiting"
            response = await client.get("/reef/scenarios/one/records/consumed", headers=headers)
            assert response.status == 200
            assert (await response.json())["payload"]["messages"][0]["content"] == "test trace"
            for route in (
                "/reef/scenarios/one/records/other",
                "/reef/scenarios/two/records/consumed",
                "/reef/scenarios/missing/learning",
            ):
                assert (await client.get(route, headers=headers)).status == 404
            for query in ("limit=0", "limit=101", "after_sequence=-1", "after_sequence=oops"):
                assert (await client.get(f"/reef/scenarios/one/learning?{query}", headers=headers)).status == 400
            scenario.records.purge_compacted("one", before=10**12)
            assert (await client.get("/reef/scenarios/one/records/consumed", headers=headers)).status == 404
            assert scenario.records.count("one") == before
        finally:
            await client.close()
            dispatcher.close()

    asyncio.run(run())
