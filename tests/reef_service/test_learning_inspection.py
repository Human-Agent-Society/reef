"""Raw history contracts preserve stored facts without dashboard decisions."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from aiohttp.test_utils import TestClient, TestServer

from reef.core import AgentRecord, RequestType
from reef.dispatcher import build_default_dispatcher
from reef.service.app import create_app
from reef.storage.commit_log import CommitLogScenarioStore
from reef.storage.commits import CommitRecord
from reef.storage.sqlite import SQLiteScenarioStorage


def test_history_reads_preserve_compacted_bodies_and_isolate_scenarios(tmp_path):
    async def run():
        dispatcher = build_default_dispatcher(
            agent_record_dir=tmp_path / "records",
            scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        )
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
        assert isinstance(scenario.store, CommitLogScenarioStore)
        log = scenario.store.commit_log
        assert log is not None
        commit = CommitRecord(
            scenario="one",
            step=1,
            artifact_ref=scenario.current_artifact_ref(),
            checkpoint=False,
            algorithm_state={"private_state": "not part of the public history"},
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
            response = await client.get("/reef/scenarios/one/records")
            assert response.status == 401
            response = await client.get("/reef/scenarios/one/records?limit=2", headers=headers)
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store"
            page = await response.json()
            assert len(page["records"]) == 2
            assert all(
                "learning_state" not in r and "reason" not in r and "learning_steps" not in r for r in page["records"]
            )
            assert all(r["compacted_at"] is not None for r in page["records"])
            contract = await (await client.get("/reef/scenarios/one/contract", headers=headers)).json()
            assert "historical_decisions_available" not in contract
            assert "training_mode" in contract
            commits = await (await client.get("/reef/scenarios/one/commits?limit=1", headers=headers)).json()
            assert commits["commits"][0]["consumed_ids"] == ["consumed"]
            assert commits["commits"][0]["metrics"] == commit.metrics
            assert "algorithm_state" not in commits["commits"][0]
            assert commits["next_after_step"] == 1
            tail_commits = await (
                await client.get("/reef/scenarios/one/commits?after_step=1&limit=1", headers=headers)
            ).json()
            assert tail_commits["commits"][0]["step"] == 2
            assert tail_commits["next_after_step"] is None
            for record_id, expected in (("consumed", [1]), ("retired", []), ("waiting", []), ("other", [])):
                filtered = await (
                    await client.get(f"/reef/scenarios/one/commits?record_id={record_id}", headers=headers)
                ).json()
                assert [c["step"] for c in filtered["commits"]] == expected
            assert (await client.get("/reef/scenarios/one/commits")).status == 401
            assert (await client.get("/reef/scenarios/missing/commits", headers=headers)).status == 404
            assert (await (await client.get("/reef/scenarios/two/commits", headers=headers)).json())["commits"] == []
            for query in (
                "limit=0",
                "limit=101",
                "after_step=-1",
                "after_step=oops",
                "record_id=",
                "&".join(["record_id=x"] * 101),
            ):
                assert (await client.get(f"/reef/scenarios/one/commits?{query}", headers=headers)).status == 400
            assert "payload" not in page["records"][0]
            response = await client.get(
                f'/reef/scenarios/one/records?after_sequence={page["next_after_sequence"]}', headers=headers
            )
            tail = await response.json()
            assert [r["agent_record_id"] for r in tail["records"]] == ["waiting"]
            assert tail["next_after_sequence"] is None
            assert tail["records"][0]["compacted_at"] is None
            response = await client.get("/reef/scenarios/one/records/consumed", headers=headers)
            assert response.status == 200
            assert (await response.json())["payload"]["messages"][0]["content"] == "test trace"
            for route in (
                "/reef/scenarios/one/records/other",
                "/reef/scenarios/two/records/consumed",
                "/reef/scenarios/missing/records",
            ):
                assert (await client.get(route, headers=headers)).status == 404
            for query in ("limit=0", "limit=101", "after_sequence=-1", "after_sequence=oops"):
                assert (await client.get(f"/reef/scenarios/one/records?{query}", headers=headers)).status == 400
            scenario.records.purge_compacted("one", before=10**12)
            assert (await client.get("/reef/scenarios/one/records/consumed", headers=headers)).status == 404
            assert scenario.records.count("one") == before
            durable = await (
                await client.get("/reef/scenarios/one/commits?record_id=consumed", headers=headers)
            ).json()
            assert durable["commits"][0]["consumed_ids"] == ["consumed"]
        finally:
            await client.close()
            dispatcher.close()

    asyncio.run(run())
