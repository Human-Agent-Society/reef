"""Harness commands use the native training route and its mode switch end to end."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from reef_service.test_harness_proposals import _dispatcher, _recipe
from reef_service.test_harness_wrapper import _ask_tree, _write_spool_entry

from reef.harness.client.wrapper import harness
from reef.service.app import create_app
from reef.train.cordis_backend import Mutation


@pytest.mark.parametrize("has_receipts", [False, True])
def test_harness_command_switches_to_manual_and_commits_native_request(tmp_path, monkeypatch, capsys, has_receipts):
    seen = []

    def propose(nodes, samples, models, *, requests=()):
        seen.append((samples, requests))
        return Mutation("create", "requested-rules", {"name": "rules", "config": {"text": "marker rules"}})

    # A large auto batch must not keep an explicit manual request waiting.
    dispatcher = _dispatcher(tmp_path, replace(_recipe(tmp_path, propose), batch_size=100))
    scenario = dispatcher.get_or_create_scenario("ask-scenario")

    async def run():
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            compose, captures = _ask_tree(tmp_path, client.server.port)
            monkeypatch.setenv("REEF_HARNESS_CAPTURES_DIR", str(captures))
            pending = _write_spool_entry(captures, "ask-scenario", "pending") if has_receipts else None
            before = pending.read_bytes() if pending is not None else None
            with pytest.raises(SystemExit, match="training_mode='manual'"):
                await asyncio.to_thread(harness, "ask-scenario", "pi", compose, "run tests first")
            assert seen == []
            response = await client.post("/reef/scenarios/ask-scenario/update", json={"training_mode": "manual"})
            assert response.status == 200
            await asyncio.to_thread(harness, "ask-scenario", "pi", compose, "run tests first")
            for _ in range(100):
                releases = scenario.releases()
                committed = [row for row in releases if row.get("metrics", {}).get("training_request")]
                if committed:
                    break
                await asyncio.sleep(0.05)
            assert len(committed) == 1
            request = committed[0]["metrics"]["training_request"]
            assert request["text"] == "run tests first"
            assert request["release_id"] == "rel-3"
            assert request["session"]
            assert seen == [((), ({**request, "untrusted": True},))]
            assert committed[0]["metrics"]["published"] is True
            assert f"training request {request['id']} accepted" in capsys.readouterr().out
            if pending is not None:
                assert pending.read_bytes() == before
            assert not (Path(scenario.trainer.training_backend.proposals.directory) / "requests").exists()
            response = await client.post("/reef/scenarios/ask-scenario/update", json={"training_mode": "auto"})
            assert response.status == 200
            assert scenario.trainer.training_mode == "auto"
        finally:
            await client.close()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
