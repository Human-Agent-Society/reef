"""Raw step file reads stay bounded, authenticated, and within the selected scenario."""

import asyncio
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from reef_service.test_harness_proposals import _dispatcher as _instruction_dispatcher
from reef_service.test_harness_proposals import _recipe
from reef_service.test_harness_requests import _request
from reef_service.test_release_page import FIRST, SCENARIO, _committed, _dispatcher

from reef.service.app import create_app
from reef.train.cordis_backend.record_history import MAX_RECORD_BYTES, read_step_records


def test_record_scope_and_limits(tmp_path: Path) -> None:
    root = tmp_path / "a"
    step = root / "1-2"
    step.mkdir(parents=True)
    (step / "proposer.json").write_text('[{"reply":"hello"}]')
    other = tmp_path / "b" / "1"
    other.mkdir(parents=True)
    (other / "secret.json").write_text('"private"')
    (step / "link.json").symlink_to(other / "secret.json")
    (step / "agents").symlink_to(other, target_is_directory=True)
    assert read_step_records(root, str(step), None)["files"] == [{"path": "proposer.json", "bytes": 19}]
    assert json.loads(read_step_records(root, str(step), "proposer.json")["text"])[0]["reply"] == "hello"
    for relative in ("../1/proposer.json", str(other / "secret.json"), "link.json", "agents/secret.json"):
        with pytest.raises(ValueError):
            read_step_records(root, str(step), relative)
    with pytest.raises(ValueError, match="outside"):
        read_step_records(root, str(other), "secret.json")
    with pytest.raises(FileNotFoundError):
        read_step_records(root, str(step), "absent.json")
    (step / "large.json").write_bytes(b"x" * (MAX_RECORD_BYTES + 1))
    with pytest.raises(ValueError, match="4 MiB"):
        read_step_records(root, str(step), "large.json")
    assert read_step_records(None, str(step), None)["status"] == "disabled"
    assert read_step_records(root, str(root / "missing"), None)["status"] == "missing"


def test_step_record_http_contract(tmp_path: Path) -> None:
    dispatcher = _dispatcher(tmp_path, keep_records=True)
    scenario = dispatcher.get_or_create_scenario(SCENARIO)
    assert scenario is not None

    async def run() -> None:
        client = TestClient(TestServer(create_app(dispatcher, tokens="test-token")))
        await client.start_server()
        headers = {"Authorization": "Bearer test-token", "x-reef-scenario": SCENARIO}
        try:
            response = await client.post("/reef/train", json=_request(FIRST), headers=headers)
            assert response.status == 200
            await _committed(scenario, 2)
            url = "/reef/harness/releases/1/records"
            assert (await client.get(url)).status == 401
            response = await client.get(url, headers=headers)
            assert response.status == 200 and response.headers["Cache-Control"] == "no-store"
            inventory = await response.json()
            assert inventory["status"] == "retained"
            assert "mutations.json" in [entry["path"] for entry in inventory["files"]]
            response = await client.get(url, params={"path": "mutations.json"}, headers=headers)
            assert response.status == 200
            assert json.loads((await response.json())["text"])
            assert (await client.get(url, params={"path": "../secret.json"}, headers=headers)).status == 400
            assert (await client.get(url, params={"path": "absent.json"}, headers=headers)).status == 404
            assert (await client.get("/reef/harness/releases/99/records", headers=headers)).status == 404
            response = await client.get("/reef/harness/releases/0/records", headers=headers)
            assert (await response.json())["status"] == "not_recorded"
        finally:
            await client.close()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_failed_proposer_records_remain_linked_after_reload_and_retry(tmp_path: Path, monkeypatch) -> None:
    replies = iter([None, "done"])

    def urlopen(request, timeout=None):
        reply = next(replies)
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": reply,
                                "reasoning": "failed call thinking" if reply is None else "retry thinking",
                            }
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    def propose(nodes, samples, models, *, requests=()):
        models.served.chat([{"role": "user", "content": requests[0]["text"]}])
        return

    recipe = replace(_recipe(tmp_path, propose), training_mode="manual", step_record_dir=str(tmp_path / "steps"))
    dispatcher = _instruction_dispatcher(tmp_path, recipe)
    dispatcher.get_or_create_scenario("s")

    async def run() -> None:
        client = TestClient(TestServer(create_app(dispatcher, tokens="test-token")))
        await client.start_server()
        headers = {"Authorization": "Bearer test-token", "x-reef-scenario": "s"}
        try:
            for step, instruction in enumerate(("fail", "retry"), 1):
                response = await client.post("/reef/train", json=_request(instruction), headers=headers)
                assert response.status == 200
                for _ in range(200):
                    scenario = dispatcher.get_or_create_scenario("s")
                    if len(scenario.releases()) > step:
                        break
                    await asyncio.sleep(0.05)
                else:
                    raise AssertionError("instruction did not commit")
                row = scenario.releases()[0]
                metrics = row["metrics"]
                assert metrics["training_request"]["text"] == instruction
                if step == 1:
                    assert metrics["error"] == "ModelBindingError: model endpoint returned non-text content"
                    assert metrics["skipped"] == "instruction failed"
                    assert Path(metrics["step_record"]).name == "1"
                else:
                    assert Path(metrics["step_record"]).name == "1-2"
                url = f"/reef/harness/releases/{step}/records"
                response = await client.get(url, headers=headers)
                assert (await response.json())["status"] == "retained"
                response = await client.get(url, params={"path": "proposer.json"}, headers=headers)
                recorded = json.loads((await response.json())["text"])[0]
                expected = "failed call thinking" if step == 1 else "retry thinking"
                assert recorded["response"]["choices"][0]["message"]["reasoning"] == expected
        finally:
            await client.close()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
