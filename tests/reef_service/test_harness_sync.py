"""The sync routes through the real HTTP app: import a person's harness, install it bound to Reef, call the model.

``POST /reef/harness/import`` publishes the files of a person's agent as the head, ``GET /reef/harness`` with
``form=client`` renders a release bound to the scenario's route, and that route records the call.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
import sys
import threading
import time
from collections.abc import Iterator

import pytest
from aiohttp import web
from reef_client import ReefClient
from reef_client.harness import HarnessInstall, HarnessSyncError

from tests.reef_service.test_harness_channel import HEADERS, PATCHES, app_client, channel_dispatcher, evaluated_step
from tests.support.harness_service import EchoInference
from tests.support.harness_steps import PI_SEED

from reef.harness.sync import TOKEN_PLACEHOLDER
from reef.service.app import create_app

SKILL = "---\nname: fix\ndescription: Fix the failing test first.\n---\n\nRun the tests, then fix.\n"
MINE = {"pi-agent/AGENTS.md": "Mine.\n", "pi-agent/skills/fix/SKILL.md": SKILL, "pi-agent/auth.json": "{}\n"}


def test_a_fresh_scenario_imports_a_persons_harness_and_the_next_step_evolves_it(tmp_path) -> None:
    async def run() -> None:
        client = app_client(channel_dispatcher(tmp_path, PATCHES[:1], seed=PI_SEED))
        await client.start_server()
        try:
            base = (await (await client.get("/reef/harness", headers=HEADERS)).json())["release_id"]
            body = {"files": MINE, "base_release_id": None}
            response = await client.post("/reef/harness/import", headers=HEADERS, json=body)
            assert response.status == 200
            imported = await response.json()
            assert imported["changed"] is True and imported["parent_release_id"] == base
            assert imported["ignored"] == ["pi-agent/auth.json"]
            # The head serves the person's files, with the seed's config: the binding owns the config.
            source = await (await client.get("/reef/harness", params={"form": "source"}, headers=HEADERS)).json()
            assert source["release_id"] == imported["release_id"]
            assert source["files"] == {
                "config/primary.json": PI_SEED.files["config/primary.json"],
                "rules.md": "Mine.\n",
                "skills/fix/SKILL.md": SKILL,
            }
            rows = (await (await client.get("/reef/scenarios/delivery/releases")).json())["releases"]
            assert rows[0]["operation"] == "import" and rows[0]["metrics"]["patch_parts"] == {
                "rules.md": "rules",
                "skills/fix/SKILL.md": "skill",
            }
            # The same files again publish nothing.
            again = await client.post(
                "/reef/harness/import", headers=HEADERS, json={**body, "base_release_id": imported["release_id"]}
            )
            assert (await again.json())["changed"] is False
            # A client on an older release, or one that installed none, pulls before it imports.
            for stale in (base, None):
                refused = await client.post(
                    "/reef/harness/import", headers=HEADERS, json={"files": MINE, "base_release_id": stale}
                )
                assert refused.status == 409
            # The next step builds on the import: its method's patch applies to the person's files.
            evolved = await evaluated_step(client)
            assert evolved["evaluation"]["head_release_id"] == imported["release_id"]
            source = await (await client.get("/reef/harness", params={"form": "source"}, headers=HEADERS)).json()
            assert source["files"]["skills/fix/SKILL.md"] == SKILL and source["files"]["rules.md"] == "marker rules\n"
        finally:
            await client.close()

    asyncio.run(run())


def test_a_client_install_calls_the_scenario_route_and_the_call_is_recorded(tmp_path) -> None:
    async def run() -> None:
        client = app_client(channel_dispatcher(tmp_path, (), seed=PI_SEED))
        await client.start_server()
        try:
            params = {"form": "client", "service_url": "http://reef.example:8900/"}
            manifest = await (await client.get("/reef/harness", params=params, headers=HEADERS)).json()
            provider = json.loads(manifest["files"]["pi-agent/models.json"])["providers"]["reef"]
            assert provider["baseUrl"] == "http://reef.example:8900/reef/scenarios/delivery/v1"
            assert provider["apiKey"] == TOKEN_PLACEHOLDER
            missing = await client.get("/reef/harness", params={"form": "client"}, headers=HEADERS)
            assert missing.status == 400
            # The installed harness sends no scenario header: the path names the scenario.
            body = {"messages": [{"role": "user", "content": "hi"}]}
            response = await client.post("/reef/scenarios/delivery/v1/chat/completions", json=body)
            assert response.status == 200
            receipt = response.headers["x-reef-agent-record-id"]
            record = await client.get(f"/reef/scenarios/delivery/records/{receipt}")
            assert record.status == 200
            unknown = await client.post("/reef/scenarios/delivery/v1/models", json={})
            assert unknown.status == 404
        finally:
            await client.close()

    asyncio.run(run())


@contextlib.contextmanager
def served(dispatcher) -> Iterator[str]:
    """The service over ``dispatcher`` on a free local port, in its own thread, as a client reaches it."""
    loop = asyncio.new_event_loop()
    runner = web.AppRunner(create_app(dispatcher, inference_handler=EchoInference()))
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "127.0.0.1", 0)
    loop.run_until_complete(site.start())
    port = site._server.sockets[0].getsockname()[1]
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(10)
        loop.close()


def test_a_person_connects_runs_pushes_and_pulls_with_reef_client(tmp_path, monkeypatch) -> None:
    """The client's whole round: import the person's pi harness, run pi on the install, push an edit, pull a step's
    release, and refuse to drop an edit that was not pushed."""
    mine = tmp_path / "home" / ".pi" / "agent"
    for path, text in MINE.items():
        (mine / path.removeprefix("pi-agent/")).parent.mkdir(parents=True, exist_ok=True)
        (mine / path.removeprefix("pi-agent/")).write_text(text, encoding="utf-8")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(mine))
    install_dir = tmp_path / "install"
    with served(channel_dispatcher(tmp_path, PATCHES[:1], seed=PI_SEED)) as url:
        install = HarnessInstall.create(install_dir, url, "delivery", "pi", token=None)
        assert "imported 2 files" in install.connect()
        agents = install_dir / "pi-agent" / "AGENTS.md"
        assert agents.read_text() == "Mine.\n" and (mine / "AGENTS.md").read_text() == "Mine.\n"
        provider = json.loads((install_dir / "pi-agent" / "models.json").read_text())["providers"]["reef"]
        assert provider["baseUrl"] == f"{url}/reef/scenarios/delivery/v1" and provider["apiKey"] == "reef"
        # The agent runs with its root at the install.
        agent = [sys.executable, "-c", "import os; print(os.environ['PI_CODING_AGENT_DIR'])"]
        shown = subprocess.run(
            [
                sys.executable,
                "-m",
                "reef_client",
                "harness",
                "run",
                "delivery",
                "--dir",
                str(install_dir),
                "--",
                *agent,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert shown.stdout.strip() == str(install_dir / "pi-agent")
        agents.write_text("Mine, briefer.\n")
        assert install.changed() == ["pi-agent/AGENTS.md"]
        assert "published pi-agent/AGENTS.md" in install.push() and install.changed() == []
        # A step evolves the pushed tree; an edit that was not pushed keeps it from installing until forced.
        client = ReefClient(url)
        _, headers = client.post("/v1/chat/completions", "delivery", {"messages": [{"role": "user", "content": "hi"}]})
        client.report("delivery", {"score": 0.0}, references=[headers["x-reef-agent-record-id"]])
        deadline = time.monotonic() + 10
        while install.head()["release_id"] == install.record["release_id"] and time.monotonic() < deadline:
            time.sleep(0.05)
        agents.write_text("Not pushed.\n")
        with pytest.raises(HarnessSyncError, match="Push them"):
            install.pull()
        assert "installed release" in install.pull(force=True)
        assert agents.read_text() == "marker rules\n" and install.changed() == []
