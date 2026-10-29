"""The sync routes through the real HTTP app: install a release as a layer, merge a client's changes, call the model.

``GET /reef/harness`` with ``form=client`` renders a release as a layer bound to the scenario's route, with the command
that runs the agent on it; that route records the call, and ``POST /reef/harness/import`` merges a client's changes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator

import pytest
from aiohttp import web
from reef_client import ReefClient
from reef_client.harness import SYNC_LOG, HarnessInstall, HarnessSyncError, commands_root, connect

from tests.reef_service.test_harness_channel import HEADERS, PATCHES, app_client, channel_dispatcher, evaluated_step
from tests.support.harness_service import EchoInference
from tests.support.harness_steps import PI_SEED

from reef.harness.sync import TOKEN_PLACEHOLDER
from reef.service.app import create_app

SKILL = "---\nname: fix\ndescription: Fix the failing test first.\n---\n\nRun the tests, then fix.\n"
MINE = {"plugin/rules.md": "Mine.\n", "plugin/skills/fix/SKILL.md": SKILL, "pi-agent/auth.json": "{}\n"}


def test_a_clients_changes_merge_onto_the_head_and_the_next_step_evolves_them(tmp_path) -> None:
    async def run() -> None:
        client = app_client(channel_dispatcher(tmp_path, PATCHES[:1], seed=PI_SEED))
        await client.start_server()
        try:
            base = (await (await client.get("/reef/harness", headers=HEADERS)).json())["release_id"]
            body = {"files": MINE, "base_release_id": base}
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
            # A client still on the seed merges onto the head: its new skill joins the person's files.
            other = SKILL.replace("name: fix", "name: other")
            on_seed = {"plugin/rules.md": "Answer briefly.\n", "plugin/skills/other/SKILL.md": other}
            response = await client.post(
                "/reef/harness/import", headers=HEADERS, json={"files": on_seed, "base_release_id": base}
            )
            merged = await response.json()
            assert merged["merged"] is True and merged["parent_release_id"] == imported["release_id"]
            # A line that both sides changed since the seed is refused by the file's name; a client names its base.
            conflicting = {**on_seed, "plugin/rules.md": "Theirs.\n"}
            for files, stale, status, named in (
                (conflicting, base, 409, "plugin/rules.md"),
                (MINE, None, 400, "base_release_id"),
            ):
                refused = await client.post(
                    "/reef/harness/import", headers=HEADERS, json={"files": files, "base_release_id": stale}
                )
                assert refused.status == status and named in await refused.text()
            # The next step builds on the merge: its method's patch applies to the person's files.
            evolved = await evaluated_step(client)
            assert evolved["evaluation"]["head_release_id"] == merged["release_id"]
            source = await (await client.get("/reef/harness", params={"form": "source"}, headers=HEADERS)).json()
            assert (
                source["files"]["skills/other/SKILL.md"] == other and source["files"]["rules.md"] == "marker rules\n"
            )
        finally:
            await client.close()

    asyncio.run(run())


def test_a_client_install_calls_the_scenario_route_and_the_call_is_recorded(tmp_path) -> None:
    async def run() -> None:
        client = app_client(channel_dispatcher(tmp_path, (), seed=PI_SEED))
        await client.start_server()
        try:
            params = {"form": "client", "service_url": "http://reef.example:8900/", "root": "/home/a/install"}
            manifest = await (await client.get("/reef/harness", params=params, headers=HEADERS)).json()
            provider = json.loads(manifest["files"]["pi-agent/models.json"])["providers"]["reef"]
            assert provider["baseUrl"] == "http://reef.example:8900/reef/scenarios/delivery/v1"
            assert provider["apiKey"] == TOKEN_PLACEHOLDER
            # pi's own command installs the package in the person's ~/.pi/agent, and a session Reef starts picks the
            # served model.
            assert manifest["run"]["binary"] == "pi" and manifest["run"]["args"] == ["--model", "reef/qwen3-8b"]
            assert manifest["run"]["register"] == [["install", "/home/a/install/plugin"]]
            assert "plugin/skills/{name}/SKILL.md" in manifest["synced"]
            # The agent's pinned install, so a client without pi can install the version the episodes run.
            assert manifest["run"]["install"]["package"] == "@earendil-works/pi-coding-agent"
            for missing in ("service_url", "root"):
                partial = {key: value for key, value in params.items() if key != missing}
                assert (await client.get("/reef/harness", params=partial, headers=HEADERS)).status == 400
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


FAKE_PI = """#!{python}
import json, pathlib, sys
# pi install and pi remove record the package in the person's own settings.json, as pi does.
settings = pathlib.Path.home() / ".pi" / "agent" / "settings.json"
packages = json.loads(settings.read_text())["packages"] if settings.exists() else []
if sys.argv[1] in ("install", "remove"):
    kept = [path for path in packages if path != sys.argv[2]]
    settings.write_text(json.dumps({{"packages": kept + [sys.argv[2]] if sys.argv[1] == "install" else kept}}))
    sys.exit(0)
# A session writes a skill into the installed package and exits 3.
package = pathlib.Path(packages[0])
(package / "skills" / "learned").mkdir(parents=True)
(package / "skills" / "learned" / "SKILL.md").write_text({learned!r})
print(" ".join(sys.argv[1:]))
sys.exit(3)
"""


def test_a_person_connects_and_each_run_syncs_with_reef_client(tmp_path, monkeypatch) -> None:
    """The client's whole round: connect sets the machine up so that the person's own pi command runs through Reef,
    pi runs on the layer and the run sends what the session wrote when it exits, an edit merges onto a release a
    step made, a line both changed is refused, discard, and disconnect gives pi back."""
    home = tmp_path / "home"
    (home / ".pi" / "agent").mkdir(parents=True)
    (home / ".pi" / "agent" / "AGENTS.md").write_text("My own rules.\n")
    learned = SKILL.replace("name: fix", "name: learned")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "pi").write_text(FAKE_PI.format(python=sys.executable, learned=learned))
    (bin_dir / "pi").chmod(0o755)
    install_dir = tmp_path / "install"
    for name, value in {"HOME": home, "XDG_DATA_HOME": tmp_path / "data", "SHELL": "/bin/zsh"}.items():
        monkeypatch.setenv(name, str(value))
    monkeypatch.delenv("ZDOTDIR", raising=False)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    with served(channel_dispatcher(tmp_path, PATCHES[:1], seed=PI_SEED)) as url:
        # One harness on this Reef and a token-free deployment: connect asks nothing and says one line.
        assert connect("delivery", url, install_dir) == "Done. Open a new terminal and run pi."
        assert str(commands_root()) in (home / ".zshrc").read_text()
        install = HarnessInstall.open(install_dir)
        assert install.record["agent"] == str(bin_dir / "pi")
        agents = install_dir / "plugin" / "rules.md"
        provider = json.loads((install_dir / "pi-agent" / "models.json").read_text())["providers"]["reef"]
        assert provider["baseUrl"] == f"{url}/reef/scenarios/delivery/v1" and provider["apiKey"] == "reef"
        # The person types pi as before. It runs with the layer's arguments and the person's own, and the run sends
        # the skill the session wrote.
        environment = {**os.environ, "PATH": f"{commands_root()}{os.pathsep}{os.environ['PATH']}"}
        shown = subprocess.run(["pi", "--resume"], capture_output=True, text=True, env=environment)
        assert shown.returncode == 3, shown.stderr
        assert shown.stdout.strip() == "--model reef/qwen3-8b --resume"
        assert "sending plugin/skills/learned/SKILL.md to Reef in the background" in shown.stderr
        # The send runs from a process of its own after the session, and reports to the log the next run prints.
        log = install_dir / SYNC_LOG
        deadline = time.monotonic() + 10
        while "sent" not in log.read_text() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert "sent plugin/skills/learned/SKILL.md as release" in log.read_text()
        assert install.head()["files"]["plugin/skills/learned/SKILL.md"] == learned
        # The person's own rules stay; pi's own command recorded the package beside them.
        assert (home / ".pi" / "agent" / "AGENTS.md").read_text() == "My own rules.\n"
        assert json.loads((home / ".pi" / "agent" / "settings.json").read_text())["packages"] == [
            str(install_dir / "plugin")
        ]
        install = HarnessInstall.open(install_dir)
        assert install.changed() == []
        # A step evolves the rules while the person edits the same line and adds a skill.
        agents.write_text("Answer briefly, always.\n")
        other = install_dir / "plugin" / "skills" / "other" / "SKILL.md"
        other.parent.mkdir()
        other.write_text(SKILL.replace("name: fix", "name: other"))
        client = ReefClient(url)
        _, headers = client.post("/v1/chat/completions", "delivery", {"messages": [{"role": "user", "content": "hi"}]})
        client.report("delivery", {"score": 0.0}, references=[headers["x-reef-agent-record-id"]])
        deadline = time.monotonic() + 10
        while install.head()["release_id"] == install.record["release_id"] and time.monotonic() < deadline:
            time.sleep(0.05)
        with pytest.raises(HarnessSyncError, match=r"both changed plugin/rules\.md"):
            install.sync()
        assert agents.read_text() == "Answer briefly, always.\n"
        # Back to the base text, the rules take the step's, and the new skill merges onto its release.
        agents.write_text("Answer briefly.\n")
        assert "merged onto release" in install.sync()
        assert agents.read_text() == "marker rules\n" and other.exists() and install.changed() == []
        agents.write_text("Dropped.\n")
        assert "installed release" in install.sync(discard=True)
        assert agents.read_text() == "marker rules\n"
        assert "pi runs on its own again" in install.disconnect() and not install_dir.exists()
        assert json.loads((home / ".pi" / "agent" / "settings.json").read_text())["packages"] == []
        assert not commands_root().exists() and "reef-client" not in (home / ".zshrc").read_text()
