"""Update-channel guarantees of the harness read routes: the release catalog
with its gate metrics, version-addressed pulls, the byte identity of a
pulled tree with the composition the gate measured, and the one-command
install script. Hermetic like test_harness_recipe.py (episodes run a fake
pi binary through the real adapter path); pulls go through the real HTTP
app with the stdlib client, and the install-script tests execute the
generated script under ``sh`` with the vendor tools shimmed on PATH."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import pytest
import yaml
from aiohttp.test_utils import TestClient, TestServer
from reef_client.client import ReefClient

import reef.harness.adapters
from reef.artifact import InMemoryRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.harness.adapters import get_adapter
from reef.harness.descriptor import DescriptorError, load_descriptor
from reef.harness.episode import EpisodeResult
from reef.harness.render import render_composition
from reef.recipe import Recipe, RecipeRegistry
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.runtime.inference import InferenceBackend
from reef.service.app import create_app
from reef.service.install_script import HARNESS_RELEASE_SIDECAR, composition_checksum, render_install_script
from reef.train.cordis_backend import CordisRecipe, Mutation
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer

# The fake harness scores itself, as in test_harness_recipe.py: its
# trajectory carries the rules text, so the evaluator can rank a composition
# by how often the marker appears and an update mutation can beat its parent.
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

#: Each mutation strictly improves the marker count, so every gated step wins
#: and publishes: step one creates the rules node, step two rewrites it.
MUTATIONS = (
    Mutation("create", "r1", {"name": "rules", "config": {"text": "marker rules"}}),
    Mutation("update", "r1", {"config": {"text": "marker marker rules"}}),
)

NODES_V1 = (("rules", {"text": "marker rules"}),)
NODES_V2 = (("rules", {"text": "marker marker rules"}),)

_ASYNC_UPDATE_TIMEOUT_S = 5.0


class _ReleaseClient(ReefClient):
    """Release/content-aware client used until the external client release lands."""

    def harness_pull(self, scenario, destination, *, release_id=None, extra_headers=None):
        path = "/reef/harness"
        if release_id is not None:
            path += f"?release_id={quote(release_id, safe='')}"
        headers = {"x-reef-scenario": scenario, **dict(extra_headers or {})}
        manifest = self.get(path, extra_headers=headers)
        files = manifest.get("files", {})
        for relative in files:
            if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
                raise ValueError(f"served path {relative!r} escapes the destination")
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        sidecar = root / HARNESS_RELEASE_SIDECAR
        if sidecar.is_file():
            try:
                previous = json.loads(sidecar.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                previous = {}
            for relative in previous.get("files", ()):
                if relative not in files:
                    stale = root / relative
                    if stale.is_file():
                        stale.unlink()
        for relative, content in files.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
        release_id = str(manifest["release_id"])
        sidecar.write_text(
            json.dumps(
                {
                    "release_id": release_id,
                    "content_id": str(manifest["content_id"]),
                    "files": sorted(files),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return release_id

    def harness_releases(self, scenario, *, extra_headers=None):
        headers = {"x-reef-scenario": scenario, **dict(extra_headers or {})}
        return self.get("/reef/harness/releases", extra_headers=headers)["releases"]


_ASYNC_UPDATE_POLL_S = 0.01


def evaluate(task: str, result: EpisodeResult) -> float:
    del task
    return float(result.trajectory[-1]["rules"].count("marker"))


class _EchoBackend(InferenceBackend):
    async def inference(self, artifact, path, payload):
        del artifact, path, payload
        return {"choices": [{"message": {"content": "ok"}}]}


def _dispatcher(
    tmp_path: Path,
    mutations: tuple[Mutation, ...],
    *,
    bootstrap_files: dict[str, str] | None = None,
    recipe_names: tuple[str, ...] = ("evolve",),
    batch_policy: str = "reports",
    batch_size: int = 1,
) -> Dispatcher:
    proposals = iter(mutations)
    binary = tmp_path / "fake-pi"
    binary.write_text(PI_FAKE)
    binary.chmod(0o755)
    recipe = CordisRecipe(
        resolve_proposer(lambda nodes, samples, model: next(proposals, None)),
        resolve_episode_scorer(evaluate),
        ("task one",),
        binary=str(binary),
        runtime=InferenceProxyRuntime(model_path="demo-model", base_url="http://localhost:8000"),
        batch_policy=batch_policy,
        batch_size=batch_size,
    )
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    for relative, text in (bootstrap_files or {}).items():
        target = bootstrap / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    dispatcher = Dispatcher(
        RecipeRegistry(dict.fromkeys(recipe_names, recipe)),
        InMemoryRepositoryBackend.factory(bootstrap, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
        # The commit log is what puts gate metrics on the release catalog.
        agent_record_dir=tmp_path / "agent-record",
    )
    dispatcher.get_or_create_scenario("delivery", recipe_names[0])
    return dispatcher


async def _gate_step(client: TestClient) -> dict:
    """Drive one gated evolution step through the wire; return the new manifest.

    One traced inference plus its failing report fills the batch (batch_size
    1, max_score 0.0). The report POST only records and schedules the step;
    the harness read channel exposes the winner after the background commit.
    """
    response = await client.get("/reef/harness", headers={"x-reef-scenario": "delivery"})
    if response.status == 200:
        previous_version = (await response.json())["release_id"]
    else:
        assert response.status == 404
        await response.read()
        previous_version = None
    response = await client.post(
        "/v1/chat/completions",
        headers={"x-reef-scenario": "delivery"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status == 200
    receipt = response.headers["x-reef-agent-record-id"]
    response = await client.post(
        "/reef/report",
        headers={"x-reef-scenario": "delivery"},
        json={"score": 0.0, "references": [receipt]},
    )
    assert response.status == 200
    deadline = asyncio.get_running_loop().time() + _ASYNC_UPDATE_TIMEOUT_S
    while True:
        response = await client.get("/reef/harness", headers={"x-reef-scenario": "delivery"})
        if response.status == 200:
            manifest = await response.json()
            if manifest["release_id"] != previous_version:
                assert manifest["gate"]["published"] is True
                return manifest
        else:
            assert response.status == 404
            await response.read()
        if asyncio.get_running_loop().time() >= deadline:
            pytest.fail("harness update did not commit")
        await asyncio.sleep(_ASYNC_UPDATE_POLL_S)


@pytest.mark.unit
def test_status_reports_a_committed_step_that_published_no_harness(tmp_path) -> None:
    async def run() -> None:
        client = TestClient(TestServer(create_app(_dispatcher(tmp_path, ()), inference_backend=_EchoBackend())))
        await client.start_server()
        try:
            response = await client.get("/reef/status")
            assert response.status == 200
            assert (await response.json())["scenarios"]["delivery"]["last_committed_step"] is None

            response = await client.post(
                "/v1/chat/completions",
                headers={"x-reef-scenario": "delivery"},
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            assert response.status == 200
            receipt = response.headers["x-reef-agent-record-id"]
            response = await client.post(
                "/reef/report",
                headers={"x-reef-scenario": "delivery"},
                json={"score": 0.0, "references": [receipt]},
            )
            assert response.status == 200

            deadline = asyncio.get_running_loop().time() + _ASYNC_UPDATE_TIMEOUT_S
            while True:
                response = await client.get("/reef/status")
                assert response.status == 200
                scenario = (await response.json())["scenarios"]["delivery"]
                committed = scenario["last_committed_step"]
                if committed is not None:
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    pytest.fail("no-proposal step did not commit")
                await asyncio.sleep(_ASYNC_UPDATE_POLL_S)

            assert scenario["scenario_step"] == committed["step"] == 1
            assert isinstance(committed["recorded_at"], float)
            assert committed["metrics"]["skipped"] == "no proposal"
            response = await client.get("/reef/harness", headers={"x-reef-scenario": "delivery"})
            assert response.status == 404
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_status_reports_a_committed_gate_rejection(tmp_path) -> None:
    async def run() -> None:
        mutation = Mutation("create", "r1", {"name": "rules", "config": {"text": "no help"}})
        client = TestClient(
            TestServer(create_app(_dispatcher(tmp_path, (mutation,)), inference_backend=_EchoBackend()))
        )
        await client.start_server()
        try:
            response = await client.post(
                "/v1/chat/completions",
                headers={"x-reef-scenario": "delivery"},
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            assert response.status == 200
            receipt = response.headers["x-reef-agent-record-id"]
            response = await client.post(
                "/reef/report",
                headers={"x-reef-scenario": "delivery"},
                json={"score": 0.0, "references": [receipt]},
            )
            assert response.status == 200

            deadline = asyncio.get_running_loop().time() + _ASYNC_UPDATE_TIMEOUT_S
            while True:
                response = await client.get("/reef/status")
                assert response.status == 200
                scenario = (await response.json())["scenarios"]["delivery"]
                committed = scenario["last_committed_step"]
                if committed is not None:
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    pytest.fail("rejected gate step did not commit")
                await asyncio.sleep(_ASYNC_UPDATE_POLL_S)

            assert scenario["scenario_step"] == committed["step"] == 1
            assert committed["metrics"]["published"] is False
            assert committed["metrics"]["selected"] is False
            assert committed["metrics"]["selection"]["outcome"] == "reject"
            response = await client.get("/reef/harness", headers={"x-reef-scenario": "delivery"})
            assert response.status == 404
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_pulled_tree_is_byte_identical_to_the_gated_composition(tmp_path) -> None:
    async def run() -> None:
        client = TestClient(
            TestServer(create_app(_dispatcher(tmp_path, MUTATIONS[:1]), inference_backend=_EchoBackend()))
        )
        await client.start_server()
        try:
            manifest = await _gate_step(client)
            destination = tmp_path / "pulled"
            puller = _ReleaseClient(str(client.server.make_url("")))
            written = await asyncio.to_thread(puller.harness_pull, "delivery", destination)
            assert written == manifest["release_id"]
            # Every pulled file carries exactly the bytes of the composition
            # the gate measured, and nothing else was written.
            expected = render_composition(NODES_V1, get_adapter("pi"))
            pulled = {
                str(path.relative_to(destination)): path.read_bytes()
                for path in sorted(destination.rglob("*"))
                if path.is_file() and path.name != HARNESS_RELEASE_SIDECAR
            }
            assert pulled == {relative: text.encode("utf-8") for relative, text in expected.items()}
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_version_addressed_pull_returns_the_superseded_tree(tmp_path) -> None:
    async def run() -> None:
        client = TestClient(TestServer(create_app(_dispatcher(tmp_path, MUTATIONS), inference_backend=_EchoBackend())))
        await client.start_server()
        try:
            first = await _gate_step(client)
            second = await _gate_step(client)
            assert first["release_id"] != second["release_id"]
            response = await client.get(
                "/reef/harness",
                params={"release_id": first["release_id"]},
                headers={"x-reef-scenario": "delivery"},
            )
            assert response.status == 200
            assert response.headers["x-reef-release-id"] == first["release_id"]
            manifest = await response.json()
            assert manifest["release_id"] == first["release_id"]
            assert manifest["files"] == render_composition(NODES_V1, get_adapter("pi"))
            assert second["files"] == render_composition(NODES_V2, get_adapter("pi"))
            # The client's version-addressed pull writes the older bytes.
            destination = tmp_path / "pinned"
            puller = _ReleaseClient(str(client.server.make_url("")))
            written = await asyncio.to_thread(
                puller.harness_pull, "delivery", destination, release_id=first["release_id"]
            )
            assert written == first["release_id"]
            assert (destination / "pi-agent" / "AGENTS.md").read_bytes() == b"marker rules\n"
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_unknown_version_is_a_404_naming_the_version(tmp_path) -> None:
    async def run() -> None:
        client = TestClient(TestServer(create_app(_dispatcher(tmp_path, ()), inference_backend=_EchoBackend())))
        await client.start_server()
        try:
            response = await client.get(
                "/reef/harness",
                params={"release_id": "no-such-version"},
                headers={"x-reef-scenario": "delivery"},
            )
            assert response.status == 404
            assert "no-such-version" in await response.text()
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_versions_catalog_carries_the_publishing_steps_gate_metrics(tmp_path) -> None:
    async def run() -> None:
        client = TestClient(
            TestServer(create_app(_dispatcher(tmp_path, MUTATIONS[:1]), inference_backend=_EchoBackend()))
        )
        await client.start_server()
        try:
            manifest = await _gate_step(client)
            response = await client.get("/reef/harness/releases", headers={"x-reef-scenario": "delivery"})
            assert response.status == 200
            catalog = await response.json()
            assert catalog["scenario"] == "delivery"
            rows = catalog["releases"]
            # Newest last: the creation version opens the catalog, the gated
            # head closes it and repeats the manifest's gate numbers.
            assert rows[0]["operation"] == "creation"
            head = rows[-1]
            assert head["release_id"] == manifest["release_id"]
            assert head["current"] is True
            assert head["metrics"] == manifest["gate"]
            assert head["metrics"]["published"] is True
            assert head["metrics"]["wins"] == 1
            # The stdlib client hands back the same parsed rows.
            puller = _ReleaseClient(str(client.server.make_url("")))
            assert await asyncio.to_thread(puller.harness_releases, "delivery") == rows
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_client_pull_writes_the_sidecar_outside_the_served_tree(tmp_path) -> None:
    async def run() -> None:
        client = TestClient(
            TestServer(create_app(_dispatcher(tmp_path, MUTATIONS[:1]), inference_backend=_EchoBackend()))
        )
        await client.start_server()
        try:
            manifest = await _gate_step(client)
            assert HARNESS_RELEASE_SIDECAR not in manifest["files"]
            destination = tmp_path / "pulled"
            puller = _ReleaseClient(str(client.server.make_url("")))
            written = await asyncio.to_thread(puller.harness_pull, "delivery", destination)
            # The sidecar records the pulled version and file list for later
            # checks and pruning; the served files around it are byte-exact.
            record = json.loads((destination / HARNESS_RELEASE_SIDECAR).read_text(encoding="utf-8"))
            assert record["release_id"] == written
            assert record["files"] == sorted(manifest["files"])
            for relative, text in manifest["files"].items():
                assert (destination / relative).read_bytes() == text.encode("utf-8")
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_crlf_content_survives_the_pull_byte_exact(tmp_path) -> None:
    """Universal newline translation must never rewrite served bytes."""
    crlf = Mutation("create", "r1", {"name": "rules", "config": {"text": "marker win\r\nrules"}})

    async def run() -> None:
        client = TestClient(TestServer(create_app(_dispatcher(tmp_path, [crlf]), inference_backend=_EchoBackend())))
        await client.start_server()
        try:
            manifest = await _gate_step(client)
            assert "marker win\r\nrules" in manifest["files"]["pi-agent/AGENTS.md"]
            destination = tmp_path / "pulled"
            puller = _ReleaseClient(str(client.server.make_url("")))
            await asyncio.to_thread(puller.harness_pull, "delivery", destination)
            assert b"marker win\r\nrules" in (destination / "pi-agent/AGENTS.md").read_bytes()
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_pull_refuses_a_manifest_path_that_escapes_the_destination(tmp_path) -> None:
    escape = {
        "release_id": "v",
        "content_id": "content-v",
        "parent_release_id": None,
        "files": {"../escape.txt": "x"},
        "gate": None,
    }
    puller = _ReleaseClient("http://127.0.0.1:1")
    puller.get = lambda path, extra_headers=None: escape  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="escapes the destination"):
        puller.harness_pull("delivery", tmp_path / "pulled")
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.unit
def test_pull_of_an_older_version_prunes_the_newer_versions_files(tmp_path) -> None:
    """Rolling back into the same directory leaves exactly the older tree."""
    v2 = {
        "release_id": "v2",
        "content_id": "content-v2",
        "parent_release_id": "v1",
        "gate": None,
        "files": {"pi-agent/AGENTS.md": "new rules", "pi-agent/prompts/helper.md": "added in v2"},
    }
    v1 = {
        "release_id": "v1",
        "content_id": "content-v1",
        "parent_release_id": None,
        "gate": None,
        "files": {"pi-agent/AGENTS.md": "old rules"},
    }
    manifests = {None: v2, "v1": v1}
    puller = _ReleaseClient("http://127.0.0.1:1")
    puller.get = (  # type: ignore[method-assign]
        lambda path, extra_headers=None: manifests["v1" if "release_id=v1" in path else None]
    )
    destination = tmp_path / "pulled"
    puller.harness_pull("delivery", destination)
    assert (destination / "pi-agent/prompts/helper.md").is_file()
    puller.harness_pull("delivery", destination, release_id="v1")
    on_disk = {
        str(path.relative_to(destination))
        for path in destination.rglob("*")
        if path.is_file() and path.name != HARNESS_RELEASE_SIDECAR
    }
    assert on_disk == {"pi-agent/AGENTS.md"}
    assert (destination / "pi-agent/AGENTS.md").read_text(encoding="utf-8") == "old rules"


# --- the one-command install script (GET /reef/harness/install) ---

#: Hostile composition text for the install-script tests: expansion syntax,
#: quotes, a line equal to a naive heredoc delimiter, no trailing newline.
HOSTILE_FILES = {
    "pi-agent/AGENTS.md": "backtick `whoami` and $HOME and $(pwd)\n'single quotes' \"doubles\"\n",
    "pi-agent/prompts/hostile.md": "line one\nEOF\nREEF_EOF\ndollar $x backslash \\ tail",
    "pi-agent/settings.json": "{}\n",
}


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


def _install_fixture(
    tmp_path: Path, *, binary_version: str | None, npm: str, scenario: str = ""
) -> tuple[Path, Path, Path, dict]:
    """A rendered script, a PATH shim dir, and an install prefix.

    ``binary_version`` seeds a fake pi at the descriptor's binary_path that
    answers ``--version`` with it (None leaves the binary absent); ``npm``
    is the shim body dropped onto PATH.
    """
    script = tmp_path / "install.sh"
    script.write_text(
        render_install_script(
            descriptor=get_adapter("pi"),
            files=HOSTILE_FILES,
            release_id="v-test",
            content_id="content-test",
            scenario=scenario,
        )
    )
    prefix = tmp_path / "prefix"
    if binary_version is not None:
        _write_executable(prefix / "node_modules/.bin/pi", f"#!/bin/sh\necho {binary_version}\n")
    shim = tmp_path / "shim"
    _write_executable(shim / "npm", npm)
    env = {**os.environ, "PATH": f"{shim}:{os.environ['PATH']}"}
    return script, tmp_path / "dest", prefix, env


def _run_install(script: Path, dest: Path, prefix: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["sh", str(script), str(dest), str(prefix)], env=env, capture_output=True, text=True, timeout=60
    )


@pytest.mark.unit
def test_install_script_golden_structure() -> None:
    """The full script for a one-file composition, byte for byte.

    The heredoc delimiters derive from each file's content hash, so hostile
    content can never terminate its own heredoc early; the checksum is the
    sha256 of the sorted relative paths, byte lengths, and file bytes the
    script re-checks after writing.
    """
    files = {"pi-agent/AGENTS.md": "hello\n"}
    sidecar = (
        json.dumps({"release_id": "v1", "content_id": "content-v1", "files": ["pi-agent/AGENTS.md"]}, indent=2) + "\n"
    )
    golden = (
        r"""#!/bin/sh
# Reef harness install: adapter pi, release v1.
# Self contained: the composition files ride inline below and the harness
# binary comes from the vendor's own channel; running this script calls no
# reef route and carries no token. Inspect freely, then run:
#     sh install.sh [DEST] [PREFIX]
set -eu

DEST="${1:-./reef-harness}"
PREFIX="${2:-$HOME/.local/share/reef-harness/pi}"
BINARY="$PREFIX/node_modules/.bin/pi"
CHECKSUM="@CHECKSUM@"
SIDECAR_CHECKSUM="@SIDECAR_CHECKSUM@"

if command -v sha256sum >/dev/null 2>&1; then
    sha256() { sha256sum | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
    sha256() { shasum -a 256 | cut -d' ' -f1; }
else
    echo 'reef: neither sha256sum nor shasum found' >&2
    exit 1
fi

# Ensure the pinned binary (@earendil-works/pi-coding-agent@0.84.2) via the vendor's channel.
installed=""
if [ -x "$BINARY" ]; then
    installed="$("$BINARY" --version 2>/dev/null || true)"
fi
case " $installed " in
    *" 0.84.2 "*)
        echo "reef: pi 0.84.2 already installed"
        ;;
    *)
        mkdir -p "$PREFIX"
        npm install --prefix "$PREFIX" '@earendil-works/pi-coding-agent@0.84.2'
        ;;
esac

# Ensure reef-client (capture proxy) and reef (harness wrapper) are installed.
python3 -c 'import reef_client.serve, reef.harness.harness_wrapper' 2>/dev/null || python3 -m pip install --quiet --user reef-client "reef @ git+https://github.com/Human-Agent-Society/reef.git" 2>/dev/null || true

# The checksum stream, as baked into CHECKSUM: each sorted relative path,
# its byte length, then its bytes, newline separated. The unquoted wc
# substitution word-splits away the padding BSD wc prints.
compose_stream() {
    :
    printf '%s\n' 'pi-agent/AGENTS.md'
    printf '%s\n' $(wc -c < "$DEST/pi-agent/AGENTS.md")
    cat "$DEST/pi-agent/AGENTS.md"
}

mkdir -p "$DEST"
mkdir -p "$DEST/pi-agent"

# A rerun on a current machine writes nothing at all, not even the sidecar.
current=""
sidecar=""
if [ -f "$DEST/.reef-harness-release" ] && [ -f "$DEST/pi-agent/AGENTS.md" ]; then
    current="$(compose_stream | sha256)"
    sidecar="$(sha256 < "$DEST/.reef-harness-release")"
fi
if [ "$current" = "$CHECKSUM" ] && [ "$sidecar" = "$SIDECAR_CHECKSUM" ]; then
    echo "reef: composition already current"
else
    # Prune the files a previous install's sidecar recorded that this
    # composition lacks, exactly like the stdlib client pull. The sidecar
    # is json.dumps at indent 2, so every file entry is one four-space
    # indented quoted line.
    if [ -f "$DEST/.reef-harness-release" ]; then
        sed -n 's/^    "\(.*\)",\{0,1\}$/\1/p' "$DEST/.reef-harness-release" |
            while IFS= read -r old; do
                case "$old" in
                    'pi-agent/AGENTS.md') ;;
                    *) rm -f "$DEST/$old" ;;
                esac
            done
    fi
cat > "$DEST/pi-agent/AGENTS.md" <<'@RULES_EOF@'
hello
@RULES_EOF@
    written="$(compose_stream | sha256)"
    if [ "$written" != "$CHECKSUM" ]; then
        echo "reef: composition checksum mismatch: $written != $CHECKSUM" >&2
        exit 1
    fi
    # Write the reef-pi wrapper: capture proxy + report command.
    BINARY_ABS="$(cd "$(dirname "$BINARY")" && pwd)/$(basename "$BINARY")"
    COMPOSE_ABS="$(mkdir -p "$DEST/pi-agent" && cd "$DEST/pi-agent" && pwd)"
    cat > "$DEST/reef-pi" <<REEF_WRAPPER_EOF
#!/bin/sh
# reef-pi: run pi with the reef-evolved composition.
# Generated by reef harness install (adapter pi, release v1).
# Usage: reef-pi -p "fix the bug"     # run the agent (receipts captured)
#        reef-pi report --score 0 --feedback "..."  # report last run's receipts
export REEF_HARNESS_BINARY="$BINARY_ABS"
export REEF_HARNESS_COMPOSE="$COMPOSE_ABS"
export REEF_HARNESS_SCENARIO="code-repair"
export REEF_HARNESS_ADAPTER="pi"
export REEF_HARNESS_ENV_VAR="PI_CODING_AGENT_DIR"
exec python3 -m reef.harness.harness_wrapper "\$@"
REEF_WRAPPER_EOF
    chmod +x "$DEST/reef-pi"
    # Symlink into ~/.local/bin so reef-pi is on PATH.
    mkdir -p "$HOME/.local/bin"
    ln -sf "$DEST/reef-pi" "$HOME/.local/bin/reef-pi"
    case ":$PATH:" in
        *":$HOME/.local/bin:"*) ;;
        *) echo "reef: add '$HOME/.local/bin' to your PATH to run reef-pi from anywhere" >&2 ;;
    esac
    # The same sidecar the stdlib client pull writes: pulled version and file list.
cat > "$DEST/.reef-harness-release" <<'@SIDECAR_EOF@'
@SIDECAR_JSON@@SIDECAR_EOF@
fi

echo "run:     $DEST/reef-pi"
echo "binary:  $BINARY"
echo "harness: $DEST"
"""
    ).replace("@CHECKSUM@", hashlib.sha256(b"pi-agent/AGENTS.md\n6\nhello\n").hexdigest())
    golden = golden.replace("@SIDECAR_CHECKSUM@", hashlib.sha256(sidecar.encode()).hexdigest())
    golden = golden.replace("@RULES_EOF@", "REEF_EOF_" + hashlib.sha256(b"hello\n").hexdigest()[:12])
    golden = golden.replace("@SIDECAR_EOF@", "REEF_EOF_" + hashlib.sha256(sidecar.encode()).hexdigest()[:12])
    golden = golden.replace("@SIDECAR_JSON@", sidecar)
    script = render_install_script(
        descriptor=get_adapter("pi"),
        files=files,
        release_id="v1",
        content_id="content-v1",
        scenario="code-repair",
    )
    assert script == golden
    assert composition_checksum(files) in script


@pytest.mark.unit
def test_install_script_skips_the_vendor_install_and_lands_hostile_content_byte_exact(tmp_path) -> None:
    """Pinned binary present: npm never runs, every file lands byte exact,
    the sidecar matches the client pull's shape, and a rerun is a no-op."""
    npm_log = tmp_path / "npm.log"
    script, dest, prefix, env = _install_fixture(
        tmp_path,
        binary_version="0.84.2",
        npm=f'#!/bin/sh\nprintf \'%s\\n\' "$@" >> "{npm_log}"\nexit 1\n',
    )
    first = _run_install(script, dest, prefix, env)
    assert first.returncode == 0, first.stderr
    assert not npm_log.exists()  # the version answered, so the install step never ran
    assert "0.84.2 already installed" in first.stdout
    assert "already current" not in first.stdout
    for relative, text in HOSTILE_FILES.items():
        assert (dest / relative).read_bytes() == text.encode("utf-8")
    sidecar = dest / HARNESS_RELEASE_SIDECAR
    record = {"release_id": "v-test", "content_id": "content-test", "files": sorted(HOSTILE_FILES)}
    assert sidecar.read_bytes() == (json.dumps(record, indent=2) + "\n").encode("utf-8")
    # Rerun on a current machine: still exit 0, writes nothing at all (the
    # read-only bits make any write attempt, sidecar included, a hard fail).
    for relative in HOSTILE_FILES:
        (dest / relative).chmod(0o444)
    sidecar.chmod(0o444)
    before = sidecar.stat().st_mtime_ns
    second = _run_install(script, dest, prefix, env)
    assert second.returncode == 0, second.stderr
    assert "already current" in second.stdout
    assert sidecar.stat().st_mtime_ns == before
    for relative, text in HOSTILE_FILES.items():
        assert (dest / relative).read_bytes() == text.encode("utf-8")


@pytest.mark.unit
def test_install_script_writes_executable_wrapper_with_baked_paths(tmp_path) -> None:
    """The reef-<adapter> wrapper is executable, bakes binary/compose/scenario
    as absolute paths, calls the harness_wrapper module, and is symlinked."""
    script, dest, prefix, env = _install_fixture(
        tmp_path,
        binary_version="0.84.2",
        npm="#!/bin/sh\nexit 1\n",
        scenario="code-repair",
    )
    result = _run_install(script, dest, prefix, env)
    assert result.returncode == 0, result.stderr
    wrapper = dest / "reef-pi"
    assert wrapper.is_file()
    assert wrapper.stat().st_mode & 0o111  # executable
    text = wrapper.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh\n")
    assert "REEF_HARNESS_BINARY" in text
    assert "REEF_HARNESS_COMPOSE" in text
    assert "REEF_HARNESS_SCENARIO" in text
    assert "REEF_HARNESS_ADAPTER" in text
    assert "REEF_HARNESS_ENV_VAR" in text
    assert "code-repair" in text
    assert '"pi"' in text
    assert "PI_CODING_AGENT_DIR" in text
    assert "python3 -m reef.harness.harness_wrapper" in text
    # compose dir is baked as an absolute path (resolved at install time)
    assert "$COMPOSE_ABS" not in text
    assert str(dest / "pi-agent") in text
    # binary path is baked as an absolute path
    binary_abs = str(prefix / "node_modules" / ".bin" / "pi")
    assert binary_abs in text
    # symlinked onto PATH
    link = Path.home() / ".local" / "bin" / "reef-pi"
    assert link.is_symlink()
    assert link.resolve() == wrapper.resolve()
    assert "run:" in result.stdout
    assert "reef-pi" in result.stdout
    # clean up the symlink so it doesn't leak between tests
    link.unlink(missing_ok=True)


@pytest.mark.unit
def test_install_script_runs_exactly_the_descriptors_vendor_install_when_the_binary_is_absent(tmp_path) -> None:
    npm_log = tmp_path / "npm.log"
    script, dest, prefix, env = _install_fixture(
        tmp_path,
        binary_version=None,
        npm=f'#!/bin/sh\nprintf \'%s\\n\' "$@" >> "{npm_log}"\n',
    )
    result = _run_install(script, dest, prefix, env)
    assert result.returncode == 0, result.stderr
    assert npm_log.read_text().splitlines() == [
        "install",
        "--prefix",
        str(prefix),
        "@earendil-works/pi-coding-agent@0.84.2",
    ]
    assert (dest / "pi-agent/AGENTS.md").read_bytes() == HOSTILE_FILES["pi-agent/AGENTS.md"].encode("utf-8")


@pytest.mark.unit
def test_install_script_reinstalls_on_a_version_mismatch(tmp_path) -> None:
    npm_log = tmp_path / "npm.log"
    script, dest, prefix, env = _install_fixture(
        tmp_path,
        binary_version="0.1.0",
        npm=f'#!/bin/sh\nprintf \'%s\\n\' "$@" >> "{npm_log}"\n',
    )
    result = _run_install(script, dest, prefix, env)
    assert result.returncode == 0, result.stderr
    assert "@earendil-works/pi-coding-agent@0.84.2" in npm_log.read_text()


def _pinned_env(tmp_path: Path) -> tuple[Path, dict]:
    """A PATH shim and install prefix whose fake pi answers the pinned version."""
    prefix = tmp_path / "prefix"
    _write_executable(prefix / "node_modules/.bin/pi", "#!/bin/sh\necho 0.84.2\n")
    shim = tmp_path / "shim"
    _write_executable(shim / "npm", "#!/bin/sh\nexit 0\n")
    env = {**os.environ, "PATH": f"{shim}:{os.environ['PATH']}"}
    return prefix, env


def _render_to(path: Path, files: dict[str, str], release_id: str) -> Path:
    path.write_text(
        render_install_script(
            descriptor=get_adapter("pi"), files=files, release_id=release_id, content_id=f"content-{release_id}"
        )
    )
    return path


@pytest.mark.unit
def test_install_of_an_older_version_prunes_the_newer_versions_files(tmp_path) -> None:
    """Running an older version's script into a DEST holding a newer install
    removes the files only the newer sidecar recorded, exactly like a repeat
    client pull, and leaves the older tree with the older sidecar."""
    prefix, env = _pinned_env(tmp_path)
    dest = tmp_path / "dest"
    v2 = _render_to(
        tmp_path / "install-v2.sh",
        {"pi-agent/AGENTS.md": "new rules\n", "pi-agent/prompts/helper.md": "added in v2\n"},
        "v2",
    )
    v1 = _render_to(tmp_path / "install-v1.sh", {"pi-agent/AGENTS.md": "old rules\n"}, "v1")
    assert _run_install(v2, dest, prefix, env).returncode == 0
    assert (dest / "pi-agent/prompts/helper.md").is_file()
    result = _run_install(v1, dest, prefix, env)
    assert result.returncode == 0, result.stderr
    on_disk = {
        str(path.relative_to(dest))
        for path in dest.rglob("*")
        if path.is_file() and path.name != HARNESS_RELEASE_SIDECAR
    }
    assert on_disk == {"pi-agent/AGENTS.md", "reef-pi"}
    assert (dest / "pi-agent/AGENTS.md").read_bytes() == b"old rules\n"
    record = {"release_id": "v1", "content_id": "content-v1", "files": ["pi-agent/AGENTS.md"]}
    assert (dest / HARNESS_RELEASE_SIDECAR).read_bytes() == (json.dumps(record, indent=2) + "\n").encode("utf-8")


@pytest.mark.unit
def test_render_refuses_a_composition_path_that_escapes_the_destination(tmp_path) -> None:
    """The generator applies the same escape rule as the client pull: an
    absolute path or any ``..`` part refuses the render, nothing is written."""
    for hostile in ("../outside-marker.txt", "/outside-marker.txt", "inner/../../outside-marker.txt"):
        with pytest.raises(ValueError, match="escapes the destination"):
            render_install_script(
                descriptor=get_adapter("pi"), files={hostile: "marker"}, release_id="v-esc", content_id="content-esc"
            )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_composition_checksum_length_framing_rejects_the_aliasing_pair(tmp_path) -> None:
    r"""{a: "1", b: "2b\n3"} and {a: "1b\n2", b: "3"} concatenate to the same
    unframed stream; the byte-length frame must keep their checksums, and the
    executed script's already-current verdict, apart."""
    aliased = {"a": "1", "b": "2b\n3"}
    files = {"a": "1b\n2", "b": "3"}
    assert composition_checksum(aliased) != composition_checksum(files)
    # Same release on purpose: the two sidecars are then identical,
    # so only the composition checksum can tell the trees apart under sh.
    prefix, env = _pinned_env(tmp_path)
    dest = tmp_path / "dest"
    first = _render_to(tmp_path / "install-aliased.sh", aliased, "v-alias")
    second = _render_to(tmp_path / "install-files.sh", files, "v-alias")
    assert _run_install(first, dest, prefix, env).returncode == 0
    result = _run_install(second, dest, prefix, env)
    assert result.returncode == 0, result.stderr
    assert "already current" not in result.stdout
    assert (dest / "a").read_bytes() == b"1b\n2"
    assert (dest / "b").read_bytes() == b"3"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [("package", "bad package; rm -rf /"), ("version", "0.84.2$(touch pwned)")],
)
def test_descriptor_install_fields_constrain_their_charset(tmp_path, field: str, value: str) -> None:
    """Install fields land inside generated shell text, so the descriptor
    parse pins their charsets and a violation names the offending field."""
    source = Path(reef.harness.adapters.__file__).parent / "pi" / "descriptor.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["install"][field] = value
    target = tmp_path / "descriptor.yaml"
    target.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(DescriptorError, match=f"install '{field}'"):
        load_descriptor(target)


@pytest.mark.unit
def test_install_route_serves_the_script_for_head_and_pinned_versions(tmp_path) -> None:
    async def run() -> None:
        client = TestClient(TestServer(create_app(_dispatcher(tmp_path, MUTATIONS), inference_backend=_EchoBackend())))
        await client.start_server()
        try:
            first = await _gate_step(client)
            second = await _gate_step(client)
            response = await client.get(
                "/reef/harness/install", params={"adapter": "pi"}, headers={"x-reef-scenario": "delivery"}
            )
            assert response.status == 200
            assert response.content_type == "text/x-shellscript"
            script = await response.text()
            assert "npm install --prefix \"$PREFIX\" '@earendil-works/pi-coding-agent@0.84.2'" in script
            assert composition_checksum(second["files"]) in script  # head by default
            assert "marker marker rules" in script  # the composition rides inline
            response = await client.get(
                "/reef/harness/install",
                params={"adapter": "pi", "release_id": first["release_id"]},
                headers={"x-reef-scenario": "delivery"},
            )
            assert response.status == 200
            assert composition_checksum(first["files"]) in await response.text()
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize("headers", ({}, {"x-reef-scenario": "   "}))
def test_install_route_creates_a_randomly_named_harness_scenario_when_header_is_missing_or_empty(
    tmp_path, monkeypatch, headers
) -> None:
    dispatcher = _dispatcher(
        tmp_path,
        (),
        bootstrap_files={"pi-agent/AGENTS.md": "starter rules\n"},
    )

    monkeypatch.setattr(
        "reef.service.request_service._random_harness_scenario_name",
        lambda: "harness-0123456789ab",
    )

    async def run() -> None:
        client = TestClient(TestServer(create_app(dispatcher, inference_backend=_EchoBackend())))
        await client.start_server()
        try:
            response = await client.get("/reef/harness/install", params={"adapter": "pi"}, headers=headers)
            assert response.status == 200
            assert 'export REEF_HARNESS_SCENARIO="harness-0123456789ab"' in await response.text()
            created = dispatcher.get_or_create_scenario("harness-0123456789ab")
            assert created is not None and created.recipe == "evolve"
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_install_route_without_scenario_rejects_multiple_harness_recipes(tmp_path, monkeypatch) -> None:
    dispatcher = _dispatcher(
        tmp_path,
        (),
        bootstrap_files={"pi-agent/AGENTS.md": "starter rules\n"},
        recipe_names=("code-harness", "support-harness"),
    )
    monkeypatch.setattr(
        "reef.service.request_service._random_harness_scenario_name",
        lambda: "harness-0123456789ab",
    )

    async def run() -> None:
        client = TestClient(TestServer(create_app(dispatcher, inference_backend=_EchoBackend())))
        await client.start_server()
        try:
            response = await client.get("/reef/harness/install", params={"adapter": "pi"})
            assert response.status == 400
            assert "multiple harness recipes are available" in await response.text()
            assert not dispatcher.has_loaded("harness-0123456789ab")
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_install_route_without_scenario_returns_404_when_no_harness_recipe_exists(tmp_path) -> None:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    dispatcher = Dispatcher(
        RecipeRegistry({"weights": Recipe()}),
        InMemoryRepositoryBackend.factory(bootstrap, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
        agent_record_dir=None,
    )
    dispatcher.get_or_create_scenario("weights-only", "weights")

    async def run() -> None:
        client = TestClient(TestServer(create_app(dispatcher, inference_backend=_EchoBackend())))
        await client.start_server()
        try:
            response = await client.get("/reef/harness/install", params={"adapter": "pi"})
            assert response.status == 404
            assert "no harness recipes are available" in await response.text()
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_install_route_refuses_an_unknown_adapter_with_a_404_naming_it(tmp_path) -> None:
    async def run() -> None:
        client = TestClient(TestServer(create_app(_dispatcher(tmp_path, ()), inference_backend=_EchoBackend())))
        await client.start_server()
        try:
            response = await client.get(
                "/reef/harness/install", params={"adapter": "codex"}, headers={"x-reef-scenario": "delivery"}
            )
            assert response.status == 404
            assert "codex" in await response.text()
            # A missing adapter parameter is a caller error, not a lookup miss.
            response = await client.get("/reef/harness/install", headers={"x-reef-scenario": "delivery"})
            assert response.status == 400
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_install_route_answers_400_when_the_adapter_declares_no_install_section(tmp_path, monkeypatch) -> None:
    """A known adapter whose descriptor has no install section is a caller
    error, not a lookup miss: HTTP 400 naming the adapter."""
    from dataclasses import replace

    monkeypatch.setattr(
        "reef.service.request_service.get_adapter", lambda name: replace(get_adapter(name), install=None)
    )

    async def run() -> None:
        client = TestClient(
            TestServer(create_app(_dispatcher(tmp_path, MUTATIONS[:1]), inference_backend=_EchoBackend()))
        )
        await client.start_server()
        try:
            await _gate_step(client)
            response = await client.get(
                "/reef/harness/install", params={"adapter": "pi"}, headers={"x-reef-scenario": "delivery"}
            )
            assert response.status == 400
            text = await response.text()
            assert "'pi'" in text
            assert "no install section" in text
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_install_route_refuses_an_unknown_version_with_a_404_naming_it(tmp_path) -> None:
    async def run() -> None:
        client = TestClient(TestServer(create_app(_dispatcher(tmp_path, ()), inference_backend=_EchoBackend())))
        await client.start_server()
        try:
            response = await client.get(
                "/reef/harness/install",
                params={"adapter": "pi", "release_id": "no-such-version"},
                headers={"x-reef-scenario": "delivery"},
            )
            assert response.status == 404
            assert "no-such-version" in await response.text()
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_record_only_traffic_fires_a_step_and_publishes(tmp_path) -> None:
    """The report-free policy end to end through the real service: recorded
    inference traffic alone fills the batch, one evolve step runs on the
    unscored samples, and the harness read channel serves the winner. No
    report is ever posted."""

    async def run() -> None:
        dispatcher = _dispatcher(tmp_path, MUTATIONS[:1], batch_policy="records", batch_size=2)
        client = TestClient(TestServer(create_app(dispatcher, inference_backend=_EchoBackend())))
        await client.start_server()
        try:
            for prompt in ("first", "second"):
                response = await client.post(
                    "/v1/chat/completions",
                    headers={"x-reef-scenario": "delivery"},
                    json={"messages": [{"role": "user", "content": prompt}]},
                )
                assert response.status == 200
                await response.read()

            deadline = asyncio.get_running_loop().time() + _ASYNC_UPDATE_TIMEOUT_S
            while True:
                response = await client.get("/reef/harness", headers={"x-reef-scenario": "delivery"})
                if response.status == 200:
                    manifest = await response.json()
                    assert manifest["gate"]["published"] is True
                    assert "marker rules" in manifest["files"]["pi-agent/AGENTS.md"]
                    break
                assert response.status == 404
                await response.read()
                if asyncio.get_running_loop().time() >= deadline:
                    pytest.fail("record-driven step did not publish")
                await asyncio.sleep(_ASYNC_UPDATE_POLL_S)
        finally:
            await client.close()

    asyncio.run(run())
