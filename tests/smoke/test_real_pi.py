"""Smoke of the real pi binary: render, episode, and trajectory against the
real thing. The hermetic suites drive a scripted fake, so every format and
environment assumption about the real binary lives here: a pinned pi runs one
episode against a stdlib OpenAI-compatible stub and must read our rendered
tree, reach the stub with the rules text, write a session our reader
parses, and leave no unexplained residue. Skipped unless REEF_REAL_PI_BINARY
points at an installed pi (.github/workflows/harness-smoke.yml provides one)."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest
from tests.support.harness_render import write_layer
from tests.support.openai_stub import MODEL, StubOpenAI

from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.episodes.run import run_episode
from reef.harness.sync import layer_install
from reef.harness.tree import HarnessTree, render_tree

REAL_PI = os.environ.get("REEF_REAL_PI_BINARY", "")

pytestmark = pytest.mark.skipif(not REAL_PI, reason="REEF_REAL_PI_BINARY does not name a real pi binary")

RULES_SENTENCE = "The reef smoke marker phrase is TIDEPOOL-STONE-41."


def capturing_wrapper(tmp_path: Path, capture: Path) -> str:
    """Wrap the real pi so the raw session file survives the root teardown.

    ``run_episode`` deletes the episode root before returning, so the raw
    bytes pi wrote are copied out by the wrapper first: they are the artifact
    the workflow uploads and the refresh source of the hermetic fixture
    (tests/harness/data/pi_session_real/README.md).
    """
    wrapper = tmp_path / "pi-capturing"
    wrapper.write_text(
        "#!/bin/sh\n"
        f'"{REAL_PI}" "$@"\n'
        "status=$?\n"
        f'find "$PI_CODING_AGENT_SESSION_DIR" -name "*.jsonl" -exec cp {{}} "{capture}" \\;\n'
        "exit $status\n"
    )
    wrapper.chmod(0o755)
    return str(wrapper)


def test_real_pi_episode_renders_runs_and_cleans_up(tmp_path: Path) -> None:
    server = StubOpenAI()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        # The model comes through Reef's binding, as in every episode: a tree may not name a provider of its own.
        binding = ModelBinding(base_url=f"http://127.0.0.1:{server.server_address[1]}", model=MODEL, api_key="dummy")
        descriptor = get_adapter("pi")
        files = render_tree(HarnessTree({"rules.md": RULES_SENTENCE}), descriptor, binding.config_overlays(descriptor))
        capture = Path(os.environ.get("REEF_REAL_PI_SESSION_OUT", tmp_path / "real-pi-session.jsonl"))
        binary = capturing_wrapper(tmp_path, capture)
        result = run_episode(
            get_adapter("pi"), files, "Reply with exactly the word READY", binary=binary, timeout=120.0
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result.exit_code == 0, (result.stdout, result.stderr)
    # (a) The rendered rules text reached the model: the whole render chain
    # (tree -> files -> relocation env -> real binary -> request) held.
    assert any(RULES_SENTENCE in body for body in server.bodies), server.bodies
    # (b) The session the real binary wrote parses through our reader and
    # carries an assistant message.
    assert result.trajectory
    roles = [event["message"]["role"] for event in result.trajectory if event.get("type") == "message"]
    assert "assistant" in roles, result.trajectory
    # (c) The cleanup audit found nothing outside the declared whitelist.
    assert result.residue == ()
    # The wrapper captured the raw session file for the workflow's artifact.
    assert capture.is_file() and capture.stat().st_size > 0


#: A tree with a marker in each part pi's adapter evolves. The prompt runs the command, so its text is the request's.
PROBE_TOOL = (
    'export default function (pi: any) { pi.registerTool({ name: "tidepool_probe", label: "probe", '
    'description: "The reef smoke extension marker is TIDEPOOL-EXT-19.", parameters: { type: "object", '
    'properties: {} }, async execute() { return { content: [{ type: "text", text: "ok" }], details: {} }; } }); }\n'
)
PARTS = {
    "rules.md": RULES_SENTENCE,
    "skills/probe/SKILL.md": "---\nname: probe\ndescription: The reef smoke skill marker is TIDEPOOL-SKILL-23.\n---\n",
    "commands/ship.md": "The reef smoke command marker is TIDEPOOL-COMMAND-31.\n",
    "extensions/probe.ts": PROBE_TOOL,
}
MARKERS = ("TIDEPOOL-STONE-41", "TIDEPOOL-SKILL-23", "TIDEPOOL-COMMAND-31", "TIDEPOOL-EXT-19")


def first_request(tmp_path: Path, form: str, tree: HarnessTree, prompt: str) -> dict[str, Any]:
    """The first request pi sends for ``tree``, as an episode or on a person's machine, where pi loads the person's
    own directory, empty here, and the release as the adapter's layer over it."""
    server = StubOpenAI()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    descriptor = get_adapter("pi")
    try:
        if form == "episode":
            binding = ModelBinding(base_url=url, model=MODEL, api_key="dummy")
            files = render_tree(tree, descriptor, binding.config_overlays(descriptor))
            result = run_episode(descriptor, files, prompt, binary=REAL_PI, timeout=120.0)
            assert result.exit_code == 0, (result.stdout, result.stderr)
        else:
            layer = layer_install(tree, descriptor, ModelBinding(base_url=url, model=MODEL), url, str(tmp_path))
            env = {**os.environ, **write_layer(tmp_path, layer), "PI_CODING_AGENT_DIR": str(tmp_path / "own")}
            for command in layer.register:
                subprocess.run([REAL_PI, *command], env=env, check=True, capture_output=True)
            argv = [REAL_PI, *layer.args, "--offline", "--mode", "json", "-p", prompt]
            done = subprocess.run(argv, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120)
            assert done.returncode == 0, done.stderr
    finally:
        server.shutdown()
        server.server_close()
    return json.loads(server.bodies[0])


@pytest.mark.parametrize("form", ["episode", "layer"])
def test_real_pi_takes_every_part_of_a_tree_in_an_episode_and_over_a_persons_own_config(
    tmp_path: Path, form: str
) -> None:
    request = json.dumps(first_request(tmp_path, form, HarnessTree(PARTS), "/ship"))
    assert [marker for marker in MARKERS if marker not in request] == []


@pytest.mark.parametrize("form", ["episode", "layer"])
def test_real_pi_offers_no_built_in_tool_when_the_settings_name_none(tmp_path: Path, form: str) -> None:
    tree = HarnessTree({"extensions/probe.ts": PROBE_TOOL, "config/primary.json": json.dumps({"defaultTools": []})})
    request = first_request(tmp_path, form, tree, "Reply with exactly the word READY")
    assert [tool["function"]["name"] for tool in request["tools"]] == ["tidepool_probe"]
