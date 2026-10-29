"""Smoke of the real opencode binary: every part an evolved tree may hold reaches the model.

The hermetic suites check the rendered files. Here the pinned opencode runs against a stdlib OpenAI-compatible stub
that records each request, once as an episode and once as a person's session: the layer over an empty own config
directory, as ``reef-client harness run`` starts it. An episode passes its task as a message, and ``opencode run``
expands a command only from ``--command``, so the command is checked where a person runs it. Skipped unless
REEF_REAL_OPENCODE_BINARY points at an installed opencode (.github/workflows/harness-smoke.yml provides one).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest
from tests.support.harness_render import write_layer
from tests.support.openai_stub import MODEL, StubOpenAI

from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.episodes.run import run_episode
from reef.harness.sync import layer_install
from reef.harness.tree import HarnessTree, render_tree

REAL_OPENCODE = os.environ.get("REEF_REAL_OPENCODE_BINARY", "")

pytestmark = pytest.mark.skipif(
    not REAL_OPENCODE, reason="REEF_REAL_OPENCODE_BINARY does not name a real opencode binary"
)

#: A plugin that adds one tool, whose description carries the extension's marker.
PROBE_PLUGIN = (
    'import { tool } from "@opencode-ai/plugin";\n'
    "export const Probe = async () => ({ tool: { tidepool_probe: tool({ "
    'description: "The reef smoke extension marker is TIDEPOOL-EXT-19.", args: {}, '
    'async execute() { return "ok"; } }) } });\n'
)
#: A tree with a marker in each part opencode's adapter evolves; the config sets the build agent's prompt.
PARTS = {
    "rules.md": "The reef smoke rules marker is TIDEPOOL-RULES-41.\n",
    "skills/probe/SKILL.md": "---\nname: probe\ndescription: The reef smoke skill marker is TIDEPOOL-SKILL-23.\n---\n",
    "commands/ship.md": "---\ndescription: Ship\n---\nThe reef smoke command marker is TIDEPOOL-COMMAND-31.\n",
    "extensions/probe.ts": PROBE_PLUGIN,
    "config/primary.json": json.dumps(
        {"agent": {"build": {"prompt": "The reef smoke config marker is TIDEPOOL-CONFIG-53."}}}
    ),
}
EPISODE_MARKERS = ("TIDEPOOL-RULES-41", "TIDEPOOL-SKILL-23", "TIDEPOOL-EXT-19", "TIDEPOOL-CONFIG-53")


@pytest.mark.parametrize("form", ["episode", "layer"])
def test_real_opencode_takes_every_part_of_a_tree_in_an_episode_and_over_a_persons_own_config(
    tmp_path: Path, form: str
) -> None:
    """Every part reaches the model, in an episode and on a person's machine, where opencode loads the person's own
    config directory, empty here, and the release's directory over it."""
    server = StubOpenAI()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    descriptor = get_adapter("opencode")
    try:
        if form == "episode":
            binding = ModelBinding(base_url=url, model=MODEL, api_key="dummy")
            files = render_tree(HarnessTree(PARTS), descriptor, binding.config_overlays(descriptor))
            result = run_episode(descriptor, files, "Reply READY", binary=REAL_OPENCODE, timeout=240.0)
            assert result.exit_code == 0, (result.stdout, result.stderr)
            assert result.trajectory and result.residue == ()
            markers = EPISODE_MARKERS
        else:
            install, home = tmp_path / "install", tmp_path / "home"
            binding = ModelBinding(base_url=url, model=MODEL)
            layer = layer_install(HarnessTree(PARTS), descriptor, binding, url, str(install))
            (home / ".config" / "opencode").mkdir(parents=True)
            own = {
                "HOME": str(home),
                "XDG_CONFIG_HOME": str(home / ".config"),
                "XDG_DATA_HOME": str(home / ".local" / "share"),
                "XDG_CACHE_HOME": str(home / ".cache"),
                "XDG_STATE_HOME": str(home / ".local" / "state"),
            }
            env = {**os.environ, **own, **write_layer(install, layer)}
            argv = [REAL_OPENCODE, *layer.args, "run", "--format", "json", "--auto", "--command", "ship"]
            done = subprocess.run(
                argv, env=env, cwd=tmp_path, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=240
            )
            assert done.returncode == 0, (done.stdout, done.stderr)
            markers = (*EPISODE_MARKERS, "TIDEPOOL-COMMAND-31")
    finally:
        server.shutdown()
        server.server_close()
    requests = "".join(server.bodies)
    assert [marker for marker in markers if marker not in requests] == []
