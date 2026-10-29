"""Smoke of the real opencode binary: every part an evolved tree may hold reaches the model.

The hermetic suites check the rendered files. Here the pinned opencode runs against a stdlib OpenAI-compatible
stub that records each request, once as an episode and once as a person's session: the plugin installed with
``opencode plugin -g`` into an empty home, as ``reef-client harness connect`` does. Skipped unless
REEF_REAL_OPENCODE_BINARY points at an installed opencode (.github/workflows/harness-smoke.yml provides one).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import replace
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

pytestmark = pytest.mark.skipif(not REAL_OPENCODE, reason="REEF_REAL_OPENCODE_BINARY does not name a real opencode")

PROBE_TOOL = (
    "export const Probe = async () => ({ tool: { tidepool_probe: { "
    'description: "The reef smoke extension marker is TIDEPOOL-EXT-19.", args: {}, execute: async () => "ok" } } });\n'
)
PARTS = {
    "rules.md": "The reef smoke rules marker is TIDEPOOL-RULES-41.\n",
    "skills/probe/SKILL.md": "---\nname: probe\ndescription: The reef smoke skill marker is TIDEPOOL-SKILL-23.\n---\n",
    "commands/ship.md": "---\ndescription: Ship it.\n---\nThe reef smoke command marker is TIDEPOOL-COMMAND-31.\n",
    "extensions/probe.ts": PROBE_TOOL,
    # A config key: without webfetch, the request offers no webfetch tool.
    "config/primary.json": json.dumps({"permission": {"webfetch": "deny"}}),
}
MARKERS = ("TIDEPOOL-RULES-41", "TIDEPOOL-SKILL-23", "TIDEPOOL-COMMAND-31", "TIDEPOOL-EXT-19")


def person_env(home: Path) -> dict[str, str]:
    """A person's machine with nothing of opencode's own yet."""
    names = ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME")
    return {"HOME": str(home), **{name: str(home / name.lower()) for name in names}}


@pytest.mark.parametrize("form", ["episode", "layer"])
def test_real_opencode_takes_every_part_of_a_tree_in_an_episode_and_over_a_persons_own_config(
    tmp_path: Path, form: str
) -> None:
    server = StubOpenAI()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    descriptor = get_adapter("opencode")
    binding = ModelBinding(base_url=url, model=MODEL, api_key="dummy")
    try:
        if form == "episode":
            files = render_tree(HarnessTree(PARTS), descriptor, binding.config_overlays(descriptor))
            command = replace(descriptor, argv=(*descriptor.argv[:-1], "--command", "ship", "{prompt}"))
            result = run_episode(command, files, "now", binary=REAL_OPENCODE, timeout=120.0)
            assert result.exit_code == 0, (result.stdout, result.stderr)
        else:
            layer = layer_install(HarnessTree(PARTS), descriptor, binding, url, str(tmp_path / "install"))
            env = {**os.environ, **person_env(tmp_path / "home"), **write_layer(tmp_path / "install", layer)}
            for command in layer.register:
                subprocess.run([REAL_OPENCODE, *command], env=env, cwd=tmp_path, check=True, capture_output=True)
            argv = [REAL_OPENCODE, *layer.args, "run", "--format", "json", "--command", "ship", "now"]
            done = subprocess.run(
                argv, env=env, cwd=tmp_path, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120
            )
            assert done.returncode == 0, (done.stdout, done.stderr)
    finally:
        server.shutdown()
        server.server_close()
    requests = [json.loads(body) for body in server.bodies]
    main = next(request for request in requests if request.get("tools"))
    text = json.dumps(main)
    assert [marker for marker in MARKERS if marker not in text] == []
    tools = [tool["function"]["name"] for tool in main["tools"]]
    assert "tidepool_probe" in tools and "webfetch" not in tools
