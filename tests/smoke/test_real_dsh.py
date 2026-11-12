"""Smoke of the real dsh binary: render, run, and trajectory against the
real thing. The hermetic suites drive scripted fakes, so every format and
environment assumption about the real DeepSeek Harness lives here: a pinned
dsh runs one run against a stdlib OpenAI-compatible stub and must read
our rendered patch layer, reach the stub with the rules text and the bound
key, write a session our reader parses, and leave no unexplained residue.
Skipped unless REEF_REAL_DSH_BINARY points at an installed dsh
(.github/workflows/harness-smoke.yml provides one)."""

from __future__ import annotations

import os
import threading

import pytest
from tests.support.openai_stub import MODEL, StubOpenAI

from reef.harness.adapters import get_adapter
from reef.harness.tasks.model_binding import ModelBinding
from reef.harness.tasks.run import run_task
from reef.harness.tree import HarnessTree, render_tree

REAL_DSH = os.environ.get("REEF_REAL_DSH_BINARY", "")

pytestmark = pytest.mark.skipif(not REAL_DSH, reason="REEF_REAL_DSH_BINARY does not name a real dsh binary")

RULES_SENTENCE = "The reef smoke marker phrase is TIDEPOOL-STONE-41."


def test_real_dsh_run_renders_runs_and_cleans_up() -> None:
    server = StubOpenAI(lists_models=False)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    descriptor = get_adapter("dsh")
    try:
        binding = ModelBinding(
            base_url=f"http://127.0.0.1:{server.server_address[1]}", model=MODEL, api_key="smoke-key"
        )
        files = render_tree(HarnessTree({"rules.md": RULES_SENTENCE}), descriptor, binding.config_overlays(descriptor))
        # The task starts with a dash on purpose: it must pass both commander layers as a positional.
        result = run_task(descriptor, files, "- Reply with exactly the word READY", binary=REAL_DSH, timeout=180.0)
    finally:
        server.shutdown()
        server.server_close()

    assert result.exit_code == 0, (result.stdout, result.stderr)
    # (a) The rendered rules text and the bound key reached the model: the
    # whole chain (patch layer, .env, relocation env, real binary, request) held.
    assert any(RULES_SENTENCE in body for body in server.bodies), server.bodies
    assert any("- Reply with exactly the word READY" in body for body in server.bodies), server.bodies
    assert server.auth and server.auth[0] == "Bearer smoke-key", server.auth
    # (b) The session the real binary wrote parses through our reader and
    # carries the assistant message and the turn end.
    kinds = [event.get("type") for event in result.trajectory]
    assert kinds and kinds[0] == "session" and "assistant/message" in kinds and kinds[-1] == "turn/end", kinds
    # (c) The cleanup audit found nothing outside the declared whitelist.
    assert result.residue == ()


def test_real_dsh_runs_a_command_that_the_model_cannot_list() -> None:
    """A command whose text carries its own frontmatter stays a command dsh lists, whatever that frontmatter
    holds: the model's skill catalog lists the skill and not the commands, and a task that types each command gets
    its text."""
    server = StubOpenAI(lists_models=False)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    descriptor = get_adapter("dsh")
    # Each header after the first is one the adapter must repair or must not break: dsh ignores a camelCase
    # invocation key and a name or description that is not a string, reads Yes and 1:30 as strings where YAML 1.1
    # does not, reads a value tagged ! as a string, and allows \r\n after a fence.
    headers = {
        "marker": "---\nname: marker\ndescription: Reply with the reef marker\n---\n",
        "legacy": "---\nname: legacy\ndescription: Legacy keys\nuserInvocable: true\nmodelInvocable: false\n---\n",
        "yes": "---\nname: yes\ndescription: Yes\n---\n",
        "clock": "---\nname: clock\ndescription: 1:30\n---\n",
        "blank": "---\nname: 123\ndescription: ''\n---\n",
        "bang": '---\nname: ! bang\ndescription: ! "true\\n"\n---\n',
        "crlf": "---\r\nname: crlf\r\ndescription: Windows line ends\r\n---\r\n",
    }
    try:
        binding = ModelBinding(
            base_url=f"http://127.0.0.1:{server.server_address[1]}", model=MODEL, api_key="smoke-key"
        )
        skill = "---\nname: notes\ndescription: Keep short notes\n---\nKeep short notes.\n"
        tree = HarnessTree(
            {
                "skills/notes/SKILL.md": skill,
                **{f"commands/{name}.md": f"{header}Start with {name}-63.\n" for name, header in headers.items()},
            }
        )
        files = render_tree(tree, descriptor, binding.config_overlays(descriptor))
        task = " ".join(f"/{name}" for name in headers) + " say hi"
        result = run_task(descriptor, files, task, binary=REAL_DSH, timeout=180.0)
    finally:
        server.shutdown()
        server.server_close()

    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert any("- `notes`: Keep short notes" in body for body in server.bodies), server.bodies
    for name in headers:
        assert not any(f"- `{name}`:" in body for body in server.bodies), (name, server.bodies)
        assert any(f'<skill_content name=\\"{name}\\">' in body for body in server.bodies), (name, server.bodies)
        assert any(f"Start with {name}-63." in body for body in server.bodies), (name, server.bodies)
        # dsh read the header as frontmatter, so it is not part of the text the command sends.
        assert not any(f"name: {name}" in body for body in server.bodies), (name, server.bodies)
