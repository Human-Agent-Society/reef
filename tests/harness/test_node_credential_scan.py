"""The credential admission gate over every node body, not only config keys."""

from __future__ import annotations

import pytest

from reef.harness.tree.nodes import NODE_KINDS


def _validate(kind: str, config: dict) -> None:
    NODE_KINDS[kind](None, config)


CREDENTIAL_BODIES = [
    "sk-abcdef1234567890ABCDEFGH",
    "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
    "AKIAIOSFODNN7EXAMPLE",
    "-----BEGIN OPENSSH " + "PRIVATE " + "KEY-----\nabc",
]


@pytest.mark.parametrize("secret", CREDENTIAL_BODIES)
def test_rules_skill_command_extension_bodies_reject_inline_credentials(secret: str) -> None:
    for kind, config in (
        ("rules", {"text": f"Use this key {secret} to call the API."}),
        ("agent_command", {"name": "deploy", "text": f"export TOKEN={secret}"}),
        ("skill", {"name": "notes", "text": f"# notes\n\n{secret}"}),
        ("code_extension", {"name": "plugin", "code": f'const key = "{secret}";'}),
    ):
        with pytest.raises(ValueError, match="inline credential"):
            _validate(kind, config)


def test_ordinary_text_about_keys_is_not_a_credential() -> None:
    # Prose that mentions keys, and the tutorial's sk-local placeholder, pass.
    for kind, config in (
        ("rules", {"text": "Set your api key in the environment before running."}),
        ("skill", {"name": "auth", "text": "# auth\n\nUse REEF_TOKEN=sk-local for the demo."}),
        ("code_extension", {"name": "read", "code": "const key = process.env.API_KEY;"}),
    ):
        _validate(kind, config)


def test_an_extension_that_writes_to_the_sessions_terminal_is_refused_unless_it_guards_the_write() -> None:
    """An extension runs inside the harness process, which owns the screen in a session with a UI: an unguarded
    console write lands in a drawn frame and takes the person's input box with it."""
    for code in (
        'pi.on("agent_end", async () => { console.error("[voice] done"); });',
        'pi.on("agent_end", async () => { console.log("done"); });',
        'pi.on("agent_end", async () => { process.stdout.write("done\\n"); });',
        'pi.on("agent_end", async () => { process.stderr.write("done\\n"); });',
    ):
        with pytest.raises(ValueError, match=r"without a ctx\.hasUI guard"):
            _validate("code_extension", {"name": "voice", "code": code})
    # The refusal names the line, so the agent reading it knows which write to move.
    with pytest.raises(ValueError, match="at line 3 "):
        _validate("code_extension", {"name": "voice", "code": 'const a = 1;\n// note\nconsole.warn("x");\n'})


def test_an_extension_keeps_console_output_for_the_session_without_a_ui() -> None:
    """The guard is what separates the two paths, on the write's own line or the line above it; reading a
    command's own stdout or stderr is not a write to the session's."""
    for code in (
        'if (ctx.hasUI) ctx.ui.notify(m, "warning"); else console.error(m);',
        "if (!ctx.hasUI) {\n  console.error(message);\n  return;\n}",
        'const result = await pi.exec("git", ["status"]);\nif (result.code !== 0) throw new Error(result.stderr);',
        'ctx.ui.setStatus("voice", "voice: on");',
    ):
        _validate("code_extension", {"name": "voice", "code": code})
