"""The credential admission gate over every node body, not only config keys."""

from __future__ import annotations

import pytest

from reef.harness.nodes import NODE_KINDS


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
