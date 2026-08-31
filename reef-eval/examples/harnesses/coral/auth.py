"""Codex authentication setup for CORAL workers."""

from __future__ import annotations

from examples.harnesses.base import TideHarnessBase


async def write_codex_auth(
    agent: TideHarnessBase, environment, env: dict[str, str]
) -> None:
    command = """python - <<'PY'
import json
import os
from pathlib import Path

home = Path(os.environ["CODEX_HOME"])
home.mkdir(parents=True, exist_ok=True)
(home / "auth.json").write_text(json.dumps({"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]}))
PY"""
    await agent._checked(environment, command, env=env)
