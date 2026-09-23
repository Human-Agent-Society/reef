"""The shipped update notice: a seedable entry per adapter.

The notice is composition, not runtime code: seeding the entry makes it part
of the tree the gate measures, and every pulled or installed copy carries it.
For pi it is a ``code_extension`` (``version_check.ts``) that offers to run
the update or skip in interactive mode, prints the instructions in headless
mode, and stays silent under ``PI_OFFLINE``, so hermetic benchmark episodes
make no network calls. For Claude Code it is a ``config`` entry: a
``SessionStart`` hook in ``settings.json`` that runs ``reef-claude notice``
through the wrapper the session exports, so an episode, which exports no
wrapper, runs nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reef.harness.adapters.descriptor import DescriptorError

VERSION_CHECK_ENTRY_ID = "reef-version-check"

_ASSETS = {
    "pi": Path(__file__).parents[1] / "adapters" / "pi" / "version_check.ts",
}

#: The Claude Code notice: one SessionStart hook. The wrapper's path arrives in the session environment
#: (``REEF_HARNESS_WRAPPER``, set by run_agent); without it, in an episode or a tree run by hand, the hook
#: exits 0 and prints nothing.
CLAUDE_NOTICE_HOOK = 'if [ -n "$REEF_HARNESS_WRAPPER" ]; then "$REEF_HARNESS_WRAPPER" notice --hook claude; fi'

#: The wrapper's own subcommands run from a session without a permission prompt: /reefine runs them in the
#: turn that invoked it, which its allowed-tools cover, and again in the turn after the person said to install.
CLAUDE_WRAPPER_RULE = "Bash(reef-claude *)"

_CONFIGS: dict[str, dict[str, Any]] = {
    "claude": {
        "target": "primary",
        "data": {
            "permissions": {"allow": [CLAUDE_WRAPPER_RULE]},
            "hooks": {
                "SessionStart": [
                    {"matcher": "startup|resume", "hooks": [{"type": "command", "command": CLAUDE_NOTICE_HOOK}]}
                ]
            },
        },
    },
}


def version_check_entry(adapter: str) -> dict[str, Any]:
    """The seed entry options for the adapter's shipped update notice."""
    asset = _ASSETS.get(adapter)
    if asset is not None:
        return {
            "id": VERSION_CHECK_ENTRY_ID,
            "name": "code_extension",
            "config": {"name": VERSION_CHECK_ENTRY_ID, "code": asset.read_text(encoding="utf-8")},
        }
    config = _CONFIGS.get(adapter)
    if config is None:
        raise DescriptorError(f"adapter {adapter!r} ships no version check extension")
    return {"id": VERSION_CHECK_ENTRY_ID, "name": "config", "config": config}


__all__ = ["CLAUDE_NOTICE_HOOK", "CLAUDE_WRAPPER_RULE", "VERSION_CHECK_ENTRY_ID", "version_check_entry"]
