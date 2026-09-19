"""Claude Code adapter quirks: boot artifacts and enforced config invariants.

Claude Code's boot writes local state beside the rendered config under
CLAUDE_CONFIG_DIR: a top-level ``.claude.json`` state file, a ``statsig/``
feature-gate cache, per-session ``todos/`` lists, and ``shell-snapshots/``.
The descriptor whitelists those so the episode inverse tolerates them and
reports anything else as residue.

``finalize_render`` enforces the traps a mutated ``settings.json`` could
reopen. The descriptor keeps a benchmark episode hermetic through
environment variables (auto-update, telemetry, and non-essential traffic all
off); a composition that sets ``settings.env`` to turn any of them back on,
or that flips ``includeCoAuthoredBy`` on, is rejected at render — the same
gate that rejects an invalid node.
"""

from __future__ import annotations

import json

from reef.harness.tree.render import RenderError

_CONFIG_PATH = "claude/settings.json"

# The env switches the descriptor relies on to keep episodes hermetic. A
# rendered settings.env that sets any of these to a falsey value would undo
# the descriptor's own env and let the binary phone home mid-campaign.
_HERMETIC_ENV = ("DISABLE_AUTOUPDATER", "DISABLE_TELEMETRY", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC")
_FALSEY = {"0", "false", "off", "no", ""}

cleanup_whitelist = (
    "claude/.claude.json",
    "claude/statsig",
    "claude/todos",
    "claude/shell-snapshots",
)


def finalize_render(files: dict[str, str]) -> dict[str, str]:
    config = json.loads(files[_CONFIG_PATH])
    if config.get("includeCoAuthoredBy") is True:
        raise RenderError("claude composition must keep includeCoAuthoredBy false for benchmark episodes")
    env = config.get("env")
    if isinstance(env, dict):
        for key in _HERMETIC_ENV:
            if key in env and str(env[key]).strip().lower() in _FALSEY:
                raise RenderError(f"claude composition must not re-enable {key} for benchmark episodes")
    return files
