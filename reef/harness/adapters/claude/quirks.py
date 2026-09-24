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
gate that rejects an invalid node. Claude Code copies ``settings.env`` over
its own environment, so a composition that turns off ``DISABLE_UPDATES`` is
rejected too: ``reef-claude`` sets it so that Claude Code's own ``update``
and ``install`` refuse to replace the pinned version. A composition that
drops ``disableDeepLinkRegistration: "disable"`` is rejected as well. ``reef-claude``
passes the same setting as ``--settings`` so that no tree can point the
person's ``claude-cli://`` link handler at the pinned binary; the default here
covers a run where the person passes their own ``--settings``.
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
    if config.get("disableDeepLinkRegistration") != "disable":
        raise RenderError(
            'claude composition must keep disableDeepLinkRegistration "disable" so a reef-claude session '
            "leaves the person's claude-cli:// handler alone"
        )
    env = config.get("env")
    if isinstance(env, dict):
        for key in _HERMETIC_ENV:
            if key in env and str(env[key]).strip().lower() in _FALSEY:
                raise RenderError(f"claude composition must not re-enable {key} for benchmark episodes")
        # reef-claude sets it so Claude Code's own update and install commands refuse to run.
        if "DISABLE_UPDATES" in env and str(env["DISABLE_UPDATES"]).strip().lower() in _FALSEY:
            raise RenderError("claude composition must keep DISABLE_UPDATES on so reef-claude keeps the pinned claude")
    return files
