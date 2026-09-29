"""Claude Code adapter quirks: boot artifacts and enforced config invariants.

Claude Code's boot writes local state beside the rendered config under
CLAUDE_CONFIG_DIR: a top-level ``.claude.json`` state file, a ``statsig/``
feature-gate cache, per-session ``todos/`` lists, and ``shell-snapshots/``.
The descriptor whitelists those so the episode inverse tolerates them and
reports anything else as residue.

``finalize_render`` enforces the traps a mutated ``settings.json`` could
reopen; a composition that breaks one is rejected at render, the same check
that rejects an invalid node. Claude Code copies ``settings.env`` over its
own environment. The episode env turns telemetry and non-essential traffic
off, so ``settings.env`` must not set either to ``0``, ``false``, ``off``,
``no`` or an empty value. The episode env and ``reef-claude`` set
``DISABLE_AUTOUPDATER`` and ``DISABLE_UPDATES`` so that Claude Code's updater
and its own ``update`` and ``install`` keep the pinned version; Claude Code
reads them as on only for ``1``, ``true``, ``yes`` or ``on``, so
``settings.env`` must not set either one at all. ``includeCoAuthoredBy`` must
stay off, and ``disableDeepLinkRegistration`` must stay ``"disable"``:
``reef-claude`` passes the same setting as ``--settings`` so that no tree can
point the person's ``claude-cli://`` link handler at the pinned binary, and
the default here covers a run where the person passes their own
``--settings``.
"""

from __future__ import annotations

import json

from reef.harness.tree.render import RenderError

_CONFIG_PATH = "claude/settings.json"

# The env switches the descriptor relies on to keep episodes hermetic. A
# rendered settings.env that sets any of these to a falsey value would undo
# the descriptor's own env and let the binary phone home mid-campaign.
_HERMETIC_ENV = ("DISABLE_TELEMETRY", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC")
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
        # The episode env and reef-claude set the updater switches, and a tree value other than 1, true, yes or on
        # would turn the updater back on. Windows matches env names in any case.
        for key in env:
            if key.upper() in ("DISABLE_AUTOUPDATER", "DISABLE_UPDATES"):
                raise RenderError(
                    f"claude composition must not set {key} in settings.env: the episode env and reef-claude "
                    "set it so Claude Code keeps the pinned version"
                )
        for key in _HERMETIC_ENV:
            if key in env and str(env[key]).strip().lower() in _FALSEY:
                raise RenderError(f"claude composition must not re-enable {key} for benchmark episodes")
    return files
