"""opencode adapter quirks: boot mutations and enforced config invariants.

opencode's boot mutates its own composition directories: it injects a
``$schema`` key into config files that lack one (an in-place edit of a
rendered file, invisible to the residue scan), writes a ``.gitignore``, and
npm-installs ``@opencode-ai/plugin`` with a ``node_modules`` tree and
lockfiles. The whitelist below names exactly those artifacts so the episode
inverse tolerates them and nothing else.

``finalize_render`` enforces the traps a mutated config node could reopen: a
benchmark episode must never autoupdate the binary mid-campaign or upload a
share link, so a composition that overrides either is rejected at render -
the same gate that rejects an invalid node.
"""

from __future__ import annotations

import json

from reef.harness.render import RenderError

_CONFIG_PATH = "opencode/opencode.json"

cleanup_whitelist = (
    "opencode/.gitignore",
    "opencode/package.json",
    "opencode/package-lock.json",
    "opencode/bun.lock",
    "opencode/node_modules/**",
)


def finalize_render(files: dict[str, str]) -> dict[str, str]:
    config = json.loads(files[_CONFIG_PATH])
    if config.get("autoupdate") is not False:
        raise RenderError("opencode composition must keep autoupdate false for benchmark episodes")
    if config.get("share") != "disabled":
        raise RenderError("opencode composition must keep share disabled for benchmark episodes")
    return files
