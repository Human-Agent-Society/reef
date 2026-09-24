"""hermes adapter quirks: the config file, skill frontmatter, plugin manifests, and the boot scaffold.

Config nodes write ``config.yaml`` as a JSON object and ``finalize_render``
emits it as YAML. It also writes the ``.no-bundled-skills`` marker, so an
episode carries only the tree's skills instead of hermes's bundled catalog;
synthesizes the ``name`` and ``description`` frontmatter hermes requires on a
SKILL.md the node text left bare, under both skill roots; and, for every
rendered plugin, writes the ``plugin.yaml`` manifest and grants the plugin
in ``config.yaml`` (``plugins.enabled`` and the ``tools.override``
capability), since hermes discovers plugins but loads none without consent.
Rules render to ``SOUL.md``, which hermes reads as the agent's identity and
seeds with its own only while the file is absent, so the rules follow that
default identity instead of replacing it.

The traps a mutated config could reopen: the scanner download, the session
title call, and the snapshot the reader parses. A composition that flips any
of them is rejected at render, the same gate that rejects an invalid node.
"""

from __future__ import annotations

import json
from typing import Any

import yaml

from reef.harness.tree.render import RenderError

_CONFIG = "hermes/config.yaml"
_MARKER = "hermes/.no-bundled-skills"
_PLUGINS = "hermes/plugins/"
_SKILL_ROOTS = ("hermes/skills/", "hermes-commands/")
_SOUL = "hermes/SOUL.md"
#: The identity hermes writes to SOUL.md on first run when the file is absent (DEFAULT_SOUL_MD in
#: hermes_cli/default_soul.py of the pinned v2026.8.31; the real hermes smoke checks it against the install).
DEFAULT_IDENTITY = (
    "You are Hermes Agent, built by Nous Research. Be direct: match the "
    "length of your reply to the weight of the ask \u2014 a one-line question "
    "gets a one-line answer, and finished work gets a short report of what "
    "changed, what's verified, and what's left, never a replay of the "
    'process. No filler ("Great question," "I\'d be happy to"), no '
    "restating the request back, no re-summarizing what you already said, "
    "no narrating tool calls the user can see. Plain claims over "
    "adjectives; when unsure, say so plainly. Agree because it's right, "
    "not because the user said it. Depth is earned \u2014 give it when the "
    "user asks for detail, teaches, or the stakes demand it, not by "
    "default."
)

# hermes's boot scaffolds the home on every start: state directories, the
# runtime and cache files, lock files beside the state store, and the seed
# skill it always writes. Episode state, not residue.
cleanup_whitelist = (
    "hermes/cron/**",
    "hermes/sessions/**",
    "hermes/pairing/**",
    "hermes/hooks/**",
    "hermes/image_cache/**",
    "hermes/audio_cache/**",
    "hermes/sandboxes/**",
    "hermes/terminal-sessions/**",
    "hermes/bin/**",
    "hermes/models_dev_cache.json*",
)


def _with_frontmatter(path: str, text: str) -> str:
    if text.startswith("---\n"):
        return text
    name = path.split("/")[-2]
    first = next((line.strip().lstrip("#").strip() for line in text.splitlines() if line.strip()), "")
    header = {"name": name, "description": first[:200] or name}
    return "---\n" + yaml.dump(header, sort_keys=False, default_flow_style=False, allow_unicode=True) + "---\n" + text


def _granted(config: dict[str, Any], plugins: list[str]) -> dict[str, Any]:
    section = dict(config.get("plugins") or {})
    enabled = [name for name in section.get("enabled") or [] if isinstance(name, str)]
    section["enabled"] = enabled + [name for name in plugins if name not in enabled]
    entries = dict(section.get("entries") or {})
    for name in plugins:
        entry = dict(entries.get(name) or {})
        granted = [item for item in entry.get("granted_capabilities") or [] if isinstance(item, str)]
        entry["granted_capabilities"] = granted + ["tools.override"] * ("tools.override" not in granted)
        entries[name] = entry
    section["entries"] = entries
    return {**config, "plugins": section}


def finalize_render(files: dict[str, str]) -> dict[str, str]:
    config = json.loads(files[_CONFIG])
    soul = files.get(_SOUL)
    if soul is not None and not soul.startswith(DEFAULT_IDENTITY):
        # A rules entry adds to the agent's identity; written alone, it would be all of it.
        files[_SOUL] = f"{DEFAULT_IDENTITY}\n\n{soul}"
    if (config.get("approval") or {}).get("tirith_enabled") is not False:
        raise RenderError("hermes composition must keep approval.tirith_enabled false for benchmark episodes")
    if ((config.get("auxiliary") or {}).get("title_generation") or {}).get("enabled") is not False:
        raise RenderError(
            "hermes composition must keep auxiliary.title_generation.enabled false for benchmark episodes"
        )
    if (config.get("sessions") or {}).get("write_json_snapshots") is not True:
        raise RenderError(
            "hermes composition must keep sessions.write_json_snapshots true so Reef can read the trajectory"
        )
    plugins = sorted(
        path[len(_PLUGINS) :].split("/")[0]
        for path in files
        if path.startswith(_PLUGINS) and path.endswith("/__init__.py") and path.count("/") == 3
    )
    for name in plugins:
        files.setdefault(f"{_PLUGINS}{name}/plugin.yaml", f"name: {name}\nversion: '0.1'\ndescription: {name}\n")
    if plugins:
        config = _granted(config, plugins)
    files[_CONFIG] = yaml.dump(config, sort_keys=True, default_flow_style=False, allow_unicode=True)
    files[_MARKER] = ""
    for path, text in list(files.items()):
        if any(path.startswith(root) for root in _SKILL_ROOTS) and path.endswith("/SKILL.md"):
            files[path] = _with_frontmatter(path, text)
    return files
