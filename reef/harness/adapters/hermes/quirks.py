"""hermes adapter quirks: the config file, skill frontmatter, plugin manifests, and the boot scaffold.

Config nodes write ``config.yaml`` as a JSON object and ``finalize_render``
emits it as YAML. It also writes the ``.no-bundled-skills`` marker, so an
episode carries only the tree's skills instead of hermes's bundled catalog;
synthesizes the ``name`` and ``description`` frontmatter hermes requires on a
SKILL.md the node text left bare, under both skill roots; adds the commands
root to ``skills.external_dirs`` after the tree's own entries, so a tree
that sets the list still has its commands; and, for every rendered plugin,
writes the ``plugin.yaml`` manifest and grants the plugin in
``config.yaml`` (``plugins.enabled`` and the ``tools.override``
capability), since hermes discovers plugins but loads none without consent.

The traps a mutated config could reopen: the scanner download, the session
title call, the background review and the curator that write skills into the
tree, and the snapshot the reader parses. A composition that flips any of
them, puts a value that is not an object where a section holding one
belongs, or puts a value that is not a list of strings where Reef adds a
name, is rejected at render, the same gate that rejects an invalid node.
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
# The commands root beside the home: HERMES_HOME/.. in an episode, and REEF_HARNESS_DEST, the install root, in a
# reef-hermes session, whose home is a temp copy. hermes skips an entry that names no directory, such as the second
# one where REEF_HARNESS_DEST is unset. A config node's list replaces the one below it, so both follow the tree's own
# skills.external_dirs, which hermes reads as one entry when it is a string.
_COMMAND_ROOTS = ("${HERMES_HOME}/../hermes-commands", "${REEF_HARNESS_DEST}/hermes-commands")

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


def _setting(config: dict[str, Any], *keys: str) -> object:
    """The value at ``keys`` in the config, None when a section on the way is absent or is not an object."""
    value: object = config
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def _strings(value: object, key: str) -> list[str]:
    """The strings at ``key``, empty when it is absent; a string or any other value is refused, not read."""
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RenderError(f"hermes composition must keep {key} a list of strings")
    return value


def _granted(config: dict[str, Any], plugins: list[str]) -> dict[str, Any]:
    section = config.get("plugins")
    entries = section.get("entries") if isinstance(section, dict) else None
    if (
        not isinstance(section, dict | None)
        or not isinstance(entries, dict | None)
        or any(not isinstance((entries or {}).get(name), dict | None) for name in plugins)
    ):
        raise RenderError(
            "hermes composition must keep plugins, plugins.entries and each rendered plugin's entry objects, "
            "so the plugin can be granted"
        )
    section = dict(section or {})
    enabled = _strings(section.get("enabled"), "plugins.enabled")
    section["enabled"] = enabled + [name for name in plugins if name not in enabled]
    entries = dict(entries or {})
    for name in plugins:
        entry = dict(entries.get(name) or {})
        granted = _strings(entry.get("granted_capabilities"), f"plugins.entries.{name}.granted_capabilities")
        entry["granted_capabilities"] = granted + ["tools.override"] * ("tools.override" not in granted)
        entries[name] = entry
    section["entries"] = entries
    return {**config, "plugins": section}


def finalize_render(files: dict[str, str]) -> dict[str, str]:
    config = json.loads(files[_CONFIG])
    if _setting(config, "security", "tirith_enabled") is not False:
        raise RenderError("hermes composition must keep security.tirith_enabled false for benchmark episodes")
    if _setting(config, "auxiliary", "title_generation", "enabled") is not False:
        raise RenderError(
            "hermes composition must keep auxiliary.title_generation.enabled false for benchmark episodes"
        )
    memory_nudge_interval = _setting(config, "memory", "nudge_interval")
    skill_nudge_interval = _setting(config, "skills", "creation_nudge_interval")
    if memory_nudge_interval != 0 or skill_nudge_interval != 0:
        raise RenderError(
            "hermes composition must keep memory.nudge_interval and skills.creation_nudge_interval 0, "
            "so no background review makes model calls or writes skills"
        )
    if _setting(config, "curator", "enabled") is not False:
        raise RenderError("hermes composition must keep curator.enabled false, so the curator leaves the skills alone")
    if _setting(config, "sessions", "write_json_snapshots") is not True:
        raise RenderError(
            "hermes composition must keep sessions.write_json_snapshots true so Reef can read the trajectory"
        )
    # skills is an object here: the nudge check above read skills.creation_nudge_interval from it.
    tree_dirs = _setting(config, "skills", "external_dirs")
    external_dirs = _strings([tree_dirs] if isinstance(tree_dirs, str) else tree_dirs, "skills.external_dirs")
    config["skills"]["external_dirs"] = external_dirs + [root for root in _COMMAND_ROOTS if root not in external_dirs]
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
