"""hermes adapter quirks: the config file, skill frontmatter, plugin manifests, and the boot scaffold.

Config nodes write ``config.yaml`` as a JSON object and the render
emits it as YAML by its suffix, and the ``env`` target as ``.env`` lines, where hermes
reads a custom provider's key. It also writes the ``.no-bundled-skills`` marker, so an
episode carries only the tree's skills instead of hermes's bundled catalog; adds the commands
root to ``skills.external_dirs`` after the tree's own entries, so a tree
that sets the list still has its commands; and, for every rendered plugin,
writes the ``plugin.yaml`` manifest and grants the plugin in
``config.yaml`` (``plugins.enabled`` and the ``tools.override``
capability), since hermes discovers plugins but loads none without consent.
Rules render to ``SOUL.md``, which hermes reads as the agent's identity and
seeds with its own only while the file is absent, so the rules follow that
default identity instead of replacing it. ``process_skill`` and
``process_command`` give a SKILL.md the node text left bare the ``name`` and
``description`` frontmatter hermes requires, under both skill roots.

``process_config`` refuses a ``skills.external_dirs`` that is not a list of
strings, where it adds the commands root.
"""

from __future__ import annotations

from typing import Any

import yaml

from reef.harness.adapters.common.frontmatter import with_skill_frontmatter
from reef.harness.adapters.descriptor import AdapterRenderer
from reef.harness.tree.render import RenderError

_CONFIG = "hermes/config.yaml"
ENV_PATH = "hermes/.env"
_MARKER = "hermes/.no-bundled-skills"
_PLUGINS = "hermes/plugins/"
# The commands root beside the home: HERMES_HOME/.. in an episode, and REEF_HARNESS_DEST, the install root, in a
# reef-hermes session, whose home is a temp copy. hermes skips an entry that names no directory, such as the second
# one where REEF_HARNESS_DEST is unset. A config node's list replaces the one below it, so both follow the tree's own
# skills.external_dirs, which hermes reads as one entry when it is a string.
COMMAND_ROOTS = ("${HERMES_HOME}/../hermes-commands", "${REEF_HARNESS_DEST}/hermes-commands")
SOUL_PATH = "hermes/SOUL.md"
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


def nested_setting(config: dict[str, Any], *keys: str) -> object:
    """The value at ``keys`` in the config, None when a section on the way is absent or is not an object."""
    value: object = config
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def string_list(value: object, key: str) -> list[str]:
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
    enabled = string_list(section.get("enabled"), "plugins.enabled")
    section["enabled"] = enabled + [name for name in plugins if name not in enabled]
    entries = dict(entries or {})
    for name in plugins:
        entry = dict(entries.get(name) or {})
        granted = string_list(entry.get("granted_capabilities"), f"plugins.entries.{name}.granted_capabilities")
        entry["granted_capabilities"] = granted + ["tools.override"] * ("tools.override" not in granted)
        entries[name] = entry
    section["entries"] = entries
    return {**config, "plugins": section}


class HermesAdapterRenderer(AdapterRenderer):
    @staticmethod
    def process_config(path: str, config: dict[str, Any]) -> dict[str, Any]:
        if path != _CONFIG:
            return config
        # skills is an object here: the descriptor's check on skills.creation_nudge_interval keeps it one.
        tree_dirs = nested_setting(config, "skills", "external_dirs")
        external_dirs = string_list([tree_dirs] if isinstance(tree_dirs, str) else tree_dirs, "skills.external_dirs")
        config["skills"]["external_dirs"] = external_dirs + [
            root for root in COMMAND_ROOTS if root not in external_dirs
        ]
        return config

    @staticmethod
    def process_skill(path: str, text: str) -> str:
        return with_skill_frontmatter(path, text)

    @staticmethod
    def process_command(path: str, text: str) -> str:
        # A command is a skill under the commands root, so it needs the same frontmatter.
        return with_skill_frontmatter(path, text)

    @staticmethod
    def finalize_render(files: dict[str, str]) -> dict[str, str]:
        soul = files.get(SOUL_PATH)
        if soul is not None and not soul.startswith(DEFAULT_IDENTITY):
            # A rules entry adds to the agent's identity; written alone, it would be all of it.
            files[SOUL_PATH] = f"{DEFAULT_IDENTITY}\n\n{soul}"
        plugins = sorted(
            path[len(_PLUGINS) :].split("/")[0]
            for path in files
            if path.startswith(_PLUGINS) and path.endswith("/__init__.py") and path.count("/") == 3
        )
        for name in plugins:
            files.setdefault(f"{_PLUGINS}{name}/plugin.yaml", f"name: {name}\nversion: '0.1'\ndescription: {name}\n")
        if plugins:
            # hermes loads a rendered plugin only when config.yaml grants it.
            config = _granted(yaml.safe_load(files[_CONFIG]), plugins)
            files[_CONFIG] = yaml.dump(config, sort_keys=True, default_flow_style=False, allow_unicode=True)
        files[_MARKER] = ""
        return files
