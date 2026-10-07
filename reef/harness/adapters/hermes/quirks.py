"""hermes adapter quirks: the config file, skill frontmatter, plugin manifests, and the boot scaffold.

Config nodes write ``config.yaml`` as a JSON object and the render
emits it as YAML by its suffix, and the ``env`` target as ``.env`` lines, where hermes
reads a custom provider's key. It also writes the ``.no-bundled-skills`` marker, so an
episode carries only the tree's skills instead of hermes's bundled catalog;
synthesizes the ``name`` and ``description`` frontmatter hermes requires on a
SKILL.md the node text left bare, under both skill roots; adds the commands
root to ``skills.external_dirs`` after the tree's own entries, so a tree
that sets the list still has its commands; and, for every rendered plugin,
writes the ``plugin.yaml`` manifest and grants the plugin in
``config.yaml`` (``plugins.enabled`` and the ``tools.override``
capability), since hermes discovers plugins but loads none without consent.
Rules render to ``SOUL.md``, which hermes reads as the agent's identity and
seeds with its own only while the file is absent, so the rules follow that
default identity instead of replacing it.

The traps a mutated config could reopen: the scanner download, the session
title call, the background review and the curator that write skills into the
tree, and the snapshot the reader parses. A composition that flips any of
them, puts a value that is not an object where a section holding one
belongs, or puts a value that is not a list of strings where Reef adds a
name, is rejected at render, the same check that rejects an invalid node.

Every model call stays on Reef's model binding. The binding writes the
``model`` section (a ``custom`` provider, the endpoint, the model and a
literal key), renders after the tree and wins every key it writes. Its key
is a credential, which a tree cannot hold because admission refuses an
inline credential, so those keys pass only beside the binding's key. The
other names hermes reads for the model, the endpoint or the key (in
``model``, and at the top level, which hermes moves into ``model``), the
model aliases, the sections that name other providers, their endpoints and
credential commands (``providers``, ``custom_providers``), the fallback
models (``fallback_model``, ``fallback_providers``) and the mixture of
agents presets, and a provider, an endpoint, a credential, a model or a
fallback chain set for an auxiliary task, for delegation, for cron or for
the curator, are the tree choosing where calls go, and are refused. An
auxiliary task may keep ``auto`` or ``main``, which run it on the main
model. The binding sets no transport, so a tree's ``api_mode`` (in
``model`` or in any of those sections) and ``model.openai_runtime`` are
refused too: for the ``custom`` provider two api modes leave the bound
endpoint, ``bedrock_converse`` for AWS Bedrock and ``codex_app_server`` for
a ``codex app-server`` subprocess. A request body a tree passes to the
bound endpoint (``extra_body``, ``delegation.request_overrides``) may not
name a model either.

These checks cover the config the tree renders. Reef's proxy forwards the
``model`` and ``models`` a request sends as they are, so a request that other
code builds with the rendered key, such as a tool the model runs, can still
name another model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import yaml

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

#: Other ``model`` keys hermes reads for the model, the endpoint or the key, and the model aliases a switch resolves.
MODEL_ALIAS_KEYS = ("aliases", "api_base", "api_key_env", "key_cmd", "key_env", "model", "name")
#: The ``model`` keys hermes reads for the transport; the binding writes neither.
TRANSPORT_KEYS = ("api_mode", "openai_runtime")
#: Top-level keys hermes moves into ``model`` (the provider and the endpoint), and the model aliases with their routes.
ROOT_ROUTE_KEYS = ("api_base", "base_url", "model_aliases", "provider")
#: Sections that name other providers, their endpoints and credential commands, or the fallback models.
PROVIDER_SECTIONS = ("custom_providers", "fallback_model", "fallback_providers", "providers")
#: The mixture of agents keys that name its models: the presets, and the older flat form of one preset.
MOA_KEYS = ("aggregator", "presets", "reference_models")
#: The keys of an auxiliary task, of delegation, of cron and of the curator that choose its provider, endpoint,
#: credential, transport or model, or a fallback chain of those.
ROUTE_KEYS = (
    "api_key",
    "api_key_env",
    "api_mode",
    "base_url",
    "fallback_chain",
    "key_cmd",
    "key_env",
    "model",
    "model_provider",
    "provider",
)
#: Request body fields that name the model: ``model`` replaces the bound one, and OpenRouter reads ``models`` as
#: fallback models.
BODY_MODEL_KEYS = ("model", "models")
#: The providers an auxiliary task may keep: hermes runs both on the main model.
MAIN_MODEL_PROVIDERS = ("auto", "main")
#: The older compression keys hermes moves into ``auxiliary.compression`` (provider, model, base_url).
COMPRESSION_KEYS = ("summary_base_url", "summary_model", "summary_provider")

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


def is_set(value: object) -> bool:
    """Whether hermes reads ``value`` as set: an empty string, list or object is its default."""
    return value is not None and value != "" and value != [] and value != {}


def section_of(config: dict[str, Any], name: str, where: str = "") -> dict[str, Any]:
    """The config section ``name`` (``where`` names it in a refusal), empty when absent; hermes reads it as an object."""
    section = config.get(name)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise RenderError(f"hermes {where or name} must be an object, got {type(section).__name__}")
    return section


def check_model_route(config: dict[str, Any]) -> None:
    """Refuse a provider, an endpoint, a credential or a model the binding did not write."""
    refusal = "Reef's model binding chooses the provider, the endpoint, the credential and the model"
    model = config.get("model")
    if model is not None and not isinstance(model, dict):
        raise RenderError(f"hermes composition must not set model: {refusal}")
    model = model or {}
    # hermes reads model.model and model.name as the model (one reader prefers model.model to model.default),
    # model.api_base as the endpoint and the key names as the key; the binding writes none of them.
    for key in MODEL_ALIAS_KEYS:
        if is_set(model.get(key)):
            raise RenderError(f"hermes composition must not set model.{key}: {refusal}")
    # The binding binds the chat completions API. For its custom provider hermes takes model.api_mode in any
    # spelling it knows, and bedrock_converse (an AWS Bedrock client) and codex_app_server (a codex subprocess)
    # leave the bound endpoint; openai_runtime picks the codex subprocess for the openai providers.
    for key in TRANSPORT_KEYS:
        if is_set(model.get(key)):
            raise RenderError(f"hermes composition must not set model.{key}: {refusal}")
    for name in (*ROOT_ROUTE_KEYS, *PROVIDER_SECTIONS):
        if is_set(config.get(name)):
            raise RenderError(f"hermes composition must not set {name}: {refusal}")
    moa = section_of(config, "moa")
    for key in MOA_KEYS:
        if is_set(moa.get(key)):
            raise RenderError(f"hermes composition must not set moa.{key}: {refusal}")
    compression = section_of(config, "compression")
    for key in COMPRESSION_KEYS:
        value = compression.get(key)
        if is_set(value) and not (key == "summary_provider" and value in MAIN_MODEL_PROVIDERS):
            raise RenderError(f"hermes composition must not set compression.{key}: {refusal}")
    auxiliary = section_of(config, "auxiliary")
    if is_set(auxiliary.get("openrouter_model")):
        raise RenderError(f"hermes composition must not set auxiliary.openrouter_model: {refusal}")
    routed = [(f"auxiliary.{task}", settings) for task, settings in auxiliary.items() if isinstance(settings, dict)]
    routed += [(name, section_of(config, name)) for name in ("delegation", "cron")]
    # The curator still reads its older per task section, curator.auxiliary.
    routed.append(("curator.auxiliary", section_of(section_of(config, "curator"), "auxiliary", "curator.auxiliary")))
    for where, settings in routed:
        for key in ROUTE_KEYS:
            value = settings.get(key)
            on_main_model = where.startswith("auxiliary.") and key == "provider" and value in MAIN_MODEL_PROVIDERS
            if is_set(value) and not on_main_model:
                raise RenderError(f"hermes composition must not set {where}.{key}: {refusal}")
        if settings.get("prefer_fast_model") is True:
            raise RenderError(f"hermes composition must not set {where}.prefer_fast_model: {refusal}")
        check_body(settings.get("extra_body"), f"{where}.extra_body", refusal)
    # Delegation sends request_overrides with each child call: its keys are call arguments, and its extra_body
    # joins the request body.
    overrides = section_of(config, "delegation").get("request_overrides")
    check_body(overrides, "delegation.request_overrides", refusal)
    if isinstance(overrides, dict):
        check_body(overrides.get("extra_body"), "delegation.request_overrides.extra_body", refusal)


def check_body(body: object, where: str, refusal: str) -> None:
    """Refuse a request body that names a model; hermes skips a body that is not an object."""
    if not isinstance(body, dict):
        return
    for key in BODY_MODEL_KEYS:
        if key in body:
            raise RenderError(f"hermes composition must not set {where}.{key}: {refusal}")


class HermesAdapterRenderer(AdapterRenderer):
    @staticmethod
    def process_config(path: str, config: dict[str, Any]) -> dict[str, Any]:
        if path != _CONFIG:
            return config
        if nested_setting(config, "security", "tirith_enabled") is not False:
            raise RenderError("hermes composition must keep security.tirith_enabled false for benchmark episodes")
        if nested_setting(config, "auxiliary", "title_generation", "enabled") is not False:
            raise RenderError(
                "hermes composition must keep auxiliary.title_generation.enabled false for benchmark episodes"
            )
        memory_nudge_interval = nested_setting(config, "memory", "nudge_interval")
        skill_nudge_interval = nested_setting(config, "skills", "creation_nudge_interval")
        if memory_nudge_interval != 0 or skill_nudge_interval != 0:
            raise RenderError(
                "hermes composition must keep memory.nudge_interval and skills.creation_nudge_interval 0, "
                "so no background review makes model calls or writes skills"
            )
        if nested_setting(config, "curator", "enabled") is not False:
            raise RenderError(
                "hermes composition must keep curator.enabled false, so the curator leaves the skills alone"
            )
        if nested_setting(config, "sessions", "write_json_snapshots") is not True:
            raise RenderError(
                "hermes composition must keep sessions.write_json_snapshots true so Reef can read the trajectory"
            )
        # skills is an object here: the nudge check above read skills.creation_nudge_interval from it.
        tree_dirs = nested_setting(config, "skills", "external_dirs")
        external_dirs = string_list([tree_dirs] if isinstance(tree_dirs, str) else tree_dirs, "skills.external_dirs")
        config["skills"]["external_dirs"] = external_dirs + [
            root for root in COMMAND_ROOTS if root not in external_dirs
        ]
        return config

    @staticmethod
    def process_skill(path: str, text: str) -> str:
        return _with_frontmatter(path, text)

    @staticmethod
    def process_command(path: str, text: str) -> str:
        # A command is a skill under the commands root, so it needs the same frontmatter.
        return _with_frontmatter(path, text)

    @staticmethod
    def check_model_route(
        configs: Mapping[str, Mapping[str, Any]], skills: Mapping[str, str], commands: Mapping[str, str]
    ) -> None:
        check_model_route(dict(configs[_CONFIG]))

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
