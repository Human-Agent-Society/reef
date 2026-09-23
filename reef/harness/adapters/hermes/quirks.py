"""hermes adapter quirks: the config file, skill frontmatter, plugin manifests, and the boot scaffold.

Config nodes write ``config.yaml`` as a JSON object and ``finalize_render``
emits it as YAML. It also writes the ``.no-bundled-skills`` marker, so an
episode carries only the tree's skills instead of hermes's bundled catalog;
synthesizes the ``name`` and ``description`` frontmatter hermes requires on a
SKILL.md the node text left bare, under both skill roots; and, for every
rendered plugin, writes the ``plugin.yaml`` manifest and grants the plugin
in ``config.yaml`` (``plugins.enabled`` and the ``tools.override``
capability), since hermes discovers plugins but loads none without consent.

The traps a mutated config could reopen: the scanner download, the session
title call, and the snapshot the reader parses. A composition that flips any
of them is rejected at render, the same gate that rejects an invalid node.

Every model call stays on Reef's model binding. The binding writes the
``model`` section (a ``custom`` provider, the endpoint, the model and a
literal key), renders after the tree and wins every key it writes. Its key
is a credential, which a tree cannot hold because admission refuses an
inline credential, so those keys pass only beside the binding's key. The
sections that name other providers, their endpoints and credential commands
(``providers``, ``custom_providers``), the fallback models
(``fallback_model``, ``fallback_providers``) and the mixture of agents
presets, and a provider, an endpoint, a credential or a model set for an
auxiliary task, for delegation or for cron, are the tree choosing where
calls go, and are refused. An auxiliary task may keep ``auto`` or ``main``,
which run it on the main model.
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

#: The model keys Reef's binding writes, and among them its credential.
BINDING_MODEL_KEYS = ("api_key", "base_url", "default", "provider")
BINDING_CREDENTIAL = "api_key"
#: Sections that name other providers, their endpoints and credential commands, or the fallback models.
PROVIDER_SECTIONS = ("custom_providers", "fallback_model", "fallback_providers", "providers")
#: The keys of an auxiliary task, of delegation and of cron that choose its provider, endpoint, credential or model.
ROUTE_KEYS = ("api_key", "base_url", "key_cmd", "key_env", "model", "model_provider", "provider")
#: The providers an auxiliary task may keep: hermes runs both on the main model.
MAIN_MODEL_PROVIDERS = ("auto", "main")

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


def is_set(value: object) -> bool:
    """Whether hermes reads ``value`` as set: an empty string, list or object is its default."""
    return value is not None and value != "" and value != [] and value != {}


def section_of(config: dict[str, Any], name: str) -> dict[str, Any]:
    """The config section ``name``, empty when absent; hermes reads each one as an object."""
    section = config.get(name)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise RenderError(f"hermes {name} must be an object, got {type(section).__name__}")
    return section


def check_model_route(config: dict[str, Any]) -> None:
    """Refuse a provider, an endpoint, a credential or a model the binding did not write."""
    refusal = "Reef's model binding chooses the provider, the endpoint, the credential and the model"
    model = config.get("model")
    if model is not None and not isinstance(model, dict):
        raise RenderError(f"hermes composition must not set model: {refusal}")
    model = model or {}
    credential = model.get(BINDING_CREDENTIAL)
    bound = isinstance(credential, str) and bool(credential.strip())
    # model.model names the model where model.default is empty; the binding always writes model.default.
    for key in (*BINDING_MODEL_KEYS, "model"):
        if key in model and (key == "model" or not bound):
            raise RenderError(f"hermes composition must not set model.{key}: {refusal}")
    for name in PROVIDER_SECTIONS:
        if is_set(config.get(name)):
            raise RenderError(f"hermes composition must not set {name}: {refusal}")
    if is_set(section_of(config, "moa").get("presets")):
        raise RenderError(f"hermes composition must not set moa.presets: {refusal}")
    auxiliary = section_of(config, "auxiliary")
    if is_set(auxiliary.get("openrouter_model")):
        raise RenderError(f"hermes composition must not set auxiliary.openrouter_model: {refusal}")
    routed = [(f"auxiliary.{task}", settings) for task, settings in auxiliary.items() if isinstance(settings, dict)]
    routed += [(name, section_of(config, name)) for name in ("delegation", "cron")]
    for where, settings in routed:
        for key in ROUTE_KEYS:
            value = settings.get(key)
            on_main_model = where.startswith("auxiliary.") and key == "provider" and value in MAIN_MODEL_PROVIDERS
            if is_set(value) and not on_main_model:
                raise RenderError(f"hermes composition must not set {where}.{key}: {refusal}")
        if settings.get("prefer_fast_model") is True:
            raise RenderError(f"hermes composition must not set {where}.prefer_fast_model: {refusal}")


def finalize_render(files: dict[str, str]) -> dict[str, str]:
    config = json.loads(files[_CONFIG])
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
    check_model_route(config)
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
