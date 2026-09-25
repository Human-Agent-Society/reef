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
    credential = model.get(BINDING_CREDENTIAL)
    bound = isinstance(credential, str) and bool(credential.strip())
    for key in BINDING_MODEL_KEYS:
        if key in model and not bound:
            raise RenderError(f"hermes composition must not set model.{key}: {refusal}")
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
