"""pi adapter quirks: what the declarative descriptor cannot state.

pi is the friendlier of the two bundled harnesses - one environment variable
relocates its whole composition and it never mutates rendered files at boot -
so the quirks reduce to the files it may create beside the composition:
``trust.json`` (trust decisions) and ``auth.json`` (provider auth caches) can
appear in the agent directory even for an offline run.

pi follows the Agent Skills spec: a ``SKILL.md`` needs ``name`` and
``description`` frontmatter or the skill is reported as a conflict at startup
and never offered. A skill node whose text carries none gets both, the
description being the text's first line, the same synthesis codex and dsh do.

Every model call stays on Reef's model binding. The binding writes the
provider ``reef`` in ``models.json`` and selects it with ``defaultProvider``
and ``defaultModel`` in ``settings.json``; it renders after the tree and wins
every key it writes. The provider's ``apiKey`` is a credential, which a tree
cannot hold because admission refuses an inline credential, so those keys
pass only beside it, and ``reef`` only with the keys the binding writes.
Another provider (its own ``baseUrl``, or pi's built in one with a new
endpoint), ``enabledModels`` and ``httpProxy``, which sends every call
through another host, are the tree choosing where calls go, and are refused.

pi 0.84.2 rewrites a few old settings keys when it loads ``settings.json`` and
writes the new form back on its next save: ``queueMode`` becomes
``steeringMode``, a boolean ``websockets`` becomes ``transport``, and a
``skills`` object becomes the array of its ``customDirectories`` (its
``enableSkillCommands`` moves to the top level). The render writes the new
form itself, so the file an install writes is the one pi keeps, and the check
of the keys pi may not change never refuses pi's own rewrite.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import yaml

from reef.harness.adapters.descriptor import AdapterRenderer
from reef.harness.tree.render import RenderError

cleanup_whitelist = (
    "pi-agent/trust.json",
    "pi-agent/auth.json",
)

SETTINGS_PATH = "pi-agent/settings.json"
MODELS_PATH = "pi-agent/models.json"

#: The provider Reef's binding writes and its keys.
BINDING_PROVIDER = "reef"
BINDING_PROVIDER_KEYS = frozenset({"api", "apiKey", "baseUrl", "models"})
#: settings.json keys that choose the models a run may use, or send every call through a proxy.
MODEL_ROUTE_SETTINGS = ("enabledModels", "httpProxy")


def _with_frontmatter(path: str, text: str) -> str:
    if text.startswith("---\n"):
        return text
    name = path.split("/")[-2]
    first = next((line.strip().lstrip("#").strip() for line in text.splitlines() if line.strip()), "")
    header: dict[str, Any] = {"name": name, "description": first[:200] or name}
    return "---\n" + yaml.dump(header, sort_keys=False, default_flow_style=False, allow_unicode=True) + "---\n" + text


def check_model_route(settings: dict[str, Any], models: dict[str, Any]) -> None:
    """Refuse a provider, an endpoint, a credential or a model the binding did not write."""
    refusal = "Reef's model binding chooses the provider, the endpoint, the credential and the model"
    providers = models.get("providers")
    if providers is not None and (not isinstance(providers, dict) or set(providers) - {BINDING_PROVIDER}):
        raise RenderError(f"pi composition must not configure a provider other than {BINDING_PROVIDER!r}: {refusal}")
    provider = (providers or {}).get(BINDING_PROVIDER)
    if provider is not None and (not isinstance(provider, dict) or set(provider) != BINDING_PROVIDER_KEYS):
        raise RenderError(f"pi composition must not configure provider {BINDING_PROVIDER!r}: {refusal}")
    for key in MODEL_ROUTE_SETTINGS:
        if key in settings:
            raise RenderError(f"pi composition must not set {key}: {refusal}")


def migrated_settings(settings: dict[str, object]) -> dict[str, object]:
    """``settings`` with the old keys pi 0.84.2 rewrites on load in their new form, as its migrateSettings does."""
    if "queueMode" in settings and "steeringMode" not in settings:
        settings["steeringMode"] = settings.pop("queueMode")
    if "transport" not in settings and isinstance(settings.get("websockets"), bool):
        settings["transport"] = "websocket" if settings.pop("websockets") else "sse"
    skills = settings.get("skills")
    if isinstance(skills, dict):
        if "enableSkillCommands" in skills and "enableSkillCommands" not in settings:
            settings["enableSkillCommands"] = skills["enableSkillCommands"]
        directories = skills.get("customDirectories")
        if isinstance(directories, list) and directories:
            settings["skills"] = directories
        else:
            settings.pop("skills")
    return settings


class PiAdapterRenderer(AdapterRenderer):
    @staticmethod
    def process_config(path: str, config: dict[str, Any]) -> dict[str, Any]:
        # Write pi's settings in the form pi keeps.
        return migrated_settings(config) if path == SETTINGS_PATH else config

    @staticmethod
    def process_skill(path: str, text: str) -> str:
        # Give every skill the frontmatter pi requires when its text has none.
        return _with_frontmatter(path, text)

    @staticmethod
    def check_model_route(
        configs: Mapping[str, Mapping[str, Any]], skills: Mapping[str, str], commands: Mapping[str, str]
    ) -> None:
        check_model_route(dict(configs[SETTINGS_PATH]), dict(configs[MODELS_PATH]))
