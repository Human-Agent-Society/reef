"""Codex adapter render steps: the model catalog, skill metadata, and the layer.

``process_config`` turns the model metadata in ``models.json`` into
Codex's model catalog. Codex skills require ``name`` and ``description``
frontmatter. ``process_skill`` adds it when an evolved skill omits it. A
command is a skill in the same root (its ``parts``), so it gets the same
frontmatter.

On a person's machine, Codex keeps the person's own ``~/.codex``. It reads a
config key from a ``-c`` argument, so ``layer_args`` passes each key of the
tree's ``config.toml``. It reads skills from no directory a flag names, only
from an installed plugin, so ``layer_files`` makes ``codex/`` a plugin in a
local marketplace at the install root, which the layer's ``register``
commands install. Codex copies the whole plugin root into its plugin cache,
so the root keeps no config file: the binding's credential stays in the
install.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tomli_w

from reef.core.model_metadata import ModelMetadata
from reef.harness.adapters.common.frontmatter import with_skill_frontmatter
from reef.harness.adapters.descriptor import AdapterRenderer
from reef.harness.tree.render import RenderError

CONFIG = "codex/config.toml"
MODELS = "codex/models.json"
#: The local marketplace at the install root, and the manifest of its one plugin, ``codex/``. The layer's
#: ``register`` commands in adapter.yaml name both as ``reef``.
MARKETPLACE = ".agents/plugins/marketplace.json"
PLUGIN_MANIFEST = "codex/.codex-plugin/plugin.json"
#: The config.toml keys a layer leaves to the person's own config: the ones Reef's model binding writes, which
#: the layer's own arguments set without the credential, and the ones the adapter.yaml checks pin to their
#: episode defaults, which keep episodes hermetic and would change a person's session.
LAYER_SKIPPED_KEYS = frozenset(
    {
        "model",
        "model_provider",
        "model_providers",
        "model_catalog_json",
        "analytics",
        "check_for_update_on_startup",
        "feedback",
        "otel",
        "features.apps",
        "features.hooks",
        "features.plugins",
        "features.shell_snapshot",
        "features.skill_mcp_dependency_install",
        "sandbox_workspace_write.network_access",
        "sandbox_workspace_write.writable_roots",
    }
)
BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


def toml_value(value: object) -> str:
    """``value``, a config value, as the TOML text that a ``-c`` argument parses."""
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{json.dumps(key)} = {toml_value(item)}" for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    return json.dumps(value)


def config_arguments(config: Mapping[str, object], prefix: str = "") -> list[tuple[str, str]]:
    """``-c`` argument groups that set each key of ``config`` but ``LAYER_SKIPPED_KEYS``, a table by its keys."""
    groups: list[tuple[str, str]] = []
    for key, value in config.items():
        if not BARE_KEY.fullmatch(key):
            raise RenderError(f"codex config key {prefix + key!r} cannot be passed as a -c argument")
        dotted = prefix + key
        if dotted in LAYER_SKIPPED_KEYS:
            continue
        if isinstance(value, Mapping):
            groups.extend(config_arguments(value, f"{dotted}."))
        else:
            groups.append(("-c", f"{dotted}={toml_value(value)}"))
    return groups


def bundled_model_catalog() -> dict[str, dict[str, object]]:
    """The pinned CLI's catalog, including native prompts and tool configuration.

    Exported with ``codex debug models --bundled``; the real-CLI test checks
    the complete resource when the install pin changes.
    """
    with Path(__file__).with_name("bundled_models.json").open(encoding="utf-8") as source:
        return {model["slug"]: model for model in json.load(source)["models"]}


def native_model_config(model: str, catalog: Mapping[str, dict[str, object]]) -> dict[str, object] | None:
    """Match the pinned CLI's longest prefix and single provider namespace rules."""
    namespace, separator, suffix = model.partition("/")
    model_names: tuple[str, ...]
    if separator and "/" not in suffix and re.fullmatch(r"[A-Za-z0-9_-]+", namespace):
        model_names = (model, suffix)
    else:
        model_names = (model,)
    for name in model_names:
        matched_slug = max((slug for slug in catalog if name.startswith(slug)), key=len, default="")
        if matched_slug:
            return dict(catalog[matched_slug])
    return None


def get_model_catalog(metadata_models: Mapping[object, object]) -> dict[str, dict[str, object]]:
    """The models.json catalog Codex reads, by model slug: the pinned CLI's bundled models, then one entry per
    model in ``metadata_models`` (model name to Reef model metadata); empty when ``metadata_models`` is.
    """
    # A catalog replaces Codex's built-ins, including when the user later selects another model.
    bundled_models = bundled_model_catalog() if metadata_models else {}
    catalog = dict(bundled_models)
    for model, value in metadata_models.items():
        if not isinstance(model, str) or not model.strip():
            raise RenderError("codex model metadata requires a non-empty model name")
        try:
            metadata = ModelMetadata.from_config(value)
        except ValueError as exc:
            raise RenderError(f"codex model {model!r}: {exc}") from exc
        native_config = native_model_config(model, bundled_models)
        if native_config is not None:
            model_config = native_config
        else:
            model_config = {
                "slug": model,
                "display_name": model,
                "supported_reasoning_levels": [],
                "default_reasoning_level": None,
                "shell_type": "unified_exec",
                "visibility": "list",
                "supported_in_api": True,
                "priority": 0,
                # Preserve the pinned CLI's unknown-model prompt; metadata must not weaken its instructions.
                "base_instructions": Path(__file__).with_name("default_instructions.md").read_text(encoding="utf-8"),
                "support_verbosity": False,
                "truncation_policy": {"mode": "bytes", "limit": 10000},
                "experimental_supported_tools": [],
            }
        if metadata.reasoning:
            model_config["supported_reasoning_levels"] = model_config["supported_reasoning_levels"] or [
                {"effort": effort, "description": effort} for effort in ("low", "medium", "high")
            ]
            model_config["default_reasoning_level"] = model_config["default_reasoning_level"] or "medium"
        else:
            model_config["supported_reasoning_levels"] = []
            model_config["default_reasoning_level"] = None
        model_config.update(
            slug=model,
            context_window=metadata.context_window,
            max_context_window=metadata.context_window,
            supports_reasoning_summary_parameter=metadata.reasoning,
        )
        catalog[model] = model_config
    return catalog


class CodexAdapterRenderer(AdapterRenderer):
    @staticmethod
    def process_config(path: str, config: dict[str, Any]) -> dict[str, Any]:
        if path == CONFIG:
            return config
        return {"models": list(get_model_catalog(config.get("models", {})).values())}

    @staticmethod
    def process_skill(path: str, text: str) -> str:
        return with_skill_frontmatter(path, text)

    @staticmethod
    def finalize_render(files: dict[str, str]) -> dict[str, str]:
        if json.loads(files[MODELS])["models"]:
            # Codex resolves this relative to config.toml, including in a relocated client session.
            files[CONFIG] = tomli_w.dumps({**tomllib.loads(files[CONFIG]), "model_catalog_json": "models.json"})
        else:
            # A catalog replaces Codex's built-ins, so none is written without model metadata.
            files.pop(MODELS)
        return files

    @staticmethod
    def layer_args(files: Mapping[str, str]) -> tuple[tuple[str, ...], ...]:
        return tuple(config_arguments(tomllib.loads(files[CONFIG])))

    @staticmethod
    def layer_files(files: dict[str, str]) -> dict[str, str]:
        # The config travels as -c arguments, and the model catalog would replace the person's built-in models.
        kept = {path: text for path, text in files.items() if path not in (CONFIG, MODELS)}
        marketplace = {"name": "reef", "plugins": [{"name": "reef", "source": {"source": "local", "path": "./codex"}}]}
        plugin = {"name": "reef", "version": "0.0.0", "description": "The skills of a Reef harness release"}
        return {
            **kept,
            MARKETPLACE: json.dumps(marketplace, indent=2) + "\n",
            PLUGIN_MANIFEST: json.dumps(plugin, indent=2) + "\n",
        }
