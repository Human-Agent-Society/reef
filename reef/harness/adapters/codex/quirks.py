"""Codex adapter quirks: the model catalog, skill metadata, and code extensions.

``process_config`` turns the model metadata in ``models.json`` into
Codex's model catalog. Codex skills require ``name`` and ``description``
frontmatter. ``process_skill`` adds it when an evolved skill omits it. An
agent_command renders as a skill in the same root, so it gets the same
frontmatter.

Codex can run lifecycle hooks, but hook subprocesses do not share Codex's
inner command sandbox. ``finalize_render`` therefore rejects ``code_extension``
nodes until Reef can run them behind a separate isolation boundary.
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
EXTENSIONS = "codex/extensions/"


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
        # An agent_command renders to the same path template, so it is processed here too.
        return with_skill_frontmatter(path, text)

    @staticmethod
    def finalize_render(files: dict[str, str]) -> dict[str, str]:
        if any(path.startswith(EXTENSIONS) for path in files):
            raise RenderError(
                "codex code_extension is not supported safely because native hooks run outside the command sandbox"
            )
        if json.loads(files[MODELS])["models"]:
            # Codex resolves this relative to config.toml, including in a relocated client session.
            files[CONFIG] = tomli_w.dumps({**tomllib.loads(files[CONFIG]), "model_catalog_json": "models.json"})
        else:
            # A catalog replaces Codex's built-ins, so none is written without model metadata.
            files.pop(MODELS)
        return files
