"""Codex adapter quirks: TOML, skill metadata, and safety invariants.

Reef config nodes are JSON objects for every adapter, while Codex reads user
configuration as TOML. ``process_config`` validates the benchmark invariants,
and the render writes ``config.toml`` as TOML by its suffix. Codex skills require ``name`` and
``description`` frontmatter, synthesized when an evolved skill omits it; an
agent_command renders as a skill in the same root, so it gets the same
frontmatter. ``web_search`` may take any value Codex reads, because the
episode argv pins it disabled and only a person's reef-codex session reads
the tree's value. ``approval_policy`` is refused: the episode argv pins it
never, and a reef-codex session keeps Codex's on-request default, so the
person approves each command that leaves the sandbox.

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
import yaml

from reef.core.model_metadata import ModelMetadata
from reef.harness.adapters.descriptor import AdapterRenderer
from reef.harness.tree.render import RenderError

CONFIG = "codex/config.toml"
MODELS = "codex/models.json"
EXTENSIONS = "codex/extensions/"
WEB_SEARCH_MODES = ("disabled", "cached", "indexed", "live")

#: The config.toml keys a tree may set.
ALLOWED_CONFIG_KEYS = {
    "analytics",
    "check_for_update_on_startup",
    "features",
    "feedback",
    "model_auto_compact_token_limit",
    "model_auto_compact_token_limit_scope",
    "model_context_window",
    "model_reasoning_effort",
    "model_reasoning_summary",
    "model_verbosity",
    "otel",
    "sandbox_workspace_write",
    "tool_output_token_limit",
    "web_search",
}
#: The config.toml keys Reef's model binding writes; the render refuses each from a tree without the binding.
BINDING_CONFIG_KEYS = {"model", "model_provider", "model_providers"}
ALLOWED_FEATURES = {
    "apps",
    "enable_request_compression",
    "hooks",
    "plugins",
    "shell_snapshot",
    "skill_mcp_dependency_install",
}
DISABLED_FEATURES = {"apps", "hooks", "plugins", "shell_snapshot", "skill_mcp_dependency_install"}
ALLOWED_PROVIDER_KEYS = {
    "base_url",
    "experimental_bearer_token",
    "name",
    "supports_websockets",
    "wire_api",
}
OTEL_DEFAULTS = {
    "exporter": "none",
    "log_user_prompt": False,
    "metrics_exporter": "none",
    "trace_exporter": "none",
}


def with_frontmatter(path: str, text: str) -> str:
    if text.startswith("---\n"):
        return text
    name = path.split("/")[-2]
    first = next((line.strip().lstrip("#").strip() for line in text.splitlines() if line.strip()), "")
    header: dict[str, Any] = {"name": name, "description": first[:200] or name}
    return "---\n" + yaml.dump(header, sort_keys=False, default_flow_style=False, allow_unicode=True) + "---\n" + text


def validate_config(config: dict[str, Any]) -> None:
    if "approval_policy" in config:
        raise RenderError(
            "codex composition may not set approval_policy: episodes pin never, and a reef-codex session keeps"
            " on-request so the person approves each command that leaves the sandbox"
        )
    extra = sorted(set(config) - ALLOWED_CONFIG_KEYS - BINDING_CONFIG_KEYS)
    if extra:
        raise RenderError(f"codex config keys are not admitted for benchmark episodes: {', '.join(extra)}")
    if config.get("web_search") not in WEB_SEARCH_MODES:
        raise RenderError(f"codex web_search must be one of {', '.join(WEB_SEARCH_MODES)}")
    if config.get("check_for_update_on_startup") is not False:
        raise RenderError("codex composition must keep check_for_update_on_startup false")
    for section in ("analytics", "feedback"):
        settings = config.get(section)
        if not isinstance(settings, dict) or settings.get("enabled") is not False:
            raise RenderError(f"codex composition must keep {section}.enabled false")
    features = config.get("features")
    if not isinstance(features, dict):
        raise RenderError("codex composition must keep features as an object")
    extra_features = sorted(set(features) - ALLOWED_FEATURES)
    if extra_features:
        raise RenderError(f"codex feature keys are not admitted: {', '.join(extra_features)}")
    for feature in DISABLED_FEATURES:
        if features.get(feature) is not False:
            raise RenderError(f"codex composition must keep features.{feature} disabled")

    otel = config.get("otel")
    if otel != OTEL_DEFAULTS:
        raise RenderError("codex composition must keep every OpenTelemetry exporter disabled")

    sandbox = config.get("sandbox_workspace_write")
    if not isinstance(sandbox, dict) or sandbox.get("network_access") is not False:
        raise RenderError("codex composition must keep sandbox_workspace_write.network_access false")
    if set(sandbox) - {"network_access", "writable_roots"}:
        raise RenderError("codex composition contains unadmitted sandbox_workspace_write fields")
    if sandbox.get("writable_roots"):
        raise RenderError("codex composition may not add sandbox_workspace_write.writable_roots")


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
            validate_config(config)
            return config
        if set(config) - {"models"}:
            raise RenderError("codex models config accepts only models")
        metadata_models = config.get("models", {})
        if not isinstance(metadata_models, dict):
            raise RenderError("codex models must map model names to metadata")
        return {"models": list(get_model_catalog(metadata_models).values())}

    @staticmethod
    def process_skill(path: str, text: str) -> str:
        # An agent_command renders to the same path template, so it is processed here too.
        return with_frontmatter(path, text)

    @staticmethod
    def check_model_route(
        configs: Mapping[str, Mapping[str, Any]], skills: Mapping[str, str], commands: Mapping[str, str]
    ) -> None:
        config = configs[CONFIG]
        providers = config.get("model_providers")
        if isinstance(providers, dict):
            extra_providers = sorted(set(providers) - {"reef"})
            if extra_providers:
                raise RenderError(
                    f"codex composition may only configure the Reef model provider: {', '.join(extra_providers)}"
                )
            for name, provider in providers.items():
                if not isinstance(provider, dict) or set(provider) - ALLOWED_PROVIDER_KEYS:
                    raise RenderError(f"codex model provider {name!r} contains unadmitted fields")
        if config.get("model_provider") not in (None, "reef"):
            raise RenderError("codex composition must use the Reef model provider")

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
