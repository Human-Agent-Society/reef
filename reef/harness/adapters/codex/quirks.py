"""Codex adapter quirks: TOML, skill metadata, and safety invariants.

Reef config nodes are JSON objects for every adapter, while Codex reads user
configuration as TOML. ``finalize_render`` validates the benchmark invariants
and performs that final serialization. Codex skills require ``name`` and
``description`` frontmatter, synthesized when an evolved skill omits it.

Codex can run lifecycle hooks, but hook subprocesses do not share Codex's
inner command sandbox. The finalizer therefore rejects ``code_extension``
nodes until Reef can run them behind a separate isolation boundary.
"""

from __future__ import annotations

import json
from typing import Any

import tomli_w
import yaml

from reef.harness.tree.render import RenderError

_CONFIG = "codex/config.toml"
_EXTENSIONS = "codex/extensions/"
_SKILLS = ".agents/skills/"

_ALLOWED_CONFIG_KEYS = {
    "analytics",
    "approval_policy",
    "check_for_update_on_startup",
    "features",
    "feedback",
    "model",
    "model_auto_compact_token_limit",
    "model_auto_compact_token_limit_scope",
    "model_context_window",
    "model_provider",
    "model_providers",
    "model_reasoning_effort",
    "model_reasoning_summary",
    "model_verbosity",
    "otel",
    "sandbox_workspace_write",
    "tool_output_token_limit",
    "web_search",
}
_ALLOWED_FEATURES = {
    "apps",
    "enable_request_compression",
    "hooks",
    "plugins",
    "shell_snapshot",
    "skill_mcp_dependency_install",
}
_DISABLED_FEATURES = {"apps", "hooks", "plugins", "shell_snapshot", "skill_mcp_dependency_install"}
_ALLOWED_PROVIDER_KEYS = {
    "base_url",
    "experimental_bearer_token",
    "name",
    "supports_websockets",
    "wire_api",
}
_OTEL_DEFAULTS = {
    "exporter": "none",
    "log_user_prompt": False,
    "metrics_exporter": "none",
    "trace_exporter": "none",
}


def _with_frontmatter(path: str, text: str) -> str:
    if text.startswith("---\n"):
        return text
    name = path.split("/")[-2]
    first = next((line.strip().lstrip("#").strip() for line in text.splitlines() if line.strip()), "")
    header: dict[str, Any] = {"name": name, "description": first[:200] or name}
    return "---\n" + yaml.dump(header, sort_keys=False, default_flow_style=False, allow_unicode=True) + "---\n" + text


def _validate_config(config: dict[str, Any]) -> None:
    extra = sorted(set(config) - _ALLOWED_CONFIG_KEYS)
    if extra:
        raise RenderError(f"codex config keys are not admitted for benchmark episodes: {', '.join(extra)}")
    if config.get("approval_policy") != "never":
        raise RenderError("codex composition must keep approval_policy never for non-interactive episodes")
    if config.get("web_search") != "disabled":
        raise RenderError("codex composition must keep web_search disabled for benchmark episodes")
    if config.get("check_for_update_on_startup") is not False:
        raise RenderError("codex composition must keep check_for_update_on_startup false")
    for section in ("analytics", "feedback"):
        settings = config.get(section)
        if not isinstance(settings, dict) or settings.get("enabled") is not False:
            raise RenderError(f"codex composition must keep {section}.enabled false")
    features = config.get("features")
    if not isinstance(features, dict):
        raise RenderError("codex composition must keep features as an object")
    extra_features = sorted(set(features) - _ALLOWED_FEATURES)
    if extra_features:
        raise RenderError(f"codex feature keys are not admitted: {', '.join(extra_features)}")
    for feature in _DISABLED_FEATURES:
        if features.get(feature) is not False:
            raise RenderError(f"codex composition must keep features.{feature} disabled")

    otel = config.get("otel")
    if otel != _OTEL_DEFAULTS:
        raise RenderError("codex composition must keep every OpenTelemetry exporter disabled")

    sandbox = config.get("sandbox_workspace_write")
    if not isinstance(sandbox, dict) or sandbox.get("network_access") is not False:
        raise RenderError("codex composition must keep sandbox_workspace_write.network_access false")
    if set(sandbox) - {"network_access", "writable_roots"}:
        raise RenderError("codex composition contains unadmitted sandbox_workspace_write fields")
    if sandbox.get("writable_roots"):
        raise RenderError("codex composition may not add sandbox_workspace_write.writable_roots")
    providers = config.get("model_providers")
    if isinstance(providers, dict):
        extra_providers = sorted(set(providers) - {"reef"})
        if extra_providers:
            raise RenderError(
                f"codex composition may only configure the Reef model provider: {', '.join(extra_providers)}"
            )
        for name, provider in providers.items():
            if not isinstance(provider, dict) or set(provider) - _ALLOWED_PROVIDER_KEYS:
                raise RenderError(f"codex model provider {name!r} contains unadmitted fields")
    if config.get("model_provider") not in (None, "reef"):
        raise RenderError("codex composition must use the Reef model provider")


def finalize_render(files: dict[str, str]) -> dict[str, str]:
    try:
        config = json.loads(files[_CONFIG])
    except (KeyError, json.JSONDecodeError) as exc:
        raise RenderError("codex primary config must be a JSON object before TOML rendering") from exc
    if not isinstance(config, dict):
        raise RenderError("codex primary config must be an object")

    if any(path.startswith(_EXTENSIONS) for path in files):
        raise RenderError(
            "codex code_extension is not supported safely because native hooks run outside the command sandbox"
        )
    _validate_config(config)
    try:
        files[_CONFIG] = tomli_w.dumps(config)
    except (TypeError, ValueError) as exc:
        raise RenderError(f"codex config cannot be represented as TOML: {exc}") from exc

    for path, text in list(files.items()):
        if path.startswith(_SKILLS) and path.endswith("/SKILL.md"):
            files[path] = _with_frontmatter(path, text)

    return files
