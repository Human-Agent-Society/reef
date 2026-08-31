"""Inference proxy runtime resolution shared by recipe implementations."""

from __future__ import annotations

from collections.abc import Mapping

from reef.runtime.base import InferenceRuntime
from reef.runtime.registry import RuntimeRegistry


def resolve_proxy_runtime(
    values: Mapping[str, str],
    runtime: InferenceRuntime | None,
) -> InferenceRuntime | None:
    """Prefer an injected runtime, else build a proxy from environment values.

    Construction goes through the runtime registry's ``inference_proxy`` kind,
    the same factory the YAML config path uses.
    """
    if runtime is not None:
        return runtime
    base_url = _env_value(values, "REEF_UPSTREAM_URL")
    if base_url is None:
        return None
    config: dict[str, object] = {"type": "inference_proxy", "base_url": base_url}
    # REEF_API_KEY is the pre-rename spelling; drop it once no deployment sets it.
    api_key = _env_value(values, "REEF_UPSTREAM_API_KEY") or _env_value(values, "REEF_API_KEY")
    if api_key is not None:
        config["api_key"] = api_key
    timeout_raw = _env_value(values, "REEF_INFERENCE_TIMEOUT_S")
    if timeout_raw is not None:
        config["timeout_s"] = float(timeout_raw)
    return RuntimeRegistry().build(
        config,
        model_path=_env_value(values, "REEF_MODEL_PATH") or "",
        environ=values,
    )


def _env_value(values: Mapping[str, str], name: str) -> str | None:
    return values.get(name, "").strip() or None
