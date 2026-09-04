"""Build the optional inference observer without importing provider SDKs."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

from reef.observability.inference import InferenceObserver, NullInferenceObserver
from reef.observability.langsmith import LangSmithClientFactory, LangSmithConfig, LangSmithInferenceObserver

logger = logging.getLogger(__name__)


def build_inference_observer(
    langsmith: object,
    *,
    environ: Mapping[str, str] | None = None,
    client_factory: LangSmithClientFactory | None = None,
) -> InferenceObserver:
    try:
        config = LangSmithConfig.from_mapping(langsmith)
    except Exception as exc:
        logger.warning("invalid LangSmith observer configuration (%s); tracing is disabled", type(exc).__name__)
        return NullInferenceObserver()
    if not config.active:
        if config.enabled:
            logger.warning("LangSmith tracing is enabled without a project; tracing is disabled")
        return NullInferenceObserver()
    env = os.environ if environ is None else environ

    def default_client_factory() -> Any:
        # Lazy import and initialization happen on the exporter thread. A base
        # Reef install therefore neither requires the SDK nor performs calls.
        from langsmith import Client

        kwargs: dict[str, Any] = {}
        endpoint = config.endpoint or env.get("LANGSMITH_ENDPOINT")
        if endpoint:
            kwargs["api_url"] = endpoint
        api_key = env.get("LANGSMITH_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        workspace_id = env.get("LANGSMITH_WORKSPACE_ID")
        if workspace_id:
            kwargs["workspace_id"] = workspace_id
        return Client(**kwargs)

    return LangSmithInferenceObserver(config, client_factory or default_client_factory)


__all__ = ["build_inference_observer"]
