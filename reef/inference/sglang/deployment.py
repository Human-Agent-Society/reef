"""Construct SGLang inference from resolved deployment input values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.inference.engine_control import engine_executor_class
from reef.inference.sglang.config import SGLangConfig
from reef.inference.sglang.executor import SGLangExecutor
from reef.inference.sglang.service import SGLangInferenceService


def create_inference(config: Mapping[str, Any]) -> SGLangInferenceService:
    """Validate native settings before allocation and return an unstarted service."""
    values = dict(config)
    values["executor"] = engine_executor_class(values.get("executor", "auto"), SGLangExecutor, "SGLang")
    return SGLangInferenceService(SGLangConfig(**values))
