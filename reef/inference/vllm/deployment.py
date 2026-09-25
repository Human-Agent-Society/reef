"""Construct vLLM inference from resolved deployment input values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.inference.vllm.config import VLLMConfig
from reef.inference.vllm.control import VLLMExecutor
from reef.inference.vllm.service import VLLMInferenceService
from reef.runtime.executor import Executor
from reef.runtime.executor.config import ExecutorSettings, select_executor


def create_inference(config: Mapping[str, Any]) -> VLLMInferenceService:
    """Validate native settings before allocation and return an unstarted service."""
    values = dict(config)
    values["executor"] = _executor(values.get("executor", "auto"))
    return VLLMInferenceService(VLLMConfig(**values))


def _executor(value: str | type[Executor]) -> type[Executor]:
    if isinstance(value, type):
        return Executor.get_class(value)
    selected = select_executor(ExecutorSettings(value or "auto"), role="rollout").settings.backend
    if selected in ("mp", "uni"):
        raise ValueError(f"vLLM inference requires ray or a native inference executor; got {selected!r}")
    return VLLMExecutor if selected == "ray" else Executor.get_class(selected)
