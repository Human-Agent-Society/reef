"""Construct SGLang inference from resolved deployment input values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.inference.sglang.config import SGLangConfig, SGLangGroupConfig, SGLangModelConfig
from reef.inference.sglang.executor import SGLangExecutor
from reef.inference.sglang.lora_schema import (
    require_lora_distributed_request_schema,
    require_lora_tensor_request_schema,
)
from reef.inference.sglang.service import SGLangInferenceService
from reef.runtime.executor import Executor
from reef.runtime.executor.config import ExecutorSettings, select_executor


def create_inference(config: Mapping[str, Any]) -> SGLangInferenceService:
    """Validate native settings before allocation and return an unstarted service.

    Model groups arrive as ordinary mappings. Parsing legacy training flags or
    selecting a training backend belongs to the caller's input boundary.
    """
    values = dict(config)
    values["models"] = tuple(_model_config(model) for model in values.get("models", ()))
    values["executor"] = _executor(values.get("executor", "auto"))
    native = SGLangConfig(**values)
    if native.options.get("enable_lora", False):
        require_lora_tensor_request_schema()
        require_lora_distributed_request_schema()
    return SGLangInferenceService(native)


def _model_config(value: Mapping[str, Any]) -> SGLangModelConfig:
    values = dict(value)
    values["groups"] = tuple(SGLangGroupConfig(**group) for group in values["groups"])
    return SGLangModelConfig(**values)


def _executor(value: str | type[Executor]) -> type[Executor]:
    if isinstance(value, type):
        return Executor.get_class(value)
    selected = select_executor(ExecutorSettings(value or "auto"), role="rollout").settings.backend
    if selected in ("mp", "uni"):
        raise ValueError(f"SGLang inference requires ray or a native inference executor; got {selected!r}")
    return SGLangExecutor if selected == "ray" else Executor.get_class(selected)
