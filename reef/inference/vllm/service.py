"""vLLM inference lifecycle with borrowed GPU reservations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from reef.inference.engine_control import EngineInferenceService, engine_executor_class
from reef.inference.vllm.backend import VLLMInferenceBackend
from reef.inference.vllm.config import VLLMSettings
from reef.inference.vllm.control import VLLMExecutor
from reef.inference.vllm.launch import engine_environment

#: The control RPC vocabulary of :class:`reef.inference.vllm.control.VLLMControl`.


class VLLMInferenceService(EngineInferenceService[VLLMSettings]):
    """Own vLLM engines and the control actor; borrow the deployment allocation."""

    engine_name = "vLLM"
    control_worker_cls = "reef.inference.vllm.control:VLLMControl"
    backend_type = VLLMInferenceBackend

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        """Validate native settings before allocation and return an unstarted service."""
        values = dict(config)
        values["executor"] = engine_executor_class(values.get("executor", "auto"), VLLMExecutor, "vLLM")
        return cls(VLLMSettings(**values))

    def engine_environment(self) -> dict[str, str]:
        return engine_environment(self.config)
