"""vLLM request capture connected to Reef's inference scheduling interface.

The runtime chooses token-native request handling; weight publication and
admission still use the backend-neutral coordinator client.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.inference.runtime import ExecutorInferenceRuntime
from reef.inference.vllm.chat import VLLMInferenceHandler
from reef.runtime.deployment import RuntimeConfigError, RuntimeConnectionConfig, RuntimeFactory
from reef.runtime.executor.connection import CoordinatorClient
from reef.runtime.interfaces import InferenceHandler, InferenceRuntime


class VLLMInferenceRuntime(ExecutorInferenceRuntime):
    """Capture vLLM tokens and log probabilities while Reef controls admission."""

    def __init__(
        self,
        *,
        control: CoordinatorClient,
        model_path: str,
        inference_url: str | None = None,
        inference_timeout_s: float = 300.0,
        inference_handler_factory: type[InferenceHandler] = VLLMInferenceHandler,
        inference_handler_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            control=control,
            model_path=model_path,
            inference_url=inference_url,
            inference_timeout_s=inference_timeout_s,
            inference_handler_factory=inference_handler_factory,
            inference_handler_config=inference_handler_config,
        )


class VLLMRuntimeFactory(RuntimeFactory):
    """Connect the selected vLLM receiver without constructing a trainer."""

    kind = "vllm"

    def config_type(self) -> type:
        return RuntimeConnectionConfig

    def parse_config(self, config: Mapping[str, Any], environ: Mapping[str, str]) -> dict[str, Any]:
        injected = {key: config[key] for key in ("control", "inference_handler_factory") if key in config}
        parsed = super().parse_config({key: value for key, value in config.items() if key not in injected}, environ)
        return {**parsed, **injected}

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> InferenceRuntime:
        control = config.get("control")
        if not isinstance(control, CoordinatorClient):
            raise RuntimeConfigError("the vLLM runtime requires a Reef coordinator client")
        return VLLMInferenceRuntime(
            control=control,
            model_path=model_path,
            inference_url=config.get("inference_url"),
            inference_timeout_s=config.get("inference_timeout_s", 300.0),
            inference_handler_factory=config.get("inference_handler_factory", VLLMInferenceHandler),
            inference_handler_config=config.get("inference_handler_config"),
        )
