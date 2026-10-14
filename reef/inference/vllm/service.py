"""vLLM inference lifecycle with borrowed GPU reservations."""

from __future__ import annotations

from reef.inference.engine_control import EngineInferenceService
from reef.inference.vllm.backend import VLLMInferenceBackend
from reef.inference.vllm.config import VLLMConfig
from reef.inference.vllm.launch import engine_environment
from reef.runtime.deployment import ADAPTER_FILES_PROTOCOL

#: The control RPC vocabulary of :class:`reef.inference.vllm.control.VLLMControl`.
INFERENCE_PROTOCOL = "reef-vllm-control-v1"


class VLLMInferenceService(EngineInferenceService[VLLMConfig]):
    """Own vLLM engines and the control actor; borrow the deployment allocation."""

    connection_protocol = INFERENCE_PROTOCOL
    #: Engines also load PEFT adapter directories, so file-delivering trainers pair with them.
    supported_transfer_protocols = (INFERENCE_PROTOCOL, ADAPTER_FILES_PROTOCOL)
    engine_name = "vLLM"
    control_worker_cls = "reef.inference.vllm.control:VLLMControl"
    backend_type = VLLMInferenceBackend

    def engine_environment(self) -> dict[str, str]:
        return engine_environment(self.config)
