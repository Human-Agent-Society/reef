"""SGLang inference lifecycle with borrowed GPU reservations."""

from __future__ import annotations

from reef.inference.engine_control import EngineInferenceService
from reef.inference.sglang.backend import SGLangInferenceBackend
from reef.inference.sglang.config import SGLangConfig
from reef.inference.sglang.launch import engine_environment
from reef.runtime.deployment import ADAPTER_FILES_PROTOCOL

# Keep the existing wire identifier while removing its implementation dependency.
INFERENCE_PROTOCOL = "slime-sglang-control-v2"


class SGLangInferenceService(EngineInferenceService[SGLangConfig]):
    """Own SGLang engines and the control actor; borrow the deployment allocation."""

    connection_protocol = INFERENCE_PROTOCOL
    #: Engines also load PEFT adapter directories, so file-delivering trainers pair with them.
    supported_transfer_protocols = (INFERENCE_PROTOCOL, ADAPTER_FILES_PROTOCOL)
    engine_name = "SGLang"
    control_worker_cls = "reef.inference.sglang.control:SGLangControl"
    backend_type = SGLangInferenceBackend

    def engine_environment(self) -> dict[str, str]:
        return engine_environment(self.config)
