"""vLLM request capture connected to Reef's inference scheduling interface.

The runtime chooses token-native request handling; weight publication and
admission still use the backend-neutral coordinator client.
"""

from __future__ import annotations

from reef.inference.runtime import EngineRuntimeFactory, ExecutorInferenceRuntime
from reef.inference.vllm.chat import VLLMInferenceHandler


class VLLMInferenceRuntime(ExecutorInferenceRuntime):
    """Capture vLLM tokens and log probabilities while Reef controls admission."""

    default_inference_handler = VLLMInferenceHandler


class VLLMRuntimeFactory(EngineRuntimeFactory):
    """Connect the selected vLLM receiver without constructing a trainer."""

    kind = "vllm"
    engine_name = "vLLM"
    runtime_type = VLLMInferenceRuntime
