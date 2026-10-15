"""SGLang request capture connected to Reef's inference scheduling interface.

The runtime chooses token-native request handling; weight publication and
admission still use the backend-neutral coordinator client.
"""

from __future__ import annotations

from reef.inference.runtime import EngineRuntimeFactory, ExecutorInferenceRuntime
from reef.inference.sglang.chat import SGLangInferenceHandler


class SGLangInferenceRuntime(ExecutorInferenceRuntime):
    """Capture SGLang tokens and log probabilities while Reef controls admission."""

    default_inference_handler = SGLangInferenceHandler


class SGLangRuntimeFactory(EngineRuntimeFactory):
    """Connect the selected SGLang receiver without constructing a trainer."""

    kind = "sglang"
    engine_name = "SGLang"
    runtime_type = SGLangInferenceRuntime
