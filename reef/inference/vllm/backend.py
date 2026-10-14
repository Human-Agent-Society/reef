"""Native vLLM inference backend used by Reef's training coordinator."""

from __future__ import annotations

from reef.inference.engine_control import EngineInferenceBackend


class VLLMInferenceBackend(EngineInferenceBackend):
    """Translate coordinator operations to the vLLM control vocabulary."""

    engine_name = "vLLM"
