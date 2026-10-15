"""Native SGLang inference backend used by Reef's training coordinator."""

from __future__ import annotations

from reef.inference.engine_control import EngineInferenceBackend


class SGLangInferenceBackend(EngineInferenceBackend):
    """Translate coordinator operations to SGLang's native control vocabulary."""

    engine_name = "SGLang"
    unload_result_optional = True
