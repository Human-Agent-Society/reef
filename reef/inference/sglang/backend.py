"""Native SGLang inference backend: the engine operations that Reef's serving runtimes call."""

from __future__ import annotations

from reef.inference.engine_control import EngineInferenceBackend


class SGLangInferenceBackend(EngineInferenceBackend):
    """Translate the serving runtimes' engine operations to SGLang's native control vocabulary."""

    engine_name = "SGLang"
    unload_result_optional = True
