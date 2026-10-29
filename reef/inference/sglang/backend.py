"""Native SGLang inference backend: the engine operations that Reef's serving runtimes call."""

from __future__ import annotations

from reef.inference.engine_control import CONTROL_TIMEOUT_S, EngineInferenceBackend


class SGLangInferenceBackend(EngineInferenceBackend):
    """Translate the serving runtimes' engine operations to SGLang's native control vocabulary."""

    engine_name = "SGLang"
    unload_result_optional = True

    def release_memory(self) -> None:
        """Release all engine memory after admission drains and generation pauses."""
        self._engines().collective_rpc("release_memory_occupation", timeout=CONTROL_TIMEOUT_S)

    def resume_memory(self) -> None:
        """Restore the same served weights before generation resumes."""
        self._engines().collective_rpc("resume_memory_occupation", timeout=CONTROL_TIMEOUT_S)
