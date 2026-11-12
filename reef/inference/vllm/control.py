"""vLLM inference control actor and its default serving executor, independent of the training backend.

Engine handles travel through control RPCs; weights reach the engines as
files that the engines read themselves.
"""

from __future__ import annotations

from reef.inference.engine_control import EngineControl, EngineExecutor
from reef.inference.vllm.config import VLLMConfig


class VLLMExecutor(EngineExecutor):
    """One control rank manages the vLLM engine group."""

    def create_worker(self) -> object:
        from reef.inference.vllm.worker import VLLMWorker

        return VLLMWorker(**dict(self.config.options))


class VLLMControl(EngineControl[VLLMConfig]):
    """Own the vLLM serving executor inside Reef's inference control actor."""
