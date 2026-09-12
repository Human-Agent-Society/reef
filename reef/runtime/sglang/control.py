"""SGLang inference control actor, independent of the training backend.

Engine handles and locks travel through control RPCs; weight tensors
continue to travel directly between training workers and inference engines.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from reef.runtime.executor import Executor, ExecutorConfig
from reef.runtime.sglang.config import SGLangConfig


class SGLangControl:
    """Own the selected serving executor inside Reef's inference control actor."""

    def __init__(self, config: SGLangConfig, pg: Any) -> None:
        serving = Executor.create(
            ExecutorConfig(
                backend=config.executor,
                options={**config.executor_options, "config": config, "pg": pg},
            )
        )
        self._serving = serving
        self._config = config
        self._prepared = False
        self._closed = False

    def check_health(self) -> None:
        self._serving.check_health(timeout=30)

    def inference_url(self) -> Any:
        return self._serving.rpc(0, "inference_url", timeout=14_400)

    def get_runtime_load_ids(self) -> Any:
        return self._serving.rpc(0, "get_runtime_load_ids", timeout=14_400)

    def pause_generation_for_update(self) -> Any:
        return self._serving.rpc(0, "pause_generation_for_update", timeout=14_400)

    def continue_generation_after_update(self) -> Any:
        return self._serving.rpc(0, "continue_generation_after_update", timeout=14_400)

    def terminate_updatable_engines(self) -> Any:
        return self._serving.rpc(0, "terminate_updatable_engines", timeout=14_400)

    def get_updatable_engines_and_lock(self) -> Any:
        return self._serving.rpc(0, "get_updatable_engines_and_lock", timeout=14_400)

    def offload(self, tags: Sequence[str] | None = None) -> Any:
        return self._serving.rpc(0, "offload", args=(tags,), timeout=14_400)

    def onload(self, tags: Sequence[str] | None = None) -> Any:
        return self._serving.rpc(0, "onload", args=(tags,), timeout=14_400)

    def onload_weights(self) -> Any:
        return self._serving.rpc(0, "onload_weights", timeout=14_400)

    def onload_kv(self) -> Any:
        return self._serving.rpc(0, "onload_kv", timeout=14_400)

    def prepare_training_connection(self) -> None:
        """Fence serving and release shared memory before training workers exist."""
        self._serving.rpc(0, "prepare_training_connection", timeout=14_400)
        if not self._prepared and self._config.check_weights:
            self.check_weights("snapshot")
            self.check_weights("reset_tensors")
        if self._config.offload:
            # Every trainer attachment needs the whole allocation, even when
            # later LoRA steps keep the frozen base resident. The engine skips
            # regions already released by an earlier attachment attempt.
            self.offload()
        self._prepared = True

    def recover_updatable_engines(self) -> Any:
        return self._serving.rpc(0, "recover_updatable_engines", timeout=14_400)

    def clear_updatable_num_new_engines(self) -> Any:
        return self._serving.rpc(0, "clear_updatable_num_new_engines", timeout=14_400)

    def health_monitoring_pause(self) -> Any:
        return self._serving.rpc(0, "health_monitoring_pause", timeout=14_400)

    def health_monitoring_resume(self) -> Any:
        return self._serving.rpc(0, "health_monitoring_resume", timeout=14_400)

    def check_weights(self, action: str) -> Any:
        return self._serving.rpc(0, "check_weights", args=(action,), timeout=14_400)

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._serving.shutdown()
