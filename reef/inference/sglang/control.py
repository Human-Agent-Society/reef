"""SGLang inference control actor, independent of the training backend.

Engine handles and locks travel through control RPCs; weight tensors
continue to travel directly between training workers and inference engines.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from reef.inference.sglang.config import CONTROL_TIMEOUT_S, SGLangConfig
from reef.runtime.executor import Executor, ExecutorConfig


class SGLangControl:
    """Own the selected serving executor inside Reef's inference control actor.

    Every method forwards one control operation to the serving worker; only
    :meth:`prepare_training_connection` adds policy of its own.
    """

    def __init__(self, config: SGLangConfig, pg: Any) -> None:
        self._serving = Executor.create(
            ExecutorConfig(
                backend=config.executor,
                options={**config.executor_options, "config": config, "pg": pg},
            )
        )
        self._config = config
        self._prepared = False
        self._closed = False

    def _call(self, method: str, *args: Any) -> Any:
        return self._serving.rpc(0, method, args=args, timeout=CONTROL_TIMEOUT_S)

    def check_health(self) -> None:
        self._serving.check_health(timeout=30)

    def inference_url(self) -> Any:
        return self._call("inference_url")

    def get_runtime_load_ids(self) -> Any:
        return self._call("get_runtime_load_ids")

    def load_adapter_from_disk(self, lora_name: str, lora_path: str, runtime_load_id: str | None = None) -> Any:
        return self._call("load_adapter_from_disk", lora_name, lora_path, runtime_load_id)

    def pause_generation_for_update(self) -> Any:
        return self._call("pause_generation_for_update")

    def continue_generation_after_update(self) -> Any:
        return self._call("continue_generation_after_update")

    def terminate_updatable_engines(self) -> Any:
        return self._call("terminate_updatable_engines")

    def get_updatable_engines_and_lock(self) -> Any:
        return self._call("get_updatable_engines_and_lock")

    def offload(self, tags: Sequence[str] | None = None) -> Any:
        return self._call("offload", tags)

    def onload(self, tags: Sequence[str] | None = None) -> Any:
        return self._call("onload", tags)

    def onload_weights(self) -> Any:
        return self._call("onload_weights")

    def onload_kv(self) -> Any:
        return self._call("onload_kv")

    def prepare_training_connection(self) -> None:
        """Fence serving and release shared memory before training workers exist."""
        self._call("prepare_training_connection")
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
        return self._call("recover_updatable_engines")

    def clear_updatable_num_new_engines(self) -> Any:
        return self._call("clear_updatable_num_new_engines")

    def health_monitoring_pause(self) -> Any:
        return self._call("health_monitoring_pause")

    def health_monitoring_resume(self) -> Any:
        return self._call("health_monitoring_resume")

    def check_weights(self, action: str) -> Any:
        return self._call("check_weights", action)

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._serving.shutdown()
