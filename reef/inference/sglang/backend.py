"""Native SGLang inference backend used by Reef's training coordinator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reef.inference.sglang.config import CONTROL_TIMEOUT_S
from reef.runtime.executor import Executor
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.interfaces import InferenceBackend


class SGLangInferenceBackend(InferenceBackend):
    """Borrow a control connection without owning publication or engine lifetime.

    The coordinator decides operation order and whether a loaded revision may
    serve requests. This adapter translates individual operations to SGLang's
    native control vocabulary and validates receiver results.
    """

    def __init__(self, executor: Executor) -> None:
        self._executor = executor

    def inference_url(self) -> str:
        value = self._call("inference_url")
        if not isinstance(value, str) or not value:
            raise RuntimeError("SGLang inference did not provide a serving endpoint")
        return value

    def runtime_load_ids(self) -> Sequence[str]:
        values = self._call("get_runtime_load_ids")
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise RuntimeError("SGLang inference returned invalid runtime load IDs")
        return tuple(values)

    def initialize_version(self, runtime_load_id: str) -> None:
        """Stamp Reef's initial version while the coordinator keeps serving paused."""
        if not isinstance(runtime_load_id, str) or not runtime_load_id:
            raise ValueError("initial runtime load ID must be a nonempty string")
        self._engines().collective_rpc("set_runtime_load_id", args=(runtime_load_id,), timeout=CONTROL_TIMEOUT_S)

    def pause(self) -> None:
        self._call("pause_generation_for_update")

    def resume(self) -> None:
        self._call("continue_generation_after_update")

    def recover(self) -> None:
        self._call("recover_updatable_engines")

    def abort(self) -> None:
        self._call("terminate_updatable_engines")

    def offload(self, tags: tuple[str, ...] | None) -> None:
        self._call("offload", tags)

    def onload_weights(self) -> None:
        self._call("onload_weights")

    def onload_kv(self) -> None:
        self._call("onload_kv")

    def load_adapter_files(self, name: str, path: Path, runtime_load_id: str | None) -> None:
        self._call("load_adapter_from_disk", name, str(path), runtime_load_id)

    def unload_adapter(self, name: str) -> None:
        results = self._engines().collective_rpc(
            "unload_lora_adapter", kwargs={"lora_name": name}, timeout=CONTROL_TIMEOUT_S
        )
        for result in results:
            if result is not None and (not isinstance(result, Mapping) or result.get("success") is not True):
                raise RuntimeError(f"engine kept adapter {name!r}: {result!r}")

    def _engines(self) -> RayExecutor:
        engines, *_ = self._call("get_updatable_engines_and_lock")
        return RayExecutor.from_workers(engines)

    def _call(self, method: str, *args: Any) -> Any:
        return self._executor.rpc(0, method, args=args, timeout=CONTROL_TIMEOUT_S)
