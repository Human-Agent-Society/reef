"""Own SGLang engines, their recovery and weight loads independently of training."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import ray

from reef.execution.executor.placement import GpuBundles
from reef.inference.process import retire_engines
from reef.inference.recovery import InferenceControl, InferenceEngineGroup
from reef.inference.sglang.config import SGLangSettings
from reef.inference.sglang.launch import SGLangCluster


class SGLangWorker:
    """Own the serving engines and the locally launched router."""

    def __init__(self, config: SGLangSettings, pg: GpuBundles) -> None:
        self.config = config
        self.cluster = SGLangCluster(config, pg)
        self._closed = False
        try:
            self.cluster.start()
            self._control = InferenceControl(_SGLangEngineGroup(self))
        except BaseException:
            with suppress(Exception):
                self.shutdown()
            raise

    @property
    def all_engines(self) -> list[Any]:
        """One slot per engine; ``None`` marks an engine that awaits recovery. Empty before the group exists."""
        group = self.cluster.group
        return [] if group is None else group.all_engines

    @property
    def engines(self) -> list[Any]:
        """Live engine actors."""
        return [engine for engine in self.all_engines if engine is not None]

    def get_runtime_load_ids(self) -> list[str]:
        return ray.get([engine.get_runtime_load_id.remote() for engine in self.engines])

    def load_adapter_from_disk(self, lora_name: str, lora_path: str, runtime_load_id: str | None = None) -> None:
        """Load one adapter directory into every engine and serve it as ``runtime_load_id``.

        The engines read the directory themselves. Generation is already
        paused by the caller, so no request observes an engine mid-load.
        """
        engines = self.engines
        if not engines:
            raise RuntimeError("no SGLang engines can load an adapter")
        results = ray.get(
            [engine.load_lora_adapter_from_disk.remote(lora_name=lora_name, lora_path=lora_path) for engine in engines]
        )
        for result in results:
            if isinstance(result, dict) and result.get("success") is False:
                raise RuntimeError(f"SGLang refused adapter {lora_name!r}: {result.get('message', result)}")
        if runtime_load_id is not None:
            ray.get([engine.set_runtime_load_id.remote(runtime_load_id) for engine in engines])

    def load_weights_from_disk(self, model_path: str, runtime_load_id: str | None = None) -> None:
        """Reload every engine's weights from a Hugging Face directory and serve them as ``runtime_load_id``.

        The engines read the directory themselves. Generation is already
        paused by the caller.
        """
        engines = self.engines
        if not engines:
            raise RuntimeError("no SGLang engines can load weights")
        results = ray.get(
            [
                engine.update_weights_from_disk.remote(model_path=model_path, runtime_load_id=runtime_load_id)
                for engine in engines
            ]
        )
        for result in results:
            if isinstance(result, dict) and result.get("success") is False:
                raise RuntimeError(f"SGLang refused weights from {model_path!r}: {result.get('message', result)}")

    def inference_url(self) -> str | None:
        return self.cluster.endpoint

    def pause_generation_for_update(self) -> Any:
        return self._control.pause()

    def continue_generation_after_update(self) -> Any:
        return self._control.resume()

    def terminate_updatable_engines(self) -> int:
        return self._control.terminate()

    def updatable_engines(self) -> list[Any]:
        return self.engines

    def recover_updatable_engines(self) -> int:
        return self._control.recover()

    def check_health(self) -> None:
        """Raise if a live engine actor is unreachable.

        Only live engines are probed. A ``None`` slot is an engine the owner
        already retired (an aborted weight update) and will replace through
        recovery; reporting it here again would turn an in-progress,
        recoverable publication into a deployment failure.
        """
        engines = self.engines
        if engines:
            ray.get([engine.__ray_ready__.remote() for engine in engines], timeout=30)

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        all_engines = self.all_engines
        retire_engines(self.engines)
        all_engines[:] = [None] * len(all_engines)
        for router in self.cluster.routers:
            if router.is_alive():
                router.terminate()
        for router in self.cluster.routers:
            router.join(timeout=5)
            if router.is_alive():
                router.kill()
                router.join(timeout=5)
        self.cluster.routers.clear()


class _SGLangEngineGroup(InferenceEngineGroup):
    """Ray fan-out and SGLang engine replacement behind Reef's control contract."""

    def __init__(self, worker: SGLangWorker) -> None:
        self._worker = worker

    def pause(self) -> Any:
        """Stop generation for a weight load, leaving no reusable cache entry.

        A ``retract`` pause releases every in-flight request's KV, which is
        what lets the shared prefix cache be cleared: an entry the previous
        weights built must not be matchable by a request running under the
        next ones. A failed flush raises, and the caller then retires the
        engines.
        """
        mode = self._worker.config.pause_mode
        engines = self._worker.engines
        result = ray.get([engine.pause_generation.remote(mode) for engine in engines])
        if mode == "retract":
            ray.get([engine.flush_cache.remote() for engine in engines])
        return result

    def resume(self) -> Any:
        return ray.get([engine.continue_generation.remote() for engine in self._worker.engines])

    def recover(self) -> int:
        group = self._worker.cluster.group
        if group is None:
            return 0
        dead = sum(engine is None for engine in group.all_engines)
        if dead:
            group.recover()
        return dead - sum(engine is None for engine in group.all_engines)

    def terminate(self) -> int:
        all_engines = self._worker.all_engines
        slots = [index for index, engine in enumerate(all_engines) if engine is not None]
        retire_engines([all_engines[index] for index in slots])
        for index in slots:
            all_engines[index] = None
        return len(slots)
