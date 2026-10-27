"""Own vLLM engines, their recovery and weight loads independently of training."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import ray

from reef.inference.process import EngineStartup, retire_engines
from reef.inference.vllm.config import VLLMConfig
from reef.inference.vllm.launch import VLLMEngineGroup, VLLMEngineHealthChecks
from reef.runtime.recovery import (
    EngineHealthMonitor,
    HealthMonitorConfig,
    InferenceControl,
    InferenceEngineGroup,
    InferenceMonitor,
)


class VLLMWorker:
    """Own the serving engines and their health monitor."""

    def __init__(self, config: VLLMConfig, pg: Any) -> None:
        self.config = config
        self.group = VLLMEngineGroup(config, pg)
        self._health_monitor: EngineHealthMonitor | None = None
        self._closed = False
        try:
            startup = EngineStartup()
            startup.start(self.group)
            startup.wait()
            self._control = InferenceControl(_VLLMEngineGroup(self), _VLLMInferenceMonitor(self))
            if config.health_enabled:
                monitor = EngineHealthMonitor(
                    VLLMEngineHealthChecks(self.group),
                    HealthMonitorConfig(
                        interval=config.health_interval,
                        timeout=config.health_timeout,
                        first_wait=config.health_first_wait,
                    ),
                )
                self._health_monitor = monitor
                monitor.start()
                monitor.resume()
        except BaseException:
            with suppress(Exception):
                self.shutdown()
            raise

    @property
    def engines(self) -> list[Any]:
        return self.group.engines

    def inference_url(self) -> str:
        if self.config.router_url:
            return self.config.router_url
        return ray.get(self.engines[0].get_url.remote())

    def get_runtime_load_ids(self) -> list[str]:
        return ray.get([engine.get_runtime_load_id.remote() for engine in self.engines])

    def load_adapter_from_disk(self, lora_name: str, lora_path: str, runtime_load_id: str | None = None) -> None:
        """Load one adapter directory into every engine and serve it as ``runtime_load_id``.

        The engines read the directory themselves. Generation is already
        paused by the caller, so no request observes an engine mid-load.
        """
        engines = self.engines
        if not engines:
            raise RuntimeError("no vLLM engines can load an adapter")
        results = ray.get(
            [engine.load_lora_adapter_from_disk.remote(lora_name=lora_name, lora_path=lora_path) for engine in engines]
        )
        for result in results:
            if result.get("success") is not True:
                raise RuntimeError(f"vLLM refused adapter {lora_name!r}: {result.get('message', result)}")
        if runtime_load_id is not None:
            ray.get([engine.set_runtime_load_id.remote(runtime_load_id) for engine in engines])

    def load_weights_from_disk(self, model_path: str, runtime_load_id: str | None = None) -> None:
        """Reload every engine's weights from a Hugging Face directory and serve them as ``runtime_load_id``.

        The engines read the directory themselves. Generation is already
        paused by the caller.
        """
        engines = self.engines
        if not engines:
            raise RuntimeError("no vLLM engines can load weights")
        ray.get(
            [engine.update_weights_from_disk.remote(model_path, runtime_load_id=runtime_load_id) for engine in engines]
        )

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

    def health_monitoring_pause(self) -> None:
        if self._health_monitor is not None:
            self._health_monitor.pause()

    def health_monitoring_resume(self) -> None:
        if self._health_monitor is not None and not self._control.paused:
            self._health_monitor.resume()

    def check_health(self) -> None:
        """Raise if the monitor failed or a live engine actor is unreachable.

        A ``None`` slot is an engine the owner already retired and will replace
        through recovery; reporting it here would turn a recoverable
        publication into a deployment failure.
        """
        if self._health_monitor is not None:
            self._health_monitor.check_health()
        engines = self.engines
        if engines:
            ray.get([engine.__ray_ready__.remote() for engine in engines], timeout=30)

    def shutdown(self) -> None:
        if self._closed:
            return
        # A failed monitor drain must prevent engine mutation and remain retryable.
        if self._health_monitor is not None:
            self._health_monitor.stop()
            self._health_monitor = None
        self._closed = True
        retire_engines(self.engines)
        self.group.all_engines = [None] * len(self.group.all_engines)


class _VLLMEngineGroup(InferenceEngineGroup):
    """Ray fan-out and engine replacement behind Reef's control contract."""

    def __init__(self, worker: VLLMWorker) -> None:
        self._worker = worker

    @property
    def owned(self) -> bool:
        return True

    def pause(self) -> Any:
        """Stop generation for a weight load; a ``retract`` pause also leaves no reusable cache entry."""
        mode = self._worker.config.pause_mode
        return ray.get([engine.pause_generation.remote(mode) for engine in self._worker.engines])

    def resume(self) -> Any:
        return ray.get([engine.continue_generation.remote() for engine in self._worker.engines])

    def recover(self) -> int:
        group = self._worker.group
        dead = sum(engine is None for engine in group.all_engines)
        if dead:
            group.recover()
        return dead - sum(engine is None for engine in group.all_engines)

    def terminate(self) -> int:
        group = self._worker.group
        slots = [index for index, engine in enumerate(group.all_engines) if engine is not None]
        retire_engines([group.all_engines[index] for index in slots])
        for index in slots:
            group.all_engines[index] = None
        return len(slots)


class _VLLMInferenceMonitor(InferenceMonitor):
    def __init__(self, worker: VLLMWorker) -> None:
        self._worker = worker

    def pause(self) -> None:
        self._worker.health_monitoring_pause()

    def resume(self) -> None:
        self._worker.health_monitoring_resume()
