"""Own vLLM engines, recovery and update control independently of training."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from typing import Any

import ray

from reef.inference.process import retire_engines
from reef.inference.vllm.config import VLLMConfig
from reef.inference.vllm.launch import VLLMEngineGroup, VLLMEngineHealthChecks, engine_environment
from reef.runtime.publication import WeightUpdateLock
from reef.runtime.recovery import (
    EngineHealthMonitor,
    HealthMonitorConfig,
    InferenceControl,
    InferenceEngines,
    InferenceMonitor,
    WeightUpdateConnection,
)


class VLLMWorker:
    """Own the serving engines, their update lock and the health monitor."""

    def __init__(self, config: VLLMConfig, pg: Any) -> None:
        self.config = config
        self.group = VLLMEngineGroup(config, pg)
        self._health_monitor: EngineHealthMonitor | None = None
        self._closed = False
        try:
            pending = self.group.start_engines({})
            if pending:
                ray.get(pending)
            self.rollout_engine_lock = self._new_rollout_engine_lock()
            self._control = InferenceControl(
                _VLLMInferenceEngines(self), _VLLMWeightUpdateConnection(self), _VLLMInferenceMonitor(self)
            )
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

        The sender that owns the files is elsewhere; the engines read the
        directory themselves. Generation is already paused by the publisher,
        so no request observes an engine mid-load.
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

    def pause_generation_for_update(self) -> Any:
        return self._control.pause()

    def continue_generation_after_update(self) -> Any:
        return self._control.resume()

    def terminate_updatable_engines(self) -> int:
        return self._control.terminate()

    def get_updatable_engines_and_lock(self) -> tuple[Any, ...]:
        engines = self.engines
        num_new_engines = self.group.num_new_engines
        if self._control.reconnect_required:
            num_new_engines = max(num_new_engines, 1)
        width = self.config.gpus_per_engine
        return (
            engines,
            self.rollout_engine_lock,
            num_new_engines,
            [width for _ in engines],
            [index * width for index, engine in enumerate(self.group.all_engines) if engine is not None],
            [self.group.parallel_config() for _ in engines],
        )

    def offload(self, tags: Sequence[str] | None = None) -> list[Any]:
        self.health_monitoring_pause()
        if not self.group.needs_offload:
            return []
        selection = list(tags) if tags else None
        return ray.get([engine.release_memory_occupation.remote(tags=selection) for engine in self.engines])

    def onload(self, tags: Sequence[str] | None = None) -> list[Any]:
        if not self.group.needs_offload:
            return []
        selection = list(tags) if tags else None
        return ray.get([engine.resume_memory_occupation.remote(tags=selection) for engine in self.engines])

    def onload_weights(self) -> list[Any]:
        return self.onload(["weights"])

    def onload_kv(self) -> list[Any]:
        return self.onload(["kv_cache", "cuda_graph"])

    def prepare_training_connection(self) -> None:
        self._control.prepare_training_connection()

    def recover_updatable_engines(self) -> tuple[Any, ...]:
        self._control.recover()
        return self.get_updatable_engines_and_lock()

    def clear_updatable_num_new_engines(self) -> None:
        self.group.num_new_engines = 0
        self._control.acknowledge_reconnect()

    def _new_rollout_engine_lock(self) -> Any:
        return (
            ray.remote(WeightUpdateLock)
            .options(num_cpus=1, num_gpus=0, runtime_env={"env_vars": engine_environment(self.config)})
            .remote()
        )

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
        lock = getattr(self, "rollout_engine_lock", None)
        if lock is not None:
            with suppress(Exception):
                ray.kill(lock, no_restart=True)
            self.rollout_engine_lock = None


class _VLLMInferenceEngines(InferenceEngines):
    """Ray fan-out and engine replacement behind Reef's control contract."""

    def __init__(self, worker: VLLMWorker) -> None:
        self._worker = worker

    @property
    def owned(self) -> bool:
        return True

    def pause(self) -> Any:
        """Stop generation for a publication; a ``retract`` pause also leaves no reusable cache entry."""
        mode = self._worker.config.pause_mode
        return ray.get([engine.pause_generation.remote(mode) for engine in self._worker.engines])

    def resume(self) -> Any:
        return ray.get([engine.continue_generation.remote() for engine in self._worker.engines])

    def recover(self) -> None:
        if any(engine is None for engine in self._worker.group.all_engines):
            self._worker.group.recover()

    def terminate(self) -> int:
        group = self._worker.group
        slots = [index for index, engine in enumerate(group.all_engines) if engine is not None]
        retire_engines([group.all_engines[index] for index in slots])
        for index in slots:
            group.all_engines[index] = None
        return len(slots)


class _VLLMWeightUpdateConnection(WeightUpdateConnection):
    """Keep Ray lock handles and their replacement private to the integration."""

    def __init__(self, worker: VLLMWorker) -> None:
        self._worker = worker

    def is_usable(self) -> bool:
        status = ray.get(self._worker.rollout_engine_lock.status.remote())
        return isinstance(status, dict) and status.get("locked") is False and status.get("poisoned") is False

    def replace(self) -> None:
        old_lock = self._worker.rollout_engine_lock
        self._worker.rollout_engine_lock = self._worker._new_rollout_engine_lock()
        with suppress(Exception):
            ray.kill(old_lock, no_restart=True)


class _VLLMInferenceMonitor(InferenceMonitor):
    def __init__(self, worker: VLLMWorker) -> None:
        self._worker = worker

    def pause(self) -> None:
        self._worker.health_monitoring_pause()

    def resume(self) -> None:
        self._worker.health_monitoring_resume()
