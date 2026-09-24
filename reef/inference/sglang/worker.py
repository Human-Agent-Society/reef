"""Own SGLang engines, recovery and update control independently of training."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import ray

from reef.inference.process import retire_engines
from reef.inference.sglang.config import SGLangConfig
from reef.inference.sglang.health import SGLangEngineHealthChecks
from reef.inference.sglang.launch import SGLangCluster, engine_environment
from reef.runtime.publication import WeightUpdateLock
from reef.runtime.recovery import (
    EngineHealthMonitor,
    HealthMonitorConfig,
    InferenceControl,
    InferenceEngines,
    InferenceMonitor,
    WeightUpdateConnection,
)


def recover_server(server) -> None:
    """Recover only when an engine is dead, preserving initial-connect state."""
    if not any(engine is None for group in server.server_groups for engine in group.all_engines):
        return
    server.recover()


class SGLangWorker:
    """Own serving engines, their update lock, monitors and locally launched routers."""

    def __init__(self, config: SGLangConfig, pg):
        self.config = config
        self.pg = pg
        self._cluster = SGLangCluster(config, pg)
        self.servers = self._cluster.servers
        self._health_monitors = []
        self._routers = self._cluster.routers
        self._closed = False
        try:
            self._cluster.start()
            self.rollout_engine_lock = self._new_rollout_engine_lock()
            self._control = self._create_control()
            if config.health_enabled and not config.external_engines:
                for server in self.servers.values():
                    for group in server.server_groups:
                        monitor = EngineHealthMonitor(
                            SGLangEngineHealthChecks(group),
                            HealthMonitorConfig(
                                interval=config.health_interval,
                                timeout=config.health_timeout,
                                first_wait=config.health_first_wait,
                            ),
                        )
                        self._health_monitors.append(monitor)
                        monitor.start()
                        monitor.resume()
        except BaseException:
            with suppress(Exception):
                self.shutdown()
            raise

    def dispose(self):
        failures = []
        for monitor in list(self._health_monitors):
            try:
                monitor.stop()
            except Exception as exc:
                failures.append(exc)
                continue
            self._health_monitors.remove(monitor)
        if failures:
            raise failures[0]

    def _get_updatable_server(self):
        return next((server for server in self.servers.values() if server.update_weights), None)

    @property
    def rollout_engines(self):
        return [engine for server in self.servers.values() for engine in server.engines]

    @property
    def updatable_rollout_engines(self):
        server = self._get_updatable_server()
        return [] if server is None else list(server.engines)

    def get_runtime_load_ids(self):
        return ray.get([engine.get_runtime_load_id.remote() for engine in self.updatable_rollout_engines])

    def load_adapter_from_disk(self, lora_name: str, lora_path: str, runtime_load_id: str | None = None) -> None:
        """Load one adapter directory into every updatable engine and serve it as ``runtime_load_id``.

        The sender that owns the files is elsewhere (a hosted trainer); the
        engines read the directory themselves. Generation is already paused
        by the publisher, so no request observes an engine mid-load.
        """
        engines = self.updatable_rollout_engines
        if not engines:
            raise RuntimeError("no updatable SGLang engines can load an adapter")
        results = ray.get(
            [engine.load_lora_adapter_from_disk.remote(lora_name=lora_name, lora_path=lora_path) for engine in engines]
        )
        for result in results:
            if isinstance(result, dict) and result.get("success") is False:
                raise RuntimeError(f"SGLang refused adapter {lora_name!r}: {result.get('message', result)}")
        if runtime_load_id is not None:
            ray.get([engine.set_runtime_load_id.remote(runtime_load_id) for engine in engines])

    def inference_url(self) -> str | None:
        return self._cluster.endpoint

    def _create_control(self) -> InferenceControl:
        return InferenceControl(
            _SGLangInferenceEngines(self), _SGLangWeightUpdateConnection(self), _SGLangInferenceMonitor(self)
        )

    def pause_generation_for_update(self):
        return self._control.pause()

    def continue_generation_after_update(self):
        return self._control.resume()

    def terminate_updatable_engines(self) -> int:
        return self._control.terminate()

    def get_updatable_engines_and_lock(self):
        server = self._get_updatable_server()
        if server is None:
            return [], self.rollout_engine_lock, 0, [], [], []
        num_new_engines = server.num_new_engines
        if self._control.reconnect_required:
            num_new_engines = max(num_new_engines, 1)
        return (
            server.engines,
            self.rollout_engine_lock,
            num_new_engines,
            server.engine_gpu_counts,
            server.engine_gpu_offsets,
            server.engine_parallel_configs,
        )

    def offload(self, tags=None):
        self.health_monitoring_pause()
        if not tags:
            return [server.offload() for server in self.servers.values()]
        # Only node-zero engines in shared allocations receive memory operations.
        handles = [
            engine.release_memory_occupation.remote(tags=list(tags))
            for server in self.servers.values()
            for group in server.server_groups
            if group.needs_offload
            for engine in group.engines
            if engine is not None
        ]
        return ray.get(handles) if handles else []

    def onload(self, tags=None):
        return [server.onload(tags) for server in self.servers.values()]

    def onload_weights(self):
        return [server.onload_weights() for server in self.servers.values()]

    def onload_kv(self):
        return [server.onload_kv() for server in self.servers.values()]

    def prepare_training_connection(self):
        self._control.prepare_training_connection()

    def recover_updatable_engines(self):
        self._control.recover()
        return self.get_updatable_engines_and_lock()

    def clear_updatable_num_new_engines(self):
        server = self._get_updatable_server()
        if server is not None:
            server.num_new_engines = 0
        self._control.acknowledge_reconnect()

    def _new_rollout_engine_lock(self):
        env_vars = engine_environment(self.config)
        return (
            ray.remote(WeightUpdateLock)
            .options(
                num_cpus=1,
                num_gpus=0,
                runtime_env={"env_vars": env_vars},
            )
            .remote()
        )

    def health_monitoring_pause(self):
        for monitor in self._health_monitors:
            monitor.pause()

    def health_monitoring_resume(self):
        if self._control.paused:
            return
        for monitor in self._health_monitors:
            monitor.resume()

    def check_weights(self, action: str):
        return ray.get([engine.check_weights.remote(action=action) for engine in self.rollout_engines])

    def check_health(self):
        """Raise if a monitor failed or a live engine actor is unreachable.

        Only live engines are probed. A ``None`` slot is an engine the owner
        already retired (a failed health probe or an aborted weight update) and
        will replace through recovery; reporting it here again would turn an
        in-progress, recoverable publication into a deployment failure.
        """
        for monitor in self._health_monitors:
            monitor.check_health()
        engines = [engine for server in self.servers.values() for engine in server.all_engines if engine is not None]
        if engines:
            ray.get([engine.__ray_ready__.remote() for engine in engines], timeout=30)

    def shutdown(self):
        if self._closed:
            return
        # A failed drain must prevent engine mutation and remain retryable.
        self.dispose()
        self._closed = True
        # External engines and shared placement groups are borrowed resources.
        if self.servers:
            retire_engines(
                [
                    engine
                    for server in self.servers.values()
                    for group in server.server_groups
                    for engine in group.all_engines
                    if engine is not None
                ]
            )
            self.servers = {}
        lock = getattr(self, "rollout_engine_lock", None)
        if lock is not None:
            with suppress(Exception):
                ray.kill(lock, no_restart=True)
            self.rollout_engine_lock = None
        # Only routers launched by this inference owner belong to it.
        for router in self._routers:
            if router.is_alive():
                router.terminate()
        for router in self._routers:
            router.join(timeout=5)
            if router.is_alive():
                router.kill()
                router.join(timeout=5)
        self._routers = []


class _SGLangInferenceEngines(InferenceEngines):
    """Ray fan-out and SGLang engine replacement behind Reef's control contract."""

    def __init__(self, worker: SGLangWorker) -> None:
        self._worker = worker

    @property
    def owned(self) -> bool:
        return not self._worker.config.external_engines

    def pause(self) -> Any:
        """Stop generation for a publication, leaving no reusable cache entry.

        A ``retract`` pause releases every in-flight request's KV, which is
        what lets the shared prefix cache be cleared: an entry the previous
        weights built must not be matchable by a request running under the
        next ones. A colocated engine drops the cache with its KV allocation
        anyway; clearing it here extends the same guarantee to a disjoint
        engine that shares prefixes. A failed flush raises, and the
        publication then retires the engines.
        """
        mode = self._worker.config.pause_mode
        engines = [engine for engine in self._worker.updatable_rollout_engines if engine is not None]
        result = ray.get([engine.pause_generation.remote(mode) for engine in engines])
        if mode == "retract":
            ray.get([engine.flush_cache.remote() for engine in engines])
        return result

    def resume(self) -> Any:
        return ray.get([engine.continue_generation.remote() for engine in self._worker.updatable_rollout_engines])

    def recover(self) -> None:
        server = self._worker._get_updatable_server()
        if server is not None:
            recover_server(server)

    def terminate(self) -> int:
        server = self._worker._get_updatable_server()
        groups = [] if server is None else server.server_groups
        slots = [
            (group, index) for group in groups for index, engine in enumerate(group.all_engines) if engine is not None
        ]
        retire_engines([group.all_engines[index] for group, index in slots])
        for group, index in slots:
            group.all_engines[index] = None
        return len(slots)


class _SGLangWeightUpdateConnection(WeightUpdateConnection):
    """Keep Ray lock handles and their replacement private to the integration."""

    def __init__(self, worker: SGLangWorker) -> None:
        self._worker = worker

    def is_usable(self) -> bool:
        status = ray.get(self._worker.rollout_engine_lock.status.remote())
        return isinstance(status, dict) and status.get("locked") is False and status.get("poisoned") is False

    def replace(self) -> None:
        old_lock = self._worker.rollout_engine_lock
        self._worker.rollout_engine_lock = self._worker._new_rollout_engine_lock()
        with suppress(Exception):
            ray.kill(old_lock, no_restart=True)


class _SGLangInferenceMonitor(InferenceMonitor):
    def __init__(self, worker: SGLangWorker) -> None:
        self._worker = worker

    def pause(self) -> None:
        self._worker.health_monitoring_pause()

    def resume(self) -> None:
        self._worker.health_monitoring_resume()
