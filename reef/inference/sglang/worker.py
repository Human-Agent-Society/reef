"""Own SGLang engines, their recovery and weight loads independently of training."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import ray

from reef.inference.process import retire_engines
from reef.inference.sglang.config import SGLangConfig
from reef.inference.sglang.health import SGLangEngineHealthChecks
from reef.inference.sglang.launch import SGLangCluster
from reef.runtime.recovery import (
    EngineHealthMonitor,
    HealthMonitorConfig,
    InferenceControl,
    InferenceEngineGroup,
    InferenceMonitor,
)


def recover_server(server) -> None:
    """Recover only when an engine is dead, preserving initial-connect state."""
    if not any(engine is None for group in server.server_groups for engine in group.all_engines):
        return
    server.recover()


class SGLangWorker:
    """Own serving engines, their monitors and locally launched routers."""

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
            self._control = InferenceControl(_SGLangEngineGroup(self), _SGLangInferenceMonitor(self))
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

        The engines read the directory themselves. Generation is already
        paused by the caller, so no request observes an engine mid-load.
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

    def load_weights_from_disk(self, model_path: str, runtime_load_id: str | None = None) -> None:
        """Reload every updatable engine's weights from a Hugging Face directory and serve them as ``runtime_load_id``.

        The engines read the directory themselves. Generation is already
        paused by the caller.
        """
        engines = self.updatable_rollout_engines
        if not engines:
            raise RuntimeError("no updatable SGLang engines can load weights")
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
        return self._cluster.endpoint

    def pause_generation_for_update(self):
        return self._control.pause()

    def continue_generation_after_update(self):
        return self._control.resume()

    def terminate_updatable_engines(self) -> int:
        return self._control.terminate()

    def updatable_engines(self):
        return self.updatable_rollout_engines

    def recover_updatable_engines(self) -> int:
        return self._control.recover()

    def health_monitoring_pause(self):
        for monitor in self._health_monitors:
            monitor.pause()

    def health_monitoring_resume(self):
        if self._control.paused:
            return
        for monitor in self._health_monitors:
            monitor.resume()

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


class _SGLangEngineGroup(InferenceEngineGroup):
    """Ray fan-out and SGLang engine replacement behind Reef's control contract."""

    def __init__(self, worker: SGLangWorker) -> None:
        self._worker = worker

    @property
    def owned(self) -> bool:
        return not self._worker.config.external_engines

    def pause(self) -> Any:
        """Stop generation for a weight load, leaving no reusable cache entry.

        A ``retract`` pause releases every in-flight request's KV, which is
        what lets the shared prefix cache be cleared: an entry the previous
        weights built must not be matchable by a request running under the
        next ones. A failed flush raises, and the caller then retires the
        engines.
        """
        mode = self._worker.config.pause_mode
        engines = [engine for engine in self._worker.updatable_rollout_engines if engine is not None]
        result = ray.get([engine.pause_generation.remote(mode) for engine in engines])
        if mode == "retract":
            ray.get([engine.flush_cache.remote() for engine in engines])
        return result

    def resume(self) -> Any:
        return ray.get([engine.continue_generation.remote() for engine in self._worker.updatable_rollout_engines])

    def recover(self) -> int:
        server = self._worker._get_updatable_server()
        if server is None:
            return 0
        dead = sum(engine is None for group in server.server_groups for engine in group.all_engines)
        recover_server(server)
        return dead - sum(engine is None for group in server.server_groups for engine in group.all_engines)

    def terminate(self) -> int:
        server = self._worker._get_updatable_server()
        groups = [] if server is None else server.generation_groups
        slots = [
            (group, index) for group in groups for index, engine in enumerate(group.all_engines) if engine is not None
        ]
        retire_engines([group.all_engines[index] for group, index in slots])
        for group, index in slots:
            group.all_engines[index] = None
        return len(slots)


class _SGLangInferenceMonitor(InferenceMonitor):
    def __init__(self, worker: SGLangWorker) -> None:
        self._worker = worker

    def pause(self) -> None:
        self._worker.health_monitoring_pause()

    def resume(self) -> None:
        self._worker.health_monitoring_resume()
