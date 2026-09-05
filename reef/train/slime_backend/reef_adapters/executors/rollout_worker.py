"""Slime/Ray SGLang launch, placement, recovery and weight-control boundary."""

from __future__ import annotations

import multiprocessing
from contextlib import suppress

import ray
from slime.ray.rollout import start_rollout_servers
from slime.ray.utils import add_default_ray_env_vars
from slime.utils.health_monitor import RolloutHealthMonitor
from slime.utils.http_utils import init_http_client
from slime.utils.logging_utils import configure_logger

from reef.train.slime_backend.reef_adapters.rollout.lock import ReefRolloutLock
from reef.train.slime_backend.reef_adapters.sglang.engine import install_sglang_extensions
from reef.train.slime_backend.reef_adapters.worker_hooks import reef_rollout_env_vars


def recover_server(server) -> None:
    """Recover only when an engine is dead, preserving initial-connect state."""
    if not any(engine is None for group in server.server_groups for engine in group.all_engines):
        return
    server.recover()


class SlimeRayRolloutWorker:
    """Own serving engines, their update lock, monitors and locally launched routers."""

    def __init__(self, args, pg):
        configure_logger()
        install_sglang_extensions()
        self.args = args
        self.pg = pg
        self.servers = {}
        self._health_monitors = []
        self._routers = []
        self._closed = False
        children_before = set(multiprocessing.active_children())
        try:
            if not args.debug_train_only:
                init_http_client(args)
                self.servers, init_handles = start_rollout_servers(args, pg)
                if init_handles:
                    ray.get(init_handles)
            self.rollout_engine_lock = self._new_rollout_engine_lock()
            self._weight_update_reconnect_required = False
            self._generation_paused_for_update = False
            if not args.debug_train_only and args.use_fault_tolerance:
                for server in self.servers.values():
                    for group in server.server_groups:
                        monitor = RolloutHealthMonitor(group, args)
                        self._health_monitors.append(monitor)
                        monitor.start()
                        monitor.resume()
        except BaseException:
            self._routers = [child for child in multiprocessing.active_children() if child not in children_before]
            with suppress(Exception):
                self.shutdown()
            raise
        self._routers = [child for child in multiprocessing.active_children() if child not in children_before]

    def dispose(self):
        for monitor in self._health_monitors:
            with suppress(Exception):
                monitor.stop()
        self._health_monitors = []

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

    def inference_url(self) -> str | None:
        """The SGLang router URL slime bound, once the servers are up.

        slime binds the router to the machine's LAN IP (never localhost) and
        records it on ``args`` inside this actor; this is the only place Reef
        can learn it without re-deriving the address itself.
        """
        ip = getattr(self.args, "sglang_router_ip", None)
        port = getattr(self.args, "sglang_router_port", None)
        return None if not ip or not port else f"http://{ip}:{port}"

    def pause_generation_for_update(self):
        mode = getattr(self.args, "weight_update_pause_mode", "retract")
        result = ray.get([engine.pause_generation.remote(mode) for engine in self.updatable_rollout_engines])
        self._generation_paused_for_update = True
        return result

    def continue_generation_after_update(self):
        result = ray.get([engine.continue_generation.remote() for engine in self.updatable_rollout_engines])
        self._generation_paused_for_update = False
        self.health_monitoring_resume()
        return result

    def terminate_updatable_engines(self) -> int:
        """Synchronously retire managed engines after an uncertain fan-out."""
        server = self._get_updatable_server()
        groups = [] if server is None else server.server_groups
        if not groups:
            return 0
        self.health_monitoring_pause()
        indexed_engines = [
            (group, index, engine)
            for group in groups
            for index, engine in enumerate(group.all_engines)
            if engine is not None
        ]
        shutdowns = []
        for _, _, engine in indexed_engines:
            with suppress(Exception):
                shutdowns.append(engine.shutdown.remote())
        if shutdowns:
            with suppress(Exception):
                ray.get(shutdowns, timeout=30)
        for group, index, engine in indexed_engines:
            with suppress(Exception):
                ray.kill(engine, no_restart=True)
            group.all_engines[index] = None
        return len(indexed_engines)

    def get_updatable_engines_and_lock(self):
        server = self._get_updatable_server()
        if server is None:
            return [], self.rollout_engine_lock, 0, [], [], []
        num_new_engines = server.num_new_engines
        if self._weight_update_reconnect_required:
            num_new_engines = max(num_new_engines, 1)
        return (
            server.engines,
            self.rollout_engine_lock,
            num_new_engines,
            server.engine_gpu_counts,
            server.engine_gpu_offsets,
            server.engine_parallel_configs,
        )

    def offload(self):
        self.health_monitoring_pause()
        return [server.offload() for server in self.servers.values()]

    def onload(self, tags=None):
        return [server.onload(tags) for server in self.servers.values()]

    def onload_weights(self):
        return [server.onload_weights() for server in self.servers.values()]

    def onload_kv(self):
        return [server.onload_kv() for server in self.servers.values()]

    def recover_updatable_engines(self):
        self.health_monitoring_pause()
        try:
            try:
                status = ray.get(self.rollout_engine_lock.status.remote())
            except Exception:
                status = None
            uncertain = (
                not isinstance(status, dict)
                or status.get("locked") is not False
                or status.get("poisoned") is not False
            )
            if uncertain and getattr(self.args, "rollout_external", False):
                raise RuntimeError(
                    "uncertain external SGLang weight update requires restarting the external deployment"
                )
            if uncertain:
                old_lock = self.rollout_engine_lock
                self.rollout_engine_lock = self._new_rollout_engine_lock()
                self._weight_update_reconnect_required = True
                with suppress(Exception):
                    ray.kill(old_lock, no_restart=True)

            server = self._get_updatable_server()
            if server is not None:
                recover_server(server)
            if self._generation_paused_for_update:
                mode = getattr(self.args, "weight_update_pause_mode", "retract")
                ray.get([engine.pause_generation.remote(mode) for engine in self.updatable_rollout_engines])
            return self.get_updatable_engines_and_lock()
        finally:
            self.health_monitoring_resume()

    def clear_updatable_num_new_engines(self):
        server = self._get_updatable_server()
        if server is not None:
            server.num_new_engines = 0
        self._weight_update_reconnect_required = False

    @staticmethod
    def _new_rollout_engine_lock():
        env_vars = add_default_ray_env_vars(reef_rollout_env_vars())
        return ReefRolloutLock.options(
            num_cpus=1,
            num_gpus=0,
            runtime_env={"env_vars": env_vars},
        ).remote()

    def health_monitoring_pause(self):
        for monitor in self._health_monitors:
            monitor.pause()

    def health_monitoring_resume(self):
        for monitor in self._health_monitors:
            monitor.resume()

    def check_weights(self, action: str):
        return ray.get([engine.check_weights.remote(action=action) for engine in self.rollout_engines])

    def check_health(self):
        engines = [engine for server in self.servers.values() for engine in server.all_engines if engine is not None]
        if engines:
            ray.get([engine.__ray_ready__.remote() for engine in engines], timeout=30)

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        self.dispose()
        # External engines and shared placement groups are borrowed resources.
        if not getattr(self.args, "rollout_external", False):
            engines = [
                engine
                for server in self.servers.values()
                for group in server.server_groups
                for engine in group.all_engines
                if engine is not None
            ]
            pending = []
            for engine in engines:
                with suppress(Exception):
                    pending.append(engine.shutdown.remote())
            if pending:
                with suppress(Exception):
                    ray.get(pending, timeout=30)
            for engine in engines:
                with suppress(Exception):
                    ray.kill(engine, no_restart=True)
            self.servers = {}
        lock = getattr(self, "rollout_engine_lock", None)
        if lock is not None:
            with suppress(Exception):
                ray.kill(lock, no_restart=True)
            self.rollout_engine_lock = None
        # Slime starts routers as multiprocessing children in this manager.
        # Only children created during this worker's launch belong to it.
        for router in self._routers:
            if router.is_alive():
                router.terminate()
        for router in self._routers:
            router.join(timeout=5)
            if router.is_alive():
                router.kill()
                router.join(timeout=5)
        self._routers = []
