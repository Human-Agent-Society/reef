"""Ray placement and native SGLang launch, independent of any trainer."""

from __future__ import annotations

import logging
import os
import random
from typing import Any

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from reef.inference.process import (
    PORT_CONFLICT_ATTEMPTS,
    PORT_CONFLICT_SPREAD,
    EngineGroup,
    EngineStartup,
    PortConflictError,
    check_server_owner,
    ports_in_use,
)
from reef.inference.sglang.config import SGLangSettings
from reef.inference.sglang.engine import ReefSGLangEngine
from reef.inference.sglang.plugin import REEF_SGLANG_PLUGIN_ENV
from reef.inference.sglang.process import launch_router, node_address_and_port, wait_ready

logger = logging.getLogger(__name__)

SGLANG_PLUGIN_NAME = "reef"


def engine_environment(config: SGLangSettings) -> dict[str, str]:
    configured = config.env_vars.get("SGLANG_PLUGINS", os.environ.get("SGLANG_PLUGINS", ""))
    plugins = [name.strip() for name in configured.split(",") if name.strip()]
    return {
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
        "RAY_USE_UVLOOP": "0",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
        "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "true",
        "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
        "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
        "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
        "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
        "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
        **config.env_vars,
        REEF_SGLANG_PLUGIN_ENV: "1",
        "SGLANG_PLUGINS": ",".join(dict.fromkeys((*plugins, SGLANG_PLUGIN_NAME))),
    }


class SGLangEngineGroup(EngineGroup):
    """One model's engines: identical single-node replicas, one actor each, on the reserved GPUs.

    ``placement`` is the deployment's inference reservation: the placement
    group, one bundle index per GPU and the physical GPU ids in bundle order.
    ``router`` is the host and port of the router that every engine registers with.
    """

    def __init__(self, config: SGLangSettings, placement: Any, router: tuple[str, int]) -> None:
        self.config, self.placement = config, placement
        self.router = router
        self.all_engines: list[Any] = [None] * config.engine_count
        self.port_ranges: dict[int, tuple[str, range]] = {}

    @property
    def engines(self) -> list[Any]:
        """Live engine actors; a ``None`` slot is an engine awaiting recovery."""
        return [engine for engine in self.all_engines if engine is not None]

    def start_engines(self, cursors: dict[str, int]) -> dict[int, Any]:
        """Launch an actor for every empty slot; return each new slot's pending ``init`` call.

        ``cursors`` tracks the next free port range per host so engines on one
        node never race for the same ports.
        """
        created = [index for index, engine in enumerate(self.all_engines) if engine is None]
        for index in created:
            self.all_engines[index] = self._launch_actor(index)
        return self.init_engines(created, cursors)

    def init_engines(self, slots: list[int], cursors: dict[str, int]) -> dict[int, Any]:
        pending = {}
        width = 34 + int(self.config.options.get("dp_size") or 1)
        for index in slots:
            actor = self.all_engines[index]
            host, _ = ray.get(actor._get_current_node_ip_and_free_port.remote())
            _, port = ray.get(
                actor._get_current_node_ip_and_free_port.remote(start_port=cursors.get(host, 15000), consecutive=width)
            )
            cursors[host] = port + width
            self.port_ranges[index] = (host, range(port, port + width))
            pending[index] = actor.init.remote(
                host=host,
                port=port,
                nccl_port=port + 1,
                dist_init_addr=f"{host}:{port + 3}",
                router_ip=self.router[0],
                router_port=self.router[1],
            )
        return pending

    def _launch_actor(self, index: int) -> Any:
        """Start the actor of engine ``index`` on the bundle and the GPUs reserved for it."""
        pg, bundles, devices = self.placement
        width = self.config.gpus_per_engine
        offset = index * width
        if offset + width > len(devices):
            raise ValueError("inference placement is smaller than its SGLang engines")
        base_gpu = int(devices[offset])
        if [int(device) for device in devices[offset : offset + width]] != list(range(base_gpu, base_gpu + width)):
            raise ValueError("SGLang engines require contiguous GPUs within each node")
        strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=bundles[offset],
            placement_group_capture_child_tasks=True,
        )
        return (
            ray.remote(ReefSGLangEngine)
            .options(
                num_cpus=0.2,
                num_gpus=0.2,
                runtime_env={"env_vars": engine_environment(self.config)},
                scheduling_strategy=strategy,
            )
            .remote(self.config, rank=index, base_gpu_id=base_gpu)
        )

    def recover(self) -> None:
        """Relaunch dead engines."""
        startup = EngineStartup()
        startup.start(self)
        startup.wait()


class SGLangCluster:
    """Own the router and the engine group, and track each of them before startup can fail.

    ``group`` is ``None`` and ``endpoint`` is ``None`` until ``start`` made the
    router; ``endpoint`` is then the router's URL, which serves every engine.
    """

    def __init__(self, config: SGLangSettings, placement: Any) -> None:
        self.config, self.placement = config, placement
        self.group: SGLangEngineGroup | None = None
        self.routers: list[Any] = []
        self.endpoint: str | None = None

    def _router(self) -> tuple[str, int]:
        from dataclasses import fields

        from sglang_router.launch_router import RouterArgs

        configured_port = self.config.router_port
        names = {item.name for item in fields(RouterArgs)}
        options = {key: value for key, value in self.config.router_options.items() if key in names}
        start_port, metrics_start_port, attempt = 3000, 4000, 1
        while True:
            host, probed_port = node_address_and_port(start_port=start_port)
            port = probed_port if configured_port is None else configured_port
            # Metrics stay near 4000: the port above a configured router port can be another stack's router port.
            _, probed_metrics_port = node_address_and_port(start_port=metrics_start_port)
            if probed_metrics_port == port:
                # The router has not bound its HTTP port yet, so the probe cannot see it.
                _, metrics_port = node_address_and_port(start_port=port + 1)
            else:
                metrics_port = probed_metrics_port
            options.update(
                host=host.strip("[]"),
                port=port,
                # Without prometheus_host, the router serves metrics on loopback, where the probe cannot see it.
                prometheus_host=host.strip("[]"),
                prometheus_port=metrics_port,
                pd_disaggregation=False,
                disable_circuit_breaker=True,
                disable_health_check=True,
            )
            process = launch_router(options)
            self.routers.append(process)
            try:
                wait_ready(f"http://{host}:{port}", process, self.config.startup_timeout, path="/health")
                # Another stack's router can answer on a port that this router failed to bind.
                check_server_owner(host, port, process.pid)
                return host, port
            except RuntimeError as error:
                # Stop the router first: its own sockets must not count as held.
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
                self.routers.remove(process)
                if isinstance(error, PortConflictError):
                    held_ports = [port]
                else:
                    held_ports = ports_in_use(host, [port, metrics_port])
                if not held_ports or (configured_port is not None and port in held_ports):
                    raise
                held_addresses = ", ".join(f"{host}:{held_port}" for held_port in held_ports)
                if attempt == PORT_CONFLICT_ATTEMPTS:
                    raise RuntimeError(
                        f"inference router ports {held_addresses} were held by other processes "
                        f"after {attempt} startup attempts"
                    ) from error
                logger.warning(
                    "inference router ports %s are held by other processes; starting attempt %d of %d on new ports",
                    held_addresses,
                    attempt + 1,
                    PORT_CONFLICT_ATTEMPTS,
                )
                # Skip ports at random so two stacks that collided usually pick different ports.
                # A fresh generator seeds from the OS; a trainer may seed the global one identically in every stack.
                skipped_ports = random.Random().randrange(1, PORT_CONFLICT_SPREAD)
                start_port, metrics_start_port = port + skipped_ports, metrics_port + skipped_ports
                attempt += 1

    def start(self) -> None:
        router = self._router()
        self.endpoint = f"http://{router[0]}:{router[1]}"
        self.group = SGLangEngineGroup(self.config, self.placement, router)
        startup = EngineStartup()
        startup.start(self.group)
        startup.wait()
