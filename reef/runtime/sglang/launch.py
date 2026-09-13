"""Ray placement and native SGLang launch, independent of any trainer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from reef.runtime.sglang.config import SGLangConfig, SGLangGroupConfig
from reef.runtime.sglang.engine import ReefSGLangEngine
from reef.runtime.sglang.plugin import REEF_SGLANG_PLUGIN_ENV, SGLANG_PLUGIN_NAME
from reef.runtime.sglang.process import launch_router, node_address_and_port, wait_ready


def engine_environment(config: SGLangConfig) -> dict[str, str]:
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


class SGLangEngineGroup:
    """Logical replicas, each backed by one actor per participating node."""

    def __init__(
        self,
        config: SGLangConfig,
        group: SGLangGroupConfig,
        placement: Any,
        gpu_offset: int,
        router: tuple[str, int],
        external: dict[str, Any] | None = None,
    ) -> None:
        self.config, self.group, self.placement = config, group, placement
        self.gpu_offset = gpu_offset
        self.router = router
        self.external = external
        self.num_gpus_per_engine = group.gpus_per_engine
        self.worker_type = group.worker_type
        self.nodes_per_engine = 1 if external else max(1, group.gpus_per_engine // config.gpus_per_node)
        local_width = min(group.gpus_per_engine, config.gpus_per_node)
        count = 1 if external else group.num_gpus // local_width
        self.all_engines: list[Any] = [] if group.worker_type == "placeholder" else [None] * count
        self.num_new_engines = 0
        self.needs_offload = bool(config.offload and gpu_offset < config.shared_gpus and not external)

    @property
    def engines(self) -> list[Any]:
        return self.all_engines[:: self.nodes_per_engine]

    def parallel_config(self) -> dict[str, int]:
        options = {**self.config.options, **self.group.options}
        pp = int(options.get("pp_size") or 1)
        return {
            "tp_size": int(options.get("tp_size") or self.num_gpus_per_engine // pp),
            "pp_size": pp,
            "ep_size": int(options.get("ep_size") or 1),
            "moe_dp_size": int(options.get("moe_dp_size") or 1),
        }

    def start_engines(self, cursors: dict[str, int]) -> list[Any]:
        self.num_new_engines = 0
        local_width = min(self.num_gpus_per_engine, self.config.gpus_per_node)
        created = []
        for index, engine in enumerate(self.all_engines):
            if engine is not None:
                continue
            options: dict[str, Any] = {
                "num_cpus": 0.2,
                "num_gpus": 0 if self.external else 0.2,
                "runtime_env": {"env_vars": engine_environment(self.config)},
            }
            base_gpu = 0
            if not self.external:
                pg, bundles, devices = self.placement
                offset = self.gpu_offset + index * local_width
                if offset + local_width > len(devices):
                    raise ValueError("inference placement is smaller than its SGLang engine groups")
                base_gpu = int(devices[offset])
                if [int(device) for device in devices[offset : offset + local_width]] != list(
                    range(base_gpu, base_gpu + local_width)
                ):
                    raise ValueError("SGLang engines require contiguous GPUs within each node")
                options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=bundles[offset],
                    placement_group_capture_child_tasks=True,
                )
            overrides = dict(self.group.options)
            if self.config.offload and not self.needs_offload:
                overrides.setdefault("enable_memory_saver", False)
            actor = (
                ray.remote(ReefSGLangEngine)
                .options(**options)
                .remote(
                    self.config,
                    rank=index,
                    worker_type=self.worker_type,
                    base_gpu_id=base_gpu,
                    sglang_overrides=overrides,
                    num_gpus_per_engine=self.num_gpus_per_engine,
                    external_url=self.external["url"] if self.external else None,
                )
            )
            self.all_engines[index] = actor
            created.append(index)
        self.num_new_engines = len(created)
        addresses: dict[int, dict[str, Any]] = {}
        for index in created:
            actor = self.all_engines[index]
            if self.external:
                host, port = self.external["host"], self.external["port"]
                addresses[index] = {
                    "host": host,
                    "port": port,
                    "nccl_port": None,
                    "dist_init_addr": f"{host}:{port}",
                    "disaggregation_bootstrap_port": self.external.get("disaggregation_bootstrap_port"),
                }
                continue
            host, _ = ray.get(actor._get_current_node_ip_and_free_port.remote())
            width = 34 + int(self.group.options.get("dp_size", self.config.options.get("dp_size")) or 1)
            _, port = ray.get(
                actor._get_current_node_ip_and_free_port.remote(start_port=cursors.get(host, 15000), consecutive=width)
            )
            cursors[host] = port + width
            addresses[index] = {
                "host": host,
                "port": port,
                "nccl_port": port + 1,
                "disaggregation_bootstrap_port": port + 2,
            }
            if index % self.nodes_per_engine == 0:
                for node in range(index, index + self.nodes_per_engine):
                    addresses.setdefault(node, {})["dist_init_addr"] = f"{host}:{port + 3}"
        # Preserve node-zero's rendezvous when subsequent per-node ports are filled.
        for index in created:
            if not self.external:
                head = index - index % self.nodes_per_engine
                root = addresses[head]
                addresses[index]["dist_init_addr"] = f"{root['host']}:{root['port'] + 3}"
        return [
            self.all_engines[index].init.remote(
                **addresses[index], router_ip=self.router[0], router_port=self.router[1]
            )
            for index in created
        ]


@dataclass
class SGLangModel:
    server_groups: list[SGLangEngineGroup] = field(default_factory=list)
    update_weights: bool = True

    @property
    def all_engines(self) -> list[Any]:
        return [engine for group in self.server_groups for engine in group.all_engines]

    @property
    def engines(self) -> list[Any]:
        return [engine for group in self.server_groups for engine in group.engines]

    @property
    def num_new_engines(self) -> int:
        return sum(group.num_new_engines for group in self.server_groups)

    @num_new_engines.setter
    def num_new_engines(self, value: int) -> None:
        for group in self.server_groups:
            group.num_new_engines = value

    @property
    def engine_gpu_counts(self) -> list[int]:
        return [group.num_gpus_per_engine for group in self.server_groups for _ in group.engines]

    @property
    def engine_gpu_offsets(self) -> list[int]:
        return [
            group.gpu_offset + index * group.num_gpus_per_engine
            for group in self.server_groups
            for index in range(len(group.engines))
        ]

    @property
    def engine_parallel_configs(self) -> list[dict[str, int]]:
        return [group.parallel_config() for group in self.server_groups for _ in group.engines]

    def offload(self):
        return ray.get(
            [
                engine.release_memory_occupation.remote()
                for group in self.server_groups
                if group.needs_offload
                for engine in group.engines
                if engine is not None
            ]
        )

    def onload(self, tags=None):
        return ray.get(
            [
                engine.resume_memory_occupation.remote(tags=tags)
                for group in self.server_groups
                if group.needs_offload
                for engine in group.engines
                if engine is not None
            ]
        )

    def onload_weights(self):
        return self.onload(["weights"])

    def onload_kv(self):
        return self.onload(["kv_cache", "cuda_graph"])

    def recover(self) -> None:
        cursors: dict[str, int] = {}
        missing = [
            (group, [i for i, engine in enumerate(group.all_engines) if engine is None])
            for group in self.server_groups
        ]
        pending = [ref for group in self.server_groups for ref in group.start_engines(cursors)]
        if pending:
            ray.get(pending)
        engines = [
            group.all_engines[index]
            for group, indices in missing
            if group.needs_offload
            for index in indices
            if index % group.nodes_per_engine == 0
        ]
        if engines:
            ray.get([engine.release_memory_occupation.remote() for engine in engines])
            ray.get([engine.resume_memory_occupation.remote(tags=["weights"]) for engine in engines])


class SGLangCluster:
    """Track each owned process/actor before startup can fail."""

    def __init__(self, config: SGLangConfig, placement: Any) -> None:
        self.config, self.placement = config, placement
        self.servers: dict[str, SGLangModel] = {}
        self.routers: list[Any] = []
        self.endpoint: str | None = None

    def _router(self, index: int, pd: bool) -> tuple[str, int]:
        if index == 0 and self.config.router_host:
            if self.config.router_port is None:
                raise ValueError("an external router needs a port")
            return self.config.router_host, self.config.router_port
        from dataclasses import fields

        from sglang_router.launch_router import RouterArgs

        host, port = node_address_and_port(start_port=3000)
        if index == 0 and self.config.router_port is not None:
            port = self.config.router_port
        _, metrics = node_address_and_port(start_port=max(4000, port + 1))
        names = {item.name for item in fields(RouterArgs)}
        options = {key: value for key, value in self.config.router_options.items() if key in names}
        options.update(
            host=host.strip("[]"),
            port=port,
            prometheus_port=metrics,
            pd_disaggregation=pd,
            disable_circuit_breaker=True,
            disable_health_check=True,
        )
        process = launch_router(options)
        self.routers.append(process)
        wait_ready(f"http://{host}:{port}", process, self.config.startup_timeout, path="/health")
        return host, port

    def start(self) -> None:
        if self.config.external_engines:
            self._attach_external()
            return
        gpu_offset = 0
        cursors: dict[str, int] = {}
        pending = []
        for index, model_config in enumerate(self.config.resolved_models):
            router = self._router(
                index, any(group.worker_type in {"prefill", "decode"} for group in model_config.groups)
            )
            if index == 0:
                self.endpoint = f"http://{router[0]}:{router[1]}"
            model = SGLangModel(update_weights=model_config.update_weights)
            self.servers[model_config.name] = model
            # Preserve allocation order; only startup order puts encoders first.
            for group_config in model_config.groups:
                group = SGLangEngineGroup(self.config, group_config, self.placement, gpu_offset, router)
                model.server_groups.append(group)
                gpu_offset += group_config.num_gpus
            encoder_urls = []
            for group in model.server_groups:
                if group.worker_type == "encoder":
                    ray.get(group.start_engines(cursors))
                    encoder_urls.extend(ray.get([engine.get_url.remote() for engine in group.engines]))
            for group in model.server_groups:
                if group.worker_type == "encoder":
                    continue
                if encoder_urls and group.worker_type in {"regular", "prefill"}:
                    group.group.options.setdefault("language_only", True)
                    group.group.options.setdefault("encoder_urls", encoder_urls)
                pending.extend(group.start_engines(cursors))
        if pending:
            ray.get(pending)

    def _attach_external(self) -> None:
        router = self._router(
            0, any(info["worker_type"] in {"prefill", "decode"} for info in self.config.external_engines)
        )
        self.endpoint = f"http://{router[0]}:{router[1]}"
        model = SGLangModel()
        self.servers["default"] = model
        offset = 0
        pending = []
        for info in self.config.external_engines:
            group_config = SGLangGroupConfig(
                info["worker_type"], info["num_gpus"], info["num_gpus"], info.get("server_info", {})
            )
            group = SGLangEngineGroup(self.config, group_config, None, offset, router, external=info)
            model.server_groups.append(group)
            pending.extend(group.start_engines({}))
            offset += info["num_gpus"]
        ray.get(pending)
