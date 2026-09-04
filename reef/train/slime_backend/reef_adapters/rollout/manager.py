"""External-batch rollout manager composed from runtime primitives."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from typing import Any

import ray
import torch
from slime.ray.rollout import _tensorize_rollout_data_for_training, start_rollout_servers
from slime.ray.utils import add_default_ray_env_vars
from slime.utils.dp_schedule import build_dp_schedule
from slime.utils.health_monitor import RolloutHealthMonitor
from slime.utils.http_utils import init_http_client
from slime.utils.logging_utils import configure_logger
from slime.utils.misc import Box

from reef.train.slime_backend.reef_adapters.rollout.lock import ReefRolloutLock
from reef.train.slime_backend.reef_adapters.sglang.engine import install_sglang_extensions
from reef.train.slime_backend.reef_adapters.worker_hooks import reef_rollout_env_vars, resolve_tensor_dtype

_PER_SAMPLE_KEYS = (
    "tokens",
    "multimodal_train_inputs",
    "response_lengths",
    "rewards",
    "truncated",
    "loss_masks",
    "round_number",
    "sample_indices",
    "rollout_ids",
    "rollout_mask_sums",
    "rollout_log_probs",
    "advantages",
    "returns",
    "rollout_top_p_token_ids",
    "rollout_top_p_token_offsets",
    "rollout_routed_experts",
    "source_names",
    "prompt",
    "teacher_log_probs",
)

_EXTERNAL_TENSOR_DTYPES = {
    "advantages": torch.float32,
    "returns": torch.float32,
}


def tensorize_external_fields(data: dict[str, Any], extra_dtypes: Mapping[str, str] | None = None) -> None:
    """Tensorize Reef payload fields absent from the runtime's rollout map.

    ``extra_dtypes`` carries the loss family's declared tensor keys
    (``args.reef_rollout_tensor_dtypes``) by dtype name. Ragged fields (e.g.
    variable-length candidate token lists) stay undeclared and ride as plain
    lists.
    """
    dtypes = dict(_EXTERNAL_TENSOR_DTYPES)
    for key, name in (extra_dtypes or {}).items():
        dtypes[key] = resolve_tensor_dtype(name)
    for key, dtype in dtypes.items():
        if key in data:
            data[key] = [torch.as_tensor(value, dtype=dtype).detach().cpu().contiguous() for value in data[key]]


def recover_server(server) -> None:
    """Recover only when an engine is dead, preserving initial-connect state."""
    if not any(engine is None for group in server.server_groups for engine in group.all_engines):
        return
    server.recover()


class ReefRolloutManagerImpl:
    """Own serving lifecycle and package Reef-produced batches by DP rank."""

    def __init__(self, args, pg):
        configure_logger()
        install_sglang_extensions()
        self.args = args
        self.pg = pg
        if args.debug_train_only:
            self.servers = {}
            init_handles = []
        else:
            init_http_client(args)
            self.servers, init_handles = start_rollout_servers(args, pg)
        if init_handles:
            ray.get(init_handles)
        self.rollout_engine_lock = self._new_rollout_engine_lock()
        self._weight_update_reconnect_required = False
        self._generation_paused_for_update = False
        self._health_monitors = []
        if not args.debug_train_only and args.use_fault_tolerance:
            for server in self.servers.values():
                for group in server.server_groups:
                    monitor = RolloutHealthMonitor(group, args)
                    monitor.start()
                    monitor.resume()
                    self._health_monitors.append(monitor)

    def dispose(self):
        for monitor in self._health_monitors:
            monitor.stop()

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
        """Stop generation for a publication, leaving nothing that outlives the weights.

        A ``retract`` pause releases every in-flight request's KV, which is
        what lets the shared prefix cache be cleared: an entry the previous
        weights built must not be matchable by a request running under the
        next ones. A colocated engine drops the cache with its KV allocation
        anyway; clearing it here is what extends the same guarantee to a
        disjoint engine that opted into sharing prefixes.
        """
        mode = getattr(self.args, "weight_update_pause_mode", "retract")
        engines = self.updatable_rollout_engines
        result = ray.get([engine.pause_generation.remote(mode) for engine in engines])
        if mode == "retract":
            ray.get([engine.flush_cache.remote() for engine in engines])
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

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    def prepare_external_train_data(self, data):
        return self._split_train_data_by_dp(dict(data))

    def _split_train_data_by_dp(self, data):
        dp_size = self.train_parallel_config["dp_size"]
        total_lengths = [len(tokens) for tokens in data["tokens"]]
        data["total_lengths"] = total_lengths
        step_sizes = self._resolve_step_sizes(data)
        partitions, micro_batch_indices, num_microbatches, global_batch_sizes = self._schedule_steps(data, step_sizes)

        references = []
        for rank in range(dp_size):
            partition = partitions[rank]
            rollout_data: dict[str, Any] = {"partition": partition}
            per_sample_keys = (*_PER_SAMPLE_KEYS, *getattr(self.args, "custom_rollout_data_keys", ()))
            for key in dict.fromkeys(per_sample_keys):
                if key in data:
                    rollout_data[key] = [data[key][index] for index in partition]
            for key in ("raw_reward", "total_lengths"):
                if key in data:
                    rollout_data[key] = data[key]
            rollout_data.update(
                global_batch_sizes=global_batch_sizes,
                num_microbatches=num_microbatches,
                micro_batch_indices=micro_batch_indices[rank],
            )
            _tensorize_rollout_data_for_training(rollout_data)
            tensorize_external_fields(rollout_data, getattr(self.args, "reef_rollout_tensor_dtypes", None))
            transport = getattr(self.args, "rollout_data_transport", "object-store")
            if transport == "nixl":
                references.append(Box(ray.put(rollout_data, _tensor_transport="nixl")))
            elif transport == "object-store":
                references.append(Box(ray.put(rollout_data)))
            else:
                raise ValueError(f"unsupported rollout data transport: {transport!r}")
        return references

    def _resolve_step_sizes(self, data) -> list[int]:
        """Rollouts per optimizer step, in order — Reef's schedule or the configured size cut here."""
        rollout_count = len(dict.fromkeys(data["rollout_ids"]))
        step_sizes = data.pop("external_step_sizes", None)
        remainder = data.pop("external_remainder", "error")
        if step_sizes is not None:
            if sum(step_sizes) != rollout_count:
                raise ValueError(f"external_step_sizes {step_sizes!r} do not cover the {rollout_count} rollouts")
            return [int(size) for size in step_sizes]
        global_batch_size = self.args.global_batch_size
        full_steps, tail = divmod(rollout_count, global_batch_size)
        if tail and remainder != "partial":
            raise ValueError(
                f"{rollout_count} external rollouts do not form complete global batches of {global_batch_size}"
            )
        return [global_batch_size] * full_steps + ([tail] if tail else [])

    def _schedule_steps(self, data, step_sizes: list[int]):
        """Run Slime's DP/micro-batch packer one step at a time and splice the results.

        ``build_dp_schedule`` packs every step independently and only knows a
        constant step size, so a schedule with a smaller final step is built
        by scheduling each step on its own rollouts. Sample indices are
        global into ``data``; micro-batch indices are local into each rank's
        partition, so they shift by the partition's length so far.
        """
        dp_size = self.train_parallel_config["dp_size"]
        rollout_ids = data["rollout_ids"]
        total_lengths = data["total_lengths"]
        rollout_order = list(dict.fromkeys(rollout_ids))
        partitions: list[list[int]] = [[] for _ in range(dp_size)]
        micro_batch_indices: list[list[list[int]]] = [[] for _ in range(dp_size)]
        num_microbatches: list[int] = []
        global_batch_sizes: list[int] = []
        cursor = 0
        for size in step_sizes:
            step_rollouts = set(rollout_order[cursor : cursor + size])
            cursor += size
            sample_indices = [index for index, rollout in enumerate(rollout_ids) if rollout in step_rollouts]
            step_partitions, step_micro_batches, step_num_microbatches, step_batch_sizes = build_dp_schedule(
                self.args,
                self.train_parallel_config,
                [total_lengths[index] for index in sample_indices],
                global_batch_size=size,
                rollout_indices=[rollout_ids[index] for index in sample_indices],
            )
            for rank in range(dp_size):
                offset = len(partitions[rank])
                partitions[rank].extend(sample_indices[local] for local in step_partitions[rank])
                micro_batch_indices[rank].extend(
                    [offset + local for local in micro_batch] for micro_batch in step_micro_batches[rank]
                )
            num_microbatches.extend(step_num_microbatches)
            global_batch_sizes.extend(step_batch_sizes)
        return partitions, micro_batch_indices, num_microbatches, global_batch_sizes


ReefRolloutManager = ray.remote(ReefRolloutManagerImpl)


def create_rollout_manager(args, pg):
    runtime_env = add_default_ray_env_vars(reef_rollout_env_vars())
    options = {
        "num_cpus": 1,
        "num_gpus": 0,
        "runtime_env": {"env_vars": runtime_env},
    }
    if getattr(args, "rollout_data_transport", "object-store") == "nixl":
        options["enable_tensor_transport"] = True
    manager = ReefRolloutManager.options(**options).remote(args, pg)
    if args.check_weight_update_equal:
        ray.get(manager.check_weights.remote(action="snapshot"))
        ray.get(manager.check_weights.remote(action="reset_tensors"))
    if args.offload_rollout:
        ray.get(manager.offload.remote())
    return manager


__all__ = [
    "ReefRolloutManager",
    "ReefRolloutManagerImpl",
    "create_rollout_manager",
    "recover_server",
    "tensorize_external_fields",
]
