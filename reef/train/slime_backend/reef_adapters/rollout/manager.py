"""External-batch rollout manager composed from runtime primitives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import ray
import torch
from slime.ray.rollout import _tensorize_rollout_data_for_training
from slime.ray.utils import add_default_ray_env_vars
from slime.utils.dp_schedule import build_dp_schedule
from slime.utils.logging_utils import configure_logger
from slime.utils.misc import Box

from reef.runtime.executor import Executor, ExecutorConfig
from reef.train.slime_backend.reef_adapters.executors.rollout import rollout_executor_class
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
    """Compatibility entry point for the default Slime/Ray recovery helper."""
    from reef.train.slime_backend.reef_adapters.executors.rollout_worker import recover_server as implementation

    implementation(server)


class ReefRolloutManagerImpl:
    """Batch scheduling is independent of the serving launch/control backend."""

    def __init__(self, args, pg):
        configure_logger()
        self.args = args
        self.pg = pg
        self._serving = Executor.create(
            ExecutorConfig(
                backend=rollout_executor_class(args),
                options={**getattr(args, "reef_rollout_executor_options", {}), "args": args, "pg": pg},
            )
        )

    def dispose(self):
        self._serving.shutdown()

    def check_health(self):
        self._serving.check_health(timeout=30)

    def inference_url(self):
        return self._serving.rpc(0, "inference_url", timeout=14_400)

    def get_runtime_load_ids(self):
        return self._serving.rpc(0, "get_runtime_load_ids", timeout=14_400)

    def pause_generation_for_update(self):
        return self._serving.rpc(0, "pause_generation_for_update", timeout=14_400)

    def continue_generation_after_update(self):
        return self._serving.rpc(0, "continue_generation_after_update", timeout=14_400)

    def terminate_updatable_engines(self):
        return self._serving.rpc(0, "terminate_updatable_engines", timeout=14_400)

    def get_updatable_engines_and_lock(self):
        return self._serving.rpc(0, "get_updatable_engines_and_lock", timeout=14_400)

    def offload(self):
        return self._serving.rpc(0, "offload", timeout=14_400)

    def onload(self, tags=None):
        return self._serving.rpc(0, "onload", args=(tags,), timeout=14_400)

    def onload_weights(self):
        return self._serving.rpc(0, "onload_weights", timeout=14_400)

    def onload_kv(self):
        return self._serving.rpc(0, "onload_kv", timeout=14_400)

    def recover_updatable_engines(self):
        return self._serving.rpc(0, "recover_updatable_engines", timeout=14_400)

    def clear_updatable_num_new_engines(self):
        return self._serving.rpc(0, "clear_updatable_num_new_engines", timeout=14_400)

    def health_monitoring_pause(self):
        return self._serving.rpc(0, "health_monitoring_pause", timeout=14_400)

    def health_monitoring_resume(self):
        return self._serving.rpc(0, "health_monitoring_resume", timeout=14_400)

    def check_weights(self, action: str):
        return self._serving.rpc(0, "check_weights", args=(action,), timeout=14_400)

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
    "rollout_executor_class",
    "tensorize_external_fields",
]
