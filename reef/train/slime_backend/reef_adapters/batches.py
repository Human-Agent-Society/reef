"""Slime tensorization and batch partitioning inside the Slime training component."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import ray

from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments

#: The loss-family arguments and learning-rate schedule state that the actor
#: workers switch to before a job; every rank receives it whole.
WORKER_SWITCH_KEY = "reef_worker_switch"

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


def tensorize_external_fields(data: dict[str, Any], extra_dtypes: Mapping[str, str] | None = None) -> None:
    """Tensorize Reef payload fields absent from the runtime's rollout map.

    ``extra_dtypes`` carries the loss family's declared tensor keys
    (``args.reef_rollout_tensor_dtypes``) by dtype name. Ragged fields (e.g.
    variable-length candidate token lists) stay undeclared and ride as plain
    lists.
    """
    import torch
    from reef.train.slime_backend.reef_adapters.worker_hooks import resolve_tensor_dtype

    dtypes = {"advantages": torch.float32, "returns": torch.float32}
    for key, name in (extra_dtypes or {}).items():
        dtypes[key] = resolve_tensor_dtype(name)
    for key, dtype in dtypes.items():
        if key in data:
            data[key] = [torch.as_tensor(value, dtype=dtype).detach().cpu().contiguous() for value in data[key]]


class TrainingBatchProcessor:
    """Split one rollout's training data across the data-parallel ranks, tensorize it and put it in Ray's store.

    ``args`` are the workers' startup arguments, from which the processor
    reads Slime's batch options. ``train_parallel_config`` is the workers'
    parallel layout, whose ``dp_size`` is a positive int. ``rollout_data_keys``
    are the per-sample keys beyond Slime's own that a rank's data carries, and
    ``tensor_dtypes`` maps a key to the dtype name it is tensorized as. Both
    start from ``args`` and grow only through ``register_keys``. The training
    component makes one processor and calls ``prepare_external_train_data``
    for each rollout.
    """

    def __init__(self, args: SlimeArguments, train_parallel_config: Mapping[str, Any]) -> None:
        dp_size = train_parallel_config.get("dp_size")
        if isinstance(dp_size, bool) or not isinstance(dp_size, int) or dp_size < 1:
            raise ValueError(f"the training parallel layout needs a positive dp_size, got {dp_size!r}")
        self.args = args
        self.train_parallel_config = dict(train_parallel_config)
        self.rollout_data_keys: tuple[str, ...] = tuple(args.custom_rollout_data_keys or ())
        self.tensor_dtypes: dict[str, str] = dict(args.reef_rollout_tensor_dtypes or {})

    def register_keys(self, keys: Sequence[str], dtypes: Mapping[str, str]) -> None:
        """Add a loss family's per-sample ``keys`` and the dtype names of its tensor keys.

        Raise ``ValueError``, and change nothing, when ``dtypes`` names a key
        that is registered with another dtype.
        """
        conflicts = {key: dtype for key, dtype in dtypes.items() if self.tensor_dtypes.get(key, dtype) != dtype}
        if conflicts:
            key, dtype = next(iter(conflicts.items()))
            raise ValueError(f"rollout key {key!r} is declared as {dtype}, and as {self.tensor_dtypes[key]} before")
        self.rollout_data_keys = tuple(dict.fromkeys((*self.rollout_data_keys, *keys)))
        self.tensor_dtypes.update(dtypes)

    def prepare_external_train_data(self, data):
        return self._split_train_data_by_dp(dict(data))

    def _split_train_data_by_dp(self, data):
        from slime.ray.rollout import _tensorize_rollout_data_for_training
        from slime.utils.misc import Box

        dp_size = self.train_parallel_config["dp_size"]
        total_lengths = [len(tokens) for tokens in data["tokens"]]
        data["total_lengths"] = total_lengths
        # Rollouts per optimizer step, in order; to_slime_rollout_data checked that they cover the rollouts.
        step_sizes = data.pop("external_step_sizes")
        partitions, micro_batch_indices, num_microbatches, global_batch_sizes = self._schedule_steps(data, step_sizes)

        references = []
        for rank in range(dp_size):
            partition = partitions[rank]
            rollout_data: dict[str, Any] = {"partition": partition}
            per_sample_keys = (*_PER_SAMPLE_KEYS, *self.rollout_data_keys)
            for key in dict.fromkeys(per_sample_keys):
                if key in data:
                    rollout_data[key] = [data[key][index] for index in partition]
            for key in ("raw_reward", "total_lengths", WORKER_SWITCH_KEY):
                if key in data:
                    rollout_data[key] = data[key]
            rollout_data.update(
                global_batch_sizes=global_batch_sizes,
                num_microbatches=num_microbatches,
                micro_batch_indices=micro_batch_indices[rank],
            )
            _tensorize_rollout_data_for_training(rollout_data)
            tensorize_external_fields(rollout_data, self.tensor_dtypes)
            transport = self.args.rollout_data_transport
            if transport == "nixl":
                references.append(Box(ray.put(rollout_data, _tensor_transport="nixl")))
            elif transport == "object-store":
                references.append(Box(ray.put(rollout_data)))
            else:
                raise ValueError(f"unsupported rollout data transport: {transport!r}")
        return references

    def _schedule_steps(self, data, step_sizes: list[int]):
        """Run Slime's DP/micro-batch packer one step at a time and splice the results.

        ``build_dp_schedule`` packs every step independently and only knows a
        constant step size, so a schedule with a smaller final step is built
        by scheduling each step on its own rollouts. Sample indices are
        global into ``data``; micro-batch indices are local into each rank's
        partition, so they shift by the partition's length so far.
        """
        from slime.utils.dp_schedule import build_dp_schedule

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
