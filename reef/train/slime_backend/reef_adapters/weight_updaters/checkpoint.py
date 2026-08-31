"""Exact-version full and delta checkpoint publication."""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import ray
import safetensors.numpy
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle
from slime.backends.megatron_utils.hf_checkpoint_saver import save_hf_model_to_path
from slime.backends.megatron_utils.update_weight.update_weight_from_disk import UpdateWeightFromDisk
from slime.backends.megatron_utils.update_weight.update_weight_from_disk_delta import (
    UpdateWeightFromDiskDelta,
    _atomic_write,
)
from slime.utils.disk_delta import make_tensor_reader
from slime.utils.distributed_utils import get_gloo_group

from reef.train.slime_backend.reef_adapters.weight_updaters.base import SynchronizedWeightUpdateMixin

logger = logging.getLogger(__name__)


class ReefUpdateWeightFromDisk(SynchronizedWeightUpdateMixin, UpdateWeightFromDisk):
    """Write full checkpoints while retaining Reef's opaque serving token."""

    def __init__(
        self,
        *args: Any,
        weight_version_incarnation: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._initialize_weight_version(weight_version_incarnation)

    @torch.no_grad()
    def update_weights(self, *, manage_generation: bool = True, force_full: bool = False) -> None:
        self.weight_update_sequence += 1
        version_dir = Path(self.args.update_weight_disk_dir) / f"weight_v{self.weight_update_sequence:06d}"
        if dist.get_rank() == 0:
            shutil.rmtree(version_dir, ignore_errors=True)
        dist.barrier(group=get_gloo_group())
        version_dir.mkdir(parents=True, exist_ok=True)
        save_hf_model_to_path(
            self.args,
            version_dir,
            self.model,
            model_name=self.model_name,
            quantization_config=self.quantization_config,
            progress_desc="Save HF weights for update from disk",
        )
        dist.barrier(group=get_gloo_group())
        if self._post_write_hook is not None:
            self._post_write_hook(self.args, str(version_dir), list(self.rollout_engines))
        dist.barrier(group=get_gloo_group())


class ReefUpdateWeightFromDiskDelta(SynchronizedWeightUpdateMixin, UpdateWeightFromDiskDelta):
    """Delta updater that can recover by republishing a complete checkpoint."""

    _baseline_captured: bool

    def __init__(
        self,
        *args: Any,
        weight_version_incarnation: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._initialize_weight_version(weight_version_incarnation)

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
        engine_parallel_configs: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        self.rollout_engines = list(rollout_engines)
        self.rollout_engine_lock = rollout_engine_lock
        self._engine_gpu_counts = engine_gpu_counts
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )

    @torch.no_grad()
    def update_weights(self, *, manage_generation: bool = True, force_full: bool = False) -> None:
        self._reset_source_phases()
        if force_full:
            self._republish_full_weights(manage_generation=manage_generation)
            return
        if not self._baseline_captured:
            self._capture_baseline()
            self._baseline_captured = True
            return
        self.weight_update_sequence += 1
        self._publish()
        self._reload_engines(manage_generation=manage_generation)
        self._record_metrics()

    def _republish_full_weights(self, *, manage_generation: bool) -> None:
        self.weight_update_sequence += 1
        version_dir = os.path.join(self.delta_dir, f"weight_v{self.weight_update_sequence:06d}")
        self._run_rank_zero_action(
            lambda: shutil.rmtree(version_dir, ignore_errors=True),
            phase="clear full recovery checkpoint",
        )

        local_error: BaseException | None = None
        try:
            os.makedirs(version_dir, exist_ok=True)
            save_hf_model_to_path(
                self.args,
                version_dir,
                self.model,
                model_name=self.model_name,
                quantization_config=self.quantization_config,
                progress_desc="Save full recovery weights",
            )
            if self._post_write_hook is not None:
                self._post_write_hook(self.args, version_dir, list(self.rollout_engines))
        except BaseException as exc:
            local_error = exc
        self._raise_synchronized_update_error(local_error, phase="publish full recovery checkpoint")

        if manage_generation:
            pause_mode = getattr(self.args, "weight_update_pause_mode", "retract")
            self._run_rank_zero_action(
                lambda: ray.get([engine.pause_generation.remote(pause_mode) for engine in self.rollout_engines]),
                phase="pause generation",
            )
            self._run_rank_zero_action(
                lambda: ray.get([engine.flush_cache.remote() for engine in self.rollout_engines]),
                phase="flush rollout cache",
            )

        def activate() -> None:
            if self.args.update_weight_local_checkpoint_dir:
                ray.get([engine.pull_weights.remote(self.weight_update_sequence) for engine in self.rollout_engines])
                model_path = self.args.update_weight_local_checkpoint_dir
            else:
                model_path = version_dir
            ray.get(
                [
                    engine.update_weights_from_disk.remote(
                        model_path=model_path,
                        weight_version=str(self.weight_version),
                    )
                    for engine in self.rollout_engines
                ]
            )

        try:
            self._run_rank_zero_action(activate, phase="activate full recovery checkpoint")
        except BaseException:
            self._poison_transport()
            raise
        if manage_generation:
            self._run_rank_zero_action(
                lambda: ray.get([engine.continue_generation.remote() for engine in self.rollout_engines]),
                phase="continue generation",
            )

        self._snapshot.clear()
        local_error = None
        try:
            for name, tensor in self._iter_hf_tensors():
                self._snapshot[name] = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().reshape(-1)
        except BaseException as exc:
            local_error = exc
        self._raise_synchronized_update_error(local_error, phase="rebuild delta snapshot")
        self._baseline_captured = True

    def _capture_baseline(self) -> None:
        def prepare_base() -> None:
            shutil.rmtree(self.delta_dir, ignore_errors=True)
            os.makedirs(self.delta_dir, exist_ok=True)
            if self._post_write_hook is not None:
                self._post_write_hook(self.args, self.delta_dir, list(self.rollout_engines))
            ray.get([engine.pull_weights.remote(target_version=0) for engine in self.rollout_engines])

        self._run_rank_zero_action(prepare_base, phase="materialize delta baseline")
        read_hf = make_tensor_reader(self.args.hf_checkpoint)
        local_error: BaseException | None = None
        try:
            for name, tensor in self._iter_hf_tensors():
                try:
                    self._snapshot[name] = read_hf(name)
                except KeyError:  # noqa: PERF203 - the fallback is intentionally per tensor name
                    self._snapshot[name] = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().reshape(-1)
                    logger.warning("seed: %s absent from hf_checkpoint; using current weights", name)
        except BaseException as exc:
            local_error = exc
        self._raise_synchronized_update_error(local_error, phase="capture delta baseline")

        def finish_base() -> None:
            ray.get([engine.set_weight_version.remote(str(self.weight_version)) for engine in self.rollout_engines])
            logger.info(
                "[disk delta] captured baseline snapshot of %d tensors from %s",
                len(self._snapshot),
                self.args.hf_checkpoint,
            )

        self._run_rank_zero_action(finish_base, phase="initialize rollout weight version")

    def _publish(self) -> None:
        local_error: BaseException | None = None
        try:
            self._encode_delta()
        except BaseException as exc:
            local_error = exc
        self._raise_synchronized_update_error(local_error, phase="encode delta weights")
        local_error = None
        try:
            self._write_delta_files()
        except BaseException as exc:
            local_error = exc
        self._raise_synchronized_update_error(local_error, phase="write delta weights")

    def _write_delta_files(self) -> None:
        group = get_gloo_group()
        world, rank = dist.get_world_size(), dist.get_rank()
        counts: list[int | None] = [None] * world
        dist.all_gather_object(counts, int(bool(self._delta)), group=group)
        concrete_counts = [0 if count is None else count for count in counts]
        offset, total = sum(concrete_counts[:rank]), sum(concrete_counts)

        filename = None
        self.wire_bytes = 0
        if self._delta:
            filename = f"model-{offset:05d}-of-{total:05d}.safetensors"
            blob = safetensors.numpy.save(self._delta, metadata=self._checksums)
            self.wire_bytes = len(blob)
            _atomic_write(os.path.join(self._version_dir, filename), blob)

        maps: list[dict[str, str | None] | None] = [None] * world
        dist.all_gather_object(maps, dict.fromkeys(self._delta, filename), group=group)
        if rank == 0:
            index = {
                "metadata": {
                    "version": f"{self.weight_update_sequence:06d}",
                    "base_version": f"{self.weight_update_sequence - 1:06d}",
                    "delta_encoding": self.delta_encoding,
                    "compression_format": "zstd",
                    "checksum_format": self.checksum_algorithm,
                },
                "weight_map": {
                    name: value for mapping in maps if mapping is not None for name, value in mapping.items()
                },
            }
            _atomic_write(
                os.path.join(self._version_dir, "model.safetensors.index.json"),
                json.dumps(index).encode(),
            )

    def _reload_engines(self, *, manage_generation: bool) -> None:
        local_error: BaseException | None = None
        try:
            if self._post_write_hook is not None:
                self._post_write_hook(self.args, self._version_dir, list(self.rollout_engines))
        except BaseException as exc:
            local_error = exc
        self._raise_synchronized_update_error(local_error, phase="publish delta checkpoint")

        def reload_engines() -> None:
            ray.get([engine.pull_weights.remote(self.weight_update_sequence) for engine in self.rollout_engines])
            if manage_generation:
                pause_mode = getattr(self.args, "weight_update_pause_mode", "retract")
                ray.get([engine.pause_generation.remote(pause_mode) for engine in self.rollout_engines])
                ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            ray.get(
                [
                    engine.update_weights_from_disk.remote(
                        model_path=self.args.update_weight_local_checkpoint_dir,
                        weight_version=str(self.weight_version),
                    )
                    for engine in self.rollout_engines
                ]
            )
            if manage_generation:
                ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])

        try:
            self._run_rank_zero_action(reload_engines, phase="activate delta checkpoint")
        except BaseException:
            self._poison_transport()
            raise

    def _iter_hf_tensors(self):
        for chunk_iter in (self._iter_non_expert_chunks(), self._iter_expert_chunks()):
            for hf_chunk in chunk_iter:
                yield from hf_chunk


__all__ = ["ReefUpdateWeightFromDisk", "ReefUpdateWeightFromDiskDelta"]
