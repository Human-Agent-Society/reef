"""Reef's Megatron training worker implemented through Slime's Ray actor hook."""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import ray
import torch
import torch.distributed as dist
from slime.backends.megatron_utils.actor import MegatronTrainRayActor
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.memory_utils import print_memory
from slime.utils.reloadable_process_group import destroy_process_groups, reload_process_groups
from slime.utils.timer import Timer, timer
from torch_memory_saver import torch_memory_saver

from reef.runtime.names import ADAPTER_SLOTS_DIRNAME
from reef.train.slime_backend.reef_adapters.megatron.adapter_slots import AdapterSlotSwitcher
from reef.train.slime_backend.reef_adapters.megatron.lora import (
    collect_lora_train_metrics,
    megatron_lora_enabled,
    zero_megatron_lora_adapters,
)
from reef.train.slime_backend.reef_adapters.megatron.lora_checkpoint import save_lora_adapter_to_path
from reef.train.slime_backend.reef_adapters.worker_hooks import (
    _loss_family_spec,
    drain_worker_metrics,
    record_worker_metrics,
    reef_node_ip_and_free_port,
    resolve_tensor_dtype,
)


def _loss_family_hook(args: Any, arg_name: str) -> Any:
    """Load a worker hook the active loss family registered via ``@objective``.

    ``resolve_objective_paths`` projects each family's registered dotted paths
    onto ``args`` during Megatron init, so families extend the actor without
    this adapter naming any of them.
    """
    path = getattr(args, arg_name, None)
    if not path:
        return None
    from slime.utils.misc import load_function

    return load_function(path)


class ReefMegatronTrainRayActor(MegatronTrainRayActor):
    """Add the small bridge-facing surface absent from the runtime."""

    _get_current_node_ip_and_free_port = staticmethod(reef_node_ip_and_free_port)

    def init(self, args, role, with_ref=False, with_opd_teacher=False):
        result = super().init(args, role, with_ref=with_ref, with_opd_teacher=with_opd_teacher)
        updater = getattr(self, "weight_updater", None)
        if updater is not None:
            updater.tie_word_embeddings = bool(getattr(self.hf_config, "tie_word_embeddings", False))
        self.adapter_slots: AdapterSlotSwitcher | None = None
        if role == "actor" and megatron_lora_enabled(args) and not args.debug_rollout_only:
            # One LoRA slot, several scenarios: snapshots live beside the
            # Megatron checkpoint so a restart restores every scenario.
            store = Path(args.save) / ADAPTER_SLOTS_DIRNAME if getattr(args, "save", None) else None
            if self.args.offload_train:
                self.wake_up()
            try:
                self.adapter_slots = AdapterSlotSwitcher(
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    store_dir=store,
                    rank=dist.get_rank(),
                )
            finally:
                if self.args.offload_train:
                    self.sleep()
        if role == "actor":
            init_hook = _loss_family_hook(args, "reef_actor_init_hook_path")
            if init_hook is not None:
                init_hook(self)
        return result

    # -- Per-scenario adapter slot --------------------------------------

    def activate_scenario(self, scenario: str) -> bool:
        """Put ``scenario``'s adapter (and optimizer state) into the LoRA slot.

        Returns whether the scenario had prior state; a first-time scenario
        starts from the pristine adapter. The CPU actor snapshot Slime reads
        for publication and export is refreshed to match.
        """
        slots = self.adapter_slots
        if slots is None:
            raise RuntimeError("activate_scenario requires Megatron LoRA on an actor")
        if slots.active == scenario:
            self.weight_updater.active_scenario = scenario
            return True
        if self.args.offload_train:
            self.wake_up()
        try:
            existed = slots.restore(scenario)
            self.weights_backuper.backup("actor")
        finally:
            if self.args.offload_train:
                self.sleep()
        self.weight_updater.active_scenario = scenario
        return existed

    def active_scenario(self) -> str | None:
        slots = self.adapter_slots
        return None if slots is None else slots.active

    def known_scenarios(self) -> tuple[str, ...]:
        slots = self.adapter_slots
        return () if slots is None else slots.scenarios

    def sync_serving_runtime_load_id(self) -> str:
        """Stamp the updater's current version token on every engine, publishing nothing.

        A LoRA bridge that has not trained yet serves the frozen base; the
        engines still need Reef's canonical ``<incarnation>:<sequence>``
        token so rollouts carry an admissible producing version.
        """
        version = str(self.weight_updater.runtime_load_id)
        if dist.get_rank() == 0:
            engines, *_ = ray.get(self.rollout_manager.get_updatable_engines_and_lock.remote())
            ray.get([engine.set_runtime_load_id.remote(version) for engine in engines])
        dist.barrier(group=get_gloo_group())
        return version

    def publish_adapter(self, scenario: str, lora_name: str) -> None:
        """Make ``scenario``'s current adapter resident under ``lora_name``.

        No serving-runtime load ID changes: this re-registers an adapter the
        engine lost (restart), it is not a training publication.
        """
        self.activate_scenario(scenario)
        self._with_lora_engines(lambda: self.weight_updater.publish_lora_adapter(lora_name))

    def train_actor(self, rollout_id, rollout_data, external_data=None):
        pre_train_hook = _loss_family_hook(self.args, "reef_actor_pre_train_hook_path")
        if pre_train_hook is not None:
            pre_train_hook(self, rollout_data)
        result = super().train_actor(rollout_id, rollout_data, external_data=external_data)
        if megatron_lora_enabled(self.args):
            record_worker_metrics(collect_lora_train_metrics(self.model))
        slots = self.adapter_slots
        if slots is not None:
            if slots.active is None:
                raise RuntimeError("LoRA training ran without an activated scenario")
            slots.capture(slots.active)
        return result

    def _get_rollout_data(self, rollout_data_ref):
        rollout_data = super()._get_rollout_data(rollout_data_ref)
        from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

        device = torch.cuda.current_device()
        # Slime's own per-response-token tensors, plus the ones the loss
        # family declared as response-aligned.
        sliced: list[tuple[str, torch.dtype]] = [("advantages", torch.float32), ("returns", torch.float32)]
        spec = _loss_family_spec(self.args)
        if spec is not None:
            sliced.extend(
                (key, resolve_tensor_dtype(spec.rollout_tensor_dtypes[key])) for key in spec.response_aligned_keys
            )
        for key, dtype in sliced:
            if key not in rollout_data:
                continue
            rollout_data[key] = [
                slice_log_prob_with_cp(value, total_length, response_length).to(
                    device=device,
                    dtype=dtype,
                    non_blocking=True,
                )
                for value, total_length, response_length in zip(
                    rollout_data[key],
                    rollout_data["total_lengths"],
                    rollout_data["response_lengths"],
                    strict=True,
                )
            ]
        return rollout_data

    def get_runtime_load_id(self) -> str:
        return str(self.weight_updater.exact_runtime_load_id)

    def restore_runtime_load_id_for_republication(self, runtime_load_id: str) -> None:
        self.weight_updater.restore_exact_runtime_load_id(runtime_load_id)

    def pop_metrics(self) -> dict[str, float]:
        metrics = drain_worker_metrics()
        updater = getattr(self, "weight_updater", None)
        if updater is not None:
            metrics.update(updater.pop_metrics())
        return metrics

    def update_weights(self, *, manage_generation: bool = True, force_full: bool = False) -> Any:
        """Pass bridge-owned generation policy through Slime's actor hook."""
        updater = self.weight_updater
        original_update = updater.update_weights
        supported = inspect.signature(original_update).parameters
        options = {
            name: value
            for name, value in {
                "manage_generation": manage_generation,
                "force_full": force_full,
            }.items()
            if name in supported
        }

        def update_with_generation_policy() -> Any:
            return original_update(**options)

        instance_attributes = vars(updater)
        previous = instance_attributes.get("update_weights")
        had_previous = "update_weights" in instance_attributes
        updater.update_weights = update_with_generation_policy
        try:
            if not megatron_lora_enabled(self.args):
                return super().update_weights()
            return self._update_lora_weights()
        finally:
            if had_previous:
                updater.update_weights = previous
            else:
                delattr(updater, "update_weights")

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        # Not ``@timer``: Slime's own ``save_model`` carries that decorator, and
        # its Timer is a singleton that refuses a second start under the same
        # name ("Timer save_model already started"). Delegating under the
        # decorator therefore failed every full-weight save — the LoRA branch
        # below never delegates, which is why only non-LoRA runs hit it. Time
        # that branch explicitly instead.
        if not megatron_lora_enabled(self.args):
            super().save_model(rollout_id, force_sync=force_sync)
            return
        with Timer().context("save_model"):
            self._save_lora_model(rollout_id, force_sync=force_sync)

    def _save_lora_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return

        if self.args.offload_train:
            self.wake_up()
        try:
            if self.args.async_save:
                from megatron.training.async_utils import maybe_finalize_async_save

                maybe_finalize_async_save(blocking=True)

            from slime.backends.megatron_utils.model import save

            save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)
            if force_sync and self.args.async_save:
                maybe_finalize_async_save(blocking=True)

            if self.args.save_hf is not None and self.role == "actor":
                output_dir = Path(self.args.save_hf.format(rollout_id=rollout_id))
                context = torch_memory_saver.disable() if self.args.offload_train else nullcontext()
                with context:
                    save_lora_adapter_to_path(
                        self.args,
                        output_dir,
                        self.weight_updater.export_lora_adapter_tensors(),
                    )
            slots = self.adapter_slots
            if slots is not None and slots.active is not None:
                # The Megatron checkpoint above holds only the active slot;
                # every scenario's state must survive a restart on its own.
                slots.capture(slots.active)
                slots.persist(slots.active)
                dist.barrier(group=get_gloo_group())
        finally:
            if self.args.offload_train:
                self.sleep()

    @timer
    def _update_lora_weights(self) -> None:
        """Publish LoRA while the offloaded actor allocation is readable.

        The runtime can publish full weights from its CPU backup while the
        actor is paused. Megatron Bridge's adapter export also inspects the
        adapter wrappers, so an offloaded LoRA actor must be resumed for the
        conversion and then returned to its previous state.
        """
        self._with_lora_engines(self.weight_updater.update_weights)

    def _with_lora_engines(self, publish) -> None:
        """Run ``publish`` with rollout engines connected and the actor readable."""
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_updatable_engines.remote())
            dist.barrier(group=get_gloo_group())

        (
            rollout_engines,
            rollout_engine_lock,
            num_new_engines,
            engine_gpu_counts,
            engine_gpu_offsets,
            engine_parallel_configs,
        ) = ray.get(self.rollout_manager.get_updatable_engines_and_lock.remote())

        reconnect_rollout_engines = self.args.offload_train and self.args.use_critic and not self.args.colocate
        if not rollout_engines and not reconnect_rollout_engines:
            return

        resumed_actor = bool(self.args.offload_train)
        if resumed_actor:
            self.wake_up()
        elif reconnect_rollout_engines:
            # Kept for symmetry with the runtime if its reconnect predicate
            # ever stops implying offload_train.
            reload_process_groups()

        try:
            if num_new_engines > 0 or reconnect_rollout_engines:
                self.weight_updater.connect_rollout_engines(
                    rollout_engines,
                    rollout_engine_lock,
                    engine_gpu_counts=engine_gpu_counts,
                    engine_gpu_offsets=engine_gpu_offsets,
                    engine_parallel_configs=engine_parallel_configs,
                )
                dist.barrier(group=get_gloo_group())
                if dist.get_rank() == 0:
                    ray.get(self.rollout_manager.clear_updatable_num_new_engines.remote())

            # CUDA IPC cannot export storage owned by torch_memory_saver's
            # pauseable pool. Keep conversion/output tensors in the ordinary
            # allocator even though the model itself has been resumed.
            with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
                print_memory("before update_weights")
                publish()
                print_memory("after update_weights")

                if getattr(self.args, "keep_old_actor", False):
                    if self.args.update_weights_interval == 1:
                        self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                        self.weights_backuper.backup("rollout_actor")
                    else:
                        self.weights_backuper.backup("old_actor")
        finally:
            if resumed_actor:
                self.sleep()
            elif reconnect_rollout_engines:
                destroy_process_groups()

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        super().load_other_checkpoint(model_tag, path)
        if model_tag == "ref" and megatron_lora_enabled(self.args):
            zero_megatron_lora_adapters(self.model)
            self.weights_backuper.backup("ref")


__all__ = ["ReefMegatronTrainRayActor"]
