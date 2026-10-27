"""Reef's Megatron training worker implemented through Slime's Ray actor hook."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from slime.utils.reloadable_process_group import monkey_patch_torch_dist

# Megatron's gradient buffer binds ``torch.distributed.reduce_scatter_tensor``,
# ``all_gather_into_tensor`` and ``_coalescing_manager`` when it is imported. Slime
# applies this patch later, inside ``MegatronTrainRayActor.init``, so the bound
# functions receive ``ReloadableProcessGroup`` objects and run through torch's
# Python process-group trampoline, whose work handle crashes in ``wait`` on
# torch 2.13. Patching before the Slime actor import makes Megatron bind the
# wrappers, which unwrap the group first. Slime's later call is a per-process no-op.
monkey_patch_torch_dist()

# isort: split
from slime.backends.megatron_utils.actor import MegatronTrainRayActor
from slime.utils.timer import Timer
from torch_memory_saver import torch_memory_saver

from reef.train.slime_backend.distill.teacher import TEACHER_TAG
from reef.train.slime_backend.reef_adapters.batches import WORKER_SWITCH_KEY
from reef.train.slime_backend.reef_adapters.megatron.hf_export import MegatronToHfWeightIterator
from reef.train.slime_backend.reef_adapters.megatron.lora import (
    collect_lora_train_metrics,
    megatron_lora_enabled,
    zero_megatron_lora_adapters,
)
from reef.train.slime_backend.reef_adapters.megatron.lora_checkpoint import (
    export_lora_adapter_tensors,
    save_lora_adapter_to_path,
)
from reef.train.slime_backend.reef_adapters.megatron.scheduler_resume import restore_scheduler_progress
from reef.train.slime_backend.reef_adapters.worker_hooks import (
    _loss_family_spec,
    drain_worker_metrics,
    record_worker_metrics,
    reef_node_ip_and_free_port,
    resolve_tensor_dtype,
    switch_loss_family,
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
    """Add what the training component calls on each worker beyond Slime's own actor."""

    _get_current_node_ip_and_free_port = staticmethod(reef_node_ip_and_free_port)

    def init(self, args, role, with_ref=False, with_opd_teacher=False):
        result = super().init(args, role, with_ref=with_ref, with_opd_teacher=with_opd_teacher)
        if isinstance(result, int):
            restore_scheduler_progress(args, result, self.opt_param_scheduler)
        # A LoRA actor's save converts its adapter through Megatron Bridge; the critic saves no adapter.
        self.lora_weight_iterator: MegatronToHfWeightIterator | None = None
        if role == "actor" and megatron_lora_enabled(args):
            self.lora_weight_iterator = MegatronToHfWeightIterator(args, self.model, None, None)
        # The loss families whose actor init hook ran; a family runs its hook on its first job.
        self.initialized_loss_families = {args.loss_family}
        if role == "actor":
            init_hook = _loss_family_hook(args, "reef_actor_init_hook_path")
            if init_hook is not None:
                init_hook(self)
        return result

    def switch_training_method(self, worker_switch: Mapping[str, Any]) -> None:
        """Switch this actor to a job's loss family and learning-rate schedule.

        ``loss_family_args`` are the family's job arguments, which the component
        derived from the startup arguments. ``learning_rate_schedule_state`` is
        the job's schedule state, or ``None`` for Slime's configured scheduler.
        """
        for name, value in worker_switch["loss_family_args"].items():
            setattr(self.args, name, value)
        switch_loss_family(self.args)
        loss_family = self.args.loss_family
        if loss_family not in self.initialized_loss_families:
            init_hook = _loss_family_hook(self.args, "reef_actor_init_hook_path")
            if init_hook is not None:
                init_hook(self)
            self.initialized_loss_families.add(loss_family)
        self.args.reef_learning_rate_schedule_state = worker_switch["learning_rate_schedule_state"]

    def reset_training_state(self) -> None:
        """Drop the optimizer's state and restart the learning-rate scheduler at step 0; the weights stay.

        Each inner optimizer of a chained optimizer, and each data-parallel
        rank's shard of a distributed one, clears its own state (Adam's moments
        and step counts); the optimizer builds a new state at its next step.
        """
        from megatron.core.optimizer import ChainedOptimizer

        optimizers = (
            self.optimizer.chained_optimizers if isinstance(self.optimizer, ChainedOptimizer) else [self.optimizer]
        )
        for optimizer in optimizers:
            optimizer.optimizer.state.clear()
        if self.opt_param_scheduler is not None:
            self.opt_param_scheduler.num_steps = 0

    def parallel_layout(self) -> dict[str, Any]:
        """Return this rank's training parallel layout, which the component's batch split reads.

        The layout is Slime's ``train_parallel_config``: the data-parallel and
        context-parallel sizes and the virtual-pipeline microbatch grouping.
        """
        return dict(self.train_parallel_config)

    def train_actor(self, rollout_id, rollout_data, external_data=None):
        pre_train_hook = _loss_family_hook(self.args, "reef_actor_pre_train_hook_path")
        if pre_train_hook is not None:
            pre_train_hook(self, rollout_data)
        result = super().train_actor(rollout_id, rollout_data, external_data=external_data)
        if megatron_lora_enabled(self.args):
            record_worker_metrics(collect_lora_train_metrics(self.model))
        return result

    def _get_rollout_data(self, rollout_data_ref):
        rollout_data = super()._get_rollout_data(rollout_data_ref)
        from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

        # A job whose family or schedule state differs from the workers' current
        # one carries a worker switch; the actor switches before its family's
        # keys are sliced. The critic keeps the startup family.
        worker_switch = rollout_data.pop(WORKER_SWITCH_KEY, None)
        if worker_switch is not None and self.role == "actor":
            self.switch_training_method(worker_switch)

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

    def pop_metrics(self) -> dict[str, float]:
        return drain_worker_metrics()

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
                if self.lora_weight_iterator is None:
                    raise RuntimeError("a LoRA actor saves its adapter, but it built no adapter export")
                output_dir = Path(self.args.save_hf.format(rollout_id=rollout_id))
                context = torch_memory_saver.disable() if self.args.offload_train else nullcontext()
                with context:
                    adapter_tensors = export_lora_adapter_tensors(
                        self.lora_weight_iterator, self.weights_backuper.get("actor")
                    )
                    save_lora_adapter_to_path(self.args, output_dir, adapter_tensors, rollout_id=rollout_id)
        finally:
            if self.args.offload_train:
                self.sleep()

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        super().load_other_checkpoint(model_tag, path)
        if model_tag in ("ref", TEACHER_TAG) and megatron_lora_enabled(self.args):
            zero_megatron_lora_adapters(self.model)
            self.weights_backuper.backup(model_tag)


__all__ = ["ReefMegatronTrainRayActor"]
