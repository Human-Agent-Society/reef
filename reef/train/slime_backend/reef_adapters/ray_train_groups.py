"""Reef extensions and construction for Slime Ray training groups."""

from __future__ import annotations

import copy
import logging
import shutil
from pathlib import Path
from typing import Any

import ray
from slime.ray.actor_group import RayTrainGroup

from reef.train.slime_backend.reef_adapters.worker_hooks import configure_critic_objective

logger = logging.getLogger(__name__)


class ReefRayTrainGroup(RayTrainGroup):
    """Expose the small control surface Reef needs from a Slime group."""

    _disk_runtime_load_id: int

    def restore_runtime_load_id_for_republication(self, runtime_load_id: str):
        return ray.get(
            [actor.restore_runtime_load_id_for_republication.remote(runtime_load_id) for actor in self._actor_handlers]
        )

    def async_pop_rank0_metrics(self):
        return self._actor_handlers[0].pop_metrics.remote()

    def activate_scenario(self, scenario: str) -> bool:
        """Put one scenario's adapter into every actor's LoRA slot."""
        existed = ray.get([actor.activate_scenario.remote(scenario) for actor in self._actor_handlers])
        if len(set(existed)) != 1:
            raise RuntimeError(f"actors disagree about prior state for scenario {scenario!r}: {existed!r}")
        return bool(existed[0])

    def sync_serving_runtime_load_id(self) -> str:
        """Stamp the group's current runtime-load-ID token on the engines."""
        versions = ray.get([actor.sync_serving_runtime_load_id.remote() for actor in self._actor_handlers])
        if len(set(versions)) != 1:
            raise RuntimeError(f"Slime workers disagree on the runtime load ID: {versions!r}")
        return str(versions[0])

    def publish_adapter(self, scenario: str, lora_name: str) -> None:
        """Re-register one scenario's adapter without a serving version bump."""
        ray.get([actor.publish_adapter.remote(scenario, lora_name) for actor in self._actor_handlers])

    def async_get_rank0_runtime_load_id(self):
        return self._actor_handlers[0].get_runtime_load_id.remote()

    def update_weights(self, *, manage_generation: bool = True, force_full: bool = False):
        if not self._full_disk_weight_update_enabled():
            if manage_generation and not force_full:
                return super().update_weights()
            return ray.get(
                [
                    actor.update_weights.remote(
                        manage_generation=manage_generation,
                        force_full=force_full,
                    )
                    for actor in self._actor_handlers
                ]
            )

        disk_sequence = self._disk_runtime_load_id + 1
        disk_weight_dir = Path(self.args.update_weight_disk_dir) / f"weight_v{disk_sequence:06d}"
        if manage_generation and not force_full:
            updates = [actor.update_weights.remote() for actor in self._actor_handlers]
        else:
            updates = [
                actor.update_weights.remote(
                    manage_generation=manage_generation,
                    force_full=force_full,
                )
                for actor in self._actor_handlers
            ]
        ray.get(updates)
        self._disk_runtime_load_id = disk_sequence
        serving_runtime_load_id = str(ray.get(self.async_get_rank0_runtime_load_id()))
        if self._release_train_enabled():
            self.release()
        return self._reload_rollout_weights_from_disk(
            disk_weight_dir,
            disk_sequence,
            serving_runtime_load_id,
            manage_generation=manage_generation,
        )

    def _reload_rollout_weights_from_disk(
        self,
        disk_weight_dir: Path,
        disk_sequence: int,
        serving_runtime_load_id: str,
        *,
        manage_generation: bool,
    ) -> None:
        if self._rollout_manager is None:
            raise RuntimeError("disk weight update requires a rollout manager")
        if self.args.offload_rollout:
            ray.get(self._rollout_manager.onload_weights.remote())
        engines, *_ = ray.get(self._rollout_manager.get_updatable_engines_and_lock.remote())
        if not engines:
            if not self.args.update_weight_disk_keep_files:
                shutil.rmtree(disk_weight_dir, ignore_errors=True)
            return

        if self.args.update_weight_local_checkpoint_dir:
            ray.get([engine.pull_weights.remote(disk_sequence) for engine in engines])
            model_path = self.args.update_weight_local_checkpoint_dir
        else:
            model_path = str(disk_weight_dir)
        if manage_generation:
            mode = getattr(self.args, "weight_update_pause_mode", "retract")
            ray.get([engine.pause_generation.remote(mode) for engine in engines])
            ray.get([engine.flush_cache.remote() for engine in engines])
        ray.get(
            [
                engine.update_weights_from_disk.remote(
                    model_path=model_path,
                    runtime_load_id=serving_runtime_load_id,
                )
                for engine in engines
            ]
        )
        if self.args.ci_test:
            engine_versions = ray.get([engine.get_runtime_load_id.remote() for engine in engines])
            mismatches = [
                f"engine {index}: {engine_version}"
                for index, engine_version in enumerate(engine_versions)
                if str(engine_version) != serving_runtime_load_id
            ]
            if mismatches:
                raise RuntimeError(
                    f"runtime load ID mismatch after disk reload; expected {serving_runtime_load_id}: "
                    + ", ".join(mismatches)
                )
        if not self.args.update_weight_disk_keep_files:
            shutil.rmtree(disk_weight_dir, ignore_errors=True)
        if manage_generation:
            ray.get([engine.continue_generation.remote() for engine in engines])


def prepare_critic_args(args: Any) -> Any:
    """Derive a critic namespace and apply Reef's role-specific policy."""
    if args.megatron_config_path is not None:
        from slime.utils.arguments import parse_megatron_role_args

        critic_args = parse_megatron_role_args(args, args.megatron_config_path, role="critic")
    else:
        critic_args = copy.deepcopy(args)
        critic_args.disable_param_buffers_cpu_backup = False

    configure_critic_objective(critic_args)
    # LoRA is an actor-only serving adapter. Restore a user's provider for the
    # critic instead of sending the critic through Reef's actor LoRA wrapper.
    critic_args.megatron_lora_rank = 0
    critic_args.megatron_lora_alpha = None
    critic_args.megatron_lora_target_modules = None
    critic_args.custom_model_provider_path = getattr(critic_args, "reef_chained_model_provider_path", None)
    _apply_critic_checkpoint_roots(critic_args)
    return critic_args


def _apply_critic_checkpoint_roots(critic_args: Any) -> None:
    critic_save = getattr(critic_args, "critic_save", None)
    if not critic_save:
        return
    critic_args.save = critic_save
    tracker = Path(critic_save).expanduser() / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        critic_args.load = critic_save
        critic_args.no_load_optim = False
        critic_args.no_load_rng = False
        critic_args.finetune = False
        critic_args.ckpt_step = None
        logger.info("critic resumes from its own checkpoint root %s", critic_save)
    else:
        logger.info(
            "critic checkpoint root %s has no checkpoint yet; using the inherited load fallback",
            critic_save,
        )


def create_train_groups(
    args: Any,
    placement_groups: Any,
    rollout_manager: Any,
    *,
    actor_cls: type,
) -> tuple[Any, Any | None]:
    """Build Slime groups while keeping Reef role policy outside Slime."""
    actor_args = args
    if args.megatron_config_path is not None:
        from slime.utils.arguments import parse_megatron_role_args

        actor_args = parse_megatron_role_args(args, args.megatron_config_path, role="actor")
    actor_group = ReefRayTrainGroup(
        args=actor_args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=placement_groups["actor"],
        num_gpus_per_actor=0.4,
        with_ref=actor_args.kl_coef != 0 or actor_args.use_kl_loss,
        with_opd_teacher=actor_args.use_opd and actor_args.opd_type == "megatron",
        actor_cls=actor_cls,
    )
    actor_start_rollout_ids = actor_group.create(rollout_manager=rollout_manager)

    critic_group = None
    start_rollout_ids = actor_start_rollout_ids
    if args.use_critic:
        critic_args = prepare_critic_args(args)
        critic_group = ReefRayTrainGroup(
            args=critic_args,
            num_nodes=args.critic_num_nodes,
            num_gpus_per_node=args.critic_num_gpus_per_node,
            pg=placement_groups["critic"],
            num_gpus_per_actor=0.4,
            role="critic",
            actor_cls=actor_cls,
        )
        start_rollout_ids = critic_group.create(rollout_manager=rollout_manager)

    if len(set(start_rollout_ids)) != 1:
        raise RuntimeError(f"Slime workers disagree on start rollout id: {start_rollout_ids!r}")
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]
    return actor_group, critic_group


__all__ = ["ReefRayTrainGroup", "create_train_groups", "prepare_critic_args"]
