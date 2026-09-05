"""Slime training policy over a replaceable, ordered worker executor.

The group owns checkpoint and publication semantics. Executors own worker
placement, launch, RPC and termination. Slime's rollout payloads and GPU
collectives remain owned by its backend, independently of control RPC.
"""

from __future__ import annotations

import copy
import logging
import shutil
from contextlib import suppress
from pathlib import Path
from typing import Any

from reef.runtime.executor import Executor, ExecutorConfig, resolve
from reef.runtime.executor.ray import RayExecutor
from reef.train.slime_backend.reef_adapters.executors.config import (
    DEFAULT_EXECUTOR_BACKEND as DEFAULT_EXECUTOR_BACKEND,
)
from reef.train.slime_backend.reef_adapters.executors.config import slime_executor_class
from reef.train.slime_backend.reef_adapters.worker_hooks import configure_critic_objective

logger = logging.getLogger(__name__)


TRAIN_RPC_TIMEOUT_S = 14_400


class SlimeTrainGroup:
    """Training operations shared by actor and critic worker groups.

    Custom executors receive the Slime launch configuration in
    ``ExecutorConfig.options`` and must provide workers with Slime's methods
    and rollout-data transport. Selecting a launcher does not reshard model
    state or change the configured parallel topology.
    """

    def __init__(
        self,
        args: Any,
        num_nodes: int,
        num_gpus_per_node: int,
        pg: Any,
        num_gpus_per_actor: float = 1,
        role: str = "actor",
        with_ref: bool = False,
        with_opd_teacher: bool = False,
        actor_cls: type | None = None,
        *,
        executor_backend: str | type[Executor] | None = None,
    ) -> None:
        if num_nodes < 1 or num_gpus_per_node < 1:
            raise ValueError("training groups require positive node and GPU counts")
        self.args = args
        self.role = role
        self._with_ref = with_ref
        self._with_opd_teacher = with_opd_teacher
        self._world_size = num_nodes * num_gpus_per_node
        backend = executor_backend if executor_backend is not None else getattr(args, "reef_executor_backend", "auto")
        # Resolve before reserving or launching any workers.
        executor_class = slime_executor_class(backend, role="training")
        self._executor_config = ExecutorConfig(
            backend=executor_class,
            options={
                **getattr(args, "reef_train_executor_options", {}),
                "args": args,
                "num_nodes": num_nodes,
                "num_gpus_per_node": num_gpus_per_node,
                "pg": pg,
                "num_gpus_per_actor": num_gpus_per_actor,
                "role": role,
                "with_ref": with_ref,
                "with_opd_teacher": with_opd_teacher,
                "actor_cls": actor_cls,
            },
        )
        self._executor: Executor | None = None
        self._rollout_manager: Any = None
        self._disk_weight_version = getattr(args, "update_weight_start_version", 0)
        self._released_runtime_load_id: str | None = None

    @property
    def executor(self) -> Executor:
        if self._executor is None:
            raise RuntimeError("training workers are not running; call create() before executing work")
        return self._executor

    def create(self, rollout_manager: Any = None) -> list[Any] | None:
        if self._executor is not None:
            return None
        if rollout_manager is not None:
            self._rollout_manager = rollout_manager
        self.args.update_weight_start_version = self._disk_weight_version
        executor = Executor.create(self._executor_config)
        self._executor = executor
        try:
            start_ids = executor.collective_rpc(
                "init",
                args=(self.args, self.role),
                kwargs={"with_ref": self._with_ref, "with_opd_teacher": self._with_opd_teacher},
                timeout=TRAIN_RPC_TIMEOUT_S,
            )
            if len(start_ids) != self._world_size or len(set(start_ids)) != 1:
                raise RuntimeError(f"Slime workers disagree on start rollout id: {start_ids!r}")
            if self._rollout_manager is not None:
                self.set_rollout_manager(self._rollout_manager)
            self._released_runtime_load_id = None
            return start_ids
        except BaseException:
            with suppress(Exception):
                self.release()
            raise

    def release(self) -> None:
        executor = self._executor
        if executor is not None:
            executor.shutdown()
            self._executor = None

    def check_health(self) -> None:
        self.executor.check_health()

    def async_train(self, rollout_id, rollout_data_ref, external_data=None):
        # Validate before dispatch: one wrong-length critic result must not
        # launch only part of a collective and strand the other ranks.
        if isinstance(external_data, list) and len(external_data) != self._world_size:
            raise ValueError("critic results must contain one entry per training worker")
        if self._executor is None and self._release_train_enabled():
            self.create()
        return [
            self.executor.rpc(
                rank,
                "train",
                args=(rollout_id, rollout_data_ref),
                kwargs={"external_data": external_data[rank] if isinstance(external_data, list) else external_data},
                non_block=True,
            )
            for rank in range(self._world_size)
        ]

    def save_model(self, rollout_id, force_sync=False):
        result = self.executor.collective_rpc(
            "save_model", args=(rollout_id,), kwargs={"force_sync": force_sync}, timeout=TRAIN_RPC_TIMEOUT_S
        )
        if self._release_train_enabled():
            self.args.load = self.args.save
            self.args.ckpt_step = None
            self.args.finetune = False
            self.args.no_load_optim = self.args.no_save_optim
            self.args.no_load_rng = False
        return result

    def onload(self):
        return self.executor.collective_rpc("wake_up", timeout=TRAIN_RPC_TIMEOUT_S)

    def offload(self):
        return self.executor.collective_rpc("sleep", timeout=TRAIN_RPC_TIMEOUT_S)

    def clear_memory(self):
        return self.executor.collective_rpc("clear_memory", timeout=TRAIN_RPC_TIMEOUT_S)

    def set_rollout_manager(self, rollout_manager):
        self._rollout_manager = rollout_manager
        return self.executor.collective_rpc(
            "set_rollout_manager", args=(rollout_manager,), timeout=TRAIN_RPC_TIMEOUT_S
        )

    def _release_train_enabled(self):
        return self.role == "actor" and getattr(self.args, "release_train", False)

    def _full_disk_weight_update_enabled(self):
        return (
            self.role == "actor"
            and self.args.update_weight_mode == "full"
            and self.args.update_weight_transport == "disk"
        )

    def restore_runtime_load_id_for_republication(self, runtime_load_id: str):
        return self.executor.collective_rpc(
            "restore_runtime_load_id_for_republication", args=(runtime_load_id,), timeout=TRAIN_RPC_TIMEOUT_S
        )

    def async_pop_rank0_metrics(self):
        return self.executor.rpc(0, "pop_metrics", non_block=True)

    def activate_scenario(self, scenario: str) -> bool:
        """Put one scenario's adapter into every actor's LoRA slot."""
        existed = self.executor.collective_rpc("activate_scenario", args=(scenario,), timeout=TRAIN_RPC_TIMEOUT_S)
        if len(set(existed)) != 1:
            raise RuntimeError(f"actors disagree about prior state for scenario {scenario!r}: {existed!r}")
        return bool(existed[0])

    def sync_serving_runtime_load_id(self) -> str:
        """Stamp the group's current runtime-load-ID token on the engines."""
        versions = self.executor.collective_rpc("sync_serving_runtime_load_id", timeout=TRAIN_RPC_TIMEOUT_S)
        if len(set(versions)) != 1:
            raise RuntimeError(f"Slime workers disagree on the runtime load ID: {versions!r}")
        return str(versions[0])

    def publish_adapter(self, scenario: str, lora_name: str) -> None:
        """Re-register one scenario's adapter without a serving version bump."""
        self.executor.collective_rpc("publish_adapter", args=(scenario, lora_name), timeout=TRAIN_RPC_TIMEOUT_S)

    def async_get_rank0_runtime_load_id(self):
        if self._executor is None and self._released_runtime_load_id is not None:
            # Full disk publication can release workers before the bridge
            # verifies the serving version. This was read before release.
            return self._released_runtime_load_id
        return self.executor.rpc(0, "get_runtime_load_id", non_block=True)

    def update_weights(self, *, manage_generation: bool = True, force_full: bool = False):
        kwargs = (
            {}
            if manage_generation and not force_full
            else {"manage_generation": manage_generation, "force_full": force_full}
        )
        if not self._full_disk_weight_update_enabled():
            return self.executor.collective_rpc("update_weights", kwargs=kwargs, timeout=TRAIN_RPC_TIMEOUT_S)

        disk_sequence = self._disk_weight_version + 1
        disk_weight_dir = Path(self.args.update_weight_disk_dir) / f"weight_v{disk_sequence:06d}"
        self.executor.collective_rpc("update_weights", kwargs=kwargs, timeout=TRAIN_RPC_TIMEOUT_S)
        self._disk_weight_version = disk_sequence
        serving_runtime_load_id = str(resolve(self.async_get_rank0_runtime_load_id(), timeout=TRAIN_RPC_TIMEOUT_S))
        if self._release_train_enabled():
            self._released_runtime_load_id = serving_runtime_load_id
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
        manager = RayExecutor.from_workers([self._rollout_manager])
        if self.args.offload_rollout:
            manager.rpc(0, "onload_weights", timeout=TRAIN_RPC_TIMEOUT_S)
        engines, *_ = manager.rpc(0, "get_updatable_engines_and_lock", timeout=TRAIN_RPC_TIMEOUT_S)
        if not engines:
            if not self.args.update_weight_disk_keep_files:
                shutil.rmtree(disk_weight_dir, ignore_errors=True)
            return

        serving = RayExecutor.from_workers(engines)

        if self.args.update_weight_local_checkpoint_dir:
            serving.collective_rpc("pull_weights", args=(disk_sequence,), timeout=TRAIN_RPC_TIMEOUT_S)
            model_path = self.args.update_weight_local_checkpoint_dir
        else:
            model_path = str(disk_weight_dir)
        if manage_generation:
            mode = getattr(self.args, "weight_update_pause_mode", "retract")
            serving.collective_rpc("pause_generation", args=(mode,), timeout=TRAIN_RPC_TIMEOUT_S)
            serving.collective_rpc("flush_cache", timeout=TRAIN_RPC_TIMEOUT_S)
        serving.collective_rpc(
            "update_weights_from_disk",
            kwargs={"model_path": model_path, "runtime_load_id": serving_runtime_load_id},
            timeout=TRAIN_RPC_TIMEOUT_S,
        )
        if self.args.ci_test:
            engine_versions = serving.collective_rpc("get_runtime_load_id", timeout=TRAIN_RPC_TIMEOUT_S)
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
            serving.collective_rpc("continue_generation", timeout=TRAIN_RPC_TIMEOUT_S)


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
    actor_group = SlimeTrainGroup(
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
    try:
        start_rollout_ids = actor_start_rollout_ids
        if args.use_critic:
            critic_args = prepare_critic_args(args)
            critic_group = SlimeTrainGroup(
                args=critic_args,
                num_nodes=args.critic_num_nodes,
                num_gpus_per_node=args.critic_num_gpus_per_node,
                pg=placement_groups["critic"],
                num_gpus_per_actor=0.4,
                role="critic",
                actor_cls=actor_cls,
            )
            start_rollout_ids = critic_group.create(rollout_manager=rollout_manager)

        if not start_rollout_ids or len(set(start_rollout_ids)) != 1:
            raise RuntimeError(f"Slime workers disagree on start rollout id: {start_rollout_ids!r}")
        if args.start_rollout_id is None:
            args.start_rollout_id = start_rollout_ids[0]
        return actor_group, critic_group
    except BaseException:
        for group in (critic_group, actor_group):
            if group is not None:
                with suppress(Exception):
                    group.release()
        raise


__all__ = ["SlimeTrainGroup", "create_train_groups", "prepare_critic_args"]
