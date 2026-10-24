"""Slime training policy over a replaceable, ordered worker executor.

The group owns checkpoint semantics. Executors own worker
placement, launch, RPC and termination. Slime's rollout payloads and GPU
collectives remain owned by its backend, independently of control RPC.
"""

from __future__ import annotations

import copy
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any

from reef.runtime.executor import Executor, ExecutorConfig
from reef.runtime.executor.failure import ExecutorFailureListener
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments
from reef.train.slime_backend.reef_adapters.executors.config import (
    DEFAULT_EXECUTOR_BACKEND as DEFAULT_EXECUTOR_BACKEND,
)
from reef.train.slime_backend.reef_adapters.executors.config import slime_executor_class
from reef.train.slime_backend.reef_adapters.worker_hooks import configure_critic_objective

logger = logging.getLogger(__name__)


#: Default worker RPC wait; startup retains this budget independently of training.timeout-s.
DEFAULT_TRAIN_RPC_TIMEOUT_S = 14_400


def merge_rank_metrics(per_rank: list[dict[str, Any]]) -> dict[str, Any]:
    """One metrics mapping from every rank's, in rank order; the lowest rank wins a shared key."""
    merged: dict[str, Any] = {}
    for metrics in reversed(per_rank):
        merged.update(metrics or {})
    return merged


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
        train_rpc_timeout_s: float = DEFAULT_TRAIN_RPC_TIMEOUT_S,
    ) -> None:
        if num_nodes < 1 or num_gpus_per_node < 1:
            raise ValueError("training groups require positive node and GPU counts")
        self.train_rpc_timeout_s = train_rpc_timeout_s
        self.args = args
        self.role = role
        self._with_ref = with_ref
        self._with_opd_teacher = with_opd_teacher
        self._world_size = num_nodes * num_gpus_per_node
        backend = executor_backend if executor_backend is not None else getattr(args, "reef_executor_backend", "auto")
        # Resolve before reserving or launching any workers.
        executor_class = slime_executor_class(backend)
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
        self._failure_listener: ExecutorFailureListener | None = None
        self._rollout_manager: Any = None
        self.train_parallel_config: dict[str, Any] = {}
        self._prepared_disk_update: tuple[Path, int, str] | None = None

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
        executor = Executor.create(self._executor_config)
        self._executor = executor
        try:
            if self._failure_listener is not None:
                executor.register_failure_listener(self._failure_listener)
            start_ids = executor.collective_rpc(
                "init",
                args=(self.args, self.role),
                kwargs={"with_ref": self._with_ref, "with_opd_teacher": self._with_opd_teacher},
                timeout=DEFAULT_TRAIN_RPC_TIMEOUT_S,
            )
            if len(start_ids) != self._world_size or len(set(start_ids)) != 1:
                raise RuntimeError(f"Slime workers disagree on start rollout id: {start_ids!r}")
            if self._rollout_manager is not None:
                self.set_rollout_manager(self._rollout_manager)
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

    def register_failure_listener(self, listener: ExecutorFailureListener) -> None:
        """Observe current and replacement workers from the operations owner."""
        self._failure_listener = listener
        if self._executor is not None:
            self._executor.register_failure_listener(listener)

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

    def reset_training_state(self) -> None:
        """Clear every rank's optimizer state and restart its learning-rate scheduler; the weights stay."""
        self.executor.collective_rpc("reset_training_state", timeout=self.train_rpc_timeout_s)

    def save_model(self, rollout_id, force_sync=False):
        result = self.executor.collective_rpc(
            "save_model",
            args=(rollout_id,),
            kwargs={"force_sync": force_sync},
            timeout=self.train_rpc_timeout_s,
        )
        if self._release_train_enabled():
            self.args.load = self.args.save
            self.args.ckpt_step = None
            self.args.finetune = False
            self.args.no_load_optim = self.args.no_save_optim
            self.args.no_load_rng = False
        return result

    def onload(self):
        return self.executor.collective_rpc("wake_up", timeout=self.train_rpc_timeout_s)

    def offload(self):
        return self.executor.collective_rpc("sleep", timeout=self.train_rpc_timeout_s)

    def clear_memory(self):
        return self.executor.collective_rpc("clear_memory", timeout=self.train_rpc_timeout_s)

    def set_rollout_manager(self, rollout_manager):
        self._rollout_manager = rollout_manager
        layouts = self.executor.collective_rpc(
            "set_rollout_manager", args=(rollout_manager,), timeout=DEFAULT_TRAIN_RPC_TIMEOUT_S
        )
        if not layouts or not isinstance(layouts[0], dict) or any(layout != layouts[0] for layout in layouts):
            raise RuntimeError(f"Slime workers disagree on training parallel config: {layouts!r}")
        self.train_parallel_config = layouts[0]
        return layouts

    def _release_train_enabled(self):
        return self.role == "actor" and getattr(self.args, "release_train", False)

    def async_pop_metrics(self):
        """Drain every rank's metrics, in rank order.

        Slime logs a train step (loss, grad norm, learning rate) on the last
        pipeline stage, which is not rank 0 under pipeline parallelism.
        """
        return self.executor.collective_rpc("pop_metrics", non_block=True)


def prepare_critic_args(args: SlimeArguments) -> SlimeArguments:
    """Derive a critic namespace and apply Reef's role-specific policy."""
    if args.megatron_config_path is not None:
        from slime.utils.arguments import parse_megatron_role_args

        critic_args = SlimeArguments(**vars(parse_megatron_role_args(args, args.megatron_config_path, role="critic")))
    else:
        critic_args = copy.deepcopy(args)
        critic_args.disable_param_buffers_cpu_backup = False

    configure_critic_objective(critic_args)
    # The value model may train faster than the policy (SAO uses 5e-6 against
    # a 1e-6 policy lr); --critic-lr scopes that to the critic role without
    # the --megatron-config-path role surgery.
    critic_lr = critic_args.critic_lr
    if critic_lr is not None:
        critic_args.lr = float(critic_lr)
    # A LoRA actor gets a LoRA critic: the value model keeps the same frozen
    # base with its own adapters and a trainable value head (Slime swaps the
    # head in after Reef's provider wraps the model), so a critic the size of
    # the policy fits beside it. Without LoRA the critic trains every
    # parameter through the user's own provider, as before.
    if not critic_args.megatron_lora_rank:
        critic_args.megatron_lora_alpha = None
        critic_args.megatron_lora_target_modules = None
        critic_args.custom_model_provider_path = critic_args.reef_chained_model_provider_path
    apply_critic_checkpoint_roots(critic_args)
    return critic_args


def apply_critic_checkpoint_roots(critic_args: SlimeArguments) -> None:
    critic_save = critic_args.critic_save
    if not critic_save:
        return
    critic_args.save = critic_save
    critic_init = critic_args.critic_init
    if (Path(critic_save).expanduser() / "latest_checkpointed_iteration.txt").is_file():
        load, reason = critic_save, "resumes from its own checkpoint root"
    elif critic_init and (Path(critic_init).expanduser() / "latest_checkpointed_iteration.txt").is_file():
        # A value model trained on earlier episodes: its weights and optimizer
        # come from --critic-init once, and every later start resumes from
        # what this run has saved since. The learning-rate schedule is this
        # run's, not the checkpoint's: the earlier run may have used another
        # batch size, and Megatron otherwise refuses a schedule whose total
        # iteration count differs.
        critic_args.override_opt_param_scheduler = True
        load, reason = critic_init, "starts from the checkpoint in --critic-init"
    else:
        logger.info(
            "critic checkpoint root %s has no checkpoint yet; using the inherited load fallback",
            critic_save,
        )
        return
    critic_args.load = load
    critic_args.no_load_optim = False
    critic_args.no_load_rng = False
    critic_args.finetune = False
    critic_args.ckpt_step = None
    logger.info("critic %s %s", reason, load)


def create_train_groups(
    args: Any,
    placement_groups: Any,
    rollout_manager: Any,
    *,
    actor_cls: type,
    train_rpc_timeout_s: float = DEFAULT_TRAIN_RPC_TIMEOUT_S,
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
        train_rpc_timeout_s=train_rpc_timeout_s,
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
                train_rpc_timeout_s=train_rpc_timeout_s,
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


__all__ = ["SlimeTrainGroup", "create_train_groups", "merge_rank_metrics", "prepare_critic_args"]
