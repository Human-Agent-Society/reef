"""Typed fields consumed from Slime's parsed argument namespace.

The pinned parser supplies native fields; Reef's parser hook supplies its
options. Derived hook fields start here before runtime configuration. Extra
native options remain on the namespace for Slime's own workers.
"""

from argparse import Namespace


class SlimeArguments(Namespace):
    """Slime's namespace with explicit Reef adapter inputs and derived hooks."""

    seed: int
    fp16: bool
    use_rollout_routing_replay: bool
    num_rollout: int
    save_hf: str
    save: str
    compute_advantages_and_returns: bool
    debug_train_only: bool
    debug_rollout_only: bool
    rollout_num_gpus: int
    rollout_external: bool
    colocate: bool
    offload_train: bool
    offload_rollout: bool
    keep_lora_base_resident: bool
    disjoint_prefix_sharing: bool
    megatron_lora_rank: int
    megatron_lora_alpha: int | None
    megatron_lora_target_modules: list[str] | None
    use_critic: bool
    critic_save: str | None
    critic_init: str | None
    critic_save_interval: int
    critic_steps_per_actor: int | None
    num_critic_only_steps: int
    critic_lr: float | None
    lr: float
    hf_checkpoint: str
    load: str | None
    start_rollout_id: int | None
    custom_megatron_init_path: str | None
    custom_critic_args_hook_path: str | None = None
    custom_model_provider_path: str | None
    custom_rollout_data_keys: tuple[str, ...] | list[str] | None = ()
    megatron_config_path: str | None
    disable_param_buffers_cpu_backup: bool
    override_opt_param_scheduler: bool
    no_load_optim: bool
    no_load_rng: bool
    finetune: bool
    ckpt_step: int | None
    reef_executor_backend: str
    advantage_estimator: str
    reef_rollout_tensor_dtypes: dict[str, str]
    reef_external_batch_keys: tuple[str, ...]
    reef_rollout_log_skip_keys: tuple[str, ...]
    loss_family: str | None = None
    reef_chained_megatron_init_path: str | None = None
    reef_chained_critic_args_hook_path: str | None = None
    reef_chained_model_provider_path: str | None = None
