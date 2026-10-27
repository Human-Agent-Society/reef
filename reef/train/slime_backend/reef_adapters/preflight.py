"""Fail-fast checks before Reef starts Slime's training workers.

Everything here runs before any placement group or GPU worker exists, so a
configuration or storage problem stops the start with a clear error instead
of a half-started cluster.
"""

from __future__ import annotations


from reef.train.slime_backend.algorithm import SlimeAlgorithm
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments
from reef.train.slime_backend.reef_adapters.training_job.storage import CheckpointStorage, RetentionConfig

CRITIC_ARGS_HOOK_PATH = "reef.train.slime_backend.reef_adapters.worker_hooks.configure_critic_objective"
REEF_ROLLOUT_DATA_KEYS = (
    "producing_runtime_load_spans",
    "producing_runtime_load_ids",
)


def validate_slime_args(args: SlimeArguments, spec: SlimeAlgorithm | None) -> None:
    """Reject Slime arguments the training component cannot run with.

    The component trains only: Slime starts no engines, and training and
    serving never share GPUs.
    """
    num_rollout = args.num_rollout
    if not isinstance(num_rollout, int) or isinstance(num_rollout, bool) or num_rollout <= 0:
        raise ValueError("Slime training requires a positive --num-rollout")
    save_hf = args.save_hf
    if not isinstance(save_hf, str) or "{rollout_id}" not in save_hf:
        raise ValueError("Slime training requires --save-hf with a {rollout_id} path template")
    validate_advantage_computation(args, spec)
    if args.debug_train_only:
        raise ValueError("Reef runs Slime's training for every job; remove --debug-train-only")
    if args.debug_rollout_only:
        raise ValueError("Slime has no rollout loop in Reef; remove --debug-rollout-only")
    for name, value in (
        ("--colocate", args.colocate),
        ("--offload-rollout", args.offload_rollout),
    ):
        if value:
            raise ValueError(f"training and serving use separate GPUs; remove {name}")
    if args.release_train:
        raise ValueError("the training workers keep running between jobs; remove --release-train")


def configure_critic_args_hook(args: SlimeArguments) -> None:
    """Install Reef's critic argument hook through Slime's public hook, after a user's own hook.

    ``finalize_reef_slime_args`` installs the Megatron init hook.
    """
    if args.loss_family is None:
        return
    critic_hook = args.custom_critic_args_hook_path
    if critic_hook and critic_hook != CRITIC_ARGS_HOOK_PATH:
        args.reef_chained_critic_args_hook_path = critic_hook
    args.custom_critic_args_hook_path = CRITIC_ARGS_HOOK_PATH


def configure_rollout_runtime(args: SlimeArguments) -> None:
    """Declare Reef's per-sample columns through Slime's payload hook."""
    configured = tuple(args.custom_rollout_data_keys or ())
    args.custom_rollout_data_keys = tuple(dict.fromkeys((*configured, *REEF_ROLLOUT_DATA_KEYS)))


def validate_advantage_computation(args: SlimeArguments, spec: SlimeAlgorithm | None) -> None:
    """Reef supplies training signals externally, with spec-declared exceptions.

    A loss family that keeps Slime's advantage pass declares
    ``allows_slime_advantage_computation``. Without a resolved family, any
    registered family that allows it is accepted.
    """
    if not args.compute_advantages_and_returns:
        return
    if spec is not None:
        if spec.allows_slime_advantage_computation:
            return
    else:
        from reef.train.slime_backend.loss_families import LOSS_FAMILIES

        if any(LOSS_FAMILIES.resolve(name).allows_slime_advantage_computation for name in LOSS_FAMILIES.names):
            return
    raise ValueError(
        "Reef supplies training signals externally; pass "
        "--disable-compute-advantages-and-returns in training.options"
    )


def prepare_checkpoint_storage(args: SlimeArguments, retention: RetentionConfig) -> CheckpointStorage:
    """Build the checkpoint store, refuse a storage that cannot hold the next checkpoint, and pin paths.

    Rewrites ``args.save_hf`` / ``args.save`` (and ``args.critic_save`` when a
    critic trains) to the storage's resolved absolute paths so the workers and
    the component agree on locations. A critic run without an explicit
    ``--critic-save`` gets ``<save>-critic`` so the critic's weights and
    optimizer survive restarts instead of cold-starting the value head.
    """
    critic_root = (args.critic_save or f"{args.save}-critic") if args.use_critic else None
    storage = CheckpointStorage(
        retention,
        hf_template=args.save_hf,
        megatron_root=args.save,
        critic_root=critic_root,
        source_hf=args.hf_checkpoint,
        source_megatron=args.load,
        lora=bool(args.megatron_lora_rank),
        critic_save_interval=args.critic_save_interval,
        saves_optimizer=not args.no_save_optim,
    )
    storage_plan = storage.validate_capacity(active_rollouts=set())
    if storage_plan["blocked"]:
        reasons = "; ".join(storage_plan["reasons"])
        raise RuntimeError(f"checkpoint storage preflight blocked Slime's start: {reasons}")
    args.save_hf, args.save = storage.hf_template, str(storage.megatron_root)
    if storage.critic_root is not None:
        args.critic_save = str(storage.critic_root)
    return storage
