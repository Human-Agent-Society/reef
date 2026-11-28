"""Fail-fast checks before Reef starts Slime's training workers.

Everything here runs before any placement group or GPU worker exists, so a
configuration or storage problem stops the start with a clear error instead
of a half-started cluster. Each training worker checks its own environment
again before it builds the model.
"""

from __future__ import annotations

from collections.abc import Mapping

from reef.train.slime_backend.algorithm import SlimeAlgorithm
from reef.train.slime_backend.loss_families import loss_family_names, resolve_loss_family
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments
from reef.train.slime_backend.reef_adapters.training_job.storage import CheckpointStorage, RetentionSettings

CRITIC_ARGS_HOOK_PATH = "reef.train.slime_backend.reef_adapters.worker_hooks.configure_critic_objective"
REEF_ROLLOUT_DATA_KEYS = ("producing_artifact_ids",)
#: The PyTorch allocator settings that can turn on expandable segments; PyTorch and torch_memory_saver read both.
PYTORCH_ALLOCATOR_ENVS = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
#: Slime's own switch: ``1`` turns on expandable segments when a training worker starts.
SLIME_EXPANDABLE_SEGMENTS_ENV = "SLIME_ENABLE_EXPANDABLE_SEGMENTS"


def validate_slime_args(args: SlimeArguments, spec: SlimeAlgorithm | None) -> None:
    """Reject Slime arguments the training component cannot run with.

    Slime places each job and starts its rollout engines itself; Reef's
    serving engines are separate.
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
    if args.rollout_num_gpus and args.megatron_lora_rank:
        # Reef's LoRA adapters are not wired into Slime's weight transfer to its engines.
        raise ValueError(
            "rollout engines need a full-weight actor; remove --megatron-lora-rank or the rollout engines"
        )
    if args.release_train:
        raise ValueError("the training workers keep running through a job; remove --release-train")


def enables_expandable_segments(allocator_settings: str) -> bool:
    """Whether PyTorch allocator settings turn on expandable segments.

    The settings are comma-separated ``key:value`` pairs, such as
    ``max_split_size_mb:512,expandable_segments:True``. Whitespace does not
    matter, and ``true`` counts as ``True``, as torch_memory_saver reads it.
    One such pair is sufficient, because PyTorch and torch_memory_saver can
    read different pairs of a repeated key.
    """
    for pair in allocator_settings.split(","):
        key, _, value = pair.partition(":")
        if key.strip() == "expandable_segments" and value.strip().lower() == "true":
            return True
    return False


def validate_training_worker_environment(args: SlimeArguments, environ: Mapping[str, str]) -> None:
    """Refuse expandable segments for training workers that Slime offloads without a CPU copy of their buffers.

    ``environ`` is the environment the workers inherit, the training
    service's or a worker's own; Slime's ``--train-env-vars`` go over it.
    Slime offloads the workers and turns off the CPU copy of Megatron's
    parameter and gradient buffers with ``--colocate``, ``--offload``,
    ``--offload-train`` or a ppo critic. torch_memory_saver then frees the
    workers' GPU memory while they sleep. With expandable segments, it decides
    the CPU copy for each 20 MiB page, not for each tensor. A tensor that shares
    a page with a buffer can then come back as zeros, and the next optimizer
    step writes the zeros into the weights.
    """
    if not (args.offload_train and (args.disable_grad_buffers_cpu_backup or args.disable_param_buffers_cpu_backup)):
        return
    worker_environ = {**environ, **args.train_env_vars}
    names = [name for name in PYTORCH_ALLOCATOR_ENVS if enables_expandable_segments(worker_environ.get(name, ""))]
    if worker_environ.get(SLIME_EXPANDABLE_SEGMENTS_ENV) == "1":
        names.append(SLIME_EXPANDABLE_SEGMENTS_ENV)
    if not names:
        return
    settings = ", ".join(
        f"{name}={worker_environ[name]} in {'--train-env-vars' if name in args.train_env_vars else 'the environment'}"
        for name in names
    )
    fixes = [
        f"set {name} to 0" if name == SLIME_EXPANDABLE_SEGMENTS_ENV else f"remove expandable_segments from {name}"
        for name in names
    ]
    raise ValueError(
        "Slime offloads these training workers without a CPU copy of their parameter or gradient buffers "
        "(--colocate, --offload, --offload-train or advantage-estimator ppo), and they cannot use expandable "
        f"segments ({settings}): torch_memory_saver keeps their CPU copy per 20 MiB page, so a tensor that shares "
        "a page with a buffer can come back as zeros after the workers sleep, and the optimizer step writes the "
        f"zeros into the weights. To fix this, {' and '.join(fixes)}, or keep the workers on the GPU: remove "
        "--offload and --offload-train, give the rollout engines GPUs of their own instead of --colocate, and "
        "keep a ppo critic on the GPU with --use-critic and --no-offload-train"
    )


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
        if any(resolve_loss_family(name).allows_slime_advantage_computation for name in loss_family_names()):
            return
    raise ValueError(
        "Reef supplies training signals externally; pass --disable-compute-advantages-and-returns in training.options"
    )


def prepare_checkpoint_storage(args: SlimeArguments, retention: RetentionSettings) -> CheckpointStorage:
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
