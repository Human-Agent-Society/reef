"""Fail-fast checks before ``start_bridge`` boots the Slime training stack.

Everything here runs before any placement group or GPU worker exists, so a
configuration or storage problem stops the driver with a clear error instead
of a half-started cluster.
"""

from __future__ import annotations

import os

from reef.train.slime_backend.algorithm import SlimeAlgorithm
from reef.train.slime_backend.reef_adapters.sglang.lora_schema import (
    require_lora_distributed_request_schema,
    require_lora_tensor_request_schema,
)
from reef.train.slime_backend.reef_adapters.sglang.plugin import REEF_SGLANG_PLUGIN_ENV, SGLANG_PLUGIN_NAME
from reef.train.slime_backend.reef_adapters.training_job.marker import marker_rollouts, read_marker
from reef.train.slime_backend.reef_adapters.training_job.storage import CheckpointStorage, RetentionConfig

MEGATRON_INIT_PATH = "reef.train.slime_backend.reef_adapters.worker_hooks.initialize_megatron_objective"
CRITIC_ARGS_HOOK_PATH = "reef.train.slime_backend.reef_adapters.worker_hooks.configure_critic_objective"
REEF_ROLLOUT_DATA_KEYS = (
    "producing_runtime_load_spans",
    "producing_runtime_load_ids",
)


def validate_bridge_args(args, spec: SlimeAlgorithm | None) -> None:
    """Reject slime driver arguments the bridge cannot run with."""
    num_rollout = getattr(args, "num_rollout", None)
    if not isinstance(num_rollout, int) or isinstance(num_rollout, bool) or num_rollout <= 0:
        raise ValueError("the Reef bridge requires a positive --num-rollout")
    save_hf = getattr(args, "save_hf", None)
    if not isinstance(save_hf, str) or "{rollout_id}" not in save_hf:
        raise ValueError("the Reef bridge requires --save-hf with a {rollout_id} path template")
    _validate_advantage_computation(args, spec)
    if getattr(args, "debug_train_only", False):
        raise ValueError("the Reef bridge requires a live inference router; remove --debug-train-only")
    if getattr(args, "debug_rollout_only", False):
        raise ValueError("the Reef bridge has no internal rollout loop; remove --debug-rollout-only")
    rollout_num_gpus = getattr(args, "rollout_num_gpus", None)
    if not getattr(args, "rollout_external", False) and (
        not isinstance(rollout_num_gpus, int) or isinstance(rollout_num_gpus, bool) or rollout_num_gpus <= 0
    ):
        raise ValueError("the Reef bridge requires a positive --rollout-num-gpus for its local inference router")
    colocate = bool(getattr(args, "colocate", False))
    if colocate and (not getattr(args, "offload_train", False) or not getattr(args, "offload_rollout", False)):
        raise ValueError("the Reef bridge requires --offload-train and --offload-rollout with --colocate")
    if getattr(args, "offload_rollout", False) and not colocate:
        raise ValueError("the Reef bridge does not support --offload-rollout because Reef needs serving to stay live")
    save = getattr(args, "save", None)
    if not isinstance(save, str) or not save.strip():
        raise ValueError("the Reef bridge requires --save for Megatron recovery checkpoints")


def configure_sglang_runtime(args) -> None:
    """Apply Reef's serving invariants through generic Slime/SGLang options.

    Disjoint/PD weight updates preserve each active request's private KV state.
    Colocated regular engines retract instead, because their KV allocation
    must leave the GPU during training, and recompute it after publication. A
    shared radix-cache entry has no runtime-load-ID identity, so Reef disables
    cross-request prefix reuse. SGLang's native plugin hook installs Reef's
    token metadata and colocated suspension policy inside each scheduler.
    """
    colocate = bool(getattr(args, "colocate", False))
    args.sglang_disable_radix_cache = True
    # Reef consumes /generate as a token-native SSE source. Disjoint chunks
    # keep text, ids, log-probs, and scheduler metadata linear in rollout
    # length and give the capture path one unambiguous wire contract.
    args.sglang_incremental_streaming_output = True
    args.weight_update_pause_mode = "retract" if colocate else "in_place"
    os.environ[REEF_SGLANG_PLUGIN_ENV] = "1"
    configured_plugins = os.environ.get("SGLANG_PLUGINS")
    plugins = (
        tuple(name.strip() for name in configured_plugins.split(",") if name.strip()) if configured_plugins else ()
    )
    os.environ["SGLANG_PLUGINS"] = ",".join(dict.fromkeys((*plugins, SGLANG_PLUGIN_NAME)))

    if int(getattr(args, "megatron_lora_rank", 0) or 0) > 0:
        require_lora_tensor_request_schema()
        require_lora_distributed_request_schema()

    if colocate and int(getattr(args, "prefill_num_servers", 0) or 0) > 0:
        raise ValueError("colocated Reef serving requires a regular SGLang engine, not PD disaggregation")

    config_path = getattr(args, "sglang_config", None)
    if config_path is None:
        return

    from slime.backends.sglang_utils.sglang_config import SglangConfig

    config = SglangConfig.from_yaml(config_path)
    if colocate and config.has_pd_disaggregation:
        raise ValueError("colocated Reef serving requires a regular SGLang engine, not PD disaggregation")
    for model in config.models:
        for group in model.server_groups:
            overrides = {key.replace("-", "_"): value for key, value in group.overrides.items()}
            if overrides.get("disable_radix_cache", True) is not True:
                raise ValueError(
                    "Reef weight updates require disable_radix_cache=true "
                    f"for SGLang model {model.name!r} group {group.worker_type!r}"
                )


def configure_megatron_runtime(args) -> None:
    """Install Reef's worker initialization through Slime's public hook."""
    if getattr(args, "loss_family", None) is None:
        return
    current = getattr(args, "custom_megatron_init_path", None)
    if current and current != MEGATRON_INIT_PATH:
        args.reef_chained_megatron_init_path = current
    args.custom_megatron_init_path = MEGATRON_INIT_PATH
    critic_hook = getattr(args, "custom_critic_args_hook_path", None)
    if critic_hook and critic_hook != CRITIC_ARGS_HOOK_PATH:
        args.reef_chained_critic_args_hook_path = critic_hook
    args.custom_critic_args_hook_path = CRITIC_ARGS_HOOK_PATH


def configure_rollout_runtime(args) -> None:
    """Declare Reef's per-sample columns through Slime's payload hook."""
    configured = tuple(getattr(args, "custom_rollout_data_keys", ()) or ())
    args.custom_rollout_data_keys = tuple(dict.fromkeys((*configured, *REEF_ROLLOUT_DATA_KEYS)))


def _validate_advantage_computation(args, spec: SlimeAlgorithm | None) -> None:
    """The bridge supplies training signals externally, with spec-declared exceptions.

    A loss family that keeps Slime's advantage pass declares
    ``allows_slime_advantage_computation``.  Without a resolved family
    (``start_bridge`` called directly), any registered family that allows
    it is accepted.
    """
    if not getattr(args, "compute_advantages_and_returns", True):
        return
    if spec is not None:
        if spec.allows_slime_advantage_computation:
            return
    else:
        from reef.train.slime_backend.loss_families import LOSS_FAMILIES

        if any(LOSS_FAMILIES.resolve(name).allows_slime_advantage_computation for name in LOSS_FAMILIES.names):
            return
    raise ValueError(
        "the Reef bridge supplies training signals externally; pass "
        "--disable-compute-advantages-and-returns to the slime driver"
    )


def prepare_checkpoint_storage(args, retention: RetentionConfig) -> CheckpointStorage:
    """Build the checkpoint store, refuse ambiguous or blocked state, pin paths.

    Rewrites ``args.save_hf`` / ``args.save`` (and ``args.critic_save`` when a
    critic trains) to the storage's resolved absolute paths so the workers and
    the bridge actor agree on locations. A critic run without an explicit
    ``--critic-save`` gets ``<save>-critic`` so the critic's weights and
    optimizer survive restarts instead of cold-starting the value head.
    """
    critic_root = None
    if getattr(args, "use_critic", False):
        critic_root = getattr(args, "critic_save", None) or f"{args.save}-critic"
    storage = CheckpointStorage(
        retention,
        hf_template=args.save_hf,
        megatron_root=args.save,
        critic_root=critic_root,
        source_hf=getattr(args, "hf_checkpoint", None),
        source_megatron=getattr(args, "load", None),
    )
    marker = read_marker(storage.marker_path)
    if marker is not None and marker["status"] == "RUNNING":
        raise RuntimeError(f"ambiguous training job {marker['job_id']}")
    storage_plan = storage.validate_capacity(active_rollouts=marker_rollouts(marker))
    if storage_plan["blocked"]:
        reasons = "; ".join(storage_plan["reasons"])
        raise RuntimeError(f"checkpoint storage preflight blocked bridge startup: {reasons}")
    args.save_hf, args.save = storage.hf_template, str(storage.megatron_root)
    if storage.critic_root is not None:
        args.critic_save = str(storage.critic_root)
    return storage
