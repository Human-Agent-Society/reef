"""Translate Slime training flags into plain deployment input values for the selected inference backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments
from reef.train.slime_backend.reef_adapters.worker_hooks import reef_rollout_env_vars

#: Slime flags that are not engine options even though they carry the prefix.
_NOT_ENGINE_OPTIONS = frozenset({"sglang_config", "sglang_model_routers"})
#: Router flags Reef binds itself rather than forwarding.
_ROUTER_BIND_OPTIONS = frozenset({"sglang_router_ip", "sglang_router_port"})
#: Inference backends the Slime trainer can publish weights to.
INFERENCE_BACKENDS = ("sglang", "vllm")


def inference_config(
    args: Any, backend: str = "sglang", inference_options: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return launch input without importing or constructing an inference backend.

    SGLang engine options arrive as Slime's ``sglang_*`` flags on ``args``. vLLM
    engine options never pass through Slime's parser: ``inference_options`` is
    Reef's resolved ``inference.options`` mapping in vLLM's own names.
    """
    if backend not in INFERENCE_BACKENDS:
        raise ValueError(f"Slime weight transfer supports inference backends {INFERENCE_BACKENDS}, not {backend!r}")
    env = reef_rollout_env_vars()
    if "SLIME_HOST_IP" in env:
        env["REEF_INFERENCE_HOST"] = env.pop("SLIME_HOST_IP")
    colocate = bool(getattr(args, "colocate", False))
    if backend == "vllm":
        return vllm_inference_config(args, inference_options or {}, env=env, colocate=colocate)
    # Colocated engines retract already, because their KV allocation must
    # leave the GPU during training. A disjoint engine preserves in-flight KV
    # unless --disjoint-prefix-sharing trades a re-prefill at each publication
    # for prefix reuse between publications.
    retracts_at_publication = colocate or bool(args.disjoint_prefix_sharing)
    return {
        "model_path": args.hf_checkpoint,
        "num_gpus": args.rollout_num_gpus,
        "gpus_per_engine": args.rollout_num_gpus_per_engine,
        "gpus_per_node": args.num_gpus_per_node,
        "options": engine_options(args),
        "models": _model_groups(args),
        "external_engines": tuple(getattr(args, "rollout_external_engine_infos", ()) or ()),
        "router_host": getattr(args, "sglang_router_ip", None),
        "router_port": getattr(args, "sglang_router_port", None),
        "router_options": {
            key.removeprefix("sglang_router_"): value
            for key, value in vars(args).items()
            if key.startswith("sglang_router_") and key not in _ROUTER_BIND_OPTIONS
        },
        "env_vars": env,
        "offload": bool(getattr(args, "offload_rollout", False)),
        "shared_gpus": args.actor_num_nodes * args.actor_num_gpus_per_node if colocate else 0,
        "check_weights": bool(getattr(args, "check_weight_update_equal", False)),
        "pause_mode": "retract" if retracts_at_publication else "in_place",
        "health_enabled": bool(getattr(args, "use_fault_tolerance", False)),
        "health_interval": getattr(args, "rollout_health_check_interval", 30),
        "health_timeout": getattr(args, "rollout_health_check_timeout", 30),
        "health_first_wait": getattr(args, "rollout_health_check_first_wait", 60),
        "request_timeout": getattr(args, "distributed_timeout_minutes", 10) * 60,
        "executor": getattr(args, "reef_rollout_executor_backend", "auto"),
        "executor_options": getattr(args, "reef_rollout_executor_options", {}),
    }


def vllm_inference_config(
    args: Any, inference_options: Mapping[str, Any], *, env: dict[str, str], colocate: bool
) -> dict[str, Any]:
    """The ``VLLMConfig`` input: one group of identical engines, Reef's placement decisions, vLLM option names.

    Only the disk weight transport reaches a vLLM engine today, so LoRA and
    weight-checker settings have no vLLM counterpart here; ``SlimeDeployment``
    rejects them before this runs.
    """
    options = {key.replace("-", "_"): value for key, value in inference_options.items()}
    options.setdefault("trust_remote_code", True)
    options.setdefault("seed", args.seed)
    if args.fp16:
        options["dtype"] = "float16"
    return {
        "model_path": args.hf_checkpoint,
        "num_gpus": args.rollout_num_gpus,
        "gpus_per_engine": args.rollout_num_gpus_per_engine,
        "gpus_per_node": args.num_gpus_per_node,
        "options": options,
        "env_vars": env,
        "offload": bool(getattr(args, "offload_rollout", False)),
        "shared_gpus": args.actor_num_nodes * args.actor_num_gpus_per_node if colocate else 0,
        "pause_mode": "retract" if (colocate or bool(args.disjoint_prefix_sharing)) else "in_place",
        "health_enabled": bool(getattr(args, "use_fault_tolerance", False)),
        "health_interval": getattr(args, "rollout_health_check_interval", 30),
        "health_timeout": getattr(args, "rollout_health_check_timeout", 30),
        "health_first_wait": getattr(args, "rollout_health_check_first_wait", 60),
        "request_timeout": getattr(args, "distributed_timeout_minutes", 10) * 60,
        "executor": getattr(args, "reef_rollout_executor_backend", "auto"),
        "executor_options": getattr(args, "reef_rollout_executor_options", {}),
    }


def engine_options(args: SlimeArguments) -> dict[str, Any]:
    """SGLang ``ServerArgs`` values: Slime's ``sglang_*`` flags plus what Reef serving needs."""
    options = {
        key.removeprefix("sglang_"): value
        for key, value in vars(args).items()
        if key.startswith("sglang_") and not key.startswith("sglang_router_") and key not in _NOT_ENGINE_OPTIONS
    }
    options.update(
        model_path=args.hf_checkpoint,
        trust_remote_code=True,
        random_seed=args.seed,
        enable_memory_saver=bool(args.offload_rollout),
        enable_draft_weights_cpu_backup=True,
        skip_server_warmup=True,
        enable_metrics=True,
        incremental_streaming_output=True,
    )
    # Slime's parser defaults the radix flag to off, which cannot express a
    # choice. Leave it unset unless the launch opts out, so the inference
    # config shares prefixes exactly where publication clears them.
    if options.get("disable_radix_cache") is not True:
        options.pop("disable_radix_cache", None)
    if args.fp16:
        options["dtype"] = "float16"
    if args.use_rollout_routing_replay:
        options["enable_return_routed_experts"] = True
    rank = args.megatron_lora_rank
    if rank > 0:
        options.update(_lora_options(args, rank))
    return options


def _lora_options(args: Any, rank: int) -> dict[str, Any]:
    from reef.train.slime_backend.reef_adapters.megatron.lora import sglang_lora_target_modules

    configured_slots = getattr(args, "max_loaded_loras", None)
    slots = 1 if configured_slots is None else int(configured_slots)
    if slots < 1:
        raise ValueError("--max-loaded-loras must be at least 1")
    return {
        "enable_lora": True,
        "max_lora_rank": rank,
        "max_loaded_loras": slots,
        "max_loras_per_batch": slots,
        "lora_target_modules": sglang_lora_target_modules(args),
        "enable_weights_cpu_backup": True,
        "tokenizer_worker_num": 1,
    }


def _model_groups(args: Any) -> tuple[dict[str, Any], ...]:
    """Explicit model groups from Slime's group file or prefill/decode split; empty means one default group."""
    config_path = getattr(args, "sglang_config", None)
    if config_path:
        # Slime's CLI still accepts its legacy group file. Resolve it only at
        # this input boundary; inference receives ordinary Reef values.
        from slime.backends.sglang_utils.sglang_config import SglangConfig

        parsed = SglangConfig.from_yaml(config_path)
        for model in parsed.models:
            model.resolve(args)
        return tuple(
            {
                "name": model.name,
                "groups": tuple(
                    {
                        "worker_type": group.worker_type,
                        "num_gpus": group.num_gpus,
                        "gpus_per_engine": group.num_gpus_per_engine,
                        "options": {key.replace("-", "_"): value for key, value in group.overrides.items()},
                    }
                    for group in model.server_groups
                ),
                "update_weights": bool(model.update_weights),
            }
            for model in parsed.models
        )
    if getattr(args, "prefill_num_servers", 0):
        prefill = args.prefill_num_servers * args.rollout_num_gpus_per_engine
        return (
            {
                "name": "default",
                "groups": (
                    {
                        "worker_type": "prefill",
                        "num_gpus": prefill,
                        "gpus_per_engine": args.rollout_num_gpus_per_engine,
                    },
                    {
                        "worker_type": "decode",
                        "num_gpus": args.rollout_num_gpus - prefill,
                        "gpus_per_engine": args.rollout_num_gpus_per_engine,
                    },
                ),
            },
        )
    return ()
