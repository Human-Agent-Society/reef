"""Opt-in FP32 head compute using Megatron's unchanged linear autograd.

Select ``provide_actor_model`` through ``custom-model-provider-path``, or
``install`` through ``custom-megatron-init-path`` for post-init binding.
Only BF16 actors with untied heads and CP1/PP1/DP1 are supported.
Native TP communication and DDP gradient accumulation remain unchanged.
This prototype still requires qualified native TP/DDP execution before use.
"""

from __future__ import annotations

from argparse import Namespace
from types import MethodType

import torch


def validate_options(args: Namespace) -> None:
    """Reject configurations outside the scoped native actor contract."""
    if args.loss_family != "sdpo" or args.use_critic:
        raise ValueError("FP32 head compute requires an SDPO actor without a critic")
    if (
        args.context_parallel_size != 1
        or args.pipeline_model_parallel_size != 1
        or args.virtual_pipeline_model_parallel_size not in (None, 1)
    ):
        raise ValueError("FP32 head compute requires CP1/PP1 without virtual pipeline interleaving")
    if args.megatron_lora_rank or not args.bf16 or not args.untie_embeddings_and_output_weights:
        raise ValueError("FP32 head compute requires BF16 training with untied weights and no LoRA")


def fp32_head_forward(
    head: torch.nn.Module,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    grad_output_buffer: list[torch.Tensor] | None = None,
    wgrad_deferral_limit: int | None = None,
    tp_group: object = None,
) -> torch.Tensor:
    """Delegate FP32 operands to the original native implementation without fusion."""
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    if weight is not head.weight or bias is not None:
        raise ValueError("FP32 head compute requires the original untied, bias-free head weight")
    if grad_output_buffer is not None or wgrad_deferral_limit is not None:
        raise ValueError("FP32 head compute does not support deferred weight gradients")
    if (
        tp_group is not head.tp_group
        or sequence_parallel != head.sequence_parallel
        or allreduce_dgrad != head.allreduce_dgrad
    ):
        raise ValueError("FP32 head compute requires the native TP group and collective flags")
    if input.ndim != 3 or input.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("FP32 head compute requires [sequence, batch, hidden] BF16 or FP32 input")
    if torch.is_grad_enabled() and weight.requires_grad and weight.grad_added_to_main_grad:
        raise ValueError("FP32 head compute requires native DDP to reset the head gradient flag")
    # Keep the leaf parameter and its DDP hooks; casts return gradients to that leaf.
    with torch.autocast(device_type=input.device.type, enabled=False):
        return ColumnParallelLinear._forward_impl(
            head,
            input=input.float(),
            weight=weight.float(),
            bias=None,
            gradient_accumulation_fusion=False,
            allreduce_dgrad=allreduce_dgrad,
            sequence_parallel=sequence_parallel,
            grad_output_buffer=None,
            wgrad_deferral_limit=None,
            tp_group=tp_group,
        )


def bind_model(model: torch.nn.Module) -> torch.nn.Module:
    """Change only the native head compute method, never modules or parameters."""
    from megatron.core import mpu
    from megatron.core.models.gpt import GPTModel
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    if not isinstance(model, GPTModel) or mpu.get_data_parallel_world_size() != 1:
        raise ValueError("FP32 head compute requires native GPT and DP1")
    if not model.post_process:
        return model
    head = model.output_layer
    if not isinstance(head, ColumnParallelLinear) or not isinstance(head._forward_impl, MethodType):
        raise ValueError("FP32 head compute requires a native ColumnParallelLinear output head")
    if head._forward_impl.__func__ not in (ColumnParallelLinear._forward_impl, fp32_head_forward):
        raise ValueError("FP32 head compute cannot replace an existing custom head implementation")
    if model.share_embeddings_and_output_weights or model.fuse_linear_cross_entropy:
        raise ValueError("FP32 head compute does not support tied weights or fused linear cross entropy")
    if (
        model.config.defer_embedding_wgrad_compute
        or model.config.cpu_offloading
        or model.config.fp8
        or model.config.fp4
        or model.config.cuda_graph_impl != "none"
    ):
        raise ValueError("FP32 head compute does not support deferral, offload, quantization, or CUDA graphs")
    if (
        head.bias is not None
        or not isinstance(head.weight, torch.nn.Parameter)
        or head.weight.dtype != torch.bfloat16
        or head.is_expert
        or head.disable_grad_reduce
        or head.embedding_activation_buffer is not None
        or head.grad_output_buffer is not None
        or head.tp_group is not model.pg_collection.tp
    ):
        raise ValueError("FP32 head compute requires the unmodified bias-free BF16 actor head")
    if head._forward_impl.__func__ is not fp32_head_forward:
        head._forward_impl = MethodType(fp32_head_forward, head)
    return model


def provide_actor_model(
    pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None
) -> torch.nn.Module:
    """Construct once through Slime's canonical actor provider using copied arguments."""
    from megatron.training.global_vars import get_args
    from slime.backends.megatron_utils.model_provider import get_model_provider_func

    args = get_args()
    validate_options(args)
    native_args = Namespace(**vars(args))
    native_args.custom_model_provider_path = None
    provider = get_model_provider_func(native_args, role="actor")
    return bind_model(provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage))


def initialize_actor(actor: object) -> None:
    """Bind after native precision conversion, DDP wrapping, and optimizer creation."""
    from megatron.core.distributed import DistributedDataParallel
    from megatron.core.transformer.module import Float16Module

    validate_options(actor.args)
    if actor.role != "actor" or len(actor.model) != 1:
        raise ValueError("FP32 head compute requires exactly one actor model chunk")
    chunk = actor.model[0]
    if not isinstance(chunk, DistributedDataParallel) or chunk.ddp_config.delay_wgrad_compute:
        raise ValueError("FP32 head compute requires native DDP without delayed weight gradients")
    if not isinstance(chunk.module, Float16Module):
        raise ValueError("FP32 head compute requires the native BF16 model wrapper")
    model = chunk.module.module
    weight = model.output_layer.weight
    if weight.requires_grad and (weight.main_grad.dtype != torch.float32 or weight.grad_added_to_main_grad):
        raise ValueError("FP32 head compute requires a reset native FP32 DDP gradient buffer")
    if actor.args.offload_train:
        actor.wake_up()
    try:
        bind_model(model)
    finally:
        if actor.args.offload_train:
            actor.sleep()


def install(args: Namespace) -> None:
    """Initialize existing capture and FP32 compute once through Reef's actor hook."""
    from recipes.sdpo.examples.agentcl import worker_capture
    from recipes.sdpo.slime import objective as native_module
    from reef.train.slime_backend.algorithm import objective

    validate_options(args)
    worker_capture.install(args)

    def _initialize(actor: object) -> None:
        worker_capture.initialize_actor(actor)
        initialize_actor(actor)

    _initialize.__module__ = native_module.__name__
    _initialize.__name__ = "agentcl_initialize_precision"
    native_module.agentcl_initialize_precision = objective("reef_actor_init_hook_path")(_initialize)
