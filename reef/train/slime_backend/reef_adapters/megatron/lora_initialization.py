"""Load unfused Qwen3.5 PEFT adapters into the matching Megatron TP shards.

Only row-parallel MLP down and attention output projections are supported.
Reject fused projections rather than silently changing the adapter's rank.
"""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file


def adapter_shard(weight: torch.Tensor, parameter: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Match a PEFT tensor to one Megatron parameter, checking the TP layout."""
    if weight.shape == parameter.shape:
        return weight
    # These attributes are attached by Megatron at the third-party boundary.
    dimension = getattr(parameter, "partition_dim", None)
    stride = getattr(parameter, "partition_stride", 1)
    parallel = getattr(parameter, "tensor_model_parallel", False)
    if not parallel or dimension not in (0, 1) or stride != 1 or weight.shape[dimension] % world_size:
        raise ValueError("Unsupported adapter tensor-parallel layout")
    shard = weight.chunk(world_size, dim=dimension)[rank]
    if shard.shape != parameter.shape:
        raise ValueError(f"Adapter shape {tuple(shard.shape)} differs from model {tuple(parameter.shape)}")
    return shard


@torch.no_grad()
def load_initial_adapter(model: torch.nn.Module, path: str, *, rank: int, alpha: int) -> None:
    """Import the complete SFT adapter before optimizer master weights are created."""
    from megatron.core import parallel_state

    if parallel_state.get_pipeline_model_parallel_world_size() != 1:
        raise ValueError("Initial PEFT adapter loading currently requires pipeline parallel size 1")
    directory = Path(path)
    config = json.loads((directory / "adapter_config.json").read_text())
    if config.get("r") != rank or config.get("lora_alpha") != alpha:
        raise ValueError("Initial adapter rank/alpha must match the OPD adapter")
    if any(
        config.get(key)
        for key in ("use_dora", "use_rslora", "fan_in_fan_out", "modules_to_save", "rank_pattern", "alpha_pattern")
    ):
        raise ValueError("Initial adapter uses unsupported PEFT options")
    if config.get("bias", "none") != "none" or config.get("lora_dropout", 0) != 0:
        raise ValueError("Initial adapter requires no bias and zero dropout")
    weights = load_file(str(directory / "adapter_model.safetensors"), device="cpu")
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    pattern = re.compile(
        r"decoder\.layers\.(\d+)\.(mlp\.linear_fc2|self_attention\.linear_proj)\.adapter\.linear_(in|out)\.weight$"
    )
    converted = []
    consumed = set()
    for name, parameter in model.named_parameters():
        if ".adapter." not in name:
            continue
        match = pattern.fullmatch(name)
        if match is None:
            raise ValueError(f"Unsupported initial adapter parameter: {name}")
        layer, projection, side = match.groups()
        target = "mlp.down_proj" if projection == "mlp.linear_fc2" else "self_attn.o_proj"
        letter = "A" if side == "in" else "B"
        key = f"base_model.model.model.language_model.layers.{layer}.{target}.lora_{letter}.weight"
        if key not in weights:
            raise ValueError(f"Initial adapter is missing {key}")
        weight = weights[key]
        if not torch.isfinite(weight).all():
            raise ValueError(f"Initial adapter contains non-finite values: {key}")
        converted.append((parameter, adapter_shard(weight, parameter, tp_rank, tp_size)))
        consumed.add(key)
    if not converted or consumed != set(weights):
        raise ValueError(f"Initial adapter has missing or unused tensors: {set(weights) - consumed}")
    # Validate everything before mutating model parameters.
    for parameter, weight in converted:
        parameter.copy_(weight.to(device=parameter.device, dtype=parameter.dtype))
