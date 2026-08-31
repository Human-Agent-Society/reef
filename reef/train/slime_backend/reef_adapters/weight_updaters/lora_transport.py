"""Topology-aware transports for publishing one SGLang LoRA adapter."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
from ray import ObjectRef
from slime.backends.megatron_utils.sglang import FlattenedTensorBucket, MultiprocessingSerializer
from slime.utils.distributed_utils import get_gloo_group


def send_lora_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine: Any,
    ipc_gather_src: int | None,
    ipc_gather_group: Any,
    lora_config: dict[str, Any],
    lora_name: str,
    lora_loaded: bool,
    expected_checksums: dict[str, str] | None,
) -> tuple[list[ObjectRef], list[dict[str, Any]] | None]:
    """Gather adapter tensors for one colocated engine without creating an NCCL peer on the same GPU."""
    if ipc_gather_group is None:
        return [], None
    dtypes = {tensor.dtype for _, tensor in hf_named_tensors}
    if len(dtypes) != 1:
        raise RuntimeError(f"SGLang tensor LoRA loading requires one adapter dtype, got {sorted(map(str, dtypes))}")
    bucket = FlattenedTensorBucket(named_tensors=hf_named_tensors)
    flattened_tensor_data = {
        "flattened_tensor": bucket.get_flattened_tensor(),
        "metadata": bucket.get_metadata(),
    }
    serialized = MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
    is_source = dist.get_rank() == ipc_gather_src
    gathered = [None] * dist.get_world_size(ipc_gather_group) if is_source else None
    dist.gather_object([serialized], object_gather_list=gathered, dst=ipc_gather_src, group=ipc_gather_group)
    refs: list[ObjectRef] = []
    if is_source:
        if ipc_engine is None:
            raise RuntimeError("colocated LoRA source rank has no SGLang engine")
        if gathered is None:
            raise RuntimeError("source rank did not allocate a LoRA gather buffer")
        serialized_named_tensors: list[str] = []
        for per_rank in gathered:
            if not per_rank:
                raise RuntimeError("LoRA tensor gather returned an empty trainer-rank payload")
            serialized_named_tensors.append(per_rank[0])
        if lora_loaded:
            validate_lora_update_results([ray.get(ipc_engine.unload_lora_adapter.remote(lora_name=lora_name))])
        refs.append(
            ipc_engine.load_lora_adapter_from_tensors.remote(
                lora_name=lora_name,
                config_dict=lora_config,
                serialized_named_tensors=serialized_named_tensors,
                load_format="flattened_bucket",
                expected_checksums=expected_checksums,
            )
        )
    return refs, [flattened_tensor_data]


def send_lora_to_distributed_engines(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    rollout_engines: Sequence[Any],
    model_update_group: Any,
    group_name: str,
    lora_config: dict[str, Any],
    lora_name: str,
) -> list[ObjectRef]:
    """Start every receiver, then broadcast adapter tensors once over NCCL."""
    if model_update_group is None:
        raise RuntimeError("distributed LoRA update group is not connected")
    refs = [
        engine.load_lora_adapter_from_distributed.remote(
            lora_name=lora_name,
            config_dict=lora_config,
            names=[name for name, _ in hf_named_tensors],
            dtypes=[tensor.dtype for _, tensor in hf_named_tensors],
            shapes=[tensor.shape for _, tensor in hf_named_tensors],
            group_name=group_name,
            upsert=True,
        )
        for engine in rollout_engines
    ]
    handles = [
        dist.broadcast(tensor.data, 0, group=model_update_group, async_op=True) for _, tensor in hf_named_tensors
    ]
    for handle in handles:
        handle.wait()
    return refs


def unload_lora_from_engines(engines: Sequence[Any], lora_name: str) -> None:
    """Remove one adapter revision from every engine; raise if any keeps it.

    The residency manager that asked for the eviction turns a failure into a
    visibly ``leaked`` slot, so this does not swallow it.
    """
    refs = [engine.unload_lora_adapter.remote(lora_name=lora_name) for engine in engines if engine is not None]
    validate_lora_update_results(ray.get(refs))


def tensor_checksums(named_tensors: list[tuple[str, torch.Tensor]]) -> dict[str, str]:
    checksums = {
        name: hashlib.sha256(
            tensor.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        for name, tensor in named_tensors
    }
    if len(checksums) != len(named_tensors):
        raise RuntimeError("LoRA publication contains duplicate tensor names")
    return checksums


def verify_replica_adapter_checksums(checksums: dict[str, str]) -> None:
    digest = hashlib.sha256()
    for name in sorted(checksums):
        digest.update(name.encode())
        digest.update(checksums[name].encode())
    local_digest = digest.hexdigest()
    replica_digests = [None] * dist.get_world_size()
    dist.all_gather_object(replica_digests, local_digest, group=get_gloo_group())
    if len(set(replica_digests)) != 1:
        raise RuntimeError(f"LoRA adapter replicas disagree before SGLang publication: {replica_digests}")


def validate_lora_update_results(results: list[Any]) -> None:
    for result in results:
        if result is None:
            continue
        if not isinstance(result, Mapping) or result.get("success") is not True:
            raise RuntimeError(f"SGLang LoRA adapter update failed: {result!r}")


__all__ = [
    "send_lora_to_colocated_engine",
    "send_lora_to_distributed_engines",
    "tensor_checksums",
    "unload_lora_from_engines",
    "validate_lora_update_results",
    "verify_replica_adapter_checksums",
]
