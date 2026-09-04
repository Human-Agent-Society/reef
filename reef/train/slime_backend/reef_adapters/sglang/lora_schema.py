"""The SGLang LoRA request schemas Reef's weight updates depend on.

Both the driver preflight (before any GPU worker exists) and each rollout
engine actor's constructor check these, so the required field sets live here
once: a loaded SGLang that drifts from them fails the same way on both paths.
"""

from __future__ import annotations

DISTRIBUTED_UPSERT_FIELDS = frozenset(
    {"lora_name", "config_dict", "names", "dtypes", "shapes", "group_name", "upsert"}
)
TENSOR_LOAD_FIELDS = frozenset(
    {"lora_name", "config_dict", "serialized_named_tensors", "load_format", "expected_checksums"}
)


def _require_struct_fields(request_type: type, required: frozenset[str], message: str) -> None:
    missing = sorted(required - set(getattr(request_type, "__struct_fields__", ())))
    if missing:
        raise RuntimeError(f"{message}; missing fields: {missing}; use Reef's pinned SGLang image")


def require_lora_distributed_request_schema() -> None:
    """Fail when SGLang cannot receive Reef's distributed LoRA upsert."""
    message = "loaded SGLang does not support Reef's distributed LoRA upsert schema"
    try:
        from sglang.srt.managers.io_struct import LoadLoRAAdapterFromDistributedReqInput
    except ImportError as exc:
        raise RuntimeError(message) from exc
    _require_struct_fields(LoadLoRAAdapterFromDistributedReqInput, DISTRIBUTED_UPSERT_FIELDS, message)


def require_lora_tensor_request_schema() -> None:
    """Fail when SGLang cannot receive Reef's colocated LoRA tensor update."""
    message = "loaded SGLang does not support Reef's colocated LoRA tensor schema"
    try:
        from sglang.srt.managers.io_struct import LoadLoRAAdapterFromTensorsReqInput
    except ImportError as exc:
        raise RuntimeError(message) from exc
    _require_struct_fields(LoadLoRAAdapterFromTensorsReqInput, TENSOR_LOAD_FIELDS, message)
