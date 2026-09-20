"""The SGLang LoRA request schemas Reef's weight updates depend on.

Both the driver preflight (before any GPU worker exists) and each rollout
engine actor's constructor check these, so the required field sets live here
once: a loaded SGLang that drifts from them fails the same way on both paths.
The adapter-scoped prefix-cache check answers whether LoRA serving may share
radix-cache entries at all.
"""

from __future__ import annotations

from array import array
from functools import cache

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


@cache
def adapter_scoped_prefix_cache_supported() -> bool:
    """Whether SGLang keys radix-cache entries per LoRA adapter.

    One engine holds several scenarios' adapters, and the same prefix has
    different KV under each. Exercise the request and radix-key APIs with
    identical tokens: requests using one adapter must share a key, while
    another adapter must not. These CPU-side request objects allocate no
    serving KV or model weights.

    Any failure answers "no". The check exists to survive a pin bump, so a
    build whose request API raises something unforeseen must leave sharing off
    rather than abort the deployment before it starts.
    """
    try:
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.sampling.sampling_params import SamplingParams

        keys = []
        for index, adapter in enumerate(("reef-probe-a", "reef-probe-a", "reef-probe-b")):
            tokens = array("q", [1])
            request = Req(
                rid=f"reef-prefix-probe-{index}",
                origin_input_text="",
                origin_input_ids=tokens,
                sampling_params=SamplingParams(max_new_tokens=1),
                lora_id=adapter,
            )
            keys.append(RadixKey(token_ids=tokens, extra_key=request.extra_key).child_key())
    except Exception:
        return False
    first_key, same_adapter_key, other_adapter_key = keys
    return first_key == same_adapter_key and first_key != other_adapter_key
