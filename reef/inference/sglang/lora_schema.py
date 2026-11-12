"""Whether the loaded SGLang lets LoRA serving share radix-cache entries."""

from __future__ import annotations

from array import array
from functools import cache


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
