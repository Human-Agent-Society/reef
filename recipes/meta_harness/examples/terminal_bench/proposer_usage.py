"""Freeze LiteLLM's provider rates and price observed proposer token usage."""

import hashlib
import json
import math


def freeze_pricing(model):
    import litellm

    prices = litellm.model_cost.get(model) or litellm.model_cost.get("openai/" + model)
    if not prices:
        raise ValueError("proposer model has no known pricing; configure a priced model before search")
    selected = {key: value for key, value in prices.items() if "cost" in key}
    for key in ("input_cost_per_token", "output_cost_per_token"):
        value = selected.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("proposer pricing must have finite positive input and output rates")
    return {
        "model": model,
        "rates": selected,
        "sha256": hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
        "source": "litellm 1.99.0 model cost table",
    }


def response_cost(response, pricing):
    """Price known usage; incomplete or unsupported billing data stays unknown.

    Cache discounts are used only when explicitly present in the frozen table;
    otherwise full input price is a conservative budget charge.
    """
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    responses = "input_tokens" in usage or "output_tokens" in usage
    counts = [
        usage.get(key)
        for key in (("input_tokens", "output_tokens") if responses else ("prompt_tokens", "completion_tokens"))
    ]
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts):
        return None
    inputs, outputs = counts
    details = usage.get("input_tokens_details" if responses else "prompt_tokens_details") or {}
    if not isinstance(details, dict):
        return None
    cached = details.get("cached_tokens", 0)
    if isinstance(cached, bool) or not isinstance(cached, int) or not 0 <= cached <= inputs:
        return None
    tier = response.get("service_tier") or "default"
    if tier not in ("default", "priority", "flex"):
        return None
    suffix = ("_above_272k_tokens" if inputs > 272_000 else "") + ("_" + tier if tier != "default" else "")
    rates = pricing["rates"]
    input_rate = rates.get("input_cost_per_token" + suffix)
    output_rate = rates.get("output_cost_per_token" + suffix)
    if input_rate is None or output_rate is None:
        return None
    cached_rate = rates.get("cache_read_input_token_cost" + suffix, input_rate)
    cost = (inputs - cached) * input_rate + cached * cached_rate + outputs * output_rate
    return cost if math.isfinite(cost) and cost >= 0 else None
