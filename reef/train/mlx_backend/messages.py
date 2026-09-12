"""Chat-message shaping shared by the render path — importable without MLX.

The engine applies the chat template in one place, but the message massaging
it needs is pure Python. Keeping it here (no ``mlx`` import) lets it be tested
without the optional extra installed, the same way the rest of the adapter
imports without MLX.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def prepare_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Copy messages, parsing OpenAI string tool-call arguments into dicts.

    An OpenAI client (Hermes is one) serialises a tool call's ``arguments`` as
    a JSON *string*; the Qwen3 template iterates ``tool_call.arguments|items``
    and needs a mapping, raising "Can only get item pairs from a mapping" on a
    string. A caller that already passes a dict (this repo's own stream driver
    did) never hit it, so the round-trip was only broken for real OpenAI
    clients. Parse the string here — the one place the template is applied —
    and leave a dict, a non-JSON string, or a missing field untouched.
    """
    prepared: list[dict[str, Any]] = []
    for message in messages:
        rendered = dict(message)
        calls = rendered.get("tool_calls")
        if isinstance(calls, list):
            rendered["tool_calls"] = [_prepare_tool_call(call) for call in calls]
        prepared.append(rendered)
    return prepared


def _prepare_tool_call(call: Any) -> Any:
    if not isinstance(call, Mapping):
        return call
    fn = call.get("function")
    if not isinstance(fn, Mapping):
        return call
    arguments = fn.get("arguments")
    if not isinstance(arguments, str):
        return call
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return call
    if not isinstance(parsed, dict):
        return call
    copy = dict(call)
    copy["function"] = {**fn, "arguments": parsed}
    return copy
