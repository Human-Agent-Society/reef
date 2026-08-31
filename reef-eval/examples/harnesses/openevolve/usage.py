"""Capture usage that OpenEvolve 0.3.2 otherwise discards."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from openevolve.llm.openai import OpenAILLM


def _usage_value(value: Any, name: str) -> int:
    result = getattr(value, name, 0) if value is not None else 0
    return int(result or 0)


def _record_usage(response: Any, fallback_model: str) -> None:
    path = os.environ.get("REEF_EVAL_USAGE_FILE")
    usage = getattr(response, "usage", None)
    if not path or usage is None:
        return
    details = getattr(usage, "prompt_tokens_details", None)
    record = {
        "model": str(getattr(response, "model", None) or fallback_model),
        "input_tokens": _usage_value(usage, "prompt_tokens"),
        "cached_input_tokens": _usage_value(details, "cached_tokens"),
        "output_tokens": _usage_value(usage, "completion_tokens"),
    }
    usage_path = Path(path)
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    with usage_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")


async def _metered_call_api(self: OpenAILLM, params: dict[str, Any]) -> str:
    if self.client is None:
        raise RuntimeError("OpenAI client is not initialized (manual_mode enabled?)")
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, lambda: self.client.chat.completions.create(**params)
    )
    _record_usage(response, str(self.model))
    logging.getLogger("openevolve.llm.openai").debug(
        "API response: %s", response.choices[0].message.content
    )
    return response.choices[0].message.content


def install_usage_tracking() -> None:
    """Instrument OpenEvolve's model call before its CLI starts."""
    if getattr(OpenAILLM, "_reef_eval_usage_patch", False):
        return
    OpenAILLM._call_api = _metered_call_api
    OpenAILLM._reef_eval_usage_patch = True
