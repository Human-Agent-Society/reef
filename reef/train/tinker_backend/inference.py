"""Text chat facade over Tinker's token-native immutable sampler."""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from reef_client.sse import synthesize_sse_events

from reef.artifact.artifact import Artifact
from reef.runtime.inference import InferenceBackend, InferenceStream, UpstreamStatusError
from reef.train.tinker_backend.client import TinkerClient
from reef.train.tinker_backend.runtime import TinkerRuntime


class TinkerInferenceBackend(InferenceBackend):
    def __init__(self, runtime: TinkerRuntime, client: TinkerClient, model: str) -> None:
        self._runtime = runtime
        self._client = client
        self._model = model

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path != "/v1/chat/completions":
            raise UpstreamStatusError("Tinker supports /v1/chat/completions", status=400)
        messages, params = _request(payload)
        return await asyncio.to_thread(self._sample, artifact, messages, params, payload)

    def _sample(
        self, artifact: Artifact, messages: list[dict[str, str]], params: dict[str, Any], payload: dict[str, Any]
    ) -> dict[str, Any]:
        checkpoint, version = self._runtime.snapshot(artifact)
        prompt = self._client.render(messages, template_kwargs=payload.get("chat_template_kwargs") or {})
        if not prompt:
            raise ValueError("Tinker chat template produced an empty prompt")
        result = self._client.sample(checkpoint, prompt, params)
        message = {"role": "assistant", "content": self._client.decode(result.tokens)}
        finish = "length" if result.stop_reason == "length" else "stop"
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model", self._model),
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(result.tokens),
                "total_tokens": len(prompt) + len(result.tokens),
            },
            "training": {
                "tokens": [*prompt, *result.tokens],
                "loss_mask": [1] * len(result.tokens),
                "rollout_log_probs": list(result.logprobs),
                "prompt_length": len(prompt),
                "response_length": len(result.tokens),
                "runtime_load_id": version,
                "request_messages": messages,
                "response_message": message,
                "finish_reason": finish,
            },
        }

    async def inference_stream(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> InferenceStream:
        response = await self.inference(artifact, path, payload)
        include_usage = bool((payload.get("stream_options") or {}).get("include_usage"))

        async def chunks() -> AsyncIterator[bytes]:
            for event in synthesize_sse_events(response, include_usage=include_usage):
                yield event.encode()

        return InferenceStream(
            status=200, headers={"Content-Type": "text/event-stream"}, chunks=chunks(), record_response=response
        )


def _request(payload: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    supported = {
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "stop",
        "stream",
        "stream_options",
        "n",
        "return_meta_info",
        "chat_template_kwargs",
    }
    unknown = payload.keys() - supported
    if unknown or payload.get("n", 1) != 1:
        raise UpstreamStatusError(f"unsupported Tinker chat options: {sorted(unknown)}; n must be 1", status=400)
    raw = payload.get("messages")
    if not isinstance(raw, list) or not raw:
        raise UpstreamStatusError("Tinker requires non-empty text messages", status=400)
    messages = []
    for message in raw:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or not isinstance(message["role"], str)
            or message["role"] not in {"system", "user", "assistant"}
            or not isinstance(message["content"], str)
        ):
            raise UpstreamStatusError("Tinker currently supports text system/user/assistant messages", status=400)
        messages.append(dict(message))
    params: dict[str, Any] = {
        key: payload[key] for key in ("temperature", "top_p", "top_k", "seed", "stop") if key in payload
    }
    params["max_tokens"] = payload.get("max_completion_tokens", payload.get("max_tokens", 1024))
    maximum = params["max_tokens"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise UpstreamStatusError("Tinker max_tokens must be a positive integer", status=400)
    template = payload.get("chat_template_kwargs", {})
    if (
        not isinstance(template, dict)
        or template.keys() - {"enable_thinking"}
        or any(not isinstance(value, bool) for value in template.values())
    ):
        raise UpstreamStatusError("Tinker chat_template_kwargs supports only boolean enable_thinking", status=400)
    stream_options = payload.get("stream_options", {})
    if not isinstance(stream_options, dict) or stream_options.keys() - {"include_usage"}:
        raise UpstreamStatusError("unsupported Tinker stream_options", status=400)
    for field, minimum, maximum_value in (("temperature", 0, None), ("top_p", 0, 1)):
        value = params.get(field, 1.0)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < minimum
            or (field == "top_p" and value == 0)
            or (maximum_value is not None and value > maximum_value)
        ):
            raise UpstreamStatusError(f"invalid Tinker {field}", status=400)
    top_k = params.get("top_k", -1)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or (top_k != -1 and top_k <= 0):
        raise UpstreamStatusError("Tinker top_k must be -1 or a positive integer", status=400)
    return messages, params
