"""The OpenAI text chat of a task job's rollouts, answered on Slime's own SGLang engines, with the training capture.

An agent's model call reaches ``EngineChat`` through ``ChatEndpoint`` (``reef.train.task_rollouts``). This module
imports no Ray, torch or Slime.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from typing import Any

from reef.train.task_rollouts import CHAT_COMPLETIONS_PATH, RolloutChat

#: The OpenAI chat sampling options that the engines take, and SGLang's name of each.
SAMPLING_OPTIONS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "stop": "stop",
    "seed": "sampling_seed",
}
#: The other OpenAI chat options a request may carry; the engines answer each request once and never stream.
REQUEST_OPTIONS = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "stream",
        "stream_options",
        "n",
        "chat_template_kwargs",
    }
)
#: The response tokens of a request that names no limit.
DEFAULT_MAX_TOKENS = 1024


class EngineChat(RolloutChat):
    """Answer a task job's OpenAI text chat calls on the rollout group's engines, with the training capture.

    ``router_url`` is the engines' router. ``model_path`` is the model directory whose tokenizer and chat template
    render each request; the tokenizer loads at the first call. ``timeout_s`` is a positive bound on one call,
    in seconds. ``topk_logprobs`` is 0, or the positive number of the engines' top log-probs to record at each
    response token, for score centering. ``distribution`` holds SGLang's ``temperature``, ``top_p`` and ``top_k`` of
    the distribution that the trainer reads: each call samples it, whatever the agent asks. ``max_response_tokens``
    is a positive bound on the response tokens of one call, or None. Each answer is a chat completion whose
    ``training`` block holds the prompt and response token IDs, the response's loss mask and sampled log-probs,
    ``runtime_load_id``, the weights the engines hold for the job, the engines' ``weight_version`` when they give it,
    and with ``topk_logprobs`` the heads ``topk_indices`` and ``topk_log_probs``, one list of ``topk_logprobs`` per
    response token. Another path gets 404, a request outside text system, user and assistant messages (or with ``n``
    other than 1) 400, an engine error status its own status, and an unusable engine answer 502.
    """

    def __init__(
        self,
        router_url: str,
        *,
        model_path: str,
        timeout_s: float,
        runtime_load_id: str,
        distribution: Mapping[str, float | int],
        max_response_tokens: int | None = None,
        topk_logprobs: int = 0,
    ) -> None:
        if not router_url or not model_path or not runtime_load_id:
            raise ValueError("EngineChat needs a router URL, a model path and a runtime load ID")
        if not timeout_s > 0:
            raise ValueError("EngineChat.timeout_s must be positive")
        if set(distribution) != {"temperature", "top_p", "top_k"}:
            raise ValueError("EngineChat.distribution must hold temperature, top_p and top_k")
        if max_response_tokens is not None and max_response_tokens <= 0:
            raise ValueError("EngineChat.max_response_tokens must be positive or None")
        if isinstance(topk_logprobs, bool) or not isinstance(topk_logprobs, int) or topk_logprobs < 0:
            raise ValueError("EngineChat.topk_logprobs must be a non-negative int")
        self.topk_logprobs = topk_logprobs
        self.router_url = router_url.rstrip("/")
        self.model_path = model_path
        self.timeout_s = timeout_s
        self.runtime_load_id = runtime_load_id
        self.distribution = dict(distribution)
        self.max_response_tokens = max_response_tokens
        # The model's Hugging Face tokenizer, loaded at the first call.
        self.tokenizer: Any = None

    def render(self, messages: list[dict[str, str]], template_kwargs: Mapping[str, Any]) -> list[int]:
        """The prompt token IDs of ``messages``; a last assistant message is continued, not answered."""
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        prefill = messages[-1]["role"] == "assistant"
        return list(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=not prefill,
                continue_final_message=prefill,
                **template_kwargs,
            )
        )

    def generate(self, body: Mapping[str, Any]) -> tuple[int, Any]:
        """POST ``body`` to the router's ``/generate``: the status and the decoded answer, or the error text."""
        request = urllib.request.Request(
            f"{self.router_url}/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode(errors="replace")[:400]

    def answer(self, path: str, payload: Any) -> tuple[int, Mapping[str, Any]]:
        if path != CHAT_COMPLETIONS_PATH:
            return 404, {"error": {"message": f"no route {path}"}}
        try:
            messages, sampling, template_kwargs = chat_request(payload)
        except ValueError as exc:
            return 400, {"error": {"message": str(exc)}}
        sampling.update(self.distribution)
        if self.max_response_tokens is not None:
            sampling["max_new_tokens"] = min(sampling["max_new_tokens"], self.max_response_tokens)
        request: dict[str, Any] = {"sampling_params": sampling, "return_logprob": True}
        if self.topk_logprobs:
            request["top_logprobs_num"] = self.topk_logprobs
        try:
            prompt = self.render(messages, template_kwargs)
        except (ValueError, TypeError) as exc:
            return 400, {"error": {"message": f"the chat template cannot render the request: {exc}"}}
        try:
            status, answer = self.generate({**request, "input_ids": prompt})
        except (OSError, ValueError) as exc:
            return 502, {"error": {"message": f"the rollout engines failed: {exc}"}}
        if status != 200:
            return status, {"error": {"message": f"the rollout engines answered {status}: {answer}"}}
        try:
            tokens, log_probs, finish = generated(answer)
            heads = recorded_heads(answer, len(tokens), self.topk_logprobs) if self.topk_logprobs else None
        except ValueError as exc:
            return 502, {"error": {"message": f"the rollout engines failed: {exc}"}}
        message = {"role": "assistant", "content": answer["text"]}
        weight_version = answer["meta_info"].get("weight_version")
        return 200, {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model", self.model_path),
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(tokens),
                "total_tokens": len(prompt) + len(tokens),
            },
            "training": {
                "tokens": [*prompt, *tokens],
                "loss_mask": [1] * len(tokens),
                "rollout_log_probs": log_probs,
                "prompt_length": len(prompt),
                "response_length": len(tokens),
                "runtime_load_id": self.runtime_load_id,
                "request_messages": messages,
                "response_message": message,
                "finish_reason": finish,
                **({} if weight_version is None else {"weight_version": str(weight_version)}),
                **({} if heads is None else {"topk_indices": heads[0], "topk_log_probs": heads[1]}),
            },
        }


def chat_request(payload: Any) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    """The messages, SGLang sampling parameters and chat template options of an OpenAI text chat request.

    Raise ``ValueError`` for a request outside text system, user and assistant messages, with another option,
    with ``n`` other than 1, or with a ``max_tokens`` that is not a positive int.
    """
    if not isinstance(payload, dict):
        raise ValueError("the request body must be a JSON object")
    unknown = payload.keys() - REQUEST_OPTIONS - SAMPLING_OPTIONS.keys()
    if unknown or payload.get("n", 1) != 1:
        raise ValueError(f"the rollout engines answer text chat only: no {sorted(unknown)}, and n must be 1")
    messages = payload.get("messages")
    if (
        not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in {"system", "user", "assistant"}
            or not isinstance(message["content"], str)
            for message in messages
        )
    ):
        raise ValueError("the rollout engines answer non-empty text system, user and assistant messages")
    maximum = payload.get("max_completion_tokens", payload.get("max_tokens", DEFAULT_MAX_TOKENS))
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise ValueError("max_tokens must be a positive int")
    sampling = {"max_new_tokens": maximum}
    sampling.update({name: payload[option] for option, name in SAMPLING_OPTIONS.items() if option in payload})
    template_kwargs = payload.get("chat_template_kwargs", {})
    if not isinstance(template_kwargs, dict):
        raise ValueError("chat_template_kwargs must be an object")
    return [dict(message) for message in messages], sampling, template_kwargs


def generated(answer: Any) -> tuple[list[int], list[float], str]:
    """The response token IDs, their finite sampled log-probs and the finish reason of an SGLang ``/generate`` answer.

    Raise ``ValueError`` for an answer without text, without one log-prob per token, or with a non-finite one.
    """
    if not isinstance(answer, dict) or not isinstance(answer.get("text"), str):
        raise ValueError("the answer has no text")
    meta = answer.get("meta_info")
    if not isinstance(meta, dict) or not isinstance(meta.get("output_token_logprobs"), list):
        raise ValueError("the answer has no output_token_logprobs")
    pairs = meta["output_token_logprobs"]
    tokens: list[int] = []
    log_probs: list[float] = []
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) < 2 or not isinstance(pair[1], int):
            raise ValueError("the answer has a malformed output_token_logprobs entry")
        if not isinstance(pair[0], int | float) or not math.isfinite(pair[0]):
            raise ValueError("the answer has a non-finite sampled log-prob")
        log_probs.append(float(pair[0]))
        tokens.append(pair[1])
    reason = meta.get("finish_reason")
    finish = "length" if isinstance(reason, dict) and reason.get("type") == "length" else "stop"
    return tokens, log_probs, finish


def recorded_heads(answer: dict[str, Any], length: int, top_k: int) -> tuple[list[list[int]], list[list[float]]]:
    """The top ``top_k`` token IDs and finite log-probs at each of the ``length`` response tokens of an answer.

    SGLang gives them in ``meta_info.output_top_logprobs`` as one list of ``[log-prob, token ID, ...]`` entries
    per response token. Raise ``ValueError`` for an answer without one list of ``top_k`` such entries per token.
    """
    rows = answer["meta_info"].get("output_top_logprobs")
    if not isinstance(rows, list) or len(rows) != length:
        raise ValueError(f"the answer has no output_top_logprobs for each of its {length} tokens")
    indices: list[list[int]] = []
    log_probs: list[list[float]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != top_k:
            raise ValueError(f"the answer has an output_top_logprobs entry without {top_k} log-probs")
        ids, values = [], []
        for entry in row:
            if (
                not isinstance(entry, list)
                or len(entry) < 2
                or not isinstance(entry[1], int)
                or not isinstance(entry[0], int | float)
                or not math.isfinite(entry[0])
            ):
                raise ValueError("the answer has a malformed output_top_logprobs entry")
            values.append(float(entry[0]))
            ids.append(entry[1])
        indices.append(ids)
        log_probs.append(values)
    return indices, log_probs
