"""Slime's own rollout group: the SGLang engines that Slime's ``RolloutManager`` starts, for the component's task jobs.

The engines belong to the training component, not to Reef's serving: they answer only the model calls of a task
job's rollouts (``reef.train.task_rollouts``), at the weights that the job trains. Slime starts, syncs and
offloads them as in its own training loop. With ``--colocate`` they share the training GPUs, and Slime moves
the GPU memory between the engines and the training workers; otherwise they run on GPUs of their own.
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

import ray

from reef.runtime.executor.placement import GpuBundles
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments
from reef.train.slime_backend.reef_adapters.train_groups import SlimeTrainGroup
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
#: The variable that loads Reef's SGLang plugin into an engine (``reef.inference.sglang.plugin``). The image's
#: SGLang takes a weight update only inside a session that Slime does not open; the plugin opens it.
SGLANG_PLUGIN_ENV = "SGLANG_REEF_PLUGIN"


def router_address(manager: Any) -> tuple[str, int]:
    """The address of the router that the rollout manager started, read inside the manager's process.

    Runs in the manager's Ray actor (``__ray_call__``), whose arguments hold the address it chose.
    """
    return manager.args.sglang_router_ip, manager.args.sglang_router_port


class SlimeRolloutGroup:
    """The rollout engines of one Slime training component, behind Slime's ``RolloutManager``.

    Values: ``manager`` is the Ray handle of the ``RolloutManager`` actor, which owns the engines and their
    router. ``colocated`` is True when the engines share the training GPUs, so they hold GPU memory only
    between ``sync`` and ``release``. ``router_url`` is the non-empty base URL of the engines' router.
    Purpose: ``start`` makes it from the workers' arguments and GPU bundles; the component syncs the engines to
    the actor's weights before a task job's rollouts and releases them before it trains.
    """

    def __init__(self, manager: Any, *, colocated: bool, router_url: str) -> None:
        if not router_url:
            raise ValueError("SlimeRolloutGroup.router_url must be non-empty")
        self.manager = manager
        self.colocated = colocated
        self.router_url = router_url

    @classmethod
    def start(cls, args: SlimeArguments, bundles: GpuBundles) -> SlimeRolloutGroup:
        """Start Slime's rollout manager and its engines on ``bundles``; colocated engines start offloaded."""
        from slime.ray.placement_group import create_rollout_manager

        manager, _ = create_rollout_manager(args, bundles)
        host, port = ray.get(manager.__ray_call__.remote(router_address))
        return cls(manager, colocated=bool(args.colocate), router_url=f"http://{host}:{port}")

    def sync(self, actor_group: SlimeTrainGroup) -> None:
        """Load the actor's weights into the engines, which then hold GPU memory and serve."""
        if self.colocated:
            ray.get(self.manager.onload_weights.remote())
        actor_group.update_weights()
        if self.colocated:
            ray.get(self.manager.onload_kv.remote())

    def release(self) -> None:
        """Free the colocated engines' GPU memory for training; engines on their own GPUs keep it."""
        if self.colocated:
            ray.get(self.manager.offload.remote())

    def close(self) -> None:
        """Stop the manager's monitors; the deployment's process cleanup then stops the manager and its engines."""
        ray.get(self.manager.dispose.remote())


class EngineChat(RolloutChat):
    """Answer a task job's OpenAI text chat calls on the rollout group's engines, with the training capture.

    ``router_url`` is the engines' router. ``model_path`` is the model directory whose tokenizer and chat template
    render each request; the tokenizer loads at the first call. ``timeout_s`` is a positive bound on one call,
    in seconds. Each answer is a chat completion whose ``training`` block holds the prompt and response token
    IDs, the response's loss mask and sampled log-probs, and ``runtime_load_id``, the weights the engines hold
    for the job. Another path gets 404, a request outside text system, user and assistant messages (or with
    ``n`` other than 1) 400, an engine error status its own status, and an unusable engine answer 502.
    """

    def __init__(self, router_url: str, *, model_path: str, timeout_s: float, runtime_load_id: str) -> None:
        if not router_url or not model_path or not runtime_load_id:
            raise ValueError("EngineChat needs a router URL, a model path and a runtime load ID")
        if not timeout_s > 0:
            raise ValueError("EngineChat.timeout_s must be positive")
        self.router_url = router_url.rstrip("/")
        self.model_path = model_path
        self.timeout_s = timeout_s
        self.runtime_load_id = runtime_load_id
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
        try:
            prompt = self.render(messages, template_kwargs)
        except (ValueError, TypeError) as exc:
            return 400, {"error": {"message": f"the chat template cannot render the request: {exc}"}}
        try:
            status, answer = self.generate({"input_ids": prompt, "sampling_params": sampling, "return_logprob": True})
        except (OSError, ValueError) as exc:
            return 502, {"error": {"message": f"the rollout engines failed: {exc}"}}
        if status != 200:
            return status, {"error": {"message": f"the rollout engines answered {status}: {answer}"}}
        try:
            tokens, log_probs, finish = generated(answer)
        except ValueError as exc:
            return 502, {"error": {"message": f"the rollout engines failed: {exc}"}}
        message = {"role": "assistant", "content": answer["text"]}
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
