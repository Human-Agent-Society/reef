"""Sampling on Tinker: the SDK sampler and the text chat request and response that Tinker serves.

The serving runtime ``reef.inference.tinker`` and the training component's rollouts
(``reef.train.tinker_backend.rollout``) both sample through these. ``chat_request`` validates an
OpenAI text chat request into messages and sampling parameters, ``TinkerSampler`` renders and samples,
and ``chat_completion`` shapes the answer with the training capture that Reef records: the token IDs,
the loss mask, the sampled log probabilities and, when asked, the sampler's top-k at each sampled token.

A request that names no reply limit may reply up to the rest of the model's context window, as on an
OpenAI-compatible server, or ``DEFAULT_MAX_TOKENS`` when the sampler does not know the window. ``sample_chat``
splits each reply into its reasoning and its answer as the token-native chat handlers do
(``reef.core.reasoning``), so the answer's ``reasoning_content`` carries the reasoning of a thinking model.
"""

from __future__ import annotations

import math
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reef.core.errors import UpstreamStatusError
from reef.core.reasoning import PROMPT_TAIL_TOKEN_COUNT, is_reasoning_pre_opened, split_reasoning
from reef.tinker.service import close_service, served_context_windows, tinker_sdk

#: The reply length of a request that names none, when the sampler does not know the model's context window.
DEFAULT_MAX_TOKENS = 1024
#: The OpenAI code and wording of a context overflow, which agents match to compact their history.
CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded: the prompt exceeds this model's maximum context length."


@dataclass(frozen=True)
class SampleResult:
    """One sampled continuation.

    Values: ``tokens`` are the sampled token IDs and ``logprobs`` the sampler's log probability of each, finite
    and as many as the tokens; ``stop_reason`` is the sampler's non-empty reason to stop, such as ``stop`` or
    ``length``. ``topk`` is empty, or holds one row per sampled token of the same width ``k >= 1``: the ``k``
    most likely ``(token ID, log probability)`` pairs at that position, with distinct non-negative IDs and finite
    log probabilities. Purpose: ``TinkerSampler.sample`` makes it; ``chat_completion`` shapes the answer and its
    training capture from it.
    """

    tokens: tuple[int, ...]
    logprobs: tuple[float, ...]
    stop_reason: str
    topk: tuple[tuple[tuple[int, float], ...], ...] = ()

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.logprobs):
            raise ValueError("Tinker must return exact log probabilities for every sampled token")
        if any(not math.isfinite(value) for value in self.logprobs):
            raise ValueError("Tinker returned non-finite sampled log probabilities")
        if not isinstance(self.stop_reason, str) or not self.stop_reason:
            raise ValueError("a sample needs the sampler's stop reason")
        if not self.topk:
            return
        widths = {len(row) for row in self.topk}
        if len(self.topk) != len(self.tokens) or len(widths) != 1 or 0 in widths:
            raise ValueError("Tinker must return top-k log probabilities of one width for every sampled token")
        for row in self.topk:
            ids = [token for token, _ in row]
            if len(set(ids)) != len(ids) or any(token < 0 for token in ids):
                raise ValueError("Tinker returned top-k token IDs that are not distinct and non-negative")
            if any(not math.isfinite(logprob) for _, logprob in row):
                raise ValueError("Tinker returned non-finite top-k log probabilities")


class TinkerSampler(ABC):
    """Remote sampling operations, also implementable by offline tests.

    ``sampler_path`` is the immutable remote sampler an artifact names, or
    None for the base model itself.
    """

    @abstractmethod
    def render(self, messages: list[dict[str, str]], *, template_kwargs: Mapping[str, Any]) -> list[int]: ...

    @abstractmethod
    def decode(self, tokens: Sequence[int], *, skip_special_tokens: bool = True) -> str: ...

    @abstractmethod
    def sample(
        self, sampler_path: str | None, prompt: list[int], params: Mapping[str, Any], *, topk_logprobs: int
    ) -> SampleResult:
        """One continuation of ``prompt`` with the SDK's ``SamplingParams`` ``params``; a positive ``topk_logprobs``
        also returns that many top log probabilities at each sampled token (``SampleResult.topk``)."""

    @abstractmethod
    def close(self) -> None: ...

    def context_window(self) -> int | None:
        """The model's context window in tokens, prompt and reply together, or None when the sampler does not know
        it."""
        return None


#: Part of the message of Tinker's refusal on a sampling client that it poisoned, for example after the client sat idle
#: too long. Only a new client samples again.
POISONED_CLIENT = "SamplingClient is poisoned"


class TinkerSDKSampler(TinkerSampler):
    """The SDK boundary for sampling; every request binds an immutable sampler path."""

    _sdk: Any
    _model: str
    _timeout_s: float
    _service: Any
    #: The base model's sampling client and its tokenizer, made at the first use.
    base_client: Any | None
    tokenizer: Any | None
    #: The sampling client of each sampler path, made at its first request.
    clients: dict[str, Any]
    #: The context window of each model that the service serves, read at the first need.
    context_windows: dict[str, int]

    def __init__(
        self,
        base_model: str,
        *,
        api_key: str,
        project_id: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        """Sample ``base_model`` on the Tinker-compatible service at ``base_url`` (``None``: hosted Tinker)."""
        self._sdk = tinker_sdk()
        self._model = base_model
        self._timeout_s = timeout_s
        # The SDK takes no client-wide timeout: each call is bounded with .result(timeout=...).
        self._service = self._sdk.ServiceClient(api_key=api_key, project_id=project_id, base_url=base_url)
        # On a service such as Spindle, a sampling client starts a sampler pool, which a component that only
        # trains never needs; and a rollout samples one checkpoint many times, so each client is kept.
        self.base_client = None
        self.tokenizer = None
        self.clients = {}
        self.context_windows = {}

    def base_sampler(self) -> Any:
        """The base model's sampling client, made at the first call."""
        if self.base_client is None:
            self.base_client = self._service.create_sampling_client(base_model=self._model)
        return self.base_client

    def base_tokenizer(self) -> Any:
        """The base model's tokenizer, from its sampling client at the first call."""
        if self.tokenizer is None:
            self.tokenizer = self.base_sampler().get_tokenizer()
        return self.tokenizer

    def render(self, messages: list[dict[str, str]], *, template_kwargs: Mapping[str, Any]) -> list[int]:
        prefill = messages[-1]["role"] == "assistant"
        return list(
            self.base_tokenizer().apply_chat_template(
                messages,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=not prefill,
                continue_final_message=prefill,
                **template_kwargs,
            )
        )

    def decode(self, tokens: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        return str(self.base_tokenizer().decode(list(tokens), skip_special_tokens=skip_special_tokens))

    def context_window(self) -> int | None:
        if not self.context_windows:
            self.context_windows = served_context_windows(self._service)
        return self.context_windows.get(self._model)

    def sampling_client(self, sampler_path: str | None) -> Any:
        """The sampling client of ``sampler_path``, or of the base model for None, made at its first use."""
        if sampler_path is None:
            return self.base_sampler()
        if sampler_path not in self.clients:
            sampler = self._service.create_sampling_client(model_path=sampler_path)
            if sampler.get_base_model() != self._model:
                raise ValueError("remote Tinker sampler checkpoint does not match the configured model")
            self.clients[sampler_path] = sampler
        return self.clients[sampler_path]

    def forget_client(self, sampler_path: str | None) -> None:
        """Drop the kept sampling client of ``sampler_path``, so that the next use makes a new one."""
        if sampler_path is None:
            self.base_client = None
        else:
            self.clients.pop(sampler_path, None)

    def sample(
        self, sampler_path: str | None, prompt: list[int], params: Mapping[str, Any], *, topk_logprobs: int
    ) -> SampleResult:
        for attempt in range(2):
            sampler = self.sampling_client(sampler_path)
            try:
                result = sampler.sample(
                    prompt=self._sdk.ModelInput.from_ints(prompt),
                    num_samples=1,
                    sampling_params=self._sdk.SamplingParams(**params),
                    topk_sample_logprobs=topk_logprobs,
                ).result(timeout=self._timeout_s)
                break
            except self._sdk.BadRequestError as exc:
                message = str(exc)
                if POISONED_CLIENT in message and attempt == 0:
                    # Tinker refuses every later call on a client that it poisoned, such as one left idle too long.
                    self.forget_client(sampler_path)
                    continue
                if "context window" in message:
                    message = f"{CONTEXT_LENGTH_EXCEEDED} {message}"
                raise UpstreamStatusError(message, status=400) from exc
        if len(result.sequences) != 1:
            raise ValueError("Tinker returned an unexpected number of sequences")
        sequence = result.sequences[0]
        if sequence.logprobs is None:
            raise ValueError("Tinker must return exact log probabilities for every sampled token")
        topk: tuple[tuple[tuple[int, float], ...], ...] = ()
        if topk_logprobs:
            rows = sequence.topk_logprobs
            if rows is None or any(row is None for row in rows):
                raise ValueError("Tinker returned no top-k log probabilities for every sampled token")
            topk = tuple(tuple((int(token), float(logprob)) for token, logprob in row) for row in rows)
        return SampleResult(tuple(sequence.tokens), tuple(sequence.logprobs), sequence.stop_reason, topk)

    def close(self) -> None:
        close_service(self._sdk, self._service, "success", timeout_s=self._timeout_s)


def chat_completion(
    model: str,
    messages: list[dict[str, str]],
    prompt: list[int],
    result: SampleResult,
    content: str,
    runtime_load_id: str,
    reasoning: str = "",
) -> dict[str, Any]:
    """An OpenAI chat completion plus the private training capture Reef records; a non-empty ``reasoning`` rides the
    message's ``reasoning_content``, as an OpenAI-compatible server with a reasoning parser returns it."""
    message = {"role": "assistant", "content": content, **({"reasoning_content": reasoning} if reasoning else {})}
    finish = "length" if result.stop_reason == "length" else "stop"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
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
            "runtime_load_id": runtime_load_id,
            "request_messages": messages,
            "response_message": message,
            "finish_reason": finish,
            **(
                {
                    "topk_indices": [[token for token, _ in row] for row in result.topk],
                    "topk_log_probs": [[logprob for _, logprob in row] for row in result.topk],
                }
                if result.topk
                else {}
            ),
        },
    }


def chat_request(payload: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Validate a text chat request into its messages and Tinker sampling parameters."""
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
    maximum = payload.get("max_completion_tokens", payload.get("max_tokens"))
    if maximum is not None:
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            raise UpstreamStatusError("Tinker max_tokens must be a positive integer", status=400)
        params["max_tokens"] = maximum
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


def sample_chat(
    sampler: TinkerSampler,
    sampler_path: str | None,
    payload: dict[str, Any],
    *,
    model: str,
    runtime_load_id: str,
    topk_logprobs: int = 0,
) -> dict[str, Any]:
    """Answer one OpenAI text chat request from ``sampler_path`` (None: the base model), with its training capture.

    A positive ``topk_logprobs`` records the sampler's top ``topk_logprobs`` log probabilities at each sampled
    token in the capture (``topk_indices``, ``topk_log_probs``). Those must be the log probabilities that the
    trainer reads, of the untruncated distribution at temperature 1, so such a request may set ``temperature``
    only to 1, ``top_p`` only to 1 and ``top_k`` only to -1. A request that names no reply limit may reply up to
    the rest of the sampler's context window (``TinkerSampler.context_window``), or ``DEFAULT_MAX_TOKENS`` tokens
    when the sampler does not know it. Raise ``UpstreamStatusError`` (400) for a request that Tinker cannot serve or
    that sets another value, and with ``CONTEXT_LENGTH_EXCEEDED`` for such a request whose prompt fills the window.
    """
    messages, params = chat_request(payload)
    if topk_logprobs and (
        params.get("temperature", 1.0) != 1 or params.get("top_p", 1.0) != 1 or params.get("top_k", -1) != -1
    ):
        raise UpstreamStatusError(
            "this sampler records the top-k log probabilities of the untruncated distribution at temperature 1: "
            "send temperature 1, top_p 1 and top_k -1, or leave them unset",
            status=400,
        )
    prompt = sampler.render(messages, template_kwargs=payload.get("chat_template_kwargs") or {})
    if not prompt:
        raise ValueError("Tinker chat template produced an empty prompt")
    if "max_tokens" not in params:
        # Like an OpenAI-compatible server, a request without a limit may use the rest of the context window.
        window = sampler.context_window()
        if window is None:
            params["max_tokens"] = DEFAULT_MAX_TOKENS
        elif window <= len(prompt):
            raise UpstreamStatusError(
                f"{CONTEXT_LENGTH_EXCEEDED} {len(prompt)} prompt tokens fill the context window of {window}",
                status=400,
            )
        else:
            params["max_tokens"] = window - len(prompt)
    result = sampler.sample(sampler_path, prompt, params, topk_logprobs=topk_logprobs)
    pre_opened = is_reasoning_pre_opened(sampler.decode(prompt[-PROMPT_TAIL_TOKEN_COUNT:], skip_special_tokens=False))
    reasoning, content = split_reasoning(sampler.decode(result.tokens), pre_opened=pre_opened)
    return chat_completion(payload.get("model", model), messages, prompt, result, content, runtime_load_id, reasoning)
