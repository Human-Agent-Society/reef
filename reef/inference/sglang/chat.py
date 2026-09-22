"""SGLang's token-native ``/generate`` behind Reef's chat facades.

The shared OpenAI and Anthropic handling lives in :mod:`reef.inference.chat`.
This module shapes SGLang's request, reads the exact sample out of its
``meta_info`` (sampled ids beside log-probs, and the per-token weight versions
Reef's scheduler plugin stamps), and relays SGLang's incremental stream so
clients see tokens as they are produced.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from reef.artifact.artifact import Artifact, is_local_release
from reef.inference.chat import (
    CapturedGeneration,
    ChatCall,
    NativeGenerateClient,
    TokenNativeChatHandler,
    _ChatStreamRelay,
    _ReasoningStreamSplitter,
    _sse_json_events,
    _stream_ids,
    finite_log_prob,
    normalize_finish_reason,
    positive_max_tokens,
)
from reef.inference.http import HttpInferenceHandler
from reef.runtime.interfaces import InferenceStream

SGLANG_GENERATE_PATH = "/generate"
#: ``meta_info`` entries that hold per-token tensors; they feed the training block, never a client.
PRIVATE_META_KEYS = frozenset({"_reef_token_runtime_load_ids", "input_token_logprobs", "output_token_logprobs"})


def output_tensors(meta: Mapping[str, Any]) -> tuple[list[int], list[float]]:
    """Split SGLang's ``output_token_logprobs`` pairs into sampled ids and their log-probs."""
    pairs = meta.get("output_token_logprobs")
    finish_type = normalize_finish_reason(meta.get("finish_reason"))
    if finish_type == "abort" and not pairs:
        return [], []
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("SGLang response is missing non-empty meta_info.output_token_logprobs")
    output_ids: list[int] = []
    log_probs: list[float] = []
    for index, pair in enumerate(pairs):
        if (
            not isinstance(pair, Sequence)
            or isinstance(pair, (str, bytes))
            or len(pair) < 2
            or not isinstance(pair[1], int)
            or isinstance(pair[1], bool)
        ):
            raise ValueError(f"invalid SGLang output_token_logprobs entry at index {index}")
        log_probs.append(finite_log_prob(pair[0], f"SGLang output_token_logprobs entry {index}"))
        output_ids.append(pair[1])
    return output_ids, log_probs


class SGLangGenerateClient(NativeGenerateClient):
    """Request and response shapes of SGLang ``/generate``."""

    generate_path = SGLANG_GENERATE_PATH
    error_label = "SGLang /generate"

    def sampling_params(self, request: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
        sampling: dict[str, Any] = {"max_new_tokens": positive_max_tokens(request, defaults, "max_new_tokens")}
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "stop",
            "stop_token_ids",
            "ignore_eos",
            "skip_special_tokens",
            "logit_bias",
        ):
            if key in request:
                sampling[key] = request[key]
        if "seed" in request:
            sampling["sampling_seed"] = request["seed"]
        for key, value in defaults.items():
            if key != "max_new_tokens":
                sampling.setdefault(key, value)
        extra = request.get("sglang_sampling_params")
        if extra is not None:
            if not isinstance(extra, Mapping):
                raise ValueError("sglang_sampling_params must be an object")
            sampling.update(extra)
        return sampling

    def payload(
        self,
        request: Mapping[str, Any],
        prompt_ids: list[int],
        sampling_params: Mapping[str, Any],
        *,
        capture_topk: int,
        stream: bool,
    ) -> dict[str, Any]:
        native_payload: dict[str, Any] = {
            "input_ids": prompt_ids,
            "sampling_params": dict(sampling_params),
            "return_logprob": True,
            "stream": stream,
        }
        top_logprobs = request.get("top_logprobs")
        if isinstance(top_logprobs, int) and not isinstance(top_logprobs, bool) and top_logprobs >= 0:
            native_payload["top_logprobs_num"] = top_logprobs
        if capture_topk > 0:
            native_payload["top_logprobs_num"] = max(capture_topk, int(native_payload.get("top_logprobs_num") or 0))
        for key in ("lora_path", "rid"):
            if key in request:
                native_payload[key] = request[key]
        return native_payload

    def parse(self, artifact: Artifact, response: Mapping[str, Any], *, capture_topk: int) -> CapturedGeneration:
        normalized = normalize_native_response(artifact, dict(response))
        meta = normalized["meta_info"]
        text = normalized.get("text")
        if not isinstance(text, str):
            raise ValueError("SGLang response is missing generated text")
        output_ids, rollout_log_probs = output_tensors(meta)
        stamped = normalized["_reef_token_runtime_load_ids"]
        if len(stamped) != len(output_ids):
            raise ValueError("SGLang response has incomplete token runtime load IDs")
        topk = captured_topk(meta, len(output_ids), capture_topk)
        return CapturedGeneration(
            output_ids=output_ids,
            rollout_log_probs=rollout_log_probs,
            token_runtime_load_ids=list(stamped),
            finish_reason=normalize_finish_reason(meta.get("finish_reason")),
            text=text,
            public_meta={key: value for key, value in meta.items() if key not in PRIVATE_META_KEYS},
            topk_indices=topk[0],
            topk_log_probs=topk[1],
        )

    def tool_parser(self, tools: list[dict[str, Any]], parser_name: str, tokenizer: Any) -> Any:
        """Load the parser shipped with the selected SGLang runtime."""
        from sglang.srt.entrypoints.openai.protocol import Tool
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        return FunctionCallParser([Tool.model_validate(tool) for tool in tools], parser_name)


def normalize_native_response(artifact: Artifact, response: dict[str, Any]) -> dict[str, Any]:
    """Require scheduler-stamped per-token versions, or SGLang's engine version for a local release."""
    exact_runtime_load_ids_required = not is_local_release(artifact.ref.release_id)
    meta = response.get("meta_info")
    if not isinstance(meta, Mapping):
        raise ValueError("SGLang /generate response lacks meta_info")
    pairs = meta.get("output_token_logprobs")
    if not isinstance(pairs, list):
        raise ValueError("SGLang /generate response lacks output token log probabilities")
    stamped = meta.get("_reef_token_runtime_load_ids")
    if stamped is None and not exact_runtime_load_ids_required:
        # Raw SGLang meta_info carries the runtime-load-ID under SGLang's
        # own field name; the normalized meta below re-keys it for Reef.
        version = meta.get("weight_version")
        stamped = [version] * len(pairs)
    if (
        not isinstance(stamped, list)
        or len(stamped) != len(pairs)
        or any(not isinstance(version, str) or not version for version in stamped)
    ):
        raise ValueError("SGLang /generate response lacks scheduler-stamped token runtime load IDs")
    normalized_meta = dict(meta)
    if stamped:
        normalized_meta["runtime_load_id"] = stamped[-1]
    response["meta_info"] = normalized_meta
    response["_reef_token_runtime_load_ids"] = stamped
    return response


def captured_topk(
    meta: Mapping[str, Any], response_length: int, capture_topk: int
) -> tuple[list[list[int]] | None, list[list[float]] | None]:
    """Generation-time top-K capture for the OpenClaw-RL top-K objective.

    SGLang's ``output_top_logprobs`` lists, per generated position, the
    top-N ``[logprob, token_id, ...]`` entries. Rows are truncated or
    absent when the engine cannot provide them; a partial capture
    disables the channel for the whole response (all-or-nothing keeps
    the training contract rectangular).
    """
    if capture_topk <= 0:
        return None, None
    entries = meta.get("output_top_logprobs")
    if not isinstance(entries, list) or len(entries) < response_length:
        return None, None
    indices: list[list[int]] = []
    log_probs: list[list[float]] = []
    for position in entries[:response_length]:
        if not isinstance(position, list) or len(position) < capture_topk:
            return None, None
        row_idx: list[int] = []
        row_lp: list[float] = []
        for entry in position[:capture_topk]:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2 or entry[0] is None:
                return None, None
            row_lp.append(float(entry[0]))
            row_idx.append(int(entry[1]))
        indices.append(row_idx)
        log_probs.append(row_lp)
    return indices, log_probs


class _NativeStreamCapture:
    """Accumulate Reef's required disjoint SGLang stream events."""

    _LIST_META_KEYS = (
        "output_token_logprobs",
        "output_top_logprobs",
        "output_token_ids_logprobs",
        "output_token_sampling_mask",
        "output_token_sampling_logprobs",
        "_reef_token_runtime_load_ids",
    )

    def __init__(self) -> None:
        self.text = ""
        self.output_ids: list[int] = []
        self.meta: dict[str, Any] = {}
        self.last_output_logprobs: list[Any] = []
        self.seen = False

    def accept(self, event: Mapping[str, Any]) -> str:
        meta = event.get("meta_info")
        if not isinstance(meta, Mapping):
            raise ValueError("SGLang stream event lacks meta_info")
        self.seen = True
        self.last_output_logprobs = []
        total = meta.get("completion_tokens")
        pairs = meta.get("output_token_logprobs")
        if not isinstance(pairs, list):
            raise ValueError("incremental SGLang stream event lacks output_token_logprobs")
        if (
            not isinstance(total, int)
            or isinstance(total, bool)
            or total != len(self.meta.get("output_token_logprobs", [])) + len(pairs)
        ):
            raise ValueError(
                "Reef requires SGLang --incremental-streaming-output; received a cumulative or malformed event"
            )

        raw_ids = event.get("output_ids")
        if isinstance(raw_ids, list):
            ids = [int(token_id) for token_id in raw_ids]
            if len(ids) != len(pairs):
                raise ValueError("incremental SGLang stream event has inconsistent output ids and log-probs")
            self.output_ids.extend(ids)

        for key, value in meta.items():
            if key in self._LIST_META_KEYS and isinstance(value, list):
                accumulated = self.meta.setdefault(key, [])
                if not isinstance(accumulated, list):
                    raise ValueError(f"incremental SGLang metadata {key} changed type")
                accumulated.extend(value)
                if key == "output_token_logprobs":
                    self.last_output_logprobs = list(value)
            else:
                self.meta[key] = value

        current_text = event.get("text")
        if current_text is None:
            return ""
        if not isinstance(current_text, str):
            raise ValueError("SGLang stream event text must be a string")
        self.text += current_text
        return current_text

    def response(self) -> dict[str, Any]:
        if not self.seen:
            raise ValueError("SGLang stream completed without a generation event")
        response: dict[str, Any] = {"text": self.text, "meta_info": dict(self.meta)}
        versions = self.meta.get("_reef_token_runtime_load_ids")
        if versions is not None:
            response["_reef_token_runtime_load_ids"] = versions
        return response


class SGLangInferenceHandler(TokenNativeChatHandler):
    """Serve OpenAI or Anthropic chat over SGLang, streaming its incremental output."""

    def __init__(
        self,
        upstream_url: str,
        *,
        model_path: str,
        timeout_s: float = 300.0,
        tokenizer: Any = None,
        tool_call_parser: str | None = None,
        capture_topk: int = 0,
        sampling_defaults: Mapping[str, Any] | None = None,
        force_reasoning: bool | None = None,
        client: NativeGenerateClient | None = None,
    ) -> None:
        super().__init__(
            upstream_url,
            client=SGLangGenerateClient() if client is None else client,
            model_path=model_path,
            timeout_s=timeout_s,
            tokenizer=tokenizer,
            tool_call_parser=tool_call_parser,
            capture_topk=capture_topk,
            sampling_defaults=sampling_defaults,
            force_reasoning=force_reasoning,
        )

    async def inference_stream(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> InferenceStream:
        self._validate_stream_path(path)
        call = self._chat_call(path, payload)
        # Force the scheduler to flush every available decode step. This is a
        # request-local knob; deployments using SGLang's disjoint incremental
        # output mode are supported by _NativeStreamCapture as well.
        call.sampling_params.setdefault("stream_interval", 1)
        native_payload = self.client.payload(
            call.request, call.prompt_ids, call.sampling_params, capture_topk=self.capture_topk, stream=True
        )
        upstream = await HttpInferenceHandler.inference_stream(self, artifact, SGLANG_GENERATE_PATH, native_payload)
        chat_id, message_id, created = _stream_ids()
        relay = _ChatStreamRelay(self._stream_writer(call, chat_id, message_id, created), call.tool_parser)

        async def chunks() -> AsyncIterator[bytes]:
            # ``stream`` is bound below, before the first chunk is pulled; the
            # relay records the completed response on it.
            async for frame in self._relay_stream(
                stream, upstream, relay, artifact, call, chat_id, message_id, created
            ):
                yield frame

        stream = InferenceStream(
            status=200,
            headers={"Content-Type": "text/event-stream; charset=utf-8"},
            chunks=chunks(),
            close=upstream.close,
            record_response_pending=True,
        )
        return stream

    async def _relay_stream(
        self,
        stream: InferenceStream,
        upstream: InferenceStream,
        relay: _ChatStreamRelay,
        artifact: Artifact,
        call: ChatCall,
        chat_id: str,
        message_id: str,
        created: int,
    ) -> AsyncIterator[bytes]:
        """Forward SGLang's incremental events as client frames, then record the exact sample."""
        capture = _NativeStreamCapture()
        reasoning = _ReasoningStreamSplitter(
            enabled=self.SPLIT_REASONING, force_reasoning=self.reasoning_is_pre_opened()
        )
        wants_logprobs = not call.anthropic and call.request.get("logprobs") is True
        try:
            for frame in relay.writer.start():
                yield frame
            async for native_event in _sse_json_events(upstream.chunks):
                for kind, value in reasoning.feed(capture.accept(native_event)):
                    for frame in relay.piece(kind, value):
                        yield frame
                if wants_logprobs and capture.last_output_logprobs:
                    token_ids, log_probs = output_tensors({"output_token_logprobs": capture.last_output_logprobs})
                    for frame in relay.writer.logprobs(self._openai_logprobs(token_ids, log_probs)):
                        yield frame
            for kind, value in reasoning.finish():
                for frame in relay.piece(kind, value):
                    yield frame

            captured = self.client.parse(artifact, capture.response(), capture_topk=self.capture_topk)
            # Streaming parsers are stateful. Parse the complete sample with a
            # fresh instance for the canonical training response.
            response = self._response_from_capture(call, captured, self._configured_tool_parser(call.request))
            provider_response, frames = self._finish_frames(call, relay, response, chat_id, message_id, created)
            stream.record_response = provider_response
            for frame in frames:
                yield frame
        finally:
            await upstream.close()


__all__ = ["SGLangGenerateClient", "SGLangInferenceHandler"]
