"""vLLM's token-native ``/inference/v1/generate`` behind Reef's chat facades.

vLLM returns the sampled token ids and one log probability per token; the
per-token weight versions arrive in ``kv_transfer_params`` once Reef's
connector is selected on the engine. vLLM returns no decoded text on this
route, so the handler decodes the ids with its own tokenizer for the client.
Streaming is served from the buffered result because vLLM's stream frames
carry no ``kv_transfer_params``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.artifact.artifact import Artifact, is_local_release
from reef.inference.chat import (
    CapturedGeneration,
    NativeGenerateClient,
    TokenNativeChatHandler,
    finite_log_prob,
    integer_tokens,
    normalize_finish_reason,
    positive_max_tokens,
)

VLLM_GENERATE_PATH = "/inference/v1/generate"
#: The ``kv_transfer_params`` entry Reef's vLLM connector fills with one version per sampled token.
TOKEN_RUNTIME_LOAD_IDS_KEY = "reef_token_runtime_load_ids"
#: vLLM renders logprob tokens as ``token_id:N`` on this route.
TOKEN_ID_PREFIX = "token_id:"


class VLLMGenerateClient(NativeGenerateClient):
    """Request and response shapes of vLLM ``/inference/v1/generate``."""

    generate_path = VLLM_GENERATE_PATH
    error_label = "vLLM /inference/v1/generate"

    def sampling_params(self, request: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
        sampling: dict[str, Any] = {"max_tokens": positive_max_tokens(request, defaults, "max_tokens")}
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
            "seed",
        ):
            if key in request:
                sampling[key] = request[key]
        for key, value in defaults.items():
            if key != "max_tokens":
                sampling.setdefault(key, value)
        extra = request.get("vllm_sampling_params")
        if extra is not None:
            if not isinstance(extra, Mapping):
                raise ValueError("vllm_sampling_params must be an object")
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
        # ``logprobs: N`` returns the sampled token's log-prob plus the top N;
        # 0 keeps only the sampled token, which is all a training record needs.
        logprobs = 0
        top_logprobs = request.get("top_logprobs")
        if isinstance(top_logprobs, int) and not isinstance(top_logprobs, bool) and top_logprobs > 0:
            logprobs = top_logprobs
        logprobs = max(logprobs, capture_topk)
        native_payload: dict[str, Any] = {
            "token_ids": prompt_ids,
            "sampling_params": {**sampling_params, "logprobs": logprobs},
            "stream": stream,
        }
        # SGLang names an adapter through ``lora_path``; vLLM selects it as the model name.
        if "lora_path" in request:
            native_payload["model"] = request["lora_path"]
        if "rid" in request:
            native_payload["request_id"] = request["rid"]
        return native_payload

    def parse(self, artifact: Artifact, response: Mapping[str, Any], *, capture_topk: int) -> CapturedGeneration:
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
            raise ValueError("vLLM generate response must carry exactly one choice")
        choice = choices[0]
        finish_reason = normalize_finish_reason(choice.get("finish_reason"))
        raw_ids = choice.get("token_ids")
        if finish_reason == "abort" and not raw_ids:
            output_ids: list[int] = []
        else:
            output_ids = integer_tokens(raw_ids, "vLLM choice token_ids")
        logprobs = choice.get("logprobs")
        entries = logprobs.get("content") if isinstance(logprobs, Mapping) else None
        if not isinstance(entries, list) or len(entries) != len(output_ids):
            raise ValueError("vLLM generate response lacks one log probability per sampled token")
        rollout_log_probs = [
            finite_log_prob(
                entry.get("logprob") if isinstance(entry, Mapping) else None, f"vLLM logprob at index {index}"
            )
            for index, entry in enumerate(entries)
        ]
        stamped = self._token_runtime_load_ids(artifact, response, len(output_ids))
        topk_indices, topk_log_probs = self._captured_topk(entries, capture_topk)
        public_meta = {
            key: value
            for key, value in (
                ("request_id", response.get("request_id")),
                ("finish_reason", choice.get("finish_reason")),
                ("usage", response.get("usage")),
            )
            if value is not None
        }
        return CapturedGeneration(
            output_ids=output_ids,
            rollout_log_probs=rollout_log_probs,
            token_runtime_load_ids=stamped,
            finish_reason=finish_reason,
            text=None,
            public_meta=public_meta,
            topk_indices=topk_indices,
            topk_log_probs=topk_log_probs,
        )

    @staticmethod
    def _token_runtime_load_ids(artifact: Artifact, response: Mapping[str, Any], count: int) -> list[str]:
        params = response.get("kv_transfer_params")
        stamped = params.get(TOKEN_RUNTIME_LOAD_IDS_KEY) if isinstance(params, Mapping) else None
        if stamped is None:
            if not is_local_release(artifact.ref.release_id):
                raise ValueError(
                    "vLLM generate response lacks per-token runtime load IDs; "
                    "select Reef's version connector in the engine's --kv-transfer-config"
                )
            # A local release serves one fixed version, so every token carries it.
            version = getattr(artifact.ref, "runtime_load_id", None) or artifact.ref.release_id
            return [str(version)] * count
        if (
            not isinstance(stamped, list)
            or len(stamped) != count
            or any(not isinstance(version, str) or not version for version in stamped)
        ):
            raise ValueError("vLLM generate response has incomplete token runtime load IDs")
        return list(stamped)

    @staticmethod
    def _captured_topk(
        entries: list[Any], capture_topk: int
    ) -> tuple[list[list[int]] | None, list[list[float]] | None]:
        """Top-K capture from vLLM's per-token ``top_logprobs``; partial rows disable the channel."""
        if capture_topk <= 0:
            return None, None
        indices: list[list[int]] = []
        log_probs: list[list[float]] = []
        for entry in entries:
            candidates = entry.get("top_logprobs") if isinstance(entry, Mapping) else None
            if not isinstance(candidates, list) or len(candidates) < capture_topk:
                return None, None
            row_idx: list[int] = []
            row_lp: list[float] = []
            for candidate in candidates[:capture_topk]:
                token = candidate.get("token") if isinstance(candidate, Mapping) else None
                if not isinstance(token, str) or not token.startswith(TOKEN_ID_PREFIX):
                    return None, None
                row_idx.append(int(token[len(TOKEN_ID_PREFIX) :]))
                row_lp.append(finite_log_prob(candidate.get("logprob"), "vLLM top_logprobs entry"))
            indices.append(row_idx)
            log_probs.append(row_lp)
        return indices, log_probs

    def tool_parser(self, tools: list[dict[str, Any]], parser_name: str, tokenizer: Any) -> Any:
        """Build the parser vLLM registers under ``--tool-call-parser``'s name, for buffered parsing."""
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.tool_parsers import ToolParserManager

        parser = ToolParserManager.get_tool_parser(parser_name)(tokenizer)
        request = ChatCompletionRequest(model="reef", messages=[{"role": "user", "content": ""}], tools=tools)
        return VLLMToolCallParser(parser, request)


@dataclass(frozen=True)
class ParsedToolCall:
    """One tool call in the shape the chat facade reads: a name and its JSON argument text."""

    name: str
    parameters: str


class VLLMToolCallParser:
    """Adapt vLLM's ``ToolParser`` to the buffered parsing interface the chat facade uses."""

    def __init__(self, parser: Any, request: Any) -> None:
        self._parser = parser
        self._request = request
        self._last: tuple[str, Any] | None = None

    def _extract(self, text: str) -> Any:
        if self._last is None or self._last[0] != text:
            self._last = (text, self._parser.extract_tool_calls(text, self._request))
        return self._last[1]

    def has_tool_call(self, text: str) -> bool:
        return bool(self._extract(text).tools_called)

    def parse_non_stream(self, text: str) -> tuple[str, list[ParsedToolCall]]:
        extracted = self._extract(text)
        calls = [
            ParsedToolCall(name=str(call.function.name), parameters=str(call.function.arguments or "{}"))
            for call in extracted.tool_calls
        ]
        return extracted.content or "", calls


class VLLMInferenceHandler(TokenNativeChatHandler):
    """Serve OpenAI or Anthropic chat over vLLM with buffered exact capture."""

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
            client=VLLMGenerateClient() if client is None else client,
            model_path=model_path,
            timeout_s=timeout_s,
            tokenizer=tokenizer,
            tool_call_parser=tool_call_parser,
            capture_topk=capture_topk,
            sampling_defaults=sampling_defaults,
            force_reasoning=force_reasoning,
        )


__all__ = [
    "TOKEN_RUNTIME_LOAD_IDS_KEY",
    "VLLM_GENERATE_PATH",
    "VLLMGenerateClient",
    "VLLMInferenceHandler",
    "VLLMToolCallParser",
]
