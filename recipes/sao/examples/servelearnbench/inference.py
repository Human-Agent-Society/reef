"""Append observations to exact Qwen3 samples instead of re-rendering old actions."""

import asyncio
from copy import deepcopy
from dataclasses import dataclass, replace

from reef.artifact import Artifact
from reef.inference.sglang.chat import SGLangInferenceHandler
from reef.runtime.interfaces import InferenceStream


@dataclass(frozen=True)
class EpisodePrefix:
    """One completed call's prefix for the next call of a serial episode.

    episode_id and runtime_load_id are nonempty strings. tokens contains the
    exact nonempty prompt plus sampled token IDs. request is the validated
    chat request that produced those tokens. Only the inference handler
    creates and reads this state; a new episode replaces it.
    """

    episode_id: str
    runtime_load_id: str
    tokens: tuple[int, ...]
    request: dict[str, object]

    def __post_init__(self) -> None:
        if not self.episode_id or not self.runtime_load_id or not self.tokens:
            raise ValueError("An episode prefix requires an identity, weight version, and tokens")
        if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in self.tokens):
            raise ValueError("Episode prefix tokens must be nonnegative integer IDs")
        if not isinstance(self.request.get("messages"), list):
            raise ValueError("An episode prefix requires a validated chat request")


class RetailInferenceHandler(SGLangInferenceHandler):
    """Nonstreaming Qwen3 inference for one serial ServeLearnBench campaign.

    The standard OpenAI user field identifies an episode. Its first request
    contains system and user messages. Each continuation appends one assistant
    message and one or more official user/tool observations. Prior messages,
    tools, model, and template settings cannot change. A single cached prefix
    contains exact sampled IDs, including length-limited actions. Restarts
    cannot resume a partial episode. A lock serializes calls to this handler.
    """

    def __init__(self, upstream_url: str, **config: object) -> None:
        super().__init__(upstream_url, **config)
        self.prefix: EpisodePrefix | None = None
        self.call_lock = asyncio.Lock()

    def continuation_prompt(self, request: dict[str, object], prefix: EpisodePrefix) -> list[int]:
        """Append Qwen3 observation delimiters and a generation prompt to saved IDs."""
        messages = request["messages"]
        previous = prefix.request["messages"]
        if not isinstance(messages, list) or not isinstance(previous, list):
            raise ValueError("Episode messages must be lists")
        count = len(previous)
        if messages[:count] != previous or len(messages) < count + 2 or messages[count]["role"] != "assistant":
            raise ValueError("Episode continuation must extend the previous conversation")
        observations = messages[count + 1 :]
        if any(message["role"] not in ("user", "tool") for message in observations):
            raise ValueError("Only user/tool observations can follow the sampled assistant turn")
        for key in ("model", "tools", "chat_template_kwargs"):
            if request.get(key) != prefix.request.get(key):
                raise ValueError(f"Episode continuation changed {key}")
        tokenizer = self._require_tokenizer()
        if tokenizer.eos_token != "<|im_end|>":
            raise ValueError("This example requires the Qwen3 ChatML end-of-turn token")
        # Render only fresh context. The dummy system message gives the
        # template a stable prefix; it is removed before encoding.
        boundary = [{"role": "system", "content": "Observation boundary."}]
        options = {**self._chat_template_kwargs, **request.get("chat_template_kwargs", {})}
        start = tokenizer.apply_chat_template(boundary, tokenize=False, add_generation_prompt=False, **options)
        rendered = tokenizer.apply_chat_template(
            [*boundary, *observations], tokenize=False, add_generation_prompt=True, **options
        )
        if not isinstance(start, str) or not isinstance(rendered, str) or not rendered.startswith(start):
            raise ValueError("The chat template cannot render an observation suffix independently")
        # A length-limited action has no sampled end-of-turn token. Supply
        # that delimiter as masked context, without discarding its output.
        delimiter = "\n" if prefix.tokens[-1] == tokenizer.eos_token_id else "<|im_end|>\n"
        context = tokenizer.encode(delimiter + rendered[len(start) :], add_special_tokens=False)
        return [*prefix.tokens, *context]

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, object]) -> dict[str, object]:
        if path != "/v1/chat/completions" or payload.get("stream", False):
            raise ValueError("Retail inference accepts nonstreaming chat completions only")
        episode_id = payload.get("user")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("Set the OpenAI user field to a distinct ID for each episode")
        async with self.call_lock:
            call = self._chat_call(path, payload)
            previous = self.prefix
            if previous is not None and previous.episode_id == episode_id:
                call = replace(call, prompt_ids=self.continuation_prompt(call.request, previous))
            else:
                messages = call.request["messages"]
                if len(messages) != 2 or [message["role"] for message in messages] != ["system", "user"]:
                    raise ValueError("No cached prefix for this episode; restart it from its initial request")
            response, _ = await self._complete(artifact, call)
            training = response["training"]
            version = training["runtime_load_id"]
            if not isinstance(version, str) or not version:
                raise ValueError("An episode call requires a single producing weight version")
            if previous is not None and previous.episode_id == episode_id and version != previous.runtime_load_id:
                raise ValueError("Model weights changed during an episode")
            self.prefix = EpisodePrefix(episode_id, version, tuple(training["tokens"]), deepcopy(call.request))
            return response

    async def inference_stream(self, artifact: Artifact, path: str, payload: dict[str, object]) -> InferenceStream:
        raise ValueError("Retail inference accepts nonstreaming chat completions only")
