"""Tinker's hosted sampler as a Reef inference runtime.

The trainer in ``reef.train.tinker_backend`` publishes checkpoints as
artifacts holding a ``tinker-checkpoint.json`` manifest that names the
immutable remote sampler; this module serves whatever sampler a frozen
artifact names, through Tinker's SDK, as an OpenAI text chat endpoint with
the exact training capture Reef records. It shares no object with the
trainer: the manifest in the artifact is the whole contract.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from reef_client.sse import synthesize_sse_events

from reef.artifact.artifact import Artifact, LiveWeightArtifactRef
from reef.core.config import config_option
from reef.runtime.deployment import RuntimeFactory, config_secret
from reef.runtime.interfaces import (
    ActivatedModel,
    InferenceHandler,
    InferenceRuntime,
    InferenceStream,
    ModelCandidate,
    StaleCandidate,
    TrainingRuntimeError,
    UpstreamStatusError,
)

#: The manifest a Tinker checkpoint artifact carries; the trainer writes it, this side reads it.
MANIFEST = "tinker-checkpoint.json"
TINKER_URL = "https://tinker.thinkingmachines.ai"


@dataclass(frozen=True)
class SampleResult:
    tokens: tuple[int, ...]
    logprobs: tuple[float, ...]
    stop_reason: str


class TinkerSampler(ABC):
    """Remote sampling operations, also implementable by offline tests.

    ``sampler_path`` is the immutable remote sampler an artifact names, or
    None for the base model itself.
    """

    @abstractmethod
    def render(self, messages: list[dict[str, str]], *, template_kwargs: Mapping[str, Any]) -> list[int]: ...

    @abstractmethod
    def decode(self, tokens: Sequence[int]) -> str: ...

    @abstractmethod
    def sample(self, sampler_path: str | None, prompt: list[int], params: Mapping[str, Any]) -> SampleResult: ...

    @abstractmethod
    def close(self) -> None: ...


class TinkerSDKSampler(TinkerSampler):
    """The SDK boundary for sampling; every request binds an immutable sampler path."""

    _sdk: Any
    _model: str
    _timeout_s: float
    _service: Any
    _base_sampler: Any
    _tokenizer: Any

    def __init__(
        self, base_model: str, *, api_key: str, project_id: str | None = None, timeout_s: float = 300.0
    ) -> None:
        try:
            self._sdk = importlib.import_module("tinker")
        except ImportError as exc:
            raise RuntimeError("install Tinker support with uv pip install 'reef-infra[tinker]'") from exc
        self._model = base_model
        self._timeout_s = timeout_s
        self._service = self._sdk.ServiceClient(api_key=api_key, project_id=project_id)
        try:
            self._base_sampler = self._service.create_sampling_client(base_model=base_model)
            self._tokenizer = self._base_sampler.get_tokenizer()
        except BaseException:
            self._service.close("errored").result(timeout=timeout_s)
            raise

    def render(self, messages: list[dict[str, str]], *, template_kwargs: Mapping[str, Any]) -> list[int]:
        prefill = messages[-1]["role"] == "assistant"
        return list(
            self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=not prefill,
                continue_final_message=prefill,
                **template_kwargs,
            )
        )

    def decode(self, tokens: Sequence[int]) -> str:
        return str(self._tokenizer.decode(list(tokens), skip_special_tokens=True))

    def sample(self, sampler_path: str | None, prompt: list[int], params: Mapping[str, Any]) -> SampleResult:
        if sampler_path is None:
            sampler = self._base_sampler
        else:
            sampler = self._service.create_sampling_client(model_path=sampler_path)
            if sampler.get_base_model() != self._model:
                raise ValueError("remote Tinker sampler checkpoint does not match the configured model")
        result = sampler.sample(
            prompt=self._sdk.ModelInput.from_ints(prompt),
            num_samples=1,
            sampling_params=self._sdk.SamplingParams(**params),
        ).result(timeout=self._timeout_s)
        if len(result.sequences) != 1:
            raise ValueError("Tinker returned an unexpected number of sequences")
        sequence = result.sequences[0]
        if sequence.logprobs is None or len(sequence.tokens) != len(sequence.logprobs):
            raise ValueError("Tinker must return exact log probabilities for every sampled token")
        if any(not math.isfinite(value) for value in sequence.logprobs):
            raise ValueError("Tinker returned non-finite sampled log probabilities")
        return SampleResult(tuple(sequence.tokens), tuple(sequence.logprobs), sequence.stop_reason)

    def close(self) -> None:
        self._service.close("success").result(timeout=self._timeout_s)


def checkpoint_sampler_path(artifact: Artifact, base_model: str) -> str | None:
    """The remote sampler a materialized artifact names; None for Reef's empty base tree."""
    path = artifact.materialize().local_path
    if path is None:
        raise ValueError("Tinker requires a materialized checkpoint manifest")
    manifest = path / MANIFEST
    if manifest.exists():
        value = json.loads(manifest.read_text())
        if not isinstance(value, Mapping) or value.get("base_model") != base_model:
            raise ValueError("Tinker checkpoint manifest does not name the served base model")
        sampler = value.get("sampler_path")
        if not isinstance(sampler, str) or not sampler.startswith("tinker://"):
            raise ValueError("Tinker checkpoint manifest must name a tinker:// sampler")
        return sampler
    # Reef's empty base tree represents the initial seeded adapter.
    # An arbitrary uploaded weight directory must not silently become base.
    contents = {entry.name for entry in path.iterdir()} - {".git", ".gitattributes", "reef-artifact.json"}
    if contents:
        raise ValueError(f"artifact is missing {MANIFEST}")
    return None


class TinkerInferenceRuntime(InferenceRuntime):
    """Serve the sampler each frozen artifact names, and follow Reef's publication of new ones.

    Every snapshot served in this process gets a runtime load ID of this
    incarnation; a rollback to an older release mints a new one. A selected
    candidate stays pending until Reef commits its training job.
    """

    def __init__(
        self,
        sampler: TinkerSampler,
        *,
        base_model: str,
        base_url: str = TINKER_URL,
        inference_timeout_s: float = 300.0,
    ) -> None:
        super().__init__(base_url=base_url, inference_timeout_s=inference_timeout_s)
        self._sampler = sampler
        self._model = base_model
        self._handler = TinkerInferenceHandler(self, sampler, base_model)
        self._lock = RLock()
        self._incarnation = uuid.uuid4().hex
        self._loads: dict[str, str | None] = {}
        self._versions: dict[str | None, str] = {}
        self._releases: dict[str, tuple[str | None, str]] = {}
        self._active_release: str | None = None
        self._active, self._version = self._remember(None)
        self._pending: str | None = None
        self._closed = False
        # The base model the runtime opens with is the published head until an artifact is bound.
        self.mark_published()

    @property
    def inference_handler(self) -> InferenceHandler:
        return self._handler

    @property
    def model_path(self) -> str:
        return self._model

    @property
    def pending_training_job_id(self) -> str | None:
        return self._pending

    def serving_runtime_load_id(self) -> str:
        with self._lock:
            return self._version

    # -- Snapshots

    def _remember(self, sampler_path: str | None) -> tuple[str | None, str]:
        """The load ID a sampler already serves under in this incarnation, else a new one."""
        version = self._versions.get(sampler_path)
        if version is None:
            _, version = self._new_load(sampler_path)
            self._versions[sampler_path] = version
        return sampler_path, version

    def _new_load(self, sampler_path: str | None) -> tuple[str | None, str]:
        version = f"{self._incarnation}:{len(self._loads)}"
        self._loads[version] = sampler_path
        return sampler_path, version

    def snapshot(self, artifact: Artifact) -> tuple[str | None, str]:
        """Resolve exactly the frozen artifact; old in-flight calls keep their sampler."""
        with self._lock:
            if isinstance(artifact.ref, LiveWeightArtifactRef):
                version = artifact.ref.runtime_load_id
                if version in self._loads:
                    return self._loads[version], version
                raise ValueError("Tinker live snapshot belongs to an unavailable serving incarnation")
            if artifact.ref.release_id in self._releases:
                return self._releases[artifact.ref.release_id]
            sampler_path = checkpoint_sampler_path(artifact, self._model)
            # A release that carries the served sampler forward (a step of
            # another component, a rollback to the same weights) was never
            # activated; it serves under the load the engine reports now,
            # not under the first load of that sampler.
            carried = sampler_path == self._active
            selected = (self._active, self._version) if carried else self._remember(sampler_path)
            self._releases[artifact.ref.release_id] = selected
            return selected

    # -- Weight updates

    def restore_checkpoint(self, artifact: Artifact) -> str:
        # Rollback validates the target first; activate_checkpoint binds the
        # republished artifact. No mutable remote serving slot needs restoring.
        return self.snapshot(artifact)[1]

    def activate_checkpoint(self, artifact: Artifact) -> str:
        """Bind the recovered or republished artifact's sampler before the scenario serves it."""
        with self._lock:
            sampler_path, version = self.snapshot(artifact)
            if self._pending is not None and sampler_path != self._active:
                # Reload after a failed publication restores the durable head.
                self._pending = None
            if self._pending is None and self._active_release != artifact.ref.release_id:
                # A rollback is a new serving update even if its immutable
                # sampler was served earlier in this incarnation.
                sampler_path, version = self._new_load(sampler_path)
            self._active, self._version = sampler_path, version
            self._active_release = artifact.ref.release_id
            self._releases[artifact.ref.release_id] = (sampler_path, version)
            if self._pending is None:
                self.mark_published()
            return version

    def activate_candidate(self, candidate: ModelCandidate) -> ActivatedModel:
        with self._lock:
            if self._pending == candidate.training_job_id:
                return ActivatedModel(candidate.candidate_id, self._version)
            if candidate.current_runtime_load_id not in (None, self._version):
                raise StaleCandidate
            sampler_path = checkpoint_sampler_path(Artifact.local(Path(candidate.checkpoint_path)), self._model)
            if sampler_path is None:
                raise ValueError("a Tinker candidate must carry a checkpoint manifest")
            self._active, self._version = self._remember(sampler_path)
            self._pending = candidate.training_job_id
            return ActivatedModel(candidate.candidate_id, self._version)

    def acknowledge_publication(self, training_job_id: str) -> None:
        """Release the activated candidate once Reef committed exactly its training job."""
        with self._lock:
            if self._pending is None:
                return
            if training_job_id != self._pending:
                raise TrainingRuntimeError("Tinker candidate does not match Reef's committed training job")
            self._pending = None

    def shutdown(self) -> None:
        self.pause_admission()
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._sampler.close()


class TinkerInferenceHandler(InferenceHandler):
    """Serve text chat from the sampler the frozen artifact resolves to."""

    def __init__(self, runtime: TinkerInferenceRuntime, sampler: TinkerSampler, model: str) -> None:
        self._runtime = runtime
        self._sampler = sampler
        self._model = model

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path != "/v1/chat/completions":
            raise UpstreamStatusError("Tinker supports /v1/chat/completions", status=400)
        messages, params = chat_request(payload)
        return await asyncio.to_thread(self._sample, artifact, messages, params, payload)

    def _sample(
        self, artifact: Artifact, messages: list[dict[str, str]], params: dict[str, Any], payload: dict[str, Any]
    ) -> dict[str, Any]:
        sampler_path, version = self._runtime.snapshot(artifact)
        prompt = self._sampler.render(messages, template_kwargs=payload.get("chat_template_kwargs") or {})
        if not prompt:
            raise ValueError("Tinker chat template produced an empty prompt")
        result = self._sampler.sample(sampler_path, prompt, params)
        content = self._sampler.decode(result.tokens)
        return chat_completion(payload.get("model", self._model), messages, prompt, result, content, version)

    async def inference_stream(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> InferenceStream:
        return chat_stream(await self.inference(artifact, path, payload), payload)


# -- Request and response shaping ---------------------------------------------


def chat_completion(
    model: str,
    messages: list[dict[str, str]],
    prompt: list[int],
    result: SampleResult,
    content: str,
    runtime_load_id: str,
) -> dict[str, Any]:
    """An OpenAI chat completion plus the private training capture Reef records."""
    message = {"role": "assistant", "content": content}
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
        },
    }


def chat_stream(response: dict[str, Any], payload: dict[str, Any]) -> InferenceStream:
    """Emit a completed response as buffered OpenAI SSE, keeping it as the record."""
    include_usage = bool((payload.get("stream_options") or {}).get("include_usage"))

    async def chunks() -> AsyncIterator[bytes]:
        for event in synthesize_sse_events(response, include_usage=include_usage):
            yield event.encode()

    return InferenceStream(
        status=200, headers={"Content-Type": "text/event-stream"}, chunks=chunks(), record_response=response
    )


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


# -- Factory ------------------------------------------------------------------


@dataclass(frozen=True)
class TinkerInferenceConfig:
    """What the sampler needs; the trainer passes these from its own ``training.options``."""

    api_key: str | None = config_option(None, help="Tinker API key; prefer api_key_env.")
    api_key_env: str | None = config_option("TINKER_API_KEY", help="Environment variable holding the Tinker API key.")
    project_id: str | None = config_option(None, help="Optional Tinker project ID.")
    timeout_s: float = config_option(300.0, help="Sampling request timeout in seconds.")

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("runtime.timeout_s must be positive")


class TinkerInferenceRuntimeFactory(RuntimeFactory):
    """Build the Tinker sampling runtime for a base model; the SDK loads only here."""

    kind = "tinker"

    def config_type(self) -> type:
        return TinkerInferenceConfig

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> InferenceRuntime:
        key = config_secret(config, environ, "api_key", "api_key_env")
        if not key:
            raise ValueError(f"Tinker requires the environment variable {config.get('api_key_env')}")
        sampler = TinkerSDKSampler(
            model_path, api_key=key, project_id=config.get("project_id"), timeout_s=config["timeout_s"]
        )
        try:
            return TinkerInferenceRuntime(sampler, base_model=model_path, inference_timeout_s=config["timeout_s"])
        except BaseException:
            sampler.close()
            raise
