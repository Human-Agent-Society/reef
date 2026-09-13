"""The optional Tinker SDK boundary; all remote mutations live here."""

from __future__ import annotations

import importlib
import math
import sys
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reef.train.tinker_backend.checkpoint import TinkerCheckpoint
from reef.train.tinker_backend.config import TinkerConfig
from reef.train.tinker_backend.losses import TinkerLoss, TokenRow


@dataclass(frozen=True)
class SampleResult:
    tokens: tuple[int, ...]
    logprobs: tuple[float, ...]
    stop_reason: str


class TinkerClient(ABC):
    """Remote operations used by the runtime, also implementable by offline tests."""

    @abstractmethod
    def initialize(self) -> TinkerCheckpoint: ...

    @abstractmethod
    def train(
        self, checkpoint: TinkerCheckpoint, batches: Sequence[Sequence[TokenRow]], loss: TinkerLoss
    ) -> tuple[TinkerCheckpoint, Mapping[str, Any]]: ...

    @abstractmethod
    def render(self, messages: list[dict[str, str]], *, template_kwargs: Mapping[str, Any]) -> list[int]: ...

    @abstractmethod
    def decode(self, tokens: Sequence[int]) -> str: ...

    @abstractmethod
    def sample(self, checkpoint: TinkerCheckpoint, prompt: list[int], params: Mapping[str, Any]) -> SampleResult: ...

    @abstractmethod
    def close(self) -> None: ...


class TinkerSDKClient(TinkerClient):
    _sdk: Any
    _model: str
    _config: TinkerConfig
    _api_key: str
    _service: Any
    _base_sampler: Any
    _tokenizer: Any

    def __init__(self, base_model: str, config: TinkerConfig, api_key: str) -> None:
        if sys.version_info < (3, 11):
            raise RuntimeError("Tinker requires Python 3.11 or newer; Reef's other backends still support 3.10")
        try:
            self._sdk = importlib.import_module("tinker")
        except ImportError as exc:
            raise RuntimeError("install Tinker support with uv pip install 'reef-infra[tinker]'") from exc
        self._model = base_model
        self._config = config
        self._api_key = api_key
        self._service = self._new_service()
        try:
            self._base_sampler = self._service.create_sampling_client(base_model=base_model)
            self._tokenizer = self._base_sampler.get_tokenizer()
        except BaseException:
            self._service.close().result(timeout=self._config.train_timeout_s)
            raise

    def _new_service(self) -> Any:
        options: dict[str, Any] = {"api_key": self._api_key, "project_id": self._config.project_id}
        if self._config.train_timeout_s is not None:
            options["timeout"] = self._config.train_timeout_s
        return self._sdk.ServiceClient(**options)

    def _save(self, trainer: Any) -> TinkerCheckpoint:
        name = f"reef-{uuid.uuid4().hex}"
        # Explicit names create durable snapshots. Ephemeral sampler exports
        # cannot back Reef's versioned artifacts. None disables TTL expiry.
        state = trainer.save_state(name, ttl_seconds=None).result(timeout=self._config.train_timeout_s)
        sampler = trainer.save_weights_for_sampler(name, ttl_seconds=None).result(timeout=self._config.train_timeout_s)
        return TinkerCheckpoint(self._model, self._config.lora_rank, state.path, sampler.path)

    def initialize(self) -> TinkerCheckpoint:
        service = self._new_service()
        try:
            trainer = service.create_lora_training_client(
                base_model=self._model, rank=self._config.lora_rank, seed=self._config.seed
            )
            return self._save(trainer)
        finally:
            service.close().result(timeout=self._config.train_timeout_s)

    def train(
        self, checkpoint: TinkerCheckpoint, batches: Sequence[Sequence[TokenRow]], loss: TinkerLoss
    ) -> tuple[TinkerCheckpoint, Mapping[str, Any]]:
        service = self._new_service()
        try:
            # Every attempt owns a new model, restored WITH optimizer state.
            # An uncertain remote result can never mutate the incumbent model.
            info = (
                service.create_rest_client()
                .get_weights_info_by_tinker_path(checkpoint.state_path)
                .result(timeout=self._config.train_timeout_s)
            )
            if info.base_model != self._model or info.is_lora is not True or info.lora_rank != self._config.lora_rank:
                raise ValueError("remote Tinker training checkpoint does not match the configured model/rank")
            trainer = service.create_training_client_from_state_with_optimizer(checkpoint.state_path)
            metrics: dict[str, Any] = {}
            for rows in batches:
                base = self._base_logprobs(rows) if loss.needs_base_logprobs and self._config.kl_coef else []
                inputs = loss.inputs(rows, base, kl_coef=self._config.kl_coef)
                if len(inputs) != len(rows):
                    raise ValueError("Tinker loss adapter must return one input per training row")
                data = [
                    self._sdk.Datum(
                        model_input=self._sdk.ModelInput.from_ints(list(row.tokens[:-1])), loss_fn_inputs=value
                    )
                    for row, value in zip(rows, inputs, strict=True)
                ]
                result = trainer.forward_backward(data, loss_fn=loss.loss_fn).result(
                    timeout=self._config.train_timeout_s
                )
                metrics.update(result.metrics)
                trainer.optim_step(self._sdk.AdamParams(learning_rate=self._config.learning_rate)).result(
                    timeout=self._config.train_timeout_s
                )
            metrics["optimizer_steps"] = len(batches)
            return self._save(trainer), metrics
        finally:
            service.close().result(timeout=self._config.train_timeout_s)

    def _base_logprobs(self, rows: Sequence[TokenRow]) -> list[list[float]]:
        result = []
        for row in rows:
            values = self._base_sampler.compute_logprobs(self._sdk.ModelInput.from_ints(list(row.tokens))).result(
                timeout=self._config.train_timeout_s
            )
            if len(values) != len(row.tokens):
                raise ValueError("Tinker base log probabilities do not align with the captured tokens")
            response = values[-len(row.mask) :]
            if any(value is None or not math.isfinite(value) for value in response):
                raise ValueError("Tinker base log probabilities must be finite for every response token")
            result.append([float(value) for value in response])
        return result

    def render(self, messages: list[dict[str, str]], *, template_kwargs: Mapping[str, Any]) -> list[int]:
        prefill = messages[-1]["role"] == "assistant"
        return list(
            self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=not prefill,
                continue_final_message=prefill,
                **template_kwargs,
            )
        )

    def decode(self, tokens: Sequence[int]) -> str:
        return str(self._tokenizer.decode(list(tokens), skip_special_tokens=True))

    def sample(self, checkpoint: TinkerCheckpoint, prompt: list[int], params: Mapping[str, Any]) -> SampleResult:
        # Each request binds an immutable sampler path, never a mutable model ID.
        sampler = self._service.create_sampling_client(model_path=checkpoint.sampler_path)
        if sampler.get_base_model() != self._model:
            raise ValueError("remote Tinker sampler checkpoint does not match the configured model")
        result = sampler.sample(
            prompt=self._sdk.ModelInput.from_ints(prompt),
            num_samples=1,
            sampling_params=self._sdk.SamplingParams(**params),
        ).result(timeout=self._config.inference_timeout_s)
        if len(result.sequences) != 1:
            raise ValueError("Tinker returned an unexpected number of sequences")
        sequence = result.sequences[0]
        if sequence.logprobs is None or len(sequence.tokens) != len(sequence.logprobs):
            raise ValueError("Tinker must return exact log probabilities for every sampled token")
        if any(not math.isfinite(value) for value in sequence.logprobs):
            raise ValueError("Tinker returned non-finite sampled log probabilities")
        return SampleResult(tuple(sequence.tokens), tuple(sequence.logprobs), sequence.stop_reason)

    def close(self) -> None:
        self._service.close().result(timeout=self._config.train_timeout_s)
