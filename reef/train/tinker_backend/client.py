"""The optional Tinker SDK boundary; all remote mutations live here."""

from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from reef.tinker.checkpoint import TinkerCheckpoint
from reef.tinker.service import close_service, tinker_sdk
from reef.train.tinker_backend.config import TinkerConfig
from reef.train.tinker_backend.losses import TinkerCustomLoss, TinkerLoss, TokenRow


class TinkerClient(ABC):
    """Remote training operations, also implementable by offline tests; sampling lives in ``reef.tinker.sampling``."""

    @abstractmethod
    def initialize(self) -> TinkerCheckpoint: ...

    @abstractmethod
    def train(
        self,
        checkpoint: TinkerCheckpoint,
        steps: Sequence[Sequence[TokenRow]],
        loss: TinkerLoss,
        learning_rates: Sequence[float],
        *,
        kl_coef: float,
    ) -> tuple[TinkerCheckpoint, Mapping[str, Any]]:
        """Train one optimizer step per entry of ``steps`` from ``checkpoint``, step ``i`` at ``learning_rates[i]``.

        ``loss`` weighs its KL term by ``kl_coef``. Return the checkpoint after the last step and Tinker's
        metrics of the steps. ``ValueError``: ``steps`` and ``learning_rates`` differ in length, or ``loss``
        refuses ``kl_coef``.
        """

    @abstractmethod
    def check_remote(self, checkpoint: TinkerCheckpoint) -> None:
        """Raise ``ValueError`` unless ``checkpoint.state_path`` resolves on the service as a LoRA of the configured
        model with the configured rank.

        An artifact import calls it before it publishes a Tinker manifest that came from outside Reef. Only the
        state path is looked up: the Tinker API describes a saved state (``weights_info``) but no sampler path,
        and Spindle refuses sampler paths there.
        """

    @abstractmethod
    def close(self) -> None: ...


class TinkerSDKClient(TinkerClient):
    _sdk: Any
    _model: str
    _config: TinkerConfig
    _api_key: str
    _service: Any
    #: The frozen base model's sampling client, made at the first KL term.
    base_client: Any | None

    def __init__(self, base_model: str, config: TinkerConfig, api_key: str) -> None:
        self._sdk = tinker_sdk()
        self._model = base_model
        self._config = config
        self._api_key = api_key
        self._service = self._new_service()
        # On a service such as Spindle, a sampling client starts a sampler pool, so the frozen base's client is
        # made at the first KL term. Sampling for requests is the inference runtime's.
        self.base_client = None

    def _new_service(self) -> Any:
        # The SDK takes no client-wide timeout: each call is bounded with .result(timeout=train_timeout_s).
        return self._sdk.ServiceClient(
            api_key=self._api_key, project_id=self._config.project_id, base_url=self._config.base_url
        )

    def _save(self, trainer: Any) -> TinkerCheckpoint:
        """Save ``trainer``'s state and its sampler weights durably."""
        name = f"reef-{uuid.uuid4().hex}"
        # Explicit names create durable snapshots. Ephemeral sampler exports
        # cannot back Reef's versioned artifacts. None disables TTL expiry.
        timeout = self._config.train_timeout_s
        state = trainer.save_state(name, ttl_seconds=None).result(timeout=timeout)
        sampler = trainer.save_weights_for_sampler(name, ttl_seconds=None).result(timeout=timeout)
        return TinkerCheckpoint(
            self._model, self._config.lora_rank, state.path, sampler.path, service=self._config.base_url
        )

    def initialize(self) -> TinkerCheckpoint:
        """Save the untrained adapter of the configured rank: the base checkpoint, which changes nothing."""
        service = self._new_service()
        status = "errored"
        try:
            trainer = service.create_lora_training_client(
                base_model=self._model, rank=self._config.lora_rank, seed=self._config.seed
            )
            checkpoint = self._save(trainer)
            status = "success"
            return checkpoint
        finally:
            close_service(self._sdk, service, status, timeout_s=self._config.train_timeout_s)

    def train(
        self,
        checkpoint: TinkerCheckpoint,
        steps: Sequence[Sequence[TokenRow]],
        loss: TinkerLoss,
        learning_rates: Sequence[float],
        *,
        kl_coef: float,
    ) -> tuple[TinkerCheckpoint, Mapping[str, Any]]:
        if len(learning_rates) != len(steps):
            raise ValueError("Tinker training needs one learning rate per optimizer step")
        loss.check_kl_coef(kl_coef)
        service = self._new_service()
        status = "errored"
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
            for rows, learning_rate in zip(steps, learning_rates, strict=True):
                base = self._base_logprobs(rows) if loss.needs_base_logprobs and kl_coef else []
                inputs = loss.inputs(rows, base, kl_coef=kl_coef)
                if len(inputs) != len(rows):
                    raise ValueError("Tinker loss adapter must return one input per training row")
                data = [
                    self._sdk.Datum(
                        model_input=self._sdk.ModelInput.from_ints(list(row.tokens[:-1])), loss_fn_inputs=value
                    )
                    for row, value in zip(rows, inputs, strict=True)
                ]
                if isinstance(loss, TinkerCustomLoss):
                    # Tinker returns logprobs; the family computes the loss here and Tinker runs backward.
                    future = trainer.forward_backward_custom(data, loss.loss)
                else:
                    future = trainer.forward_backward(data, loss_fn=loss.loss_fn)
                result = future.result(timeout=self._config.train_timeout_s)
                metrics.update(result.metrics)
                trainer.optim_step(self._sdk.AdamParams(learning_rate=learning_rate)).result(
                    timeout=self._config.train_timeout_s
                )
            result_checkpoint = self._save(trainer)
            status = "success"
            return result_checkpoint, metrics
        finally:
            close_service(self._sdk, service, status, timeout_s=self._config.train_timeout_s)

    def _base_logprobs(self, rows: Sequence[TokenRow]) -> list[list[float]]:
        if self.base_client is None:
            self.base_client = self._service.create_sampling_client(base_model=self._model)
        result = []
        for row in rows:
            values = self.base_client.compute_logprobs(self._sdk.ModelInput.from_ints(list(row.tokens))).result(
                timeout=self._config.train_timeout_s
            )
            if len(values) != len(row.tokens):
                raise ValueError("Tinker base log probabilities do not align with the captured tokens")
            response = values[-len(row.mask) :]
            if any(value is None or not math.isfinite(value) for value in response):
                raise ValueError("Tinker base log probabilities must be finite for every response token")
            result.append([float(value) for value in response])
        return result

    def check_remote(self, checkpoint: TinkerCheckpoint) -> None:
        path = checkpoint.state_path
        rest = self._service.create_rest_client()
        try:
            info = rest.get_weights_info_by_tinker_path(path).result(timeout=self._config.train_timeout_s)
        except self._sdk.NotFoundError as exc:
            raise ValueError(f"the Tinker path {path} does not resolve on the configured service") from exc
        except self._sdk.BadRequestError as exc:
            raise ValueError(f"the configured service refuses the Tinker path {path}: {exc}") from exc
        if info.base_model != self._model or info.is_lora is not True or info.lora_rank != self._config.lora_rank:
            raise ValueError(f"the Tinker path {path} is no rank {self._config.lora_rank} LoRA of {self._model}")

    def close(self) -> None:
        close_service(self._sdk, self._service, "success", timeout_s=self._config.train_timeout_s)
