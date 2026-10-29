"""The optional Tinker SDK boundary; all remote mutations live here."""

from __future__ import annotations

import importlib
import math
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from reef.tinker.checkpoint import FreshStart, TinkerCheckpoint
from reef.tinker.service import close_service, tinker_sdk
from reef.train.tinker_backend.config import TinkerServiceConfig
from reef.train.tinker_backend.losses import TinkerCustomLoss, TinkerLoss, TokenRow


class TinkerClient(ABC):
    """Remote training operations, also implementable by offline tests; sampling lives in ``reef.tinker.sampling``."""

    @abstractmethod
    def initialize(self) -> TinkerCheckpoint:
        """Save the untrained LoRA adapter: the base checkpoint, which changes nothing.

        ``ValueError`` for a client of all the parameters, whose base is the base model itself (``FreshStart``).
        """

    @abstractmethod
    def train(
        self,
        start: TinkerCheckpoint | FreshStart,
        steps: Sequence[Sequence[TokenRow]],
        loss: TinkerLoss,
        learning_rates: Sequence[float],
        *,
        kl_coef: float,
    ) -> tuple[TinkerCheckpoint, Mapping[str, Any]]:
        """Train one optimizer step per entry of ``steps`` from ``start``, step ``i`` at ``learning_rates[i]``.

        A ``TinkerCheckpoint`` start restores its weights and optimizer state; a ``FreshStart`` creates a new
        model of all the parameters of its base model. ``loss`` weighs its KL term by ``kl_coef``. Return the
        checkpoint after the last step and Tinker's metrics of the steps. ``ValueError``: ``steps`` and
        ``learning_rates`` differ in length, ``loss`` refuses ``kl_coef``, or ``start`` is of another model or
        parameterization than the client's.
        """

    @abstractmethod
    def check_remote(self, checkpoint: TinkerCheckpoint) -> None:
        """Raise ``ValueError`` unless ``checkpoint.state_path`` resolves on the service as a checkpoint of the
        configured model with the configured parameterization: a LoRA of the configured rank, or all parameters.

        An artifact import calls it before it publishes a Tinker manifest that came from outside Reef. Only the
        state path is looked up: the Tinker API describes a saved state (``weights_info``) but no sampler path,
        and Spindle refuses sampler paths there.
        """

    @abstractmethod
    def close(self) -> None: ...


def create_full_training_client(sdk: Any, service: Any, base_model: str) -> Any:
    """A new training client of all the parameters of ``base_model`` on the Spindle service of ``service``.

    Spindle extends Tinker's ``create_model`` with ``parameterization: {"type": "full"}``, which the Tinker SDK
    cannot send, so this posts the request as ``spindle.client.create_full_training_client`` does, through the
    SDK's own session and request future.
    """
    futures = importlib.import_module("tinker.lib.api_future_impl")
    pools = importlib.import_module("tinker.lib.client_connection_pool_type")
    queue_logs = importlib.import_module("tinker.lib.queue_state_logger")
    holder = service.holder
    session_id = holder.get_session_id()
    model_seq_id = holder.get_training_client_id()

    async def create() -> Any:
        started = time.time()
        request = sdk.types.CreateModelRequest(session_id=session_id, model_seq_id=model_seq_id, base_model=base_model)
        with holder.aclient(pools.ClientConnectionPoolType.TRAIN) as client:
            future = await client.post(
                "/api/v1/create_model",
                body={**request.model_dump(exclude_none=True, mode="json"), "parameterization": {"type": "full"}},
                cast_to=sdk.types.UntypedAPIFuture,
            )
        response = await futures._APIFuture(
            sdk.types.CreateModelResponse,
            holder,
            future,
            request_start_time=started,
            request_type="CreateModel",
            queue_state_observer=queue_logs.QueueStateLogger(base_model, "Model creation"),
        ).result_async()
        return sdk.TrainingClient(holder, model_seq_id=model_seq_id, model_id=response.model_id)

    return holder.run_coroutine_threadsafe(create()).result()


class TinkerSDKClient(TinkerClient):
    """The remote training operations through Tinker's SDK, on a LoRA of rank ``lora_rank`` or on all parameters.

    ``lora_rank`` is a positive int for a LoRA, which starts with ``seed``, or ``None`` for all the parameters,
    which only Spindle trains (``create_full_training_client``).
    """

    _sdk: Any
    _model: str
    _config: TinkerServiceConfig
    _api_key: str
    _service: Any
    #: The frozen base model's sampling client, made at the first KL term.
    base_client: Any | None

    def __init__(
        self,
        base_model: str,
        config: TinkerServiceConfig,
        api_key: str,
        *,
        lora_rank: int | None,
        seed: int | None = None,
    ) -> None:
        self._sdk = tinker_sdk()
        self._model = base_model
        self._config = config
        self._lora_rank = lora_rank
        self._seed = seed
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
        return TinkerCheckpoint(self._model, self._lora_rank, state.path, sampler.path, service=self._config.base_url)

    def _new_model(self, service: Any) -> Any:
        """A new training client of the base model: an untrained LoRA, or all the parameters with their own weights."""
        if self._lora_rank is None:
            return create_full_training_client(self._sdk, service, self._model)
        return service.create_lora_training_client(base_model=self._model, rank=self._lora_rank, seed=self._seed)

    def _check_info(self, info: Any, path: str) -> None:
        """Raise ``ValueError`` unless the service's ``weights_info`` of ``path`` is of the configured model and parameterization."""
        lora = self._lora_rank is not None
        if info.base_model != self._model or info.is_lora is not lora or (lora and info.lora_rank != self._lora_rank):
            expected = f"rank {self._lora_rank} LoRA" if lora else "full-parameter checkpoint"
            raise ValueError(f"the Tinker path {path} is no {expected} of {self._model}")

    def initialize(self) -> TinkerCheckpoint:
        if self._lora_rank is None:
            raise ValueError("a full-parameter client starts from the base model itself; it saves no base checkpoint")
        service = self._new_service()
        status = "errored"
        try:
            trainer = self._new_model(service)
            checkpoint = self._save(trainer)
            status = "success"
            return checkpoint
        finally:
            close_service(self._sdk, service, status, timeout_s=self._config.train_timeout_s)

    def train(
        self,
        start: TinkerCheckpoint | FreshStart,
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
            if isinstance(start, FreshStart):
                if start.base_model != self._model:
                    raise ValueError(f"a fresh start of {start.base_model} is no start of {self._model}")
                trainer = self._new_model(service)
            else:
                info = (
                    service.create_rest_client()
                    .get_weights_info_by_tinker_path(start.state_path)
                    .result(timeout=self._config.train_timeout_s)
                )
                self._check_info(info, start.state_path)
                trainer = service.create_training_client_from_state_with_optimizer(start.state_path)
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
        self._check_info(info, path)

    def close(self) -> None:
        close_service(self._sdk, self._service, "success", timeout_s=self._config.train_timeout_s)
