"""The optional Tinker SDK boundary; all remote mutations live here."""

from __future__ import annotations

import math
import shutil
import urllib.request
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reef.runtime.serving import ADAPTER_DIR
from reef.tinker.checkpoint import TinkerCheckpoint
from reef.tinker.service import close_service, tinker_sdk
from reef.train.tinker_backend.config import TinkerConfig
from reef.train.tinker_backend.losses import TinkerCustomLoss, TinkerLoss, TokenRow

#: The directory of a release that holds Tinker's adapter as Tinker saved it, in Tinker's key names. The cookbook's
#: merge into full weights (``tinker_cookbook.weights.build_hf_model``) reads this form, not the PEFT conversion.
TINKER_ADAPTER_DIR = "tinker-adapter"


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
        export_adapter: bool,
    ) -> tuple[TinkerCheckpoint, Mapping[str, Any]]:
        """Train one optimizer step per entry of ``steps`` from ``checkpoint``, step ``i`` at ``learning_rates[i]``.

        ``loss`` weighs its KL term by ``kl_coef``. Return the checkpoint after the last step, whose
        ``adapter_path`` names its exported adapter exactly when ``export_adapter`` is set, and Tinker's
        metrics of the steps. ``ValueError``: ``steps`` and ``learning_rates`` differ in length, or ``loss``
        refuses ``kl_coef``.
        """

    @abstractmethod
    def download(self, checkpoint: TinkerCheckpoint, release: Path) -> None:
        """Write the adapter that training exported for ``checkpoint`` into the release directory ``release``.

        ``release / TINKER_ADAPTER_DIR`` gets the adapter as Tinker exported it, and ``release / ADAPTER_DIR`` its
        conversion into a PEFT adapter in the base model's parameter names, which inference engines load. Neither
        directory exists before the call. ``ValueError``: ``checkpoint.adapter_path`` is ``None``.
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

    def _save(self, trainer: Any, *, export_adapter: bool) -> TinkerCheckpoint:
        """Save ``trainer``'s weights durably, and with ``export_adapter`` also its adapter for ``download``."""
        name = f"reef-{uuid.uuid4().hex}"
        # Explicit names create durable snapshots. Ephemeral sampler exports
        # cannot back Reef's versioned artifacts. None disables TTL expiry.
        timeout = self._config.train_timeout_s
        state = trainer.save_state(name, ttl_seconds=None).result(timeout=timeout)
        sampler = trainer.save_weights_for_sampler(name, ttl_seconds=None).result(timeout=timeout)
        adapter = (
            trainer.save_weights_external(name, ttl_seconds=None).result(timeout=timeout) if export_adapter else None
        )
        return TinkerCheckpoint(
            self._model,
            self._config.lora_rank,
            state.path,
            sampler.path,
            adapter_path=None if adapter is None else adapter.path,
            service=self._config.base_url,
        )

    def initialize(self) -> TinkerCheckpoint:
        """Save the untrained adapter of the configured rank: the base checkpoint, which changes nothing.

        With ``export_adapter``, the checkpoint also names the exported adapter, as every saved checkpoint does.
        """
        service = self._new_service()
        status = "errored"
        try:
            trainer = service.create_lora_training_client(
                base_model=self._model, rank=self._config.lora_rank, seed=self._config.seed
            )
            checkpoint = self._save(trainer, export_adapter=self._config.export_adapter)
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
        export_adapter: bool,
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
            result_checkpoint = self._save(trainer, export_adapter=export_adapter)
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

    def download(self, checkpoint: TinkerCheckpoint, release: Path) -> None:
        """Fetch the checkpoint's exported adapter into ``TINKER_ADAPTER_DIR``, then convert it into ``ADAPTER_DIR``.

        ``checkpoint.adapter_path`` names the adapter that training exported
        (``save_weights_external``) in Tinker's key names; Tinker signs one
        download URL per file. The cookbook's converter renames the tensors to
        the base model's parameter names, which is what SGLang and vLLM load.
        It keeps projections that the engines fuse at load time, such as
        Qwen3.5's ``in_proj_q/k/v``, apart, so a merge into full weights reads
        ``TINKER_ADAPTER_DIR`` instead. ``ValueError``: the checkpoint has no
        exported adapter, or a file path leaves the adapter's directory.
        """
        if checkpoint.adapter_path is None:
            raise ValueError("the Tinker checkpoint was saved without an exported adapter (training.export_adapter)")
        try:
            from tinker_cookbook import weights
        except ImportError as exc:
            raise RuntimeError(
                "serving Tinker checkpoints on a local engine needs tinker-cookbook: "
                "uv pip install 'reef-infra[tinker]'"
            ) from exc
        urls = (
            self._service.create_rest_client()
            .get_external_weights_urls(checkpoint.adapter_path)
            .result(timeout=self._config.train_timeout_s)
            .urls
        )
        staging = release.parent / f".{release.name}.download"
        shutil.rmtree(staging, ignore_errors=True)
        raw = staging / "tinker"
        raw.mkdir(parents=True)
        try:
            for name, url in urls.items():
                target = (raw / name).resolve()
                if not target.is_relative_to(raw.resolve()):
                    raise ValueError(f"the exported Tinker adapter has a file outside its directory: {name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(url, target)
            weights.build_lora_adapter(
                base_model=self._model, adapter_path=str(raw), output_path=str(release / ADAPTER_DIR)
            )
            shutil.move(raw, release / TINKER_ADAPTER_DIR)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def close(self) -> None:
        close_service(self._sdk, self._service, "success", timeout_s=self._config.train_timeout_s)
