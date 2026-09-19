"""The optional Tinker SDK boundary; all remote mutations live here."""

from __future__ import annotations

import importlib
import math
import shutil
import tarfile
import urllib.request
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reef.train.tinker_backend.checkpoint import TinkerCheckpoint
from reef.train.tinker_backend.config import TinkerConfig
from reef.train.tinker_backend.losses import TinkerCustomLoss, TinkerLoss, TokenRow


class TinkerClient(ABC):
    """Remote training operations, also implementable by offline tests; sampling lives in ``reef.inference.tinker``."""

    @abstractmethod
    def initialize(self) -> TinkerCheckpoint: ...

    @abstractmethod
    def train(
        self, checkpoint: TinkerCheckpoint, batches: Sequence[Sequence[TokenRow]], loss: TinkerLoss
    ) -> tuple[TinkerCheckpoint, Mapping[str, Any]]: ...

    @abstractmethod
    def download(self, checkpoint: TinkerCheckpoint, directory: Path) -> None:
        """Materialize the checkpoint's sampler weights as a PEFT adapter directory."""

    @abstractmethod
    def close(self) -> None: ...


def _extract_archive(archive: Path, directory: Path) -> None:
    """Unpack a checkpoint archive, refusing links and paths outside ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    base = directory.resolve()
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise ValueError(f"checkpoint archive contains a link: {member.name}")
            if not (base / member.name).resolve().is_relative_to(base):
                raise ValueError(f"checkpoint archive escapes its directory: {member.name}")
        tar.extractall(path=base, members=members)


class TinkerSDKClient(TinkerClient):
    _sdk: Any
    _model: str
    _config: TinkerConfig
    _api_key: str
    _service: Any
    _base_sampler: Any

    def __init__(self, base_model: str, config: TinkerConfig, api_key: str) -> None:
        try:
            self._sdk = importlib.import_module("tinker")
        except ImportError as exc:
            raise RuntimeError("install Tinker support with uv pip install 'reef-infra[tinker]'") from exc
        self._model = base_model
        self._config = config
        self._api_key = api_key
        self._service = self._new_service()
        try:
            # The frozen base scores KL terms; sampling for requests is the inference runtime's.
            self._base_sampler = self._service.create_sampling_client(base_model=base_model)
        except BaseException:
            self._service.close("errored").result(timeout=self._config.train_timeout_s)
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
        status = "errored"
        try:
            trainer = service.create_lora_training_client(
                base_model=self._model, rank=self._config.lora_rank, seed=self._config.seed
            )
            checkpoint = self._save(trainer)
            status = "success"
            return checkpoint
        finally:
            service.close(status).result(timeout=self._config.train_timeout_s)

    def train(
        self, checkpoint: TinkerCheckpoint, batches: Sequence[Sequence[TokenRow]], loss: TinkerLoss
    ) -> tuple[TinkerCheckpoint, Mapping[str, Any]]:
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
                if isinstance(loss, TinkerCustomLoss):
                    # Tinker returns logprobs; the family computes the loss here and Tinker runs backward.
                    future = trainer.forward_backward_custom(data, loss.loss)
                else:
                    future = trainer.forward_backward(data, loss_fn=loss.loss_fn)
                result = future.result(timeout=self._config.train_timeout_s)
                metrics.update(result.metrics)
                trainer.optim_step(self._sdk.AdamParams(learning_rate=self._config.learning_rate)).result(
                    timeout=self._config.train_timeout_s
                )
            metrics["optimizer_steps"] = len(batches)
            result_checkpoint = self._save(trainer)
            status = "success"
            return result_checkpoint, metrics
        finally:
            service.close(status).result(timeout=self._config.train_timeout_s)

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

    def download(self, checkpoint: TinkerCheckpoint, directory: Path) -> None:
        """Fetch the sampler archive, then convert it with the cookbook into PEFT layout.

        Tinker's archive holds the adapter in its own key naming; the
        cookbook's converter renames tensors to the base model's parameter
        names, which is what SGLang and vLLM load.
        """
        try:
            from tinker_cookbook import weights
        except ImportError as exc:
            raise RuntimeError(
                "serving Tinker checkpoints on a local engine needs tinker-cookbook: "
                "uv pip install 'reef-infra[tinker]'"
            ) from exc
        response = (
            self._service.create_rest_client()
            .get_checkpoint_archive_url_from_tinker_path(checkpoint.sampler_path)
            .result(timeout=self._config.train_timeout_s)
        )
        staging = directory.parent / f".{directory.name}.download"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            archive = staging / "checkpoint.tar"
            urllib.request.urlretrieve(response.url, archive)
            raw = staging / "tinker"
            _extract_archive(archive, raw)
            weights.build_lora_adapter(base_model=self._model, adapter_path=str(raw), output_path=str(directory))
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def close(self) -> None:
        self._service.close("success").result(timeout=self._config.train_timeout_s)
