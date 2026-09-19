"""Tinker's components for Reef's model driver: a GPU-less trainer beside a local engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.runtime.deployment import (
    ADAPTER_FILES_PROTOCOL,
    DeploymentResources,
    InferenceResources,
    TrainingService,
    WeightTransferSession,
)
from reef.runtime.executor.placement import ModelGpuLayout, ModelGpuReservation, reserve_model_gpus
from reef.runtime.interfaces import TrainingBackend
from reef.train.tinker_backend.client import TinkerClient
from reef.train.tinker_backend.config import TinkerConfig


class TinkerDeploymentResources(InferenceResources):
    """One Ray session and the placement group the local engines run on; the trainer needs no GPU."""

    def __init__(self, layout: ModelGpuLayout, *, ray_address: str, namespace: str) -> None:
        if layout.training_gpus != 0:
            raise ValueError("a hosted trainer reserves inference GPUs only")
        self.layout = layout
        self.ray_address = ray_address
        self.namespace = namespace
        self._reservation: ModelGpuReservation | None = None
        self._started = False
        self._closed = False

    @property
    def inference_placement(self) -> Any:
        return None if self._reservation is None else self._reservation.inference

    def start(self) -> None:
        import ray

        if self._started or self._closed:
            raise RuntimeError("deployment resources can only be started once")
        if ray.is_initialized():
            raise RuntimeError("the model driver requires its own Ray client session")
        self._started = True
        ray.init(address=self.ray_address, namespace=self.namespace)
        self._reservation = reserve_model_gpus(self.layout)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._reservation is not None:
                self._reservation.release()
        finally:
            if self._started:
                import ray

                ray.shutdown()


class TinkerTrainingService(TrainingService):
    """Own the SDK client; the trainer delivers adapter files, so it needs no engine handle."""

    weight_transfer_protocol = ADAPTER_FILES_PROTOCOL

    def __init__(
        self,
        base_model: str,
        config: TinkerConfig,
        api_key: str,
        *,
        loss_family: str | None = None,
        loss_reference: str | None = None,
        client: TinkerClient | None = None,
    ) -> None:
        self._model = base_model
        self._config = config
        self._loss_family = loss_family
        self._loss_reference = loss_reference
        self._api_key = api_key
        self._client = client
        self._session_id: str | None = None
        self._backend: TrainingBackend | None = None
        self._started = False
        self._closed = False

    def start(self, resources: DeploymentResources) -> None:
        if self._started or self._closed:
            raise RuntimeError("training service can only be started once")
        self._started = True

    def attach_weight_transport(self, session: WeightTransferSession) -> None:
        if not self._started or self._closed:
            raise RuntimeError("start the training service before attaching its weight transport")
        if session.protocol != self.weight_transfer_protocol:
            raise ValueError(
                "Tinker delivers adapter files; the deployment must pair it with a receiver that loads them"
            )
        self._session_id = session.session_id

    def backend(self) -> TrainingBackend:
        """The backend the coordinator starts in its own process; the SDK session opens there."""
        if not self._started or self._session_id is None or self._closed:
            raise RuntimeError("training backend requires a started service with attached engines")
        if self._backend is None:
            from reef.train.tinker_backend.backend import TinkerTrainingBackend

            self._backend = TinkerTrainingBackend(
                self._model,
                self._config,
                loss_family=self._loss_family,
                loss_reference=self._loss_reference,
                api_key=self._api_key,
                client=self._client,
            )
        return self._backend

    def check_health(self) -> None:
        self.poll()

    def poll(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError("training service is not running")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True


def sglang_inference_config(reef: Mapping[str, Any], config: TinkerConfig) -> dict[str, Any]:
    """The managed SGLang input for serving Tinker's adapters: single-node tensor-parallel engines."""
    num_gpus = int(reef["inference_num_gpus"])
    gpus_per_engine = int(reef["tensor_parallel_size"])
    options = {key.replace("-", "_"): value for key, value in dict(reef.get("inference_options") or {}).items()}
    options.update(
        model_path=reef["model_path"],
        trust_remote_code=True,
        skip_server_warmup=True,
        enable_metrics=True,
        enable_lora=True,
        max_lora_rank=config.lora_rank,
        max_loaded_loras=config.max_loaded_adapters,
        max_loras_per_batch=config.max_loaded_adapters,
    )
    options.setdefault("lora_target_modules", ["all"])
    return {
        "model_path": reef["model_path"],
        "num_gpus": num_gpus,
        "gpus_per_engine": gpus_per_engine,
        "gpus_per_node": max(num_gpus, gpus_per_engine),
        "options": options,
        "executor": "auto",
    }
