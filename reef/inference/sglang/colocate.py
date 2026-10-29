"""Synchronous GPU handoff for Slime jobs and one local SGLang serving engine.

The trainer owns the reservation. Serving borrows it and retains its committed
weights across training. Native memory-saver calls move the models; Reef only
orders those calls and keeps requests out until the handoff finishes.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from reef.artifact.artifact import Artifact
from reef.inference.remote.runtime import serving_engine_config
from reef.inference.sglang.backend import SGLangInferenceBackend
from reef.inference.sglang.chat import SGLangInferenceHandler
from reef.inference.sglang.deployment import create_inference
from reef.inference.sglang.service import SGLangInferenceService
from reef.runtime.deployment import InferenceResources, RuntimeConfigError, RuntimeFactory
from reef.runtime.interfaces import InferenceRuntime, SharedGpuInferenceRuntime
from reef.runtime.jobs import ReleaseFormat
from reef.runtime.serving import FullWeightServingRuntime

logger = logging.getLogger(__name__)


class ColocatedServingRuntime(FullWeightServingRuntime, SharedGpuInferenceRuntime):
    """A full-weight serving runtime that lends its GPUs to one serialized training job.

    ``phase`` is serving, draining, training, restoring, or failed. Only
    serving admits requests. A failed handoff leaves admission paused until
    deployment restart. The Slime component uses ``training_phase`` around
    each train-and-save operation; publication keeps the inherited artifact
    loading behavior. ``memory_backend`` controls the same serving engines.
    """

    def __init__(
        self,
        backend: SGLangInferenceBackend,
        handler: SGLangInferenceHandler,
        *,
        service: SGLangInferenceService,
        model_path: str,
        inference_timeout_s: float,
    ) -> None:
        self.memory_backend = backend
        self.phase = "serving"
        super().__init__(
            backend, handler, service=service, model_path=model_path, inference_timeout_s=inference_timeout_s
        )

    @contextmanager
    def training_phase(self) -> Iterator[None]:
        """Drain requests, lend GPU memory to training, then restore the served weights.

        The caller must use native train offload, including checkpoint export.
        Any exception leaves admission paused: an RPC timeout does not prove
        that the training workers stopped using the GPUs.
        """
        if self.phase != "serving":
            raise RuntimeError(f"cannot train while shared GPU phase is {self.phase}")
        self.phase = "draining"
        try:
            self.pause_and_drain_inference(self.inference_timeout_s)
            # Drain before taking the publication lock: admitted requests may still need a version lookup.
            with self._lock:
                self.memory_backend.pause()
                self.memory_backend.release_memory()
                self.phase = "training"
                logger.info("Shared GPUs: inference released memory; training starts")
                yield
                self.phase = "restoring"
                self.memory_backend.resume_memory()
                self.memory_backend.resume()
                self.phase = "serving"
                self.resume_inference()
                logger.info("Shared GPUs: served weights restored; inference resumes")
        except BaseException:
            self.phase = "failed"
            self.close_admission()
            raise

    def recover_engines(self) -> None:
        with self._lock:
            if self.phase != "serving":
                raise RuntimeError(f"cannot recover inference during shared GPU phase {self.phase}")
            super().recover_engines()

    def activate_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        with self._lock:
            if self.phase == "failed":
                raise RuntimeError("cannot publish after a shared GPU handoff failed; restart the deployment")
            return super().activate_checkpoint(artifact, scenario=scenario)


def build_colocated_serving(
    resources: InferenceResources,
    release_format: ReleaseFormat,
    config: Mapping[str, Any],
    *,
    model_path: str,
    environ: Mapping[str, str],
) -> ColocatedServingRuntime:
    """Start full-weight SGLang on the trainer's borrowed, single-node GPU bundles."""
    if release_format.layout != "full":
        raise ValueError("colocated serving requires full-weight releases")
    options = dict(config["options"])
    options["enable-memory-saver"] = True
    # A trained candidate is not necessarily published. Preserve the incumbent
    # weights while releasing their GPU storage, then restore that same model.
    options["enable-weights-cpu-backup"] = True
    engine_config = serving_engine_config(
        "sglang",
        model_path=model_path,
        num_gpus=config["num_gpus"],
        tensor_parallel_size=config["tensor_parallel_size"],
        options=options,
        release_format=release_format,
        adapter_capacity=1,
        prefix_sharing=False,
        executor=config["executor"],
        executor_options=config["executor_options"],
        environ=environ,
    )
    service = create_inference(engine_config)
    try:
        backend = service.backend(service.start(resources))
        if not isinstance(backend, SGLangInferenceBackend):
            raise TypeError("colocated serving requires the SGLang memory-control backend")
        handler = SGLangInferenceHandler.from_config(
            backend.inference_url(),
            model_path=model_path,
            timeout_s=config["inference_timeout_s"],
            **dict(config["inference_handler_config"]),
        )
        return ColocatedServingRuntime(
            backend,
            handler,
            service=service,
            model_path=model_path,
            inference_timeout_s=config["inference_timeout_s"],
        )
    except BaseException:
        service.close()
        raise


class ColocatedServingFactory(RuntimeFactory):
    """Local SGLang serving built only on an owner's borrowed GPU reservation."""

    kind = "reef.inference.sglang.colocate:ColocatedServingFactory"

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> InferenceRuntime:
        raise RuntimeConfigError("colocated SGLang requires the training component's GPU reservation")

    def build_with_resources(
        self,
        resources: InferenceResources,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> InferenceRuntime:
        return build_colocated_serving(
            resources,
            ReleaseFormat.from_dict(config["release_format"]),
            config,
            model_path=model_path,
            environ=environ,
        )
