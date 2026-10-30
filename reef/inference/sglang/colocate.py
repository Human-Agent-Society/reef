"""Synchronous GPU handoff for Slime jobs and SGLang serving engines.

The trainer owns the reservation. Serving borrows it and retains its committed
weights across training. Native memory-saver calls move the models; Reef only
orders those calls and keeps requests out until the handoff finishes.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from reef.artifact.artifact import Artifact
from reef.inference.http import ProviderRequestHeaders
from reef.inference.remote.config import RemoteInferenceSettings
from reef.inference.remote.runtime import serving_engine_config
from reef.inference.remote.service import RemoteInferenceService
from reef.inference.sglang.chat import SGLangInferenceHandler
from reef.inference.sglang.deployment import create_inference
from reef.runtime.deployment import (
    InferenceResources,
    InferenceService,
    RuntimeConfigError,
    RuntimeFactory,
    select_inference_resources,
)
from reef.runtime.interfaces import InferenceRuntime, MemoryManagedInferenceBackend, SharedGpuInferenceRuntime
from reef.runtime.jobs import ReleaseFormat
from reef.runtime.serving import AdapterServingRuntime, FullWeightServingRuntime, SavedWeightServingRuntime

logger = logging.getLogger(__name__)


class SharedSGLangServingRuntime(SavedWeightServingRuntime, SharedGpuInferenceRuntime):
    """Shared GPU handoff added to an existing saved-weight serving runtime.

    Concrete subclasses select full weights or adapters through their existing
    serving base. ``phase`` starts as serving, then becomes draining, training,
    restoring, or failed. Only serving admits requests. Each instance assigns
    its own phase when a handoff starts. A failed handoff requires restart.
    """

    phase: str = "serving"

    @property
    def memory_backend(self) -> MemoryManagedInferenceBackend:
        """The native backend that controls every engine in this allocation."""
        if not isinstance(self._backend, MemoryManagedInferenceBackend):
            raise TypeError("shared GPU serving requires a memory-control backend")
        return self._backend

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
            return self.publish_checkpoint(artifact, scenario=scenario)

    @abstractmethod
    def publish_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        """Publish through the selected saved-weight runtime under the handoff lock."""


class ColocatedServingRuntime(SharedSGLangServingRuntime, FullWeightServingRuntime):
    """Full-weight publication with the shared SGLang GPU handoff."""

    def publish_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        return FullWeightServingRuntime.activate_checkpoint(self, artifact, scenario=scenario)


class ColocatedAdapterServingRuntime(SharedSGLangServingRuntime, AdapterServingRuntime):
    """Existing adapter publication and residency with the shared GPU handoff.

    SGLang retains its adapter registry and buffers across native memory-saver
    calls. The base weights use CPU backup. Adapter allocations outside the
    native managed regions remain resident and need training memory headroom.
    """

    def publish_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        return AdapterServingRuntime.activate_checkpoint(self, artifact, scenario=scenario)


def build_colocated_serving(
    resources: InferenceResources,
    release_format: ReleaseFormat,
    config: Mapping[str, Any],
    *,
    model_path: str,
    environ: Mapping[str, str],
) -> SharedSGLangServingRuntime:
    """Start local or remote SGLang serving on the trainer's borrowed GPU bundles."""
    borrowed = select_inference_resources(resources, config["num_gpus"])
    remote = RemoteInferenceSettings.from_mapping(config["remote"]) if config.get("remote") else None
    options = dict(config["options"])
    options["enable-memory-saver"] = True
    # A trained candidate is not necessarily published. Preserve the incumbent
    # weights while releasing their GPU storage, then restore that same model.
    options["enable-weights-cpu-backup"] = True
    engine_config = serving_engine_config(
        "sglang",
        model_path=(remote.model_path or model_path) if remote is not None else model_path,
        num_gpus=config["num_gpus"],
        tensor_parallel_size=config["tensor_parallel_size"],
        options=options,
        release_format=release_format,
        adapter_capacity=config["adapter_capacity"],
        prefix_sharing=config.get("prefix_sharing", False),
        executor=config["executor"],
        executor_options=config["executor_options"],
        environ=environ,
    )
    engine_config["gpus_per_node"] = borrowed.node_width(config["tensor_parallel_size"])
    service: InferenceService
    handler_options = dict(config["inference_handler_config"])
    if remote is None:
        service = create_inference(engine_config)
    else:
        service = RemoteInferenceService(
            remote,
            engine="sglang",
            engine_config=engine_config,
            num_gpus=config["num_gpus"],
            control_token=remote.token(remote.control_token_env, environ),
        )
        handler_options["request_headers"] = ProviderRequestHeaders(remote.token(remote.inference_token_env, environ))
    try:
        backend = service.backend(service.start(borrowed))
        if not isinstance(backend, MemoryManagedInferenceBackend):
            raise TypeError("colocated serving requires the SGLang memory-control backend")
        handler_factory = config.get("inference_handler_factory") or SGLangInferenceHandler
        handler = handler_factory.from_config(
            backend.inference_url(),
            model_path=model_path,
            timeout_s=config["inference_timeout_s"],
            **handler_options,
        )
        if release_format.layout == "adapter":
            return ColocatedAdapterServingRuntime(
                backend,
                handler,
                adapter_capacity=config["adapter_capacity"],
                service=service,
                model_path=model_path,
                inference_timeout_s=config["inference_timeout_s"],
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
    """SGLang serving built only on an owner's borrowed GPU reservation."""

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
