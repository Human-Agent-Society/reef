"""SGLang inference lifecycle with borrowed GPU reservations."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

import ray

from reef.runtime.deployment import DeploymentResources, InferenceConnection
from reef.runtime.executor import ExecutorConfig, WorkerSpec
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.sglang.config import SGLangConfig
from reef.runtime.sglang.launch import engine_environment

# Keep the existing wire identifier while removing its implementation dependency.
INFERENCE_PROTOCOL = "slime-sglang-control-v2"


@runtime_checkable
class SGLangResources(Protocol):
    @property
    def inference_placement(self) -> Any: ...


class RayHealthProbe:
    """Keep one outstanding RPC; a busy actor is not presumed dead."""

    def __init__(self) -> None:
        self.pending: Any = None

    def poll(self, actor: Any, method: str) -> None:
        if self.pending is None:
            self.pending = getattr(actor, method).remote()
        ready, _ = ray.wait([self.pending], timeout=0)
        if not ready:
            return
        pending, self.pending = self.pending, None
        result = ray.get(pending)
        if isinstance(result, dict) and result.get("ok") is False and result.get("recoverable") is not True:
            raise RuntimeError(f"model component failed its health check: {result!r}")


class SGLangInferenceService:
    """Own engines and the control actor; borrow the deployment allocation."""

    connection_protocol = INFERENCE_PROTOCOL

    def __init__(self, config: SGLangConfig) -> None:
        self.config = config
        self._inference: RayExecutor | None = None
        self._started = False
        self._closed = False
        self._probe = RayHealthProbe()

    def start(self, resources: DeploymentResources) -> InferenceConnection:
        if self._started or self._closed:
            raise RuntimeError("inference service can only be started once")
        if not isinstance(resources, SGLangResources) or resources.inference_placement is None:
            raise ValueError("SGLang inference requires its supplied model reservations")
        self._started = True
        self._inference = RayExecutor(
            ExecutorConfig(
                backend=RayExecutor,
                workers=(
                    WorkerSpec(
                        worker_cls="reef.runtime.sglang.control:SGLangControl",
                        args=(self.config, resources.inference_placement),
                    ),
                ),
                options={
                    "num_cpus": 1,
                    "num_gpus": 0,
                    "runtime_env": {"env_vars": engine_environment(self.config)},
                },
                launch_timeout_s=14_400,
            )
        )
        return InferenceConnection(self.connection_protocol, RayExecutor.from_workers(self._inference.workers))

    def check_health(self) -> None:
        if self._inference is None or self._closed:
            raise RuntimeError("inference service is not running")
        self._inference.rpc(0, "check_health", timeout=30)

    def poll(self) -> None:
        if self._inference is None or self._closed:
            raise RuntimeError("inference service is not running")
        self._probe.poll(self._inference.workers[0], "check_health")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._inference is not None:
            try:
                self._inference.rpc(0, "shutdown", timeout=90)
            except Exception:
                # The resource owner confirms retirement of native children.
                logging.getLogger(__name__).exception("Inference shutdown failed; retiring its owned process groups")
            finally:
                self._inference.shutdown()
