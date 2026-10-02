"""SGLang inference lifecycle with borrowed GPU reservations."""

from __future__ import annotations

import logging


from reef.inference.process import RayHealthProbe
from reef.inference.sglang.backend import SGLangInferenceBackend
from reef.inference.sglang.config import CONTROL_TIMEOUT_S, SGLangConfig
from reef.inference.sglang.launch import engine_environment
from reef.runtime.deployment import (
    ADAPTER_FILES_PROTOCOL,
    DeploymentResources,
    InferenceConnection,
    InferenceResources,
    InferenceService,
)
from reef.runtime.executor import Executor, ExecutorConfig, WorkerSpec
from reef.runtime.executor.ray import RayExecutor

# Keep the existing wire identifier while removing its implementation dependency.
INFERENCE_PROTOCOL = "slime-sglang-control-v2"


class SGLangInferenceService(InferenceService):
    """Own engines and the control actor; borrow the deployment allocation."""

    connection_protocol = INFERENCE_PROTOCOL
    #: Engines also load PEFT adapter directories, so file-delivering trainers pair with them.
    supported_transfer_protocols = (INFERENCE_PROTOCOL, ADAPTER_FILES_PROTOCOL)

    def __init__(self, config: SGLangConfig) -> None:
        self.config = config
        self._inference: RayExecutor | None = None
        self._started = False
        self._closed = False
        self._probe = RayHealthProbe()

    def start(self, resources: DeploymentResources) -> InferenceConnection:
        if self._started or self._closed:
            raise RuntimeError("inference service can only be started once")
        if not isinstance(resources, InferenceResources) or resources.inference_placement is None:
            raise ValueError("SGLang inference requires its supplied model reservations")
        self._started = True
        self._inference = RayExecutor(
            ExecutorConfig(
                backend=RayExecutor,
                workers=(
                    WorkerSpec(
                        worker_cls="reef.inference.sglang.control:SGLangControl",
                        args=(self.config, resources.inference_placement),
                    ),
                ),
                options={
                    "num_cpus": 1,
                    "num_gpus": 0,
                    "runtime_env": {"env_vars": engine_environment(self.config)},
                },
                launch_timeout_s=CONTROL_TIMEOUT_S,
            )
        )
        return InferenceConnection(self.connection_protocol, RayExecutor.from_workers(self._inference.workers))

    def _control(self, connection: InferenceConnection) -> Executor:
        if connection.protocol != self.connection_protocol:
            raise ValueError(f"incompatible SGLang inference connection: {connection.protocol!r}")
        return connection.control

    def backend(self, connection: InferenceConnection) -> SGLangInferenceBackend:
        """Adapt a compatible borrowed connection for Reef's coordinator."""
        return SGLangInferenceBackend(self._control(connection))

    def prepare_weight_transfer(self, connection: InferenceConnection) -> None:
        """Fence engines and release shared memory before trainer allocation."""
        self._control(connection).rpc(0, "prepare_training_connection", timeout=CONTROL_TIMEOUT_S)

    def _running(self) -> RayExecutor:
        if self._inference is None or self._closed:
            raise RuntimeError("inference service is not running")
        return self._inference

    def check_health(self) -> None:
        self._running().rpc(0, "check_health", timeout=30)

    def poll(self) -> None:
        self._probe.poll(self._running().workers[0], "check_health")

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
