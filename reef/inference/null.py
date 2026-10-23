"""An inference component that runs no engine: ``inference.backend: none``.

A deployment selects it to train on data that does not come from its own model,
such as uploaded trajectories (``POST /reef/trajectories``) or imported records
(``POST /reef/records``). The component takes no GPUs and serves no requests.
It still implements the receiver contract, so Reef's coordinator publishes and
commits every training step as it does with engines:

- It accepts only ``reef-no-weight-transfer-v1``. No weights leave the trainer;
  in place of a send, the trainer records the version it holds in the
  component's :class:`ServedVersionRecord`, and the backend reports it.
- Pause, resume, abort, recovery and adapter unloads have no engine to act on.
- Every inference request fails with
  :class:`~reef.core.errors.InferenceUnavailable`, which the HTTP service
  returns as 503.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.errors import InferenceUnavailable
from reef.inference.runtime import EngineRuntimeFactory, ExecutorInferenceRuntime
from reef.runtime.deployment import (
    NO_WEIGHT_TRANSFER_PROTOCOL,
    DeploymentResources,
    InferenceConnection,
    InferenceService,
)
from reef.runtime.executor import Executor, ExecutorConfig, WorkerSpec
from reef.runtime.executor.failure import ExecutorFailedError
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.interfaces import InferenceAdmissionHandle, InferenceBackend, InferenceHandler

#: The response body of every inference request.
UNAVAILABLE_MESSAGE = (
    "this deployment runs no inference engine (inference.backend: none) and serves no inference requests; "
    "serve a published checkpoint from a separate deployment"
)
#: The base URL the coordinator reports. It names no endpoint: every request fails before it would reach one.
NULL_INFERENCE_URL = "none://no-inference-engine"
#: What a new record reports before Reef stamps a version: the value an SGLang engine reports at boot.
UNSTAMPED_RUNTIME_LOAD_ID = "default"
#: Bound on one call to the record.
RECORD_TIMEOUT_S = 30


class ServedVersionRecord:
    """The one runtime load ID that the trainer's weights carry; the receiver end of the no-transfer protocol.

    Values: ``runtime_load_id`` is a non-empty string. It is
    ``UNSTAMPED_RUNTIME_LOAD_ID`` until the first ``set_runtime_load_id``.
    Purpose: the trainer sets it in place of a weight send, and
    :meth:`NullInferenceBackend.initialize_version` sets it at startup.
    :class:`NullInferenceBackend` reads it, so the coordinator verifies a
    publication as it does with engines. It runs in its own executor worker,
    so the trainer and the coordinator reach the same value from their
    processes. A restarted deployment starts a new record.
    """

    def __init__(self) -> None:
        self.runtime_load_id = UNSTAMPED_RUNTIME_LOAD_ID

    def set_runtime_load_id(self, runtime_load_id: str) -> None:
        """Afterwards ``get_runtime_load_ids()`` returns ``[runtime_load_id]``."""
        if not isinstance(runtime_load_id, str) or not runtime_load_id:
            raise ValueError("a runtime load ID must be a non-empty string")
        self.runtime_load_id = runtime_load_id

    def get_runtime_load_ids(self) -> list[str]:
        """One entry, as one engine would report it."""
        return [self.runtime_load_id]

    def check_health(self) -> None:
        """The record has nothing to fail; a reply means its worker is alive."""


class NullInferenceBackend(InferenceBackend):
    """Receiver operations without an engine, for Reef's coordinator.

    Values: ``record`` is a borrowed executor whose rank 0 is a
    :class:`ServedVersionRecord`.
    Purpose: :meth:`NullInferenceService.backend` makes it, and Reef's
    coordinator calls it. ``initialize_version(i)`` sets the record to ``i``;
    ``runtime_load_ids()`` returns the record's one ID. Pause, resume,
    recovery, abort and adapter unloads change nothing: no engine generates,
    and no adapter is ever loaded. The inherited ``load_weight_files``
    refuses, because no files reach this receiver.
    """

    def __init__(self, record: Executor) -> None:
        self.record = record

    def initialize_version(self, runtime_load_id: str) -> None:
        self.record.rpc(0, "set_runtime_load_id", args=(runtime_load_id,), timeout=RECORD_TIMEOUT_S)

    def inference_url(self) -> str:
        return NULL_INFERENCE_URL

    def runtime_load_ids(self) -> Sequence[str]:
        return tuple(self.record.rpc(0, "get_runtime_load_ids", timeout=RECORD_TIMEOUT_S))

    def pause(self) -> None:
        """No engine generates, so nothing pauses."""

    def resume(self) -> None:
        """No engine generates, so nothing resumes."""

    def recover(self) -> None:
        """No engine can fail, so nothing recovers."""

    def abort(self) -> None:
        """No engine runs, so nothing terminates."""

    def unload_adapter(self, name: str) -> None:
        """No adapter is ever loaded, so nothing unloads."""


class NullInferenceService(InferenceService):
    """Run the served-version record and no engine; borrow no GPUs.

    Values: ``record`` is None before ``start``, then the owned Ray executor of
    one :class:`ServedVersionRecord`; ``closed`` becomes true at ``close``.
    Purpose: ``inference.backend: none`` selects it as the deployment's
    inference component. ``start`` puts the record in a Ray actor, which the
    trainer's and the coordinator's processes both reach. It returns a
    connection of protocol ``reef-no-weight-transfer-v1`` whose control
    borrows that actor, so the trainer receives a native session with the
    record as its receiver. ``close`` stops the record.
    """

    connection_protocol = NO_WEIGHT_TRANSFER_PROTOCOL

    def __init__(self) -> None:
        self.record: RayExecutor | None = None
        self.started = False
        self.closed = False

    def start(self, resources: DeploymentResources) -> InferenceConnection:
        """Start the record; ``resources`` holds no inference GPUs to use."""
        if self.started or self.closed:
            raise RuntimeError("inference service can only be started once")
        self.started = True
        self.record = RayExecutor(
            ExecutorConfig(backend=RayExecutor, workers=(WorkerSpec(ServedVersionRecord),), options={"num_cpus": 0})
        )
        return InferenceConnection(self.connection_protocol, RayExecutor.from_workers(self.record.workers))

    def prepare_weight_transfer(self, connection: InferenceConnection) -> None:
        """No engine holds GPU memory, so nothing is released before the trainer starts."""
        self.require_record(connection)

    def backend(self, connection: InferenceConnection) -> InferenceBackend:
        return NullInferenceBackend(self.require_record(connection))

    def check_health(self) -> None:
        self.running().rpc(0, "check_health", timeout=RECORD_TIMEOUT_S)

    def poll(self) -> None:
        failure = self.running().failure
        if failure is not None:
            raise ExecutorFailedError(failure)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.record is not None:
            self.record.shutdown()

    def running(self) -> RayExecutor:
        if self.record is None or self.closed:
            raise RuntimeError("inference service is not running")
        return self.record

    def require_record(self, connection: InferenceConnection) -> Executor:
        if connection.protocol != self.connection_protocol or connection.control is None:
            raise ValueError(f"incompatible inference connection for inference.backend none: {connection.protocol!r}")
        return connection.control


def create_inference(config: Mapping[str, Any]) -> NullInferenceService:
    """Build the service for ``inference.backend: none``; it takes no settings, so ``config`` is empty."""
    if config:
        raise ValueError(f"inference.backend none takes no engine settings; got {', '.join(sorted(config))}")
    return NullInferenceService()


class NullInferenceHandler(InferenceHandler):
    """A request handler that refuses every request with :class:`InferenceUnavailable`.

    It holds no state. :class:`NullInferenceRuntime` uses it as its handler.
    """

    @classmethod
    def from_config(
        cls,
        upstream_url: str,
        *,
        model_path: str,
        timeout_s: float,
        **config: Any,
    ) -> InferenceHandler:
        return cls()

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise InferenceUnavailable(UNAVAILABLE_MESSAGE)


class NullInferenceRuntime(ExecutorInferenceRuntime):
    """The HTTP service's inference runtime for ``inference.backend: none``.

    Publication and commit acknowledgement go to the coordinator as with
    engines. Admission refuses every request at once, before it waits for
    serving.
    """

    default_inference_handler = NullInferenceHandler

    async def acquire_inference(self) -> InferenceAdmissionHandle:
        raise InferenceUnavailable(UNAVAILABLE_MESSAGE)


class NullRuntimeFactory(EngineRuntimeFactory):
    """Connect the HTTP service to a deployment with ``inference.backend: none``."""

    kind = "none"
    engine_name = "none"
    runtime_type = NullInferenceRuntime
